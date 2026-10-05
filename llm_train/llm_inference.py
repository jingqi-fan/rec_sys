import argparse
import json
import os
import random

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from generative.csr_utils import build_static_index
from generative.decoding_pt import _gather_beams


DEFAULT_BASE_MODEL_PATH = os.getenv(
    "BASE_LLM_MODEL",
    "Qwen/Qwen2.5-0.5B",
)
DEFAULT_SFT_ADAPTER_PATH = os.getenv(
    "KUAI_RAND_SFT_OUTPUT_DIR",
    os.path.join("artifacts", "qwen2.5-0.5b-kuairand-sft-lora"),
)
DEFAULT_SFT_JSONL_PATH = os.getenv("KUAI_RAND_SFT_JSONL", os.path.join("artifacts", "kuairand_sft.jsonl"))
DEFAULT_TRIE_CODES_PATH = os.getenv(
    "KUAI_RAND_TRIE_CODES",
    os.path.join("artifacts", "kuairand_rqvae_trie_codes.npy"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run constrained inference for the KuaiRand SFT LoRA model.")
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--sft_adapter_path", type=str, default=DEFAULT_SFT_ADAPTER_PATH)
    parser.add_argument("--data_path", type=str, default=DEFAULT_SFT_JSONL_PATH)
    parser.add_argument("--trie_codes_path", type=str, default=DEFAULT_TRIE_CODES_PATH)
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Directly provide an inference prompt. If omitted, a sample prompt is loaded from --data_path.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Which sample to load from the jsonl file. Ignored when --random_sample is enabled.",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Randomly choose one prompt from the jsonl file.",
    )
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument(
        "--tokens_per_beam",
        type=int,
        default=10,
        help="Reserved for compatibility with STATIC terminology. Current scorer expands all valid children.",
    )
    parser.add_argument(
        "--dense_lookup_layers",
        type=int,
        default=2,
        help="Must be smaller than semantic id length. For current KuaiRand RQ-VAE ids, 2 is the natural choice.",
    )
    parser.add_argument("--code_vocab_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_model(base_model_path: str, sft_adapter_path: str):
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    model = PeftModel.from_pretrained(model, sft_adapter_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def load_prompt_from_jsonl(data_path: str, sample_index: int = 0, random_sample: bool = False):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"SFT jsonl not found: {data_path}")

    samples = []
    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    if not samples:
        raise ValueError(f"No valid samples found in jsonl: {data_path}")

    if random_sample:
        chosen_index = random.randrange(len(samples))
    else:
        if sample_index < 0 or sample_index >= len(samples):
            raise IndexError(
                f"sample_index {sample_index} is out of range. Valid range: [0, {len(samples) - 1}]"
            )
        chosen_index = sample_index

    sample = samples[chosen_index]
    return sample["prompt"], sample.get("target"), chosen_index


def load_trie_codes(trie_codes_path: str) -> np.ndarray:
    if not os.path.exists(trie_codes_path):
        raise FileNotFoundError(f"Trie code file not found: {trie_codes_path}")

    trie_codes = np.load(trie_codes_path)
    if trie_codes.ndim != 2:
        raise ValueError(f"Expected 2D trie code matrix, got shape {trie_codes.shape}")
    if trie_codes.shape[0] == 0:
        raise ValueError("Trie code matrix is empty")

    sort_order = np.lexsort([trie_codes[:, i] for i in range(trie_codes.shape[1] - 1, -1, -1)])
    return trie_codes[sort_order].astype(np.int32)


def build_static_components(trie_codes: np.ndarray, vocab_size: int, dense_lookup_layers: int):
    semantic_length = int(trie_codes.shape[1])
    if dense_lookup_layers >= semantic_length:
        raise ValueError(
            "dense_lookup_layers must be smaller than semantic id length: "
            f"dense_lookup_layers={dense_lookup_layers}, semantic_length={semantic_length}"
        )

    max_code = int(trie_codes.max())
    if max_code >= vocab_size:
        raise ValueError(
            f"code_vocab_size={vocab_size} is too small for the trie matrix max code {max_code}"
        )

    packed_csr, indptr, layer_max_branches, start_mask, dense_mask, dense_states = build_static_index(
        trie_codes,
        vocab_size=vocab_size,
        dense_lookup_layers=dense_lookup_layers,
    )
    return {
        "semantic_length": semantic_length,
        "packed_csr": packed_csr,
        "indptr": indptr,
        "layer_max_branches": layer_max_branches,
        "start_mask": start_mask,
        "dense_mask": dense_mask,
        "dense_states": dense_states,
        "dense_lookup_layers": dense_lookup_layers,
        "vocab_size": vocab_size,
    }


def get_allowed_children(step: int, prefix_codes: list[int], current_state: int, static_index: dict):
    dense_lookup_layers = static_index["dense_lookup_layers"]

    if step == 0:
        allowed_codes = np.flatnonzero(static_index["start_mask"]).astype(np.int32)
        next_states = allowed_codes + 1
        return allowed_codes, next_states

    if step < dense_lookup_layers:
        dense_mask = static_index["dense_mask"]
        dense_states = static_index["dense_states"]

        if step == 1:
            parent_code = prefix_codes[0]
            allowed_codes = np.flatnonzero(dense_mask[parent_code]).astype(np.int32)
            next_states = dense_states[parent_code, allowed_codes].astype(np.int32)
            return allowed_codes, next_states

        raise NotImplementedError(
            "Current adapter supports dense_lookup_layers <= 2. "
            f"Received dense_lookup_layers={dense_lookup_layers}."
        )

    indptr = static_index["indptr"]
    packed_csr = static_index["packed_csr"]
    start = int(indptr[current_state])
    end = int(indptr[current_state + 1])
    transitions = packed_csr[start:end]

    allowed_codes = transitions[:, 0].astype(np.int32)
    next_states = transitions[:, 1].astype(np.int32)
    valid_mask = allowed_codes < static_index["vocab_size"]
    return allowed_codes[valid_mask], next_states[valid_mask]


def build_prefix_text(prompt: str, generated_codes: list[int]) -> str:
    prompt_text = prompt.rstrip() + "\n"
    if not generated_codes:
        return prompt_text
    return prompt_text + "-".join(str(code) for code in generated_codes)


def score_candidate_fragment(
    model,
    tokenizer,
    device,
    prompt: str,
    generated_codes: list[int],
    candidate_code: int,
    cache: dict[tuple[str, str], float],
) -> float:
    prefix_text = build_prefix_text(prompt, generated_codes)
    fragment_text = str(candidate_code) if not generated_codes else f"-{candidate_code}"
    cache_key = (prefix_text, fragment_text)
    if cache_key in cache:
        return cache[cache_key]

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    fragment_ids = tokenizer(fragment_text, add_special_tokens=False)["input_ids"]
    input_ids = prefix_ids + fragment_ids

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_tensor).logits

    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_tensor[:, 1:]
    log_probs = torch.log_softmax(shifted_logits, dim=-1)

    prefix_length = len(prefix_ids)
    fragment_target = shifted_labels[:, prefix_length - 1: prefix_length - 1 + len(fragment_ids)]
    fragment_log_probs = log_probs[:, prefix_length - 1: prefix_length - 1 + len(fragment_ids), :]
    token_log_probs = fragment_log_probs.gather(dim=-1, index=fragment_target.unsqueeze(-1)).squeeze(-1)
    score = float(token_log_probs.sum().item())
    cache[cache_key] = score
    return score


def constrained_beam_search(
    model,
    tokenizer,
    device,
    prompt: str,
    static_index: dict,
    beam_size: int,
):
    semantic_length = static_index["semantic_length"]
    code_buffer = torch.full((1, 1, semantic_length), -1, dtype=torch.long, device=device)
    state_buffer = torch.zeros((1, 1), dtype=torch.long, device=device)
    score_buffer = torch.zeros((1, 1), dtype=torch.float32, device=device)
    score_cache = {}

    for step in range(semantic_length):
        candidate_scores = []
        candidate_codes = []
        candidate_states = []
        candidate_parents = []

        current_beam_size = code_buffer.shape[1]
        for beam_idx in range(current_beam_size):
            prefix_codes = code_buffer[0, beam_idx, :step].tolist()
            prefix_codes = [int(code) for code in prefix_codes if int(code) >= 0]
            current_state = int(state_buffer[0, beam_idx].item())
            allowed_codes, next_states = get_allowed_children(
                step=step,
                prefix_codes=prefix_codes,
                current_state=current_state,
                static_index=static_index,
            )

            for code, next_state in zip(allowed_codes.tolist(), next_states.tolist()):
                delta_score = score_candidate_fragment(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt,
                    generated_codes=prefix_codes,
                    candidate_code=int(code),
                    cache=score_cache,
                )
                total_score = float(score_buffer[0, beam_idx].item()) + delta_score
                candidate_scores.append(total_score)
                candidate_codes.append(int(code))
                candidate_states.append(int(next_state))
                candidate_parents.append(beam_idx)

        if not candidate_scores:
            raise RuntimeError(f"No valid candidates remained at decoding step {step}")

        topk = min(beam_size, len(candidate_scores))
        flat_scores = torch.tensor(candidate_scores, dtype=torch.float32, device=device).view(1, -1)
        parent_tensor = torch.tensor(candidate_parents, dtype=torch.long, device=device).view(1, -1)
        code_tensor = torch.tensor(candidate_codes, dtype=torch.long, device=device).view(1, -1)
        state_tensor = torch.tensor(candidate_states, dtype=torch.long, device=device).view(1, -1)

        top_scores, top_candidate_indices = torch.topk(flat_scores, topk, dim=-1)
        top_parent_indices = _gather_beams(parent_tensor, top_candidate_indices)

        code_buffer = _gather_beams(code_buffer, top_parent_indices)
        state_buffer = _gather_beams(state_tensor, top_candidate_indices)
        selected_codes = _gather_beams(code_tensor, top_candidate_indices)
        code_buffer[:, :, step] = selected_codes
        score_buffer = top_scores

    final_codes = code_buffer[0].tolist()
    final_scores = score_buffer[0].tolist()
    results = []
    seen_sids = set()
    for codes, score in sorted(zip(final_codes, final_scores), key=lambda item: item[1], reverse=True):
        sid = "-".join(str(int(code)) for code in codes)
        if sid in seen_sids:
            continue
        seen_sids.add(sid)
        results.append({"sid": sid, "codes": [int(code) for code in codes], "score": float(score)})
    return results


def run_unconstrained_generation(model, tokenizer, device, prompt: str, max_new_tokens: int = 32):
    prompt_text = prompt.rstrip() + "\n"
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return generated_text


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = build_tokenizer(args.sft_adapter_path)
    model, device = build_model(
        base_model_path=args.base_model_path,
        sft_adapter_path=args.sft_adapter_path,
    )

    if args.prompt is not None:
        prompt = args.prompt
        ground_truth = None
        sample_index = None
    else:
        prompt, ground_truth, sample_index = load_prompt_from_jsonl(
            data_path=args.data_path,
            sample_index=args.sample_index,
            random_sample=args.random_sample,
        )

    trie_codes = load_trie_codes(args.trie_codes_path)
    static_index = build_static_components(
        trie_codes=trie_codes,
        vocab_size=args.code_vocab_size,
        dense_lookup_layers=args.dense_lookup_layers,
    )

    topk_results = constrained_beam_search(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        static_index=static_index,
        beam_size=args.beam_size,
    )
    unconstrained_text = run_unconstrained_generation(model=model, tokenizer=tokenizer, device=device, prompt=prompt)

    if sample_index is not None:
        print(f"Loaded sample index: {sample_index}")
    print("===== Prompt =====")
    print(prompt)
    print()
    print("===== Constrained TopK =====")
    for rank, result in enumerate(topk_results, start=1):
        print(f"{rank}. sid={result['sid']}, score={result['score']:.4f}, codes={result['codes']}")

    print()
    print("===== Unconstrained Output =====")
    print(unconstrained_text)

    if ground_truth is not None:
        predicted_sid = topk_results[0]["sid"] if topk_results else None
        print()
        print("===== Ground Truth =====")
        print(ground_truth)
        print()
        print(f"Top1 exact match: {predicted_sid == ground_truth}")

    print()
    print("===== Compatibility Note =====")
    print(
        "This script uses csr_utils.build_static_index() and decoding_pt._gather_beams(). "
        "It does not call decoding_pt.sparse_transition_torch() directly because that function "
        "assumes the model outputs logits over the semantic-code vocabulary itself, while the "
        "current Qwen SFT model outputs tokenizer-vocabulary logits over text."
    )


if __name__ == "__main__":
    main()
