import json
import os

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DEFAULT_BASE_MODEL_PATH = os.getenv(
    "BASE_LLM_MODEL",
    "Qwen/Qwen2.5-0.5B",
)
DEFAULT_CPT_ADAPTER_PATH = os.getenv(
    "KUAI_RAND_CPT_ADAPTER_DIR",
    os.path.join("artifacts", "qwen2.5-0.5b-kuairand-cpt-lora"),
)
DEFAULT_SFT_JSONL_PATH = os.getenv(
    "KUAI_RAND_SFT_JSONL",
    os.path.join("artifacts", "kuairand_sft.jsonl"),
)
DEFAULT_OUTPUT_DIR = os.getenv(
    "KUAI_RAND_SFT_OUTPUT_DIR",
    os.path.join("artifacts", "qwen2.5-0.5b-kuairand-sft-lora"),
)


MAX_LENGTH = 384
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LOGGING_STEPS = 10
SAVE_STEPS = 100
HISTORY_SEQ_LENGTH = 10
MIN_HISTORY_LENGTH = 5
SAMPLE_BY_REWARD = True
SAMPLE_SEED = 42


def load_sft_dataset(data_path: str = DEFAULT_SFT_JSONL_PATH):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"SFT jsonl not found: {data_path}")
    dataset_dict = load_dataset("json", data_files={"train": data_path})
    return dataset_dict["train"]


def sample_dataset_by_reward(dataset, seed: int = SAMPLE_SEED):
    if "reward" not in dataset.column_names:
        print("Reward column not found in SFT jsonl. Skip reward-aware sampling.")
        return dataset

    rewards = np.asarray(dataset["reward"], dtype=np.float32)
    rewards = np.clip(rewards, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    sampled_mask = rng.random(rewards.shape[0]) < rewards
    sampled_indices = np.flatnonzero(sampled_mask).tolist()

    if not sampled_indices:
        sampled_indices = [int(np.argmax(rewards))]

    sampled_dataset = dataset.select(sampled_indices)
    print(
        "Reward-aware sampling kept "
        f"{len(sampled_indices)}/{len(dataset)} samples "
        f"({len(sampled_indices) / max(len(dataset), 1):.2%})."
    )
    print(f"Final SFT train dataset size: {len(sampled_dataset)}")
    return sampled_dataset


def build_tokenizer(model_path: str = DEFAULT_CPT_ADAPTER_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_dataset(dataset, tokenizer, max_length):
    def tokenize_fn(batch):
        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for prompt, target in zip(batch["prompt"], batch["target"]):
            prompt_text = prompt.rstrip() + "\n"
            target_text = target.strip() + tokenizer.eos_token

            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
            target_tokens = tokenizer(target_text, add_special_tokens=False)

            input_ids = prompt_tokens["input_ids"] + target_tokens["input_ids"]
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_tokens["input_ids"]) + target_tokens["input_ids"]

            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
                attention_mask = attention_mask[-max_length:]
                labels = labels[-max_length:]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(labels)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }

    remove_columns = [column for column in dataset.column_names if column in {"prompt", "target", "text", "reward"}]
    return dataset.map(tokenize_fn, batched=True, remove_columns=remove_columns)


def build_training_arguments(output_dir: str = DEFAULT_OUTPUT_DIR):
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
        else:
            use_fp16 = True

    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        bf16=use_bf16,
        fp16=use_fp16,
    )


class SFTDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        padded_features = {"input_ids": [], "attention_mask": [], "labels": []}

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            padded_features["input_ids"].append(
                feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_length
            )
            padded_features["attention_mask"].append(feature["attention_mask"] + [0] * pad_length)
            padded_features["labels"].append(feature["labels"] + [-100] * pad_length)

        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in padded_features.items()
        }


def build_model(
    base_model_path: str = DEFAULT_BASE_MODEL_PATH,
    cpt_adapter_path: str = DEFAULT_CPT_ADAPTER_PATH,
):
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    base_model.config.use_cache = False

    model = PeftModel.from_pretrained(
        base_model,
        cpt_adapter_path,
        is_trainable=True,
    )
    model.print_trainable_parameters()
    return model


def run_sft(
    base_model_path: str = DEFAULT_BASE_MODEL_PATH,
    cpt_adapter_path: str = DEFAULT_CPT_ADAPTER_PATH,
    data_path: str = DEFAULT_SFT_JSONL_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_length: int = MAX_LENGTH,
    sample_by_reward: bool = SAMPLE_BY_REWARD,
    sample_seed: int = SAMPLE_SEED,
):
    dataset = load_sft_dataset(data_path=data_path)
    if sample_by_reward:
        dataset = sample_dataset_by_reward(dataset, seed=sample_seed)
    tokenizer = build_tokenizer(model_path=cpt_adapter_path)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer=tokenizer, max_length=max_length)
    model = build_model(base_model_path=base_model_path, cpt_adapter_path=cpt_adapter_path)
    training_args = build_training_arguments(output_dir=output_dir)
    data_collator = SFTDataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("SFT training started...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    run_sft()
