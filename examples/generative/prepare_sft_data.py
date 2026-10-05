import json
import os

from data.dataset_kuairand import prepare_kuairand_llm_post_training_data


DEFAULT_SFT_JSONL_PATH = os.path.join("artifacts", "kuairand_sft.jsonl")


def build_or_load_kuairand_sft_jsonl(
    output_path: str = DEFAULT_SFT_JSONL_PATH,
    history_seq_length: int = 5,
    min_history_length: int = 3,
    force_rebuild: bool = False,
) -> str:
    """
    Reuse cached SFT jsonl when it already exists; otherwise generate it from
    the KuaiRand LLM post-training dataset builder.
    """
    # if os.path.exists(output_path) and not force_rebuild:
    #     print(f"Using cached SFT jsonl: {output_path}")
    #     return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sft_data = prepare_kuairand_llm_post_training_data(
        history_seq_length=history_seq_length,
        min_history_length=min_history_length,
    )

    with open(output_path, "w", encoding="utf-8") as file:
        for row in sft_data.itertuples(index=False):
            file.write(
                json.dumps(
                    {
                        "prompt": row.prompt,
                        "target": row.target,
                        "text": row.text,
                        "reward": row.reward,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Saved SFT jsonl to: {output_path}")
    print(f"Num samples: {sft_data.shape[0]}")
    return output_path


if __name__ == "__main__":
    build_or_load_kuairand_sft_jsonl()
