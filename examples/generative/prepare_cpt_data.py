import json
import os

from data.dataset_kuairand import prepare_kuairand_cpt_data


DEFAULT_CPT_JSONL_PATH = os.path.join("artifacts/kuairand_cpt.jsonl")


def build_or_load_kuairand_cpt_jsonl(
    output_path: str = DEFAULT_CPT_JSONL_PATH,
    history_seq_length: int = 7,  # upper bound
    min_history_length: int = 4,  # lower bound
    force_rebuild: bool = False,
) -> str:
    """
    Reuse cached CPT jsonl when it already exists; otherwise generate it from
    the KuaiRand CPT dataset builder.
    """
    # if os.path.exists(output_path) and not force_rebuild:
    #     print(f"Using cached CPT jsonl: {output_path}")
    #     return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cpt_data = prepare_kuairand_cpt_data(
        history_seq_length=history_seq_length,
        min_history_length=min_history_length,
    )

    with open(output_path, "w", encoding="utf-8") as file:
        for row in cpt_data.itertuples(index=False):
            file.write(json.dumps({"text": row.text}, ensure_ascii=False) + "\n")

    print(f"Saved CPT jsonl to: {output_path}")
    print(f"Num samples: {cpt_data.shape[0]}")  # There are "Num samples" number of users satisfying the condition
    return output_path


if __name__ == "__main__":
    build_or_load_kuairand_cpt_jsonl()
