import argparse
import os

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot.")
    parser.add_argument(
        "--repo_id",
        default=os.getenv("BASE_LLM_MODEL", "Qwen/Qwen2.5-0.5B"),
        help="Hugging Face repository id to download.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = snapshot_download(repo_id=args.repo_id)
    print("Model downloaded to:", path)
