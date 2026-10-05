import json
import os

import numpy as np

from configs.model import OUTPUT_MODEL_PATH

# Export semantic ID codes into matrix/list formats for vectorized prefix-trie decoding.

DEFAULT_VIDEO_IDS_PATH = os.path.join(OUTPUT_MODEL_PATH, "kuairand_rqvae_video_ids.npy")
DEFAULT_CODES_PATH = os.path.join(OUTPUT_MODEL_PATH, "kuairand_rqvae_codes.npy")
DEFAULT_SEMANTIC_IDS_PATH = os.path.join(OUTPUT_MODEL_PATH, "kuairand_rqvae_semantic_ids.npy")

DEFAULT_TRIE_MATRIX_PATH = os.path.join(OUTPUT_MODEL_PATH, "kuairand_rqvae_trie_codes.npy")
DEFAULT_TRIE_LISTS_PATH = os.path.join(OUTPUT_MODEL_PATH, "kuairand_rqvae_trie_code_lists.json")
DEFAULT_VIDEO_TO_CODES_PATH = os.path.join(OUTPUT_MODEL_PATH, "kuairand_video_to_trie_codes.json")


def _load_required_array(path: str, allow_pickle: bool = False) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return np.load(path, allow_pickle=allow_pickle)


def _parse_semantic_id(semantic_id: str) -> list[int]:
    parts = str(semantic_id).strip().split("-")
    if not parts or any(part == "" for part in parts):
        raise ValueError(f"Invalid semantic id: {semantic_id}")
    return [int(part) for part in parts]


def load_rqvae_semantic_mapping(
    video_ids_path: str = DEFAULT_VIDEO_IDS_PATH,
    codes_path: str = DEFAULT_CODES_PATH,
    semantic_ids_path: str = DEFAULT_SEMANTIC_IDS_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    video_ids = _load_required_array(video_ids_path)
    codes = _load_required_array(codes_path)
    semantic_ids = _load_required_array(semantic_ids_path, allow_pickle=True)

    if video_ids.shape[0] != codes.shape[0] or video_ids.shape[0] != semantic_ids.shape[0]:
        raise ValueError(
            "RQVAE export files are misaligned: "
            f"video_ids={video_ids.shape}, codes={codes.shape}, semantic_ids={semantic_ids.shape}"
        )
    if codes.ndim != 2:
        raise ValueError(f"Expected 2D code matrix, got shape {codes.shape}")

    return video_ids, codes.astype(np.int32), semantic_ids


def build_trie_code_matrix(codes: np.ndarray, semantic_ids: np.ndarray) -> np.ndarray:
    parsed_semantic_ids = np.asarray([_parse_semantic_id(value) for value in semantic_ids], dtype=np.int32)
    if parsed_semantic_ids.shape != codes.shape:
        raise ValueError(
            "Parsed semantic ids do not match exported code matrix: "
            f"parsed={parsed_semantic_ids.shape}, codes={codes.shape}"
        )
    if not np.array_equal(parsed_semantic_ids, codes):
        raise ValueError("Semantic id strings and RQVAE code matrix are inconsistent")
    return parsed_semantic_ids


def export_trie_ready_semantic_codes(
    video_ids_path: str = DEFAULT_VIDEO_IDS_PATH,
    codes_path: str = DEFAULT_CODES_PATH,
    semantic_ids_path: str = DEFAULT_SEMANTIC_IDS_PATH,
    trie_matrix_path: str = DEFAULT_TRIE_MATRIX_PATH,
    trie_lists_path: str = DEFAULT_TRIE_LISTS_PATH,
    video_to_codes_path: str = DEFAULT_VIDEO_TO_CODES_PATH,
) -> tuple[str, str, str]:
    video_ids, codes, semantic_ids = load_rqvae_semantic_mapping(
        video_ids_path=video_ids_path,
        codes_path=codes_path,
        semantic_ids_path=semantic_ids_path,
    )
    trie_code_matrix = build_trie_code_matrix(codes=codes, semantic_ids=semantic_ids)

    code_lists = trie_code_matrix.tolist()
    video_to_codes = {
        str(int(video_id)): code_list
        for video_id, code_list in zip(video_ids.tolist(), code_lists)
    }

    np.save(trie_matrix_path, trie_code_matrix)

    with open(trie_lists_path, "w", encoding="utf-8") as file:
        json.dump(code_lists, file, ensure_ascii=False)

    with open(video_to_codes_path, "w", encoding="utf-8") as file:
        json.dump(video_to_codes, file, ensure_ascii=False)

    print(f"Loaded {video_ids.shape[0]} aligned video ids")
    print(f"Semantic code length: {trie_code_matrix.shape[1]}")
    print(f"Saved trie code matrix to: {trie_matrix_path}")
    print(f"Saved trie code lists to: {trie_lists_path}")
    print(f"Saved video-to-code mapping to: {video_to_codes_path}")
    return trie_matrix_path, trie_lists_path, video_to_codes_path


if __name__ == "__main__":
    export_trie_ready_semantic_codes()
