# Hybrid Recommendation System Pipeline

This project is an end-to-end recommendation system pipeline built around
classic embedding retrieval, deep ranking models, generative retrieval, and
adaptive diversity reranking. It is designed as a portfolio-quality project:
the codebase separates data processing, model training, retrieval serving,
reranking, and LLM-based candidate generation into clear modules.

## Highlights

- **Multi-stage recommender pipeline**: feature engineering, two-tower retrieval,
  DIN/TransAct ranking, FAISS or DuckDB candidate retrieval, and FastAPI serving.
- **Generative retrieval channel**: item embeddings are compressed into semantic
  IDs with RQ-VAE, then converted into CPT/SFT datasets for causal language
  model fine-tuning.
- **Personalized diversity reranking**: DPP reranking is combined with Linear
  Thompson Sampling to adapt the relevance-diversity tradeoff per user context.
- **Two public datasets**: MovieLens is used for movie recommendation examples;
  KuaiRand-1K is used for short-video ranking, generative retrieval, and offline
  replay experiments.

## Architecture

```text
Raw datasets
  -> feature engineering
  -> retrieval training / ranking training
  -> item embedding export
  -> FAISS or DuckDB candidate retrieval
  -> DIN / TransAct scoring
  -> DPP + Thompson Sampling reranking
  -> FastAPI recommendation service

Generative retrieval branch
  -> two-tower item embeddings
  -> RQ-VAE semantic IDs
  -> CPT and SFT jsonl generation
  -> LoRA fine-tuning
  -> constrained decoding utilities
```

## Repository Layout

```text
configs/        Shared model and runtime configuration.
data/           Dataset downloaders and dataset builders.
feature/        MovieLens and KuaiRand feature engineering.
model/          Two-tower, DIN, DCNv2, TransAct, and RQ-VAE models.
trainer/        Training loops and embedding export utilities.
server/         Retrieval engines, model manager, and FastAPI serving.
rerank/         DPP reranking, reward functions, and Linear Thompson Sampling.
generative/     Constrained decoding utilities for generative retrieval.
llm_train/      CPT/SFT LoRA training scripts for the generative branch.
examples/       Runnable pipeline entry points.
experiments/    Optional API experiments that are not required by the core flow.
tests/          Unit tests and small fixtures.
```


## Setup

This project targets Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create local environment settings:

```bash
copy .env.template .env
```

Important environment variables:

- `OUTPUT_MODEL_PATH`: output directory for checkpoints, embeddings, indexes,
  and generated jsonl files. The default is `./tmp` in `.env.template`.
- `TMDB_API_KEY`: optional, only needed for MovieLens metadata crawling.
- `DASHSCOPE_API_KEY`: optional, only needed for DashScope embedding API
  experiments.
- `BASE_LLM_MODEL`: base causal LM used by `llm_train/`, defaulting to
  `Qwen/Qwen2.5-0.5B`.

## Example Workflows

Run commands from the repository root with `python -m ...`.

### MovieLens Ranking and Serving

Train a small DIN ranking model:

```bash
python -m examples.training.train_movielens_din
```

Build MovieLens item embeddings and a FAISS index:

```bash
python -m examples.movielens.build_embeddings
python -m examples.movielens.build_faiss_index
```

Start the recommendation API:

```bash
python -m server.inference_engine
```

Example request:

```bash
curl -X POST http://localhost:8000/recommend/ ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\": 1}"
```

### KuaiRand Retrieval and Reranking

Train the KuaiRand two-tower retrieval model and export item embeddings:

```bash
python -m examples.kuairand.build_embeddings
```

Build a FAISS index:

```bash
python -m examples.kuairand.build_faiss_index
```

Run offline replay for personalized diversity reranking:

```bash
python -m examples.kuairand.rerank_replay
```

The replay script learns which DPP `alpha` value to use from a compact user
context vector and updates the arm posterior with observed replay rewards.

### Generative Retrieval Branch

Train RQ-VAE on exported item embeddings and write semantic IDs:

```bash
python -m examples.kuairand.train_rqvae_semantic_ids
python -m examples.kuairand.export_trie_semantic_codes
```

Prepare CPT and SFT data:

```bash
python -m examples.generative.prepare_cpt_data
python -m examples.generative.prepare_sft_data
```

Fine-tune the language model with LoRA:

```bash
python -m llm_train.cpt
python -m llm_train.sft
```

Run constrained semantic-ID inference:

```bash
python -m llm_train.llm_inference --sample_index 0 --beam_size 10
```

The generated data teaches the model to map a user's history and context to the
next item's semantic ID. The `generative/` package contains constrained decoding
helpers that can restrict generated outputs to valid semantic ID sequences.

## Testing

Run the offline fixture-based unit tests:

```bash
pytest tests/test_kuairand_features.py
```
