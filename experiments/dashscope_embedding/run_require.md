# DashScope Embedding Experiment Notes

This directory contains optional experiments for generating MovieLens text
embeddings through a DashScope OpenAI-compatible endpoint.

## Environment

```bash
conda create -n recsys python=3.12
conda activate recsys
pip install -r requirements.txt
```

Create a local `.env` file from `.env.template` and set `DASHSCOPE_API_KEY`.

## Retrieval Modes

DuckDB retrieval:

```bash
python -m server.inference_engine
```

FAISS retrieval:

```bash
python -m server.ebr_server
```

The active retrieval backend is controlled by
`enable_duckdb_retrieval_engine` in the retrieval configuration.
