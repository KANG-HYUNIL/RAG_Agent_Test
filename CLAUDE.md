# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build a RAG Agent system that answers Korean legal multiple-choice questions (A/B/C/D) via a FastAPI inference server, packaged in Docker.

- **Generation model**: `gpt-4o-mini`
- **Embedding model**: `text-embedding-3-small`
- **API contract**: `POST /` with `{ "query": str }` → `{ "answer": "A" | "B" | "C" | "D" }`

## Commands

All `make` commands must be run from `ai-assigment+/ai-assignment/`:

```bash
make lint        # ruff check --fix (fails if fixes applied)
make format      # ruff format --check
make type-check  # pyright (strict mode)
```

Running the server (once implemented):

```bash
docker compose up --build
```

## Toolchain

- **Python 3.13**, managed via `uv`
- **ruff** — linting and formatting (config: `ai-assigment+/ai-assignment/ruff.toml`); rules: E, F, B, SIM, UP, I; ignores E501
- **pyright** — strict type checking (`typeCheckingMode: "strict"`)
- `.env` file is auto-loaded by the Makefile if present (use this for `OPENAI_API_KEY`)

## Architecture

```
src/
  index.py          # entrypoint (server startup)
  app/
    server.py       # FastAPI app definition
ai-assigment+/ai-assignment/
  data/
    train.csv       # RAG knowledge base (Korean law Q&A with answers)
    dev.csv         # evaluation set (same schema, used to measure accuracy)
```

### Data Schema

Both CSVs share: `question, answer, A, B, C, D, Category, Human Accuracy`

- `answer` is an integer index (1–4) mapping to A/B/C/D
- `Category` is always `Law`
- `Human Accuracy` is a float baseline per question

### RAG Pipeline (to be built)

1. **Indexing**: embed `train.csv` questions+options using `text-embedding-3-small`, store in a vector store
2. **Retrieval**: at inference time, embed the incoming query and retrieve top-k similar train examples
3. **Generation**: pass retrieved examples as few-shot context to `gpt-4o-mini`, prompt it to output exactly one of A/B/C/D

### Evaluation

Run inference over `dev.csv` by calling the running server, compare returned answers against the `answer` column (converted: 1→A, 2→B, 3→C, 4→D), report accuracy.
