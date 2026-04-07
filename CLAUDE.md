# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

RAG Agent that answers Korean legal multiple-choice questions (A/B/C/D) via a FastAPI inference server, packaged in Docker.

- **Generation model**: `gpt-4o-mini`  (temp=0.0)
- **Embedding model**: `text-embedding-3-small` (dim=1536)
- **API contract**: `POST /` with `{ "query": str }` → `{ "answer": "A" | "B" | "C" | "D" }`

## Commands

All commands run from the **project root**:

```bash
make lint        # ruff check --fix (exits non-zero if fixes applied)
make format      # ruff format --check
make type-check  # pyright (strict mode)

# Run the full benchmark once (uses configs/config.yaml defaults)
python test/benchmark.py

# Run with Hydra overrides (any combination of axes)
python test/benchmark.py serialization=narrative retrieval=mmr prompt=few_shot_envelope

# Run the OAAT sweep (baseline + one-axis-at-a-time, 14 experiments total)
python test/oaat_sweep.py --yes

# Run only baseline
python test/oaat_sweep.py --mode baseline_only --yes

# Run the FastAPI server locally
uv run python -m src.main

# Run in Docker
docker compose up --build
```

`.env` at project root is auto-loaded by the Makefile. Set `OPENAI_API_KEY` there.

## Architecture

### Data Flow

```
data/train.csv
  → DataLoader.load_csv()     # List[Dict[str, str]]
  → Chunker.chunk_data()      # List[{"chunk_id", "content_dict"}]
  → Embedder.preprocess()     # serialization strategy → str (or List[str] for dual)
  → Embedder.embed_batch()    # OpenAI text-embedding-3-small → float[1536]
  → Retriever.add_documents() # FAISS IndexFlatIP (L2-normalized = cosine sim)

data/dev.csv (query)
  → Embedder.embed()          # single query vector
  → Retriever.search()        # retrieval strategy → List[Dict] (top-k contexts)
  → PromptBuilder.build_prompt() # prompt strategy → PromptResult(system, user)
  → OpenAIService.generate_text() # gpt-4o-mini → "A"/"B"/"C"/"D"
```

### Three Experimental Axes (Hydra config groups)

Each axis is a pluggable strategy selected via Hydra config groups under `configs/`:

| Axis | Config group | Available strategies |
|---|---|---|
| **Serialization** (embedding text format) | `configs/serialization/` | `raw`, `kv_pairs` (baseline), `narrative` (→`narrativized_lite`), `weighted` (→`field_weighted_kv`), `dual` (→`dual_representation`), `kv_pairs_no_category` (ablation) |
| **Retrieval** (FAISS search strategy) | `configs/retrieval/` | `top_k` (baseline), `score_threshold`, `mmr`, `top_k_category_filter`, `hybrid` (미구현) |
| **Prompt** (LLM context assembly) | `configs/prompt/` | `raw_stuffing` (baseline), `labeled_context`, `structured_context`, `few_shot_envelope` |
| **Query Representation** (query embedding text) | `configs/query_representation/` | `question_only` (baseline), `question_plus_choices` |

All serialization strategies accept `exclude_fields: list[str]` from config (default: `["answer", "Human Accuracy"]`). `category_filter` in retrieval configs enables Law/Criminal Law post-filter via `metadata_filter` passed from `benchmark.py`.


The baseline is defined in `configs/config.yaml` (`serialization=kv_pairs`, `retrieval=top_k`, `prompt=raw_stuffing`).

### Registry Pattern

All three axes use the same pattern — decorator-based auto-registration, no elif chains:

**Adding a new serialization strategy:**
1. Create `src/agent/embedder/strategy_<name>.py`
2. Decorate the class: `@register_preprocess("name")`
3. Create `configs/serialization/<name>.yaml` with at least `method: <name>`
4. Import the module in `src/agent/embedder/__init__.py`

Same pattern for retrieval (`@register_strategy`) and prompt (`@register_prompt`).

The registries:
- `src/agent/embedder/_registry.py` — `_PREPROCESS_REGISTRY`
- `src/agent/retriever/_registry.py` — `RETRIEVAL_STRATEGIES`
- `src/agent/prompt_builder/_registry.py` — `_PROMPT_REGISTRY`

### Hydra Config

`test/benchmark.py` is the `@hydra.main` entrypoint. Hydra changes CWD to `hydra.run.dir` (default: `outputs/{date}/{time}/`). The benchmark writes its CSV to that CWD.

To control where a run's output lands:
```bash
python test/benchmark.py hydra.run.dir=outputs/my_run
```

### OAAT Sweep (`test/oaat_sweep.py`)

Runs baseline + each axis variant independently (holding the other two axes at baseline). Results aggregated to `outputs/oaat_sweep/oaat_summary_{ts}.csv` and `.json`. Each run's Hydra output goes under `outputs/oaat_runs/<name>/`.

### FastAPI Server

`src/main.py` → `src/app/server.py` (create_app factory) → `src/app/router/inference_router.py` → `src/app/controller/inference_controller.py` → `src/app/service/openai_service.py`

The server's `POST /` endpoint currently delegates to `OpenAIService.infer()`. The RAG pipeline (DataLoader → Embedder → Retriever → PromptBuilder) needs to be wired into the server lifespan and controller — currently it only exists in `test/benchmark.py`.

### PYTHONPATH Note

`src/` is not a package root by default. `test/benchmark.py` adds it to `sys.path` manually. The Dockerfile sets `PYTHONPATH=/app` so `from app.server import ...` and `from config.config import ...` resolve correctly.

## Data Schema

Both `data/train.csv` and `data/dev.csv`:

| Field | Notes |
|---|---|
| `question` | Korean legal MCQ body |
| `A`, `B`, `C`, `D` | Answer choices |
| `answer` | Integer 1–4 → mapped to A–D via `_LABEL_MAP` in benchmark.py |
| `Category` | Always `Law` |
| `Human Accuracy` | Float 0–1, excluded from embedding by serialization strategies |

## Toolchain

- **Python 3.13**, managed via `uv`
- **ruff** — linting and formatting; `ruff.toml` at project root; rules: E, F, B, SIM, UP, I; ignores E501
- **pyright** — strict type checking
