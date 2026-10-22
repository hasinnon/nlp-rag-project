# NLP Course Project — QA Baselines

A baseline question-answering pipeline for the course passage collection.
It supports closed-book generation, BM25 retrieval, dense retrieval, and
hybrid retrieval with Reciprocal Rank Fusion (RRF).

These are initial baselines. Final Phase 1 requirements must be checked
against the course handout.

## Pipeline

Question → ranked passages → token-budgeted context → short answer

All methods share the configured language model. Dense and hybrid retrieval
share the configured embedding model. Gold labels are used only by the
official evaluator, never by the prediction pipeline.

## Setup

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Place the course data in `course_pack_v1/` and the unmodified official
scoring script in `starter/score.py`.

Default models:
- LLM: `Qwen/Qwen2.5-3B-Instruct`
- Embedder: `BAAI/bge-small-en-v1.5`

These are configurable examples, not course-mandated model choices.
Default generation uses 4-bit GPU loading.

## Run

From the repository root:

```bash
python src/run.py --method bm25 --split dev --out runs/bm25_dev
python starter/score.py runs/bm25_dev/predictions.jsonl \
  --pack course_pack_v1 --split dev
```

Other methods: `closed_book`, `dense`, and `hybrid`.

For a retrieval-only smoke test:

```bash
python src/run.py --method bm25 --split dev \
  --limit 10 --retrieval-only --out runs/bm25_smoke
python starter/score.py runs/bm25_smoke/predictions.jsonl \
  --pack course_pack_v1 --split dev --partial
```

Retrieval-only runs intentionally contain empty answers and are not full
QA baselines. `--limit` is allowed only on dev.

## Outputs

Each run produces:

```text
runs/<name>/
├── predictions.jsonl
└── run_config.yaml
```

The run configuration records settings, package versions, input hashes,
prompt hashes, and the execution command. Use a new folder for each run.

## Kaggle

Enable a supported GPU and Internet access for the initial model download.
Attach the course pack as a Kaggle dataset and pass its path with `--pack`.
Write results under `/kaggle/working/`.

## Data and Results

Course data is not included. Check redistribution rules before publishing it.
No measured results are reported until the pipeline has been executed.