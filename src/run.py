"""Phase 1 runner: closed-book, oracle-context and dense RAG systems.

Example:
    python src/run.py --pack course_pack_v1 --method dense --split dev --out runs/dense_dev
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import random
import sys
from collections import defaultdict
from datetime import date
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("closed_book", "oracle_context", "dense")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def read_jsonl(path):
    """Read non-empty JSONL lines."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def file_hash(path):
    """Short SHA-256 hash of a file (recorded in run_config.yaml)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def set_seed(seed):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def package_versions():
    """Versions of the main packages, for run_config.yaml."""
    versions = {}
    for name in ("torch", "transformers", "sentence-transformers",
                 "bitsandbytes", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_scorer(path):
    """Import the official score.py as a module (metrics are NOT rewritten)."""
    spec = importlib.util.spec_from_file_location("course_score", path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    return scorer


# ---------------------------------------------------------------------------
# Exact dense retrieval with a document mask
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Exact cosine search over all passage embeddings."""

    def __init__(self, corpus, corpus_path, model_name, batch_size,
                 device, query_prefix):
        self.doc_ids = [doc["doc_id"] for doc in corpus]
        self.query_prefix = query_prefix
        self.encoder = SentenceTransformer(model_name, device=device)

        # Cache key: same model + same corpus file -> same embeddings.
        key = hashlib.sha256(
            (model_name + file_hash(corpus_path)).encode()
        ).hexdigest()[:16]
        cache_path = ROOT / ".cache" / f"embeddings_{key}.npy"

        if cache_path.exists():
            self.embeddings = np.load(cache_path)
        else:
            texts = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
            self.embeddings = self.encoder.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,  # L2 normalization
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self.embeddings)

    def dense_search(self, query, k, exclude_ids=frozenset()):
        """Return ranked doc ids (best first), skipping excluded ids."""
        # 1) embed the query (with the model's query prefix)
        query_vector = self.encoder.encode(
            [self.query_prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        # 2) cosine scores against ALL passages
        #    (dot product of normalized vectors = cosine similarity)
        scores = self.embeddings @ query_vector

        # 3) apply the mask BEFORE taking the top-k
        masked = [i for i, doc_id in enumerate(self.doc_ids)
                  if doc_id in exclude_ids]
        if masked:
            scores[masked] = -np.inf

        # 4) k highest scores; stable sort keeps ties reproducible
        order = np.argsort(-scores, kind="stable")[:k]
        return [self.doc_ids[i] for i in order if scores[i] != -np.inf]


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

class AnswerGenerator:
    """Generate short answers with one shared language model."""

    def __init__(self, model_name, use_4bit, max_input_tokens, max_new_tokens):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {}
        if torch.cuda.is_available():
            if use_4bit:
                # 4-bit quantization so the model fits on a Kaggle T4.
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def render_prompt(self, system_prompt, question, context):
        """Apply the model's chat template."""
        if context:
            user_text = (
                f"Passages:\n{context}\n\n"
                f"Question: {question}"
            )
        else:
            user_text = f"Question: {question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        template_kwargs = {}
        # Qwen3 can skip its thinking mode (we want short answers only).
        if "qwen3" in self.model_name.lower():
            template_kwargs["enable_thinking"] = False

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

    def build_prompt(self, system_prompt, question, documents):
        """Give every selected passage an equal share of the token budget."""
        if not documents:
            return self.render_prompt(system_prompt, question, "")

        # Cost of everything except the passage texts (headers included).
        headers = [f"[{d['doc_id']}] {d['title']}" for d in documents]
        base_prompt = self.render_prompt(
            system_prompt, question, "\n\n".join(headers)
        )
        base_size = len(
            self.tokenizer.encode(base_prompt, add_special_tokens=False)
        )

        # Small safety margin for tokenization differences.
        available = self.max_input_tokens - base_size - 32
        if available < len(documents):
            raise ValueError("max_input_tokens is too small for the context.")
        tokens_per_doc = available // len(documents)

        parts = []
        for doc, header in zip(documents, headers):
            ids = self.tokenizer.encode(doc["text"], add_special_tokens=False)
            text = self.tokenizer.decode(
                ids[:tokens_per_doc], skip_special_tokens=True
            )
            parts.append(f"{header}\n{text}")

        return self.render_prompt(system_prompt, question, "\n\n".join(parts))

    def answer(self, system_prompt, question, documents):
        """Greedy decoding; only the newly generated tokens are returned."""
        prompt = self.build_prompt(system_prompt, question, documents)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        text = self.tokenizer.decode(
            output[0, input_length:], skip_special_tokens=True
        )
        return clean_answer(text)


def clean_answer(text):
    """Keep the first non-empty line; NO_ANSWER becomes an abstention ('')."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    answer = lines[0] if lines else ""
    return "" if answer == "NO_ANSWER" else answer


# ---------------------------------------------------------------------------
# Evaluation (all metrics come from the official score.py)
# ---------------------------------------------------------------------------

def evaluate(scorer, predictions, labels):
    """Score predictions with score.py and print overall / per-type tables."""
    rows = []
    by_type = defaultdict(list)

    for prediction in predictions:
        label = labels[prediction["qid"]]
        metrics = scorer.per_question(prediction, label)
        rows.append(metrics)
        by_type[label["type"]].append(metrics)

    scorer.table(rows, "overall")
    for question_type in sorted(by_type):
        scorer.table(by_type[question_type], f"type: {question_type}")

    return {
        name: sum(row[name] for row in rows) / len(rows)
        for name in rows[0]
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default="course_pack_v1")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--split", default="dev", choices=("dev", "hidden"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--llm", default=None,
                        help="Override models.llm from the config file.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N questions (dev only).")
    parser.add_argument("--score-script",
                        default=str(ROOT / "starter" / "score.py"))
    return parser.parse_args()


def main():
    args = parse_args()

    if args.method == "oracle_context" and args.split != "dev":
        raise ValueError("Oracle-context is only available on dev.")
    if args.limit is not None and args.split != "dev":
        raise ValueError("--limit is allowed only on dev.")

    # All settings come from the YAML file, not from the code.
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    llm_name = args.llm or cfg["models"]["llm"]
    gen_cfg = cfg["generation"]
    ret_cfg = cfg["retrieval"]

    if ret_cfg["top_k"] < 10:
        raise ValueError("retrieval.top_k must be at least 10.")

    set_seed(cfg["seed"])

    out_dir = Path(args.out)
    predictions_path = out_dir / "predictions.jsonl"
    if predictions_path.exists():
        raise FileExistsError(f"{predictions_path} exists. Use a new folder.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prompts are read from files so they can be hashed and reported.
    prompt_key = "closed_book" if args.method == "closed_book" else "rag"
    prompt_path = ROOT / cfg["prompts"][prompt_key]
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    # Data
    pack = Path(args.pack)
    corpus_path = pack / "corpus.jsonl"
    corpus = read_jsonl(corpus_path)
    all_questions = read_jsonl(pack / f"questions_{args.split}.jsonl")
    questions = all_questions[:args.limit] if args.limit else all_questions
    documents_by_id = {doc["doc_id"]: doc for doc in corpus}

    # Hidden labels are never loaded.
    labels = {}
    if args.split == "dev":
        labels = {row["qid"]: row
                  for row in read_jsonl(pack / "labels_dev.jsonl")}

    scorer = load_scorer(args.score_script)

    # Models
    retriever = None
    if args.method == "dense":
        retriever = DenseRetriever(
            corpus,
            corpus_path,
            cfg["models"]["embedder"],
            ret_cfg["embedding_batch_size"],
            ret_cfg["embedding_device"],
            ret_cfg["query_prefix"],
        )
    gc.collect()

    generator = AnswerGenerator(
        llm_name,
        gen_cfg["load_in_4bit"],
        gen_cfg["max_input_tokens"],
        gen_cfg["max_new_tokens"],
    )

    # run_config.yaml (same layout as starter/example_run_config.yaml)
    run_config = {
        "system_name": args.method,
        "phase": cfg["phase"],
        "split": args.split,
        "description": cfg["description"],
        "models": {
            "llm": {
                "name": llm_name,
                "where": ("Kaggle GPU, bitsandbytes 4-bit NF4"
                          if torch.cuda.is_available() and gen_cfg["load_in_4bit"]
                          else "full precision"),
                "temperature": 0,
            },
            "embedder": cfg["models"]["embedder"],
            "reranker": None,
        },
        "settings": {
            "retrieval": ({"unit": "passage", "type": "exact_cosine",
                           "top_k": ret_cfg["top_k"],
                           "context_k": ret_cfg["context_k"]}
                          if args.method == "dense" else None),
            "max_input_tokens": gen_cfg["max_input_tokens"],
            "max_new_tokens": gen_cfg["max_new_tokens"],
            "context_policy": "equal token share per passage",
            "prompt_file": str(prompt_path.relative_to(ROOT)),
            "prompt_hash": file_hash(prompt_path),
            "config_hash": file_hash(args.config),
            "corpus_hash": file_hash(corpus_path),
            "seed": cfg["seed"],
            "n_questions": len(questions),
            "partial": len(questions) < len(all_questions),
        },
        "versions": package_versions(),
        "command": "python " + " ".join(sys.argv),
        "date": date.today().isoformat(),
    }
    with open(out_dir / "run_config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(run_config, file, allow_unicode=True, sort_keys=False)

    # Prediction loop
    predictions = []
    with open(predictions_path, "w", encoding="utf-8") as file:
        for question in tqdm(questions, desc=args.method):
            qid = question["qid"]

            if args.method == "closed_book":
                retrieved, context_ids = [], []
            elif args.method == "dense":
                retrieved = retriever.dense_search(
                    question["question"], ret_cfg["top_k"]
                )
                context_ids = retrieved[:ret_cfg["context_k"]]
            else:
                # Oracle: the gold passages themselves are the context.
                retrieved = list(dict.fromkeys(labels[qid]["gold_doc_ids"]))
                context_ids = retrieved

            answer = generator.answer(
                system_prompt,
                question["question"],
                [documents_by_id[doc_id] for doc_id in context_ids],
            )

            # Only fields allowed by predictions_format.md.
            row = {"qid": qid, "answer": answer, "retrieved": retrieved}
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            predictions.append(row)

    print(f"\nSaved predictions: {predictions_path}")

    # Validate with the official checker (works for dev and hidden).
    errors = scorer.validate(
        predictions,
        {q["qid"] for q in all_questions},
        set(documents_by_id),
        partial=len(questions) < len(all_questions),
    )
    if errors:
        raise ValueError("Invalid predictions:\n" + "\n".join(errors[:20]))
    print("predictions.jsonl passed score.py validation.")

    # Dev only: score with the official functions.
    if args.split == "dev":
        averages = evaluate(scorer, predictions, labels)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(averages, file, indent=2)


if __name__ == "__main__":
    main()