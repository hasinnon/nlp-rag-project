"""Phase 1: closed-book, oracle-context and dense RAG baselines."""

import argparse
import datetime
import importlib.util
import json
import math
import random
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# NOTE: torch / transformers / sentence-transformers are imported inside the
# functions that need them, so tests can import this file without heavy libs.

ROOT = Path(__file__).resolve().parent.parent  # repository root
KS = (2, 5, 10)
METHODS = ("closed_book", "oracle_context", "dense")
SYSTEM_NAMES = {
    "closed_book": "closed_book",
    "oracle_context": "oracle_context",
    "dense": "dense_rag",
}

# Tokens kept free for the system prompt, question and chat template.
PROMPT_RESERVE = 600


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def read_jsonl(path):
    """Read a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def set_seed(seed):
    """Fix random seeds for reproducibility."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize(vectors):
    """Scale every row to unit length (so dot product = cosine similarity)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# Handwritten metrics (no metric libraries)
# ---------------------------------------------------------------------------

def recall_at_k(ranked, gold, k):
    """Fraction of gold documents found in the top-k."""
    return len(gold & set(ranked[:k])) / len(gold)


def hitall_at_k(ranked, gold, k):
    """1 if ALL gold documents are in the top-k, else 0."""
    return float(gold <= set(ranked[:k]))


def mrr(ranked, gold):
    """Reciprocal rank of the first gold document (0 if none is retrieved)."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked, gold, k):
    """nDCG with binary gain; ideal DCG uses min(|G|, k) top ranks."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in gold
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal


def normalize_answer(text):
    """Lowercase, remove punctuation and articles, fix whitespace."""
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, reference):
    """Token-level F1 (same rules as the course scorer)."""
    pred = normalize_answer(prediction)
    gold = normalize_answer(reference)

    # yes / no / noanswer must match exactly, otherwise the score is 0.
    special = ("yes", "no", "noanswer")
    if (pred in special or gold in special) and pred != gold:
        return 0.0

    common = sum((Counter(pred.split()) & Counter(gold.split())).values())
    if common == 0:
        return 0.0

    precision = common / len(pred.split())
    recall = common / len(gold.split())
    return 2 * precision * recall / (precision + recall)


def question_metrics(prediction, label):
    """All metrics for ONE question (names match score.py)."""
    gold = set(label["gold_doc_ids"])
    ranked = prediction["retrieved"]

    metrics = {}
    for k in KS:
        metrics[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        metrics[f"hitall@{k}"] = hitall_at_k(ranked, gold, k)
    metrics["mrr"] = mrr(ranked, gold)
    metrics["ndcg@10"] = ndcg_at_k(ranked, gold, 10)

    answer = prediction["answer"]
    references = [label["answer"]] + label.get("aliases", [])
    if answer:
        metrics["em"] = float(any(
            normalize_answer(answer) == normalize_answer(r) for r in references
        ))
        metrics["f1"] = max(token_f1(answer, r) for r in references)
    else:
        metrics["em"] = 0.0
        metrics["f1"] = 0.0
    return metrics


# ---------------------------------------------------------------------------
# Exact dense retrieval with document mask
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Exact cosine search over ALL passages (no approximate index)."""

    def __init__(self, corpus, model_name, batch_size, device, query_prefix):
        from sentence_transformers import SentenceTransformer

        self.doc_ids = [doc["doc_id"] for doc in corpus]
        self.index_of = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        self.query_prefix = query_prefix
        self.encoder = SentenceTransformer(model_name, device=device)

        # Passages are embedded as "title + newline + text" (no prefix).
        cache = ROOT / ".cache" / (
            f"emb_{model_name.replace('/', '_')}_{len(corpus)}.npy"
        )
        if cache.exists():
            self.embeddings = np.load(cache)
        else:
            texts = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
            vectors = self.encoder.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            self.embeddings = l2_normalize(vectors)
            cache.parent.mkdir(exist_ok=True)
            np.save(cache, self.embeddings)

    def dense_search(self, query, k, exclude_ids=frozenset()):
        """Return ids of the k best passages, best first, skipping exclude_ids."""
        # 1) embed the query (bge needs the instruction prefix) and normalize
        q = self.encoder.encode([self.query_prefix + query], convert_to_numpy=True)
        q = l2_normalize(q)[0]

        # 2) cosine scores against ALL passages (dot product of unit vectors)
        scores = self.embeddings @ q

        # 3) document mask: excluded ids get -inf BEFORE the top-k step
        for doc_id in exclude_ids:
            if doc_id in self.index_of:
                scores[self.index_of[doc_id]] = -np.inf

        # 4) k highest scores; stable sort keeps tie order reproducible
        order = np.argsort(-scores, kind="stable")[:k]
        return [self.doc_ids[i] for i in order if scores[i] != -np.inf]


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

class AnswerGenerator:
    """One shared LLM used by all three systems."""

    def __init__(self, model_name, load_in_4bit, max_input_tokens, max_new_tokens):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self.torch = torch
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {}
        self.quantized = bool(load_in_4bit and torch.cuda.is_available())
        if self.quantized:
            # 4-bit weights so the model fits on a Kaggle T4.
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["device_map"] = {"": 0}
        elif torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = {"": 0}

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def build_context(self, documents):
        """Put passages in rank order; cut the text when the budget is used up."""
        remaining = self.max_input_tokens - PROMPT_RESERVE
        parts = []
        for doc in documents:
            if remaining <= 0:
                break
            ids = self.tokenizer.encode(doc["text"], add_special_tokens=False)
            ids = ids[:remaining]
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            parts.append(f"[{doc['doc_id']}] {doc['title']}\n{text}")
            remaining -= len(ids)
        return "\n\n".join(parts)

    def build_prompt(self, system_prompt, question, documents):
        """Create the chat prompt (with or without passages)."""
        if documents:
            user_text = (
                f"Passages:\n{self.build_context(documents)}\n\n"
                f"Question: {question}"
            )
        else:
            user_text = f"Question: {question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        extra = {}
        if "qwen3" in self.model_name.lower():
            extra["enable_thinking"] = False  # short answers, no reasoning

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra
        )

    def answer(self, system_prompt, question, documents):
        """Greedy decoding; return only the newly generated text."""
        prompt = self.build_prompt(system_prompt, question, documents)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        text = self.tokenizer.decode(output[0, input_length:], skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # safety
        return text.strip()


# ---------------------------------------------------------------------------
# Validation and comparison with score.py
# ---------------------------------------------------------------------------

def load_scorer(path):
    """Import the official score.py as a module."""
    spec = importlib.util.spec_from_file_location("course_score", path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    return scorer


def validate_predictions(scorer, predictions, all_questions, corpus):
    """Use the official validator to check the predictions format."""
    errors = scorer.validate(
        predictions,
        {q["qid"] for q in all_questions},
        {d["doc_id"] for d in corpus},
        partial=len(predictions) < len(all_questions),
    )
    if errors:
        raise ValueError("Invalid predictions:\n" + "\n".join(errors[:20]))
    print("Predictions format is valid.")


def evaluate_dev(scorer, predictions, labels):
    """Compute our metrics and check they equal score.py (per question)."""
    rows = []
    max_difference = 0.0

    for prediction in predictions:
        label = labels[prediction["qid"]]
        own = question_metrics(prediction, label)
        reference = scorer.per_question(prediction, label)
        for name, value in own.items():
            max_difference = max(max_difference, abs(value - reference[name]))
        rows.append((label["type"], own))

    if max_difference > 1e-9:
        raise AssertionError(f"Metric mismatch with score.py: {max_difference}")

    def averages(metric_rows):
        return {name: mean(r[name] for r in metric_rows) for name in metric_rows[0]}

    overall = averages([m for _, m in rows])
    by_type = defaultdict(list)
    for question_type, metrics in rows:
        by_type[question_type].append(metrics)

    print(f"\nOverall (n={len(rows)}), percent:")
    for name, value in overall.items():
        print(f"  {name:10s} {100 * value:6.2f}")

    for question_type in sorted(by_type):
        values = averages(by_type[question_type])
        print(f"\nType {question_type} (n={len(by_type[question_type])}):")
        print("  " + "  ".join(f"{k}={100 * v:.1f}" for k, v in values.items()))

    print(f"\nMax difference from score.py: {max_difference:.2e}")

    return {
        "n": len(rows),
        "overall": overall,
        "by_type": {t: averages(m) for t, m in by_type.items()},
        "max_difference_from_score_py": max_difference,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default="course_pack_v1")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--split", default="dev", choices=("dev", "hidden"))
    parser.add_argument("--out", "--output", dest="out", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--score-script", default=str(ROOT / "starter" / "score.py"))
    parser.add_argument("--limit", type=int, default=None,
                        help="dev only: run the first N questions (smoke test)")
    return parser.parse_args()


def write_run_config(path, args, cfg, generator, n_questions):
    """Write run_config.yaml in the format of example_run_config.yaml."""
    is_dense = args.method == "dense"
    where = "Kaggle T4, HF transformers" if generator.quantized else "HF transformers"
    if generator.quantized:
        where += ", 4-bit bitsandbytes (nf4)"

    prompt_key = "closed_book" if args.method == "closed_book" else "rag"
    settings = {
        "retrieval": {
            "unit": "passage",
            "type": "exact_cosine" if is_dense else None,
            "k": cfg["retrieval"]["top_k"] if is_dense else None,
            "context_k": cfg["retrieval"]["context_k"] if is_dense else None,
        },
        "max_input_tokens": cfg["generation"]["max_input_tokens"],
        "max_new_tokens": cfg["generation"]["max_new_tokens"],
        "prompt_file": cfg["prompts"][prompt_key],
        "n_questions": n_questions,
        "seed": cfg["seed"],
    }
    if is_dense:
        settings["query_prefix"] = cfg["retrieval"]["query_prefix"]

    run_config = {
        "system_name": SYSTEM_NAMES[args.method],
        "phase": cfg["phase"],
        "split": args.split,
        "description": f"{args.method} baseline: {cfg['description']}",
        "models": {
            "llm": {"name": cfg["models"]["llm"], "where": where, "temperature": 0},
            "embedder": cfg["models"]["embedder"],
            "reranker": None,
        },
        "settings": settings,
        "command": "python " + " ".join(sys.argv),
        "date": datetime.date.today().isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, allow_unicode=True, sort_keys=False)


def main():
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    retrieval_cfg = cfg["retrieval"]
    gen_cfg = cfg["generation"]

    # ---- basic checks --------------------------------------------------
    if args.method == "oracle_context" and args.split != "dev":
        raise ValueError("Oracle-context needs gold labels: dev only.")
    if args.limit is not None and args.split != "dev":
        raise ValueError("--limit is allowed only on dev.")
    if retrieval_cfg["top_k"] < 10:
        raise ValueError("top_k must be at least 10.")
    if not 1 <= retrieval_cfg["context_k"] <= retrieval_cfg["top_k"]:
        raise ValueError("context_k must be between 1 and top_k.")

    set_seed(cfg["seed"])

    out_dir = Path(args.out)
    predictions_path = out_dir / "predictions.jsonl"
    if predictions_path.exists():
        raise FileExistsError(f"{predictions_path} exists. Use a new folder.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ------------------------------------------------------------
    pack = Path(args.pack)
    corpus = read_jsonl(pack / "corpus.jsonl")
    all_questions = read_jsonl(pack / f"questions_{args.split}.jsonl")
    questions = all_questions[:args.limit] if args.limit else all_questions
    docs_by_id = {doc["doc_id"]: doc for doc in corpus}

    # Hidden labels are never loaded.
    labels = {}
    if args.split == "dev":
        labels = {r["qid"]: r for r in read_jsonl(pack / "labels_dev.jsonl")}

    # ---- prompt, retriever, LLM -----------------------------------------
    prompt_key = "closed_book" if args.method == "closed_book" else "rag"
    system_prompt = (ROOT / cfg["prompts"][prompt_key]).read_text(encoding="utf-8").strip()

    retriever = None
    if args.method == "dense":
        retriever = DenseRetriever(
            corpus,
            cfg["models"]["embedder"],
            retrieval_cfg["embedding_batch_size"],
            retrieval_cfg["embedding_device"],
            retrieval_cfg["query_prefix"],
        )

    generator = AnswerGenerator(
        cfg["models"]["llm"],
        gen_cfg["load_in_4bit"],
        gen_cfg["max_input_tokens"],
        gen_cfg["max_new_tokens"],
    )

    write_run_config(out_dir / "run_config.yaml", args, cfg, generator, len(questions))

    # ---- main loop -------------------------------------------------------
    predictions = []
    with open(predictions_path, "w", encoding="utf-8") as f:
        for q in tqdm(questions, desc=args.method):
            if args.method == "closed_book":
                retrieved, context_ids = [], []
            elif args.method == "dense":
                retrieved = retriever.dense_search(q["question"], retrieval_cfg["top_k"])
                context_ids = retrieved[:retrieval_cfg["context_k"]]
            else:
                # Oracle: the gold passages are the context and the "retrieved" field.
                retrieved = list(dict.fromkeys(labels[q["qid"]]["gold_doc_ids"]))
                context_ids = retrieved

            documents = [docs_by_id[doc_id] for doc_id in context_ids]
            answer = generator.answer(system_prompt, q["question"], documents)

            # Only the fields allowed by predictions_format.md.
            row = {"qid": q["qid"], "answer": answer, "retrieved": retrieved}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            predictions.append(row)

    print(f"\nSaved: {predictions_path}")

    # ---- validation / evaluation ----------------------------------------
    score_path = Path(args.score_script)
    if not score_path.is_file():
        print(f"WARNING: {score_path} not found, skipping validation.")
        return

    scorer = load_scorer(score_path)
    validate_predictions(scorer, predictions, all_questions, corpus)

    if args.split == "dev":
        metrics = evaluate_dev(scorer, predictions, labels)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()