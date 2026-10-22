"""Phase 1: closed-book, exact dense RAG, and oracle-context baselines."""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import random
import re
import shlex
import string
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("closed_book", "dense", "oracle_context")
KS = (2, 5, 10)


# ---------------------------------------------------------------------------
# Files and configuration
# ---------------------------------------------------------------------------

def read_jsonl(path):
    """Read non-empty JSONL lines."""
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_json(path, data):
    """Save a JSON document."""
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_jsonl(path, rows):
    """Save diagnostic rows in JSONL format."""
    with Path(path).open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path, rows, fields):
    """Save a table with an explicit column order."""
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    """Calculate a SHA-256 hash without loading the full file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value):
    """Resolve project resources independently of the working directory."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path):
    """Load the existing project YAML configuration."""
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping.")

    for section in ("models", "generation", "retrieval", "prompts"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")

    if config.get("phase") != 1:
        raise ValueError("This runner implements Phase 1 only.")

    for name in ("llm", "embedder"):
        if not config["models"].get(name):
            raise ValueError(f"Missing model setting: models.{name}")

    generation = config["generation"]
    retrieval = config["retrieval"]

    for name in ("max_input_tokens", "max_new_tokens"):
        if int(generation[name]) < 1:
            raise ValueError(f"generation.{name} must be positive.")

    if not isinstance(generation.get("load_in_4bit"), bool):
        raise ValueError("generation.load_in_4bit must be true or false.")

    if int(retrieval["top_k"]) < 10:
        raise ValueError("retrieval.top_k must be at least 10.")

    if not 1 <= int(retrieval["context_k"]) <= int(retrieval["top_k"]):
        raise ValueError("context_k must be between 1 and top_k.")

    if int(retrieval["embedding_batch_size"]) < 1:
        raise ValueError("embedding_batch_size must be positive.")

    return config


def load_scorer(path):
    """Import the unmodified official scorer."""
    spec = importlib.util.spec_from_file_location("course_score", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import the official scorer: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_versions():
    """Record relevant installed package versions."""
    names = (
        "numpy",
        "PyYAML",
        "torch",
        "transformers",
        "sentence-transformers",
        "accelerate",
        "bitsandbytes",
        "tqdm",
    )
    versions = {"python": sys.version.split()[0]}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit():
    """Record the repository commit when Git is available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def set_seed(seed):
    """Set random seeds without promising bitwise GPU reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def top_indices(scores, k):
    """Return stable descending indices."""
    if k < 0:
        raise ValueError("k must be non-negative.")
    return np.argsort(
        -np.asarray(scores, dtype=np.float64),
        kind="stable",
    )[:k].tolist()


def rrf_scores(rankings, size, constant=60):
    """Retain the existing RRF helper for compatibility with project tests."""
    if constant <= 0:
        raise ValueError("The RRF constant must be positive.")

    scores = np.zeros(size, dtype=np.float64)
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            scores[index] += 1.0 / (constant + rank)
    return scores


class Retriever:
    """Exact dense retrieval with a pre-ranking document mask.

    BM25 is retained only for compatibility with the existing unit tests.
    The Phase 1 CLI exposes only the three required systems.
    """

    def __init__(
        self,
        corpus,
        config,
        method,
        corpus_hash=None,
        cache_dir=None,
    ):
        self.method = method
        self.doc_ids = [doc["doc_id"] for doc in corpus]
        self.id_to_index = {
            doc_id: index for index, doc_id in enumerate(self.doc_ids)
        }
        self.config = config.get("retrieval", {})
        self.query_prefix = self.config.get("query_prefix", "")

        if len(self.doc_ids) != len(set(self.doc_ids)):
            raise ValueError("The corpus contains duplicate document IDs.")

        if method == "closed_book":
            return

        texts = [
            f"{doc.get('title', '')}\n{doc['text']}"
            for doc in corpus
        ]

        if method == "bm25":
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi([
                re.findall(r"\w+", text.lower()) for text in texts
            ])
            return

        if method != "dense":
            raise ValueError(f"Unsupported retriever: {method}")

        from sentence_transformers import SentenceTransformer

        model_name = config["models"]["embedder"]
        device = self.config.get("embedding_device", "cpu")
        batch_size = int(self.config["embedding_batch_size"])

        self.encoder = SentenceTransformer(model_name, device=device)

        # Cache only passage embeddings, never predictions or gold labels.
        cache_path = None
        if cache_dir is not None and corpus_hash is not None:
            key_data = {
                "corpus_hash": corpus_hash,
                "model": model_name,
                "format": "title + newline + text",
                "normalization": "l2",
                "sentence_transformers_version": package_versions()[
                    "sentence-transformers"
                ],
            }
            key = hashlib.sha256(
                json.dumps(key_data, sort_keys=True).encode()
            ).hexdigest()[:24]

            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"passages_{key}.npy"

        if cache_path is not None and cache_path.is_file():
            print(f"Loading passage embeddings: {cache_path}")
            embeddings = np.load(cache_path, allow_pickle=False)
        else:
            embeddings = self.encoder.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            if cache_path is not None:
                np.save(cache_path, embeddings, allow_pickle=False)

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2 or embeddings.shape[0] != len(corpus):
            raise ValueError("Cached embedding dimensions do not match corpus.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Passage embeddings contain non-finite values.")

        # Normalize explicitly so the search implementation is self-contained.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("A passage has a zero embedding.")

        self.embeddings = embeddings / norms

    def rank_scores(self, scores, k, exclude_ids=frozenset()):
        """Apply the mask before selecting the highest-scoring documents."""
        if k < 0:
            raise ValueError("k must be non-negative.")

        scores = np.asarray(scores, dtype=np.float32).copy()
        if scores.shape != (len(self.doc_ids),):
            raise ValueError("Scores must contain one value per document.")

        for doc_id in exclude_ids:
            index = self.id_to_index.get(doc_id)
            if index is not None:
                scores[index] = -np.inf

        # Full sorting is exact and keeps ties in corpus order.
        order = top_indices(scores, len(scores))
        selected = [
            index for index in order
            if np.isfinite(scores[index])
        ][:k]

        return [self.doc_ids[index] for index in selected]

    def dense_search(self, query, k, exclude_ids=frozenset()):
        """Search every passage using exact normalized dot products."""
        if self.method != "dense":
            raise ValueError("dense_search requires a dense retriever.")

        query_vector = self.encoder.encode(
            [self.query_prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        query_vector = np.asarray(query_vector, dtype=np.float32)

        norm = np.linalg.norm(query_vector)
        if not np.isfinite(query_vector).all() or norm == 0:
            raise ValueError("The query embedding is invalid.")

        query_vector = query_vector / norm

        # Cosine scores are computed against ALL passage embeddings.
        scores = self.embeddings @ query_vector
        return self.rank_scores(scores, k, exclude_ids)

    def retrieve(self, query, k=10, exclude_ids=frozenset()):
        """Provide the interface used by the existing project tests."""
        if self.method == "closed_book":
            return []

        if self.method == "dense":
            return self.dense_search(query, k, exclude_ids)

        tokens = re.findall(r"\w+", query.lower())
        return self.rank_scores(
            self.bm25.get_scores(tokens),
            k,
            exclude_ids,
        )


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def clean_answer(text):
    """Remove explicit reasoning blocks and standardize abstention."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # An unfinished reasoning block is not a valid final answer.
    if "<think>" in text:
        return ""

    text = text.strip()
    if text.upper() == "NO_ANSWER":
        return ""

    return text


class AnswerGenerator:
    """Use one shared LLM and the project's existing prompt files."""

    def __init__(self, config, prompts):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self.model_name = config["models"]["llm"]
        self.prompts = prompts
        self.max_input_tokens = int(
            config["generation"]["max_input_tokens"]
        )
        self.max_new_tokens = int(
            config["generation"]["max_new_tokens"]
        )
        self.thinking_disabled = "qwen3" in self.model_name.lower()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        requested_4bit = config["generation"]["load_in_4bit"]
        self.effective_4bit = requested_4bit and torch.cuda.is_available()

        kwargs = {}

        if torch.cuda.is_available():
            kwargs["device_map"] = {"": 0}
            kwargs["torch_dtype"] = torch.float16

            if requested_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
        else:
            kwargs["torch_dtype"] = torch.float32
            if requested_4bit:
                print(
                    "WARNING: CUDA is unavailable. "
                    "Using CPU float32 instead of 4-bit GPU loading."
                )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **kwargs,
        )
        self.model.eval()

        context_limit = getattr(
            self.model.config, "max_position_embeddings", None
        )
        if context_limit is not None:
            if self.max_input_tokens + self.max_new_tokens > context_limit:
                raise ValueError(
                    "Input plus output budgets exceed the model context."
                )

    def render_prompt(self, question, context, method):
        """Apply the model chat template with the selected file prompt."""
        prompt_name = "closed_book" if method == "closed_book" else "rag"

        user_text = f"Question: {question}"
        if method != "closed_book":
            user_text = (
                f"Passages:\n{context}\n\n"
                f"Question: {question}"
            )

        messages = [
            {"role": "system", "content": self.prompts[prompt_name]},
            {"role": "user", "content": user_text},
        ]

        kwargs = {}
        if self.thinking_disabled:
            kwargs["enable_thinking"] = False

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )

    def token_count(self, text):
        """Count prompt tokens without inserting extra special tokens."""
        return len(self.tokenizer.encode(
            text,
            add_special_tokens=False,
        ))

    def build_prompt(self, question, documents, method):
        """Allocate text tokens equally while preserving every document header."""
        if not documents:
            prompt = self.render_prompt(question, "", method)
            if self.token_count(prompt) > self.max_input_tokens:
                raise ValueError("The question exceeds the input budget.")
            return prompt, []

        headers = [
            f"[{doc['doc_id']}] {doc.get('title', '')}"
            for doc in documents
        ]
        text_tokens = [
            self.tokenizer.encode(
                doc["text"],
                add_special_tokens=False,
            )
            for doc in documents
        ]

        empty_prompt = self.render_prompt(
            question,
            "\n\n".join(headers),
            method,
        )
        available = self.max_input_tokens - self.token_count(empty_prompt) - 64

        if available < len(documents):
            raise ValueError(
                "The input budget cannot fit all selected document headers."
            )

        per_document = available // len(documents)

        # Recheck the rendered prompt because token boundaries can change.
        while per_document >= 1:
            parts = []
            context_info = []

            for doc, header, tokens in zip(documents, headers, text_tokens):
                kept = tokens[:per_document]
                text = self.tokenizer.decode(
                    kept,
                    skip_special_tokens=True,
                )
                parts.append(f"{header}\n{text}")
                context_info.append({
                    "doc_id": doc["doc_id"],
                    "original_text_tokens": len(tokens),
                    "used_text_tokens": len(kept),
                    "truncated": len(kept) < len(tokens),
                })

            prompt = self.render_prompt(
                question,
                "\n\n".join(parts),
                method,
            )
            if self.token_count(prompt) <= self.max_input_tokens:
                return prompt, context_info

            per_document = max(0, per_document - 32)

        raise ValueError("Unable to fit the selected context into the budget.")

    def answer(self, question, documents, method):
        """Decode only newly generated tokens."""
        import torch

        prompt, context_info = self.build_prompt(
            question,
            documents,
            method,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = inputs.to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        raw_answer = self.tokenizer.decode(
            output[0, input_length:],
            skip_special_tokens=True,
        )
        return clean_answer(raw_answer), raw_answer, context_info


# ---------------------------------------------------------------------------
# Handwritten metrics
# ---------------------------------------------------------------------------

def normalize_answer(text):
    """Match the official answer normalization."""
    text = "".join(
        char for char in text.lower()
        if char not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, reference):
    """Calculate token F1 using token overlap."""
    pred = normalize_answer(prediction)
    gold = normalize_answer(reference)

    special = ("yes", "no", "noanswer")
    if (pred in special or gold in special) and pred != gold:
        return 0.0

    pred_tokens = pred.split()
    gold_tokens = gold.split()
    overlap = sum(
        (Counter(pred_tokens) & Counter(gold_tokens)).values()
    )

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_metrics(prediction, label):
    """Calculate answer EM and F1 independently of score.py."""
    answer = "" if prediction.get("abstained") else prediction["answer"]
    references = [label["answer"]] + label.get("aliases", [])

    if not answer:
        return {"em": 0.0, "f1": 0.0}

    return {
        "em": float(any(
            normalize_answer(answer) == normalize_answer(reference)
            for reference in references
        )),
        "f1": max(token_f1(answer, reference) for reference in references),
    }


def retrieval_metrics(ranked, gold_doc_ids):
    """Calculate Recall, HitAll, MRR, and binary-gain nDCG manually."""
    gold = set(gold_doc_ids)
    if not gold:
        raise ValueError("Retrieval metrics require non-empty gold_doc_ids.")

    metrics = {}

    for k in KS:
        top = set(ranked[:k])
        metrics[f"recall@{k}"] = len(top & gold) / len(gold)
        metrics[f"hitall@{k}"] = float(gold <= top)

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, doc_id in enumerate(ranked[:k], start=1)
            if doc_id in gold
        )
        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(len(gold), k) + 1)
        )
        metrics[f"ndcg@{k}"] = dcg / idcg

    metrics["mrr"] = next(
        (
            1.0 / rank
            for rank, doc_id in enumerate(ranked, start=1)
            if doc_id in gold
        ),
        0.0,
    )
    return metrics


def summarize(rows, metric_names, scorer):
    """Use the official bootstrap routine for report confidence intervals."""
    result = {"n": len(rows), "metrics": {}}

    for name in metric_names:
        values = [row[name] for row in rows]
        low, high = scorer.bootstrap(values)
        result["metrics"][name] = {
            "mean": sum(values) / len(values),
            "ci95_low": low,
            "ci95_high": high,
        }

    return result


def evaluate_dev(predictions, labels, traces, method, scorer, output_dir):
    """Verify handwritten metrics and save report-ready diagnostic outputs."""
    rows = []
    reference_rows = []
    diagnostic_rows = []
    max_differences = defaultdict(float)

    for prediction in predictions:
        qid = prediction["qid"]
        label = labels[qid]

        own = answer_metrics(prediction, label)
        if method == "dense":
            own.update(retrieval_metrics(
                prediction["retrieved"],
                label["gold_doc_ids"],
            ))

        reference = scorer.per_question(prediction, label)

        # score.py exposes nDCG@10 but not nDCG@2 or nDCG@5.
        for name, value in own.items():
            if name not in reference:
                continue

            difference = abs(value - reference[name])
            max_differences[name] = max(
                max_differences[name],
                difference,
            )
            if difference > 1e-12:
                raise AssertionError(
                    f"Metric mismatch: qid={qid}, metric={name}, "
                    f"own={value}, official={reference[name]}"
                )

        rows.append({"qid": qid, "type": label["type"], **own})
        reference_rows.append(reference)

        trace = traces[qid]
        gold = set(label["gold_doc_ids"])
        top10 = set(prediction["retrieved"][:10])
        context_ids = set(trace["context_ids"])

        if method != "dense":
            category = "not_applicable"
        elif own["em"] == 1.0:
            category = "answer_correct"
        elif not gold <= top10:
            category = "missing_gold_in_top10"
        elif not gold <= context_ids:
            category = "gold_in_top10_but_missing_from_context"
        elif any(item["truncated"] for item in trace["context_info"]):
            category = "all_gold_in_context_with_text_truncation"
        else:
            category = "all_gold_in_context_answer_wrong"

        diagnostic_rows.append({
            "qid": qid,
            "type": label["type"],
            "question": trace["question"],
            "answer": prediction["answer"],
            "raw_answer": trace["raw_answer"],
            "gold_answer": label["answer"],
            "aliases": label.get("aliases", []),
            "gold_doc_ids": label["gold_doc_ids"],
            "retrieved": prediction["retrieved"],
            "context_ids": trace["context_ids"],
            "context_info": trace["context_info"],
            "missing_gold_in_top10": sorted(gold - top10),
            "missing_gold_in_context": sorted(gold - context_ids),
            "diagnostic_category": category,
            "metrics": own,
        })

    metric_names = ["em", "f1"]
    if method == "dense":
        metric_names += [
            "recall@2", "recall@5", "recall@10",
            "hitall@2", "hitall@5", "hitall@10",
            "mrr", "ndcg@2", "ndcg@5", "ndcg@10",
        ]

    by_type = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)

    summary = {
        "method": method,
        "units": "fractions; CSV and console use percentages",
        "overall": summarize(rows, metric_names, scorer),
        "by_type": {
            name: summarize(type_rows, metric_names, scorer)
            for name, type_rows in sorted(by_type.items())
        },
        "max_difference_from_score_py": dict(max_differences),
    }

    save_json(output_dir / "metrics.json", summary)
    save_jsonl(output_dir / "per_question.jsonl", diagnostic_rows)

    # Long-form tables are easy to inspect and copy into a report.
    table_rows = []
    groups = {"overall": summary["overall"], **summary["by_type"]}

    for group, group_summary in groups.items():
        for name, values in group_summary["metrics"].items():
            table_rows.append({
                "group": group,
                "n": group_summary["n"],
                "metric": name,
                "mean_percent": 100 * values["mean"],
                "ci95_low_percent": 100 * values["ci95_low"],
                "ci95_high_percent": 100 * values["ci95_high"],
            })

    save_csv(
        output_dir / "results.csv",
        table_rows,
        [
            "group", "n", "metric", "mean_percent",
            "ci95_low_percent", "ci95_high_percent",
        ],
    )

    agreement_rows = []
    for name in max_differences:
        own_mean = sum(row[name] for row in rows) / len(rows)
        official_mean = sum(
            row[name] for row in reference_rows
        ) / len(reference_rows)

        agreement_rows.append({
            "metric": name,
            "handwritten": own_mean,
            "score_py": official_mean,
            "absolute_mean_difference": abs(own_mean - official_mean),
            "max_per_question_difference": max_differences[name],
        })

    save_csv(
        output_dir / "metric_agreement.csv",
        agreement_rows,
        [
            "metric", "handwritten", "score_py",
            "absolute_mean_difference", "max_per_question_difference",
        ],
    )

    if method == "dense":
        bridge_rows = [
            row for row in diagnostic_rows
            if row["type"] == "bridge_comparison"
        ]

        save_jsonl(
            output_dir / "bridge_comparison_errors.jsonl",
            [
                row for row in bridge_rows
                if row["metrics"]["em"] == 0.0
                or row["metrics"]["hitall@10"] == 0.0
            ],
        )

        bridge_summary = {
            "n": len(bridge_rows),
            "hitall@10": (
                sum(row["metrics"]["hitall@10"] for row in bridge_rows)
                / len(bridge_rows)
                if bridge_rows else None
            ),
            "diagnostic_counts": dict(Counter(
                row["diagnostic_category"] for row in bridge_rows
            )),
            "note": (
                "Categories identify observable failure conditions, "
                "not proven causal explanations. Inspect passage text "
                "and model outputs before writing the final analysis."
            ),
        }
        save_json(
            output_dir / "bridge_comparison_summary.json",
            bridge_summary,
        )

    print("\nOverall results (percent, 95% bootstrap interval):")
    for name, values in summary["overall"]["metrics"].items():
        print(
            f"  {name:12s} {100 * values['mean']:6.2f} "
            f"[{100 * values['ci95_low']:.2f}, "
            f"{100 * values['ci95_high']:.2f}]"
        )

    print("Handwritten metrics match score.py within 1e-12.")
    return summary


# ---------------------------------------------------------------------------
# Validation and official evaluation
# ---------------------------------------------------------------------------

def validate_predictions(scorer, predictions, questions, corpus, partial):
    """Validate the official output schema and question coverage."""
    errors = scorer.validate(
        predictions,
        {question["qid"] for question in questions},
        {doc["doc_id"] for doc in corpus},
        partial=partial,
    )
    if errors:
        raise ValueError("\n".join(errors[:20]))


def run_official_score(
    score_path,
    predictions_path,
    pack,
    split,
    partial,
    output_dir,
    compare_path=None,
):
    """Save the official evaluation or hidden schema validation."""
    command = [
        sys.executable,
        str(score_path),
        str(predictions_path),
        "--pack", str(pack),
        "--split", split,
    ]

    if split == "hidden":
        command.append("--validate-only")
        output_name = "validation.txt"
    else:
        if partial:
            command.append("--partial")
        output_name = "official_scores.txt"

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    (output_dir / output_name).write_text(
        result.stdout + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Official scorer failed:\n{result.stdout}\n{result.stderr}"
        )

    print(f"Official scorer output: {output_dir / output_name}")

    if compare_path is not None:
        comparison = subprocess.run(
            command + ["--compare", str(compare_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        (output_dir / "paired_comparison.txt").write_text(
            comparison.stdout + comparison.stderr,
            encoding="utf-8",
        )
        if comparison.returncode != 0:
            raise RuntimeError(
                "Paired comparison failed:\n"
                + comparison.stdout + comparison.stderr
            )
        print(f"Paired comparison: {output_dir / 'paired_comparison.txt'}")


# ---------------------------------------------------------------------------
# Lightweight implementation checks
# ---------------------------------------------------------------------------

def implementation_checks(scorer):
    """Check masking, ranking, metrics, and parsing without loading models."""
    retriever = Retriever(
        corpus=[
            {"doc_id": "a"},
            {"doc_id": "b"},
            {"doc_id": "c"},
        ],
        config={},
        method="closed_book",
    )

    assert retriever.rank_scores(
        [3.0, 2.0, 1.0], 2, {"a"}
    ) == ["b", "c"]
    assert retriever.rank_scores(
        [3.0, 2.0, 1.0], 10, {"a", "b", "c"}
    ) == []
    assert top_indices([1.0, 3.0, 3.0], 3) == [1, 2, 0]

    label = {
        "gold_doc_ids": ["a", "b"],
        "answer": "Paris",
        "aliases": ["the Paris"],
    }

    for ranked in ([], ["c"], ["a", "c", "b"], ["a", "b"]):
        prediction = {
            "qid": "check",
            "answer": "Paris",
            "retrieved": ranked,
        }
        own = {
            **answer_metrics(prediction, label),
            **retrieval_metrics(ranked, label["gold_doc_ids"]),
        }
        reference = scorer.per_question(prediction, label)

        for name, value in reference.items():
            assert abs(own[name] - value) <= 1e-12

    assert clean_answer("NO_ANSWER") == ""
    assert clean_answer("<think>reasoning</think>Paris") == "Paris"
    assert clean_answer("<think>unfinished") == ""

    return {
        "status": "passed",
        "checks": [
            "document mask before top-k",
            "fully masked corpus",
            "stable ranking",
            "handwritten metrics against official synthetic examples",
            "answer parser",
        ],
        "scope": (
            "Implementation checks only. These do not replace the "
            "course-specific model self-check instructions."
        ),
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "base.yaml"),
    )
    parser.add_argument(
        "--pack",
        default=str(ROOT / "course_pack_v1"),
    )
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--split", choices=("dev", "hidden"), default="dev")
    parser.add_argument("--out", "--output", dest="out")
    parser.add_argument(
        "--score-script",
        default=str(ROOT / "starter" / "score.py"),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / ".cache" / "embeddings"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument(
        "--compare",
        help="Another dev predictions.jsonl file for official paired scoring.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run implementation checks without loading models or course data.",
    )

    args = parser.parse_args()

    if not args.self_test and (args.method is None or args.out is None):
        parser.error("--method and --out are required for an experiment.")

    return args


def main():
    args = parse_args()
    score_path = Path(args.score_script).expanduser().resolve()
    scorer = load_scorer(score_path)
    check_results = implementation_checks(scorer)

    if args.self_test:
        print(json.dumps(check_results, indent=2))
        return

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    if args.method == "oracle_context" and args.split != "dev":
        raise ValueError("Oracle-context is available only on dev.")

    if args.limit is not None:
        if args.split != "dev" or args.limit < 1:
            raise ValueError("--limit requires dev and a positive value.")

    if args.retrieval_only:
        if args.split != "dev" or args.method != "dense":
            raise ValueError("--retrieval-only is a dense dev smoke test.")
        if args.compare:
            raise ValueError("Do not compare retrieval-only and full QA runs.")

    if args.compare and args.split != "dev":
        raise ValueError("--compare is available only on dev.")

    pack = Path(args.pack).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()

    # Require a fresh directory to avoid mixing different experiments.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = pack / "corpus.jsonl"
    questions_path = pack / f"questions_{args.split}.jsonl"

    corpus = read_jsonl(corpus_path)
    all_questions = read_jsonl(questions_path)
    questions = (
        all_questions[:args.limit]
        if args.limit is not None else all_questions
    )
    partial = len(questions) < len(all_questions)

    if not corpus or not questions:
        raise ValueError("The corpus and selected questions must be non-empty.")

    documents_by_id = {doc["doc_id"]: doc for doc in corpus}
    if len(documents_by_id) != len(corpus):
        raise ValueError("Duplicate corpus document IDs.")

    if len({q["qid"] for q in all_questions}) != len(all_questions):
        raise ValueError("Duplicate question IDs.")

    retrieval = config["retrieval"]
    top_k = int(retrieval["top_k"])
    context_k = int(retrieval["context_k"])

    if args.method == "dense" and len(corpus) < top_k:
        raise ValueError("The corpus is smaller than retrieval.top_k.")

    prompt_paths = {
        name: project_path(config["prompts"][name])
        for name in ("closed_book", "rag")
    }
    prompts = {
        name: path.read_text(encoding="utf-8").strip()
        for name, path in prompt_paths.items()
    }
    if not all(prompts.values()):
        raise ValueError("Prompt files must not be empty.")

    # Gold labels are needed before generation only for oracle-context.
    labels = {}
    if args.method == "oracle_context":
        labels = {
            row["qid"]: row
            for row in read_jsonl(pack / "labels_dev.jsonl")
        }
        for question in questions:
            gold_ids = labels[question["qid"]]["gold_doc_ids"]
            if not gold_ids or len(gold_ids) != len(set(gold_ids)):
                raise ValueError("Oracle gold IDs must be non-empty and unique.")
            if any(doc_id not in documents_by_id for doc_id in gold_ids):
                raise ValueError("Oracle labels contain unknown document IDs.")

    compare_path = None
    if args.compare:
        compare_path = Path(args.compare).expanduser().resolve()
        other = read_jsonl(compare_path)
        validate_predictions(
            scorer, other, all_questions, corpus, partial
        )
        if {row["qid"] for row in other} != {q["qid"] for q in questions}:
            raise ValueError(
                "Paired runs must contain exactly the same question IDs."
            )

    set_seed(int(config["seed"]))
    corpus_digest = file_hash(corpus_path)

    run_config = {
        "system_name": {
            "closed_book": "closed_book",
            "dense": "dense_rag",
            "oracle_context": "oracle_context",
        }[args.method],
        "phase": config["phase"],
        "split": args.split,
        "description": config.get("description", ""),
        "models": {
            "llm": {
                "name": config["models"]["llm"],
                "where": "Kaggle/local, Hugging Face Transformers",
                "temperature": 0,
            },
            "embedder": config["models"]["embedder"],
            "reranker": None,
        },
        "settings": {
            "source_config": config,
            "generation_used": not args.retrieval_only,
            "retrieval": {
                "unit": "passage",
                "algorithm": (
                    "exact normalized dot product"
                    if args.method == "dense" else args.method
                ),
                "retrieved_k": top_k if args.method == "dense" else None,
                "context_k": (
                    context_k if args.method == "dense"
                    else "all gold passages"
                    if args.method == "oracle_context"
                    else 0
                ),
                "mask_policy": "exclude document IDs before top-k",
                "query_prefix": retrieval.get("query_prefix", ""),
                "query_prefix_used": args.method == "dense",
                "document_format": "title + newline + text",
            },
            "context_policy": "equal text-token allocation with final budget check",
            "do_sample": False,
            "thinking_disabled": (
                "qwen3" in config["models"]["llm"].lower()
                and not args.retrieval_only
            ),
            "answer_parser": "remove think blocks; NO_ANSWER becomes empty string",
        },
        "seed": int(config["seed"]),
        "n_questions": len(questions),
        "partial": partial,
        "retrieval_only": args.retrieval_only,
        "submission_ready": not partial and not args.retrieval_only,
        "pack": str(pack),
        "config_file": str(config_path),
        "command": shlex.join([sys.executable, *sys.argv]),
        "date": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "versions": package_versions(),
        "hashes": {
            "config": file_hash(config_path),
            "corpus": corpus_digest,
            "questions": file_hash(questions_path),
            "run_py": file_hash(Path(__file__)),
            "score_py": file_hash(score_path),
            "prompts": {
                name: file_hash(path)
                for name, path in prompt_paths.items()
            },
        },
        "prompts": prompts,
        "implementation_checks": check_results,
        "compare": str(compare_path) if compare_path else None,
        "status": "started",
    }

    config_output = output_dir / "run_config.yaml"

    def save_run_config():
        with config_output.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                run_config,
                file,
                allow_unicode=True,
                sort_keys=False,
            )

    save_run_config()
    save_json(output_dir / "implementation_checks.json", check_results)

    retriever = None
    if args.method == "dense":
        retriever = Retriever(
            corpus=corpus,
            config=config,
            method="dense",
            corpus_hash=corpus_digest,
            cache_dir=Path(args.cache_dir).expanduser().resolve(),
        )

    generator = None
    if not args.retrieval_only:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        generator = AnswerGenerator(config, prompts)
        run_config["settings"]["effective_load_in_4bit"] = (
            generator.effective_4bit
        )
        run_config["settings"]["llm_device"] = str(generator.model.device)
        run_config["settings"]["quantization"] = (
            "bitsandbytes NF4"
            if generator.effective_4bit else None
        )
        save_run_config()

    from tqdm.auto import tqdm

    predictions = []
    traces = {}
    predictions_path = output_dir / "predictions.jsonl"

    with predictions_path.open("w", encoding="utf-8") as file:
        for question in tqdm(questions, desc=f"{args.method}/{args.split}"):
            qid = question["qid"]

            if args.method == "closed_book":
                retrieved = []
                context_ids = []
            elif args.method == "dense":
                retrieved = retriever.dense_search(
                    question["question"],
                    top_k,
                    exclude_ids=frozenset(),
                )
                if len(retrieved) < 10:
                    raise ValueError("Dense predictions require at least 10 IDs.")
                context_ids = retrieved[:context_k]
            else:
                retrieved = list(labels[qid]["gold_doc_ids"])
                context_ids = retrieved

            documents = [
                documents_by_id[doc_id] for doc_id in context_ids
            ]

            if args.retrieval_only:
                answer, raw_answer, context_info = "", "", []
            else:
                answer, raw_answer, context_info = generator.answer(
                    question["question"],
                    documents,
                    args.method,
                )

            # Only the official required fields are written here.
            prediction = {
                "qid": qid,
                "answer": answer,
                "retrieved": retrieved,
            }
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            file.flush()
            predictions.append(prediction)

            traces[qid] = {
                "question": question["question"],
                "raw_answer": raw_answer,
                "context_ids": context_ids,
                "context_info": context_info,
            }

    validate_predictions(
        scorer, predictions, all_questions, corpus, partial
    )

    # For non-oracle systems, dev labels are loaded only after predictions.
    # Hidden labels are never accessed.
    if args.split == "dev":
        if not labels:
            labels = {
                row["qid"]: row
                for row in read_jsonl(pack / "labels_dev.jsonl")
            }

        run_config["hashes"]["dev_labels"] = file_hash(
            pack / "labels_dev.jsonl"
        )
        evaluate_dev(
            predictions,
            labels,
            traces,
            args.method,
            scorer,
            output_dir,
        )

    run_official_score(
        score_path,
        predictions_path,
        pack,
        args.split,
        partial,
        output_dir,
        compare_path,
    )

    run_config["status"] = "completed"
    save_run_config()
    print(f"\nSaved run: {output_dir}")


if __name__ == "__main__":
    main()