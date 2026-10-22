"""Phase 1: closed-book, dense RAG, and oracle-context evaluation."""

import argparse
import gc
import importlib.util
import json
import math
import random
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm


METHODS = ("closed_book", "dense", "oracle_context")
KS = (2, 5, 10)

SYSTEM_PROMPT = (
    "Answer the question with only a short final answer: "
    "an entity, place, date, number, or yes/no. "
    "Do not provide explanations or reasoning."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_jsonl(path):
    """Read non-empty JSONL lines."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl_row(file, row):
    """Write one prediction and flush it to disk."""
    file.write(json.dumps(row, ensure_ascii=False) + "\n")
    file.flush()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Exact dense retrieval
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Search all normalized document vectors using cosine similarity."""

    def __init__(self, corpus, model_name, batch_size):
        self.doc_ids = [doc["doc_id"] for doc in corpus]

        # CPU embedding keeps GPU memory available for the language model.
        self.encoder = SentenceTransformer(model_name, device="cpu")

        texts = [
            f"{doc['title']}\n{doc['text']}"
            for doc in corpus
        ]

        self.embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def dense_search(self, question, k, exclude_ids=frozenset()):
        """Return the top-k document IDs, excluding masked documents."""
        query = self.encoder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        # Dot product equals cosine similarity for normalized vectors.
        scores = self.embeddings @ query

        # Remove excluded documents before ranking.
        allowed = np.array([
            index
            for index, doc_id in enumerate(self.doc_ids)
            if doc_id not in exclude_ids
        ], dtype=np.int64)

        if len(allowed) == 0:
            return []

        # Stable sorting makes equal-score ordering reproducible.
        order = np.argsort(-scores[allowed], kind="stable")
        selected = allowed[order[:k]]

        return [self.doc_ids[index] for index in selected]


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

class AnswerGenerator:
    """Generate short answers using one shared language model."""

    def __init__(self, model_name, max_input_tokens, max_new_tokens):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}

        if torch.cuda.is_available():
            # Quantize the model to reduce memory use on a Kaggle T4.
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = {"": 0}
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        self.model.eval()

        context_limit = getattr(
            self.model.config, "max_position_embeddings", None
        )

        if context_limit is not None:
            if max_input_tokens + max_new_tokens > context_limit:
                raise ValueError(
                    "Input and generation budgets exceed the model context."
                )

    def render_prompt(self, question, context):
        """Apply the model's chat template."""
        if context:
            user_text = (
                "Use the following evidence to answer the question.\n\n"
                f"Evidence:\n{context}\n\n"
                f"Question: {question}"
            )
        else:
            user_text = f"Question: {question}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        template_kwargs = {}

        # Qwen3 supports disabling thinking for short-answer evaluation.
        if "qwen3" in self.model_name.lower():
            template_kwargs["enable_thinking"] = False

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

    def build_prompt(self, question, documents):
        """Allocate context space equally across the selected documents."""
        if not documents:
            return self.render_prompt(question, "")

        # Include document headers when estimating the fixed prompt cost.
        headers = [
            f"[{doc['doc_id']}] {doc['title']}"
            for doc in documents
        ]
        empty_context = "\n\n".join(headers)
        base_prompt = self.render_prompt(question, empty_context)
        base_size = len(
            self.tokenizer.encode(base_prompt, add_special_tokens=False)
        )

        # Reserve a small margin for tokenization boundary differences.
        available = self.max_input_tokens - base_size - 128

        if available < len(documents):
            raise ValueError(
                "The input budget is too small for the selected documents."
            )

        tokens_per_doc = available // len(documents)
        parts = []

        for doc, header in zip(documents, headers):
            text_ids = self.tokenizer.encode(
                doc["text"],
                add_special_tokens=False,
            )
            text = self.tokenizer.decode(
                text_ids[:tokens_per_doc],
                skip_special_tokens=True,
            )
            parts.append(f"{header}\n{text}")

        return self.render_prompt(question, "\n\n".join(parts))

    def answer(self, question, documents):
        """Generate only tokens following the input prompt."""
        prompt = self.build_prompt(question, documents)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        if inputs["input_ids"].shape[1] > self.max_input_tokens:
            raise ValueError(
                "Prompt exceeds max_input_tokens; increase the input budget."
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

        generated = output[0, input_length:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()


# ---------------------------------------------------------------------------
# Handwritten metrics
# ---------------------------------------------------------------------------

def normalize_answer(text):
    """Match the answer normalization used by the course scorer."""
    text = "".join(
        char for char in text.lower()
        if char not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, reference):
    """Calculate token-level F1 without a metric library."""
    pred = normalize_answer(prediction)
    gold = normalize_answer(reference)

    special_answers = ("yes", "no", "noanswer")

    if (pred in special_answers or gold in special_answers) and pred != gold:
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


def question_metrics(prediction, label):
    """Calculate answer and retrieval metrics for one question."""
    gold_ids = set(label["gold_doc_ids"])
    ranked = prediction["retrieved"]

    if not gold_ids:
        raise ValueError("Retrieval evaluation requires gold documents.")

    metrics = {}

    for k in KS:
        top_ids = set(ranked[:k])
        metrics[f"recall@{k}"] = len(top_ids & gold_ids) / len(gold_ids)
        metrics[f"hitall@{k}"] = float(gold_ids <= top_ids)

    metrics["mrr"] = next(
        (
            1.0 / rank
            for rank, doc_id in enumerate(ranked, start=1)
            if doc_id in gold_ids
        ),
        0.0,
    )

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:10], start=1)
        if doc_id in gold_ids
    )
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(gold_ids), 10) + 1)
    )
    metrics["ndcg@10"] = dcg / ideal_dcg

    answer = prediction["answer"]
    references = [label["answer"]] + label.get("aliases", [])

    metrics["em"] = (
        float(any(
            normalize_answer(answer) == normalize_answer(reference)
            for reference in references
        ))
        if answer else 0.0
    )
    metrics["f1"] = (
        max(token_f1(answer, reference) for reference in references)
        if answer else 0.0
    )

    return metrics


def evaluate_dev(predictions, labels, score_path, questions, corpus):
    """Validate predictions and compare handwritten metrics with score.py."""
    spec = importlib.util.spec_from_file_location(
        "course_score", score_path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load scorer: {score_path}")

    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)

    errors = scorer.validate(
        predictions,
        {question["qid"] for question in questions},
        {doc["doc_id"] for doc in corpus},
        partial=len(predictions) < len(questions),
    )

    if errors:
        raise ValueError("\n".join(errors[:20]))

    rows = []
    max_difference = 0.0

    for prediction in predictions:
        label = labels[prediction["qid"]]

        # Compute our metrics independently, then check the staff reference.
        own = question_metrics(prediction, label)
        reference = scorer.per_question(prediction, label)

        for name, value in own.items():
            difference = abs(value - reference[name])
            max_difference = max(max_difference, difference)

        rows.append(own)

    if max_difference >= 0.001:
        raise AssertionError(
            f"Metric mismatch with score.py: {max_difference}"
        )

    averages = {
        name: sum(row[name] for row in rows) / len(rows)
        for name in rows[0]
    }

    print("\nHandwritten metrics (percent):")
    for name, value in averages.items():
        print(f"  {name:12s}: {100 * value:.2f}")

    print(f"Maximum difference from score.py: {max_difference:.8f}")

    return {
        "n": len(rows),
        "metrics": averages,
        "max_difference_from_score_py": max_difference,
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--pack", required=True)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--split", default="dev", choices=("dev", "hidden"))
    parser.add_argument("--output", required=True)

    parser.add_argument("--llm", default="Qwen/Qwen3-8B")
    parser.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=32)

    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run a small test subset instead of the full split.",
    )
    parser.add_argument(
        "--score-script",
        default=str(Path(__file__).resolve().with_name("score.py")),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.method == "oracle_context" and args.split != "dev":
        raise ValueError("Oracle-context is only available on dev.")

    if args.top_k < 10:
        raise ValueError("Use at least 10 retrieved documents for dense.")

    if not 1 <= args.context_k <= args.top_k:
        raise ValueError("context-k must be between 1 and top-k.")

    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive.")

    if args.max_input_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError("Token budgets must be positive.")

    if args.embedding_batch_size < 1:
        raise ValueError("Embedding batch size must be positive.")

    set_seed(args.seed)

    pack = Path(args.pack)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.jsonl"

    # Prevent accidental replacement of an existing experiment.
    if predictions_path.exists():
        raise FileExistsError(
            f"{predictions_path} already exists. Use a new output folder."
        )

    corpus = read_jsonl(pack / "corpus.jsonl")
    all_questions = read_jsonl(pack / f"questions_{args.split}.jsonl")
    questions = (
        all_questions[:args.limit]
        if args.limit is not None else all_questions
    )

    if not questions:
        raise ValueError("No questions were loaded.")

    documents_by_id = {doc["doc_id"]: doc for doc in corpus}

    # Hidden labels are never loaded.
    labels = {}
    if args.split == "dev":
        labels = {
            label["qid"]: label
            for label in read_jsonl(pack / "labels_dev.jsonl")
        }

        if not Path(args.score_script).is_file():
            raise FileNotFoundError(args.score_script)

    retriever = None

    if args.method == "dense":
        retriever = DenseRetriever(
            corpus,
            args.embedder,
            args.embedding_batch_size,
        )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generator = AnswerGenerator(
        args.llm,
        args.max_input_tokens,
        args.max_new_tokens,
    )

    config = {
        "method": args.method,
        "split": args.split,
        "pack": str(pack.resolve()),
        "llm": args.llm,
        "quantization": "bitsandbytes_nf4" if torch.cuda.is_available() else None,
        "embedder": args.embedder if args.method == "dense" else None,
        "embedding_device": "cpu" if args.method == "dense" else None,
        "document_format": "title + newline + text",
        "retrieval": {
            "type": "exact_cosine" if args.method == "dense" else args.method,
            "top_k": args.top_k if args.method == "dense" else None,
            "context_k": args.context_k if args.method == "dense" else None,
        },
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "context_policy": "equal token allocation across selected documents",
        "do_sample": False,
        "qwen3_thinking_disabled": "qwen3" in args.llm.lower(),
        "system_prompt": SYSTEM_PROMPT,
        "seed": args.seed,
        "n_questions": len(questions),
        "partial": len(questions) < len(all_questions),
        "versions": {
            "torch": str(torch.__version__),
        },
    }

    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    predictions = []

    with open(predictions_path, "w", encoding="utf-8") as file:
        for question in tqdm(questions, desc=args.method):
            qid = question["qid"]

            if args.method == "closed_book":
                retrieved = []
                context_ids = []

            elif args.method == "dense":
                retrieved = retriever.dense_search(
                    question["question"],
                    args.top_k,
                )
                context_ids = retrieved[:args.context_k]

            else:
                # Oracle uses every gold document, not dense context-k.
                retrieved = list(dict.fromkeys(labels[qid]["gold_doc_ids"]))
                context_ids = retrieved

            context_documents = [
                documents_by_id[doc_id]
                for doc_id in context_ids
            ]

            answer = generator.answer(
                question["question"],
                context_documents,
            )

            # Only fields allowed by predictions_format.md are written.
            prediction = {
                "qid": qid,
                "answer": answer,
                "retrieved": retrieved,
            }

            write_jsonl_row(file, prediction)
            predictions.append(prediction)

    print(f"\nSaved predictions: {predictions_path}")

    if args.split == "dev":
        metrics = evaluate_dev(
            predictions,
            labels,
            args.score_script,
            all_questions,
            corpus,
        )

        with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()