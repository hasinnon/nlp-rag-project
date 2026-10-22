"""Run closed-book, BM25, dense, or hybrid QA baselines."""

import argparse
import hashlib
import importlib.metadata
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("closed_book", "bm25", "dense", "hybrid")


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON at line {number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {number} must be an object")
            rows.append(row)
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_text(doc):
    return f"{doc['title']}\n{doc['text']}"


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def top_indices(scores, k):
    # Stable sorting gives deterministic ordering for ties.
    return np.argsort(-np.asarray(scores), kind="stable")[:k].tolist()


def rrf_scores(rankings, size, constant=60):
    scores = np.zeros(size, dtype=np.float64)
    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            scores[index] += 1.0 / (constant + rank)
    return scores


def validate_inputs(corpus, questions):
    if not corpus or not questions:
        raise ValueError("Corpus and question split must not be empty.")

    for doc in corpus:
        for field in ("doc_id", "title", "text"):
            if not isinstance(doc.get(field), str):
                raise ValueError(f"Invalid corpus field: {field}")

    for question in questions:
        for field in ("qid", "question"):
            if not isinstance(question.get(field), str):
                raise ValueError(f"Invalid question field: {field}")

    if len({d["doc_id"] for d in corpus}) != len(corpus):
        raise ValueError("Duplicate document IDs in corpus.")
    if len({q["qid"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs in question split.")


class Retriever:
    def __init__(self, corpus, config, method, corpus_hash, cache_dir):
        self.corpus = corpus
        self.method = method
        self.config = config
        self.bm25 = None
        self.embedder = None
        self.embeddings = None

        if method in ("bm25", "hybrid"):
            tokenized = [tokenize(document_text(d)) for d in corpus]
            if not any(tokenized):
                raise ValueError("Corpus has no indexable tokens.")
            self.bm25 = BM25Okapi(tokenized)

        if method in ("dense", "hybrid"):
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(
                config["models"]["embedder"],
                device=config["retrieval"]["embedding_device"],
            )

            # Corpus text, embedding model, and encoding policy define the cache.
            identity = {
                "corpus_sha256": corpus_hash,
                "model": config["models"]["embedder"],
                "encoding": "title-newline-text;normalized;v1",
                "sentence_transformers": importlib.metadata.version(
                    "sentence-transformers"
                ),
            }
            key = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode("utf-8")
            ).hexdigest()

            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{key}.npy"

            if cache_path.exists():
                self.embeddings = np.load(cache_path, allow_pickle=False)
            else:
                self.embeddings = self.embedder.encode(
                    [document_text(d) for d in corpus],
                    batch_size=config["retrieval"]["embedding_batch_size"],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                )
                np.save(cache_path, self.embeddings)

            if self.embeddings.shape[0] != len(corpus):
                raise ValueError("Embedding cache does not match corpus size.")

    def retrieve(self, question, k):
        if self.method == "closed_book":
            return []

        k = min(k, len(self.corpus))
        rankings = []

        if self.bm25 is not None:
            scores = self.bm25.get_scores(tokenize(question))
            rankings.append(top_indices(scores, len(self.corpus)))

        if self.embedder is not None:
            prefix = self.config["retrieval"].get("query_prefix", "")
            query = self.embedder.encode(
                prefix + question,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            scores = self.embeddings @ query
            rankings.append(top_indices(scores, len(self.corpus)))

        if self.method == "hybrid":
            scores = rrf_scores(
                rankings,
                len(self.corpus),
                self.config["retrieval"]["rrf_k"],
            )
            indices = top_indices(scores, k)
        else:
            indices = rankings[0][:k]

        return [self.corpus[i]["doc_id"] for i in indices]


class Generator:
    def __init__(self, config, method):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            set_seed,
        )

        set_seed(config["seed"])
        self.torch = torch
        self.settings = config["generation"]
        model_name = config["models"]["llm"]

        if self.settings["load_in_4bit"] and not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit mode requires a supported GPU environment. "
                "Enable a Kaggle GPU, or set load_in_4bit: false."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"device_map": "auto"}
        if self.settings["load_in_4bit"]:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            kwargs["torch_dtype"] = (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

        max_positions = getattr(
            self.model.config, "max_position_embeddings", None
        )
        total_tokens = (
            self.settings["max_input_tokens"] + self.settings["max_new_tokens"]
        )
        if max_positions is not None and total_tokens > max_positions:
            raise ValueError("Configured input + output exceeds model context.")

        prompt_key = "closed_book" if method == "closed_book" else "rag"
        prompt_path = ROOT / config["prompts"][prompt_key]
        self.system_prompt = prompt_path.read_text(encoding="utf-8")

    def render(self, question, blocks):
        if blocks:
            user_text = (
                "Passages:\n\n"
                + "\n\n".join(blocks)
                + f"\n\nQuestion: {question}"
            )
        else:
            user_text = f"Question: {question}"

        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def token_count(self, prompt):
        return len(
            self.tokenizer.encode(prompt, add_special_tokens=False)
        )

    def build_prompt(self, question, documents):
        limit = self.settings["max_input_tokens"]
        if self.token_count(self.render(question, [])) > limit:
            raise ValueError("Question and instructions exceed input budget.")

        blocks = []
        for doc in documents:
            header = f"[{doc['doc_id']}] {doc['title']}\n"
            full_block = header + doc["text"]

            candidate = self.render(question, blocks + [full_block])
            if self.token_count(candidate) <= limit:
                blocks.append(full_block)
                continue

            # Fit a prefix of the final passage while preserving the question.
            text_tokens = self.tokenizer.encode(
                doc["text"], add_special_tokens=False
            )
            low, high = 0, len(text_tokens)
            while low < high:
                middle = (low + high + 1) // 2
                prefix = self.tokenizer.decode(
                    text_tokens[:middle], skip_special_tokens=True
                )
                candidate = self.render(question, blocks + [header + prefix])
                if self.token_count(candidate) <= limit:
                    low = middle
                else:
                    high = middle - 1

            if low > 0:
                prefix = self.tokenizer.decode(
                    text_tokens[:low], skip_special_tokens=True
                )
                blocks.append(header + prefix)
            break

        return self.render(question, blocks)

    def answer(self, question, documents):
        prompt = self.build_prompt(question, documents)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        device = self.model.get_input_embeddings().weight.device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.settings["max_new_tokens"],
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0, inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()

        abstained = not answer or answer == "NO_ANSWER"
        return ("" if abstained else answer), abstained


def environment_versions():
    packages = (
        "numpy",
        "PyYAML",
        "rank-bm25",
        "torch",
        "transformers",
        "sentence-transformers",
        "accelerate",
        "bitsandbytes",
    )
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--pack", default="course_pack_v1")
    parser.add_argument("--split", choices=("dev", "hidden"), default="dev")
    parser.add_argument("--method", choices=METHODS, default="bm25")
    parser.add_argument("--out", required=True, help="Output run directory")
    parser.add_argument("--cache", default=".cache/embeddings")
    parser.add_argument("--limit", type=int, help="Dev smoke test only")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM; write empty answers for retrieval debugging",
    )
    args = parser.parse_args()

    if args.limit is not None:
        if args.split != "dev" or args.limit < 1:
            parser.error("--limit must be positive and is allowed only on dev.")

    if args.retrieval_only and args.method == "closed_book":
        parser.error("--retrieval-only is not useful with closed_book.")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = config["retrieval"]

    if not 1 <= settings["context_k"] <= settings["top_k"]:
        parser.error("Require 1 <= context_k <= top_k.")
    if settings["rrf_k"] <= 0:
        parser.error("rrf_k must be positive.")

    random.seed(config["seed"])
    np.random.seed(config["seed"])

    pack = Path(args.pack)
    corpus_path = pack / "corpus.jsonl"
    question_path = pack / f"questions_{args.split}.jsonl"

    corpus = read_jsonl(corpus_path)
    questions = read_jsonl(question_path)
    validate_inputs(corpus, questions)

    if args.limit is not None:
        questions = questions[:args.limit]

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    run_config_path = output_dir / "run_config.yaml"

    if prediction_path.exists() or run_config_path.exists():
        raise FileExistsError("Use a new output directory for each run.")

    corpus_hash = sha256_file(corpus_path)
    retriever = Retriever(
        corpus, config, args.method, corpus_hash, args.cache
    )
    generator = None if args.retrieval_only else Generator(config, args.method)
    by_id = {doc["doc_id"]: doc for doc in corpus}

    temporary_path = output_dir / "predictions.jsonl.tmp"
    with temporary_path.open("w", encoding="utf-8") as handle:
        for index, question in enumerate(questions, 1):
            ranked = retriever.retrieve(
                question["question"], settings["top_k"]
            )
            context = [by_id[d] for d in ranked[:settings["context_k"]]]

            if generator is None:
                answer, abstained = "", True
            else:
                answer, abstained = generator.answer(
                    question["question"], context
                )

            row = {
                "qid": question["qid"],
                "answer": answer,
                "retrieved": ranked,
                "abstained": abstained,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(questions)}] {question['qid']}", flush=True)

    # Preserve model declarations and actual execution settings.
    metadata = {
        "system_name": args.method,
        "phase": config["phase"],
        "split": args.split,
        "description": config["description"],
        "models": {
            "llm": {
                "name": config["models"]["llm"],
                "backend": "transformers",
                "load_in_4bit": config["generation"]["load_in_4bit"],
                "do_sample": False,
            },
            "embedder": config["models"]["embedder"],
            "reranker": None,
        },
        "settings": config,
        "execution": {
            "method": args.method,
            "retrieval_only": args.retrieval_only,
            "llm_loaded": generator is not None,
            "embedder_loaded": retriever.embedder is not None,
            "n_questions": len(questions),
            "partial": len(questions) < len(read_jsonl(question_path)),
            "arguments": vars(args),
        },
        "provenance": {
            "corpus_sha256": corpus_hash,
            "questions_sha256": sha256_file(question_path),
            "config_sha256": sha256_file(config_path),
            "code_sha256": sha256_file(Path(__file__)),
            "prompt_sha256": {
                key: sha256_file(ROOT / path)
                for key, path in config["prompts"].items()
            },
        },
        "environment": environment_versions(),
        "date": datetime.now(timezone.utc).isoformat(),
    }

    import shlex
    import sys

    metadata["command"] = shlex.join([sys.executable, *sys.argv])
    run_config_path.write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary_path.replace(prediction_path)
    print(f"Saved predictions: {prediction_path}")


if __name__ == "__main__":
    main()