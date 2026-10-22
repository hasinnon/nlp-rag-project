"""Minimal example: writes a valid predictions file (random retrieval, fixed answer).

    python example_predict.py --pack course_pack_v1 --split dev --out runs/example/predictions.jsonl
    python score.py runs/example/predictions.jsonl --pack course_pack_v1 --split dev

Replace `retrieve` and `answer` with your own system. Scores will be near zero: that is the point.
"""

import argparse
import json
import random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--pack", default="course_pack_v1")
ap.add_argument("--split", default="dev")
ap.add_argument("--out", default="runs/example/predictions.jsonl")
a = ap.parse_args()

pack = Path(a.pack)
corpus = [json.loads(line) for line in open(pack / "corpus.jsonl", encoding="utf-8")]
questions = [json.loads(line) for line in open(pack / f"questions_{a.split}.jsonl", encoding="utf-8")]
doc_ids = [d["doc_id"] for d in corpus]
rng = random.Random(42)


def retrieve(question, k=10):          # your retriever goes here: return ranked doc ids
    return rng.sample(doc_ids, k)


def answer(question, doc_ids):         # your reader (LLM) goes here: return a short answer string
    return "yes"


Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    for q in questions:
        ranked = retrieve(q["question"])
        row = {"qid": q["qid"], "answer": answer(q["question"], ranked), "retrieved": ranked}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {len(questions)} predictions to {a.out}")
