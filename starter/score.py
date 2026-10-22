"""Course scoring script (owned by the course staff; students use it on dev).

    python score.py runs/dense/predictions.jsonl --pack course_pack_v1 --split dev
    python score.py runs/hybrid/predictions.jsonl --pack course_pack_v1 --split dev --compare runs/dense/predictions.jsonl
    python score.py runs/selfcheck/predictions.jsonl --pack course_pack_v1 --split dev --partial   # subset of dev
    python score.py preds.jsonl --pack course_pack_v1 --split hidden --labels labels_hidden.jsonl   # staff only

Step 1 validates the predictions file (see predictions_format.md); it stops with a list of problems if invalid.
Step 2 prints metrics overall and per question type, with 95 % bootstrap intervals.
--compare B prints the paired difference (this file minus B) on the same questions.
Only the Python standard library is used.
"""

import argparse
import json
import math
import random
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

KS = (2, 5, 10)
SCHEMA = {  # machine-readable summary of predictions_format.md
    "required": {"qid": str, "answer": str, "retrieved": list},
    "optional": {"cited": list, "abstained": bool},
}


# ---------------------------------------------------------------- loading and validation
def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    sys.exit(f"{path}: line {n} is not valid JSON ({e.msg})")
    return rows


def validate(preds, qids, doc_ids, partial=False):
    errs, seen = [], Counter(p.get("qid") for p in preds)
    for q in sorted(qids - set(seen)) if not partial else []:
        errs.append(f"missing prediction for {q}")
    for q, n in seen.items():
        if q not in qids:
            errs.append(f"unknown qid {q!r} (not in this split)")
        elif n > 1:
            errs.append(f"{q}: {n} predictions (expected exactly 1)")
    for p in preds:
        q = p.get("qid", "?")
        for k, t in SCHEMA["required"].items():
            if not isinstance(p.get(k), t):
                errs.append(f"{q}: field '{k}' missing or not a {t.__name__}")
        for k, t in SCHEMA["optional"].items():
            if k in p and not isinstance(p[k], t):
                errs.append(f"{q}: field '{k}' is not a {t.__name__}")
        for k in ("retrieved", "cited"):
            ids = p.get(k) if isinstance(p.get(k), list) else []
            bad = [d for d in ids if d not in doc_ids]
            if bad:
                errs.append(f"{q}: '{k}' contains unknown doc ids {bad[:3]}")
            if len(set(map(str, ids))) != len(ids):
                errs.append(f"{q}: '{k}' contains duplicates")
        unknown = set(p) - set(SCHEMA["required"]) - set(SCHEMA["optional"])
        if unknown:
            errs.append(f"{q}: unexpected fields {sorted(unknown)} (allowed: see predictions_format.md)")
    return errs


# ---------------------------------------------------------------- metrics
def normalize(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if (p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer")) and p != g:
        return 0.0
    common = sum((Counter(p.split()) & Counter(g.split())).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p.split()), common / len(g.split())
    return 2 * pr * rc / (pr + rc)


def per_question(p, lab):
    gold, ranked = set(lab["gold_doc_ids"]), p["retrieved"]
    m = {}
    for k in KS:
        top = set(ranked[:k])
        m[f"recall@{k}"] = len(gold & top) / len(gold)
        m[f"hitall@{k}"] = float(gold <= top)
    m["mrr"] = next((1 / i for i, d in enumerate(ranked, 1) if d in gold), 0.0)
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(ranked[:10], 1) if d in gold)
    m["ndcg@10"] = dcg / sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), 10) + 1))
    answer = "" if p.get("abstained") else p["answer"]
    golds = [lab["answer"]] + lab.get("aliases", [])
    m["em"] = float(any(normalize(answer) == normalize(g) for g in golds)) if answer else 0.0
    m["f1"] = max(f1(answer, g) for g in golds) if answer else 0.0
    if "cited" in p:
        c = set(p["cited"])
        m["cite_precision"] = len(c & gold) / len(c) if c else 0.0
        m["cite_recall"] = len(c & gold) / len(gold)
    if "abstained" in p:
        m["abstained"] = float(p["abstained"])
    return m


def bootstrap(values, n=1000, seed=0):
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def table(rows, title):
    names = list(rows[0]) if rows else []
    print(f"\n{title} (n={len(rows)})")
    for name in names:
        vals = [r[name] for r in rows if name in r]
        lo, hi = bootstrap(vals)
        print(f"  {name:15s} {100 * sum(vals) / len(vals):6.1f}   [{100 * lo:5.1f}, {100 * hi:5.1f}]")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("predictions")
    ap.add_argument("--pack", default="course_pack_v1")
    ap.add_argument("--split", default="dev", choices=["dev", "hidden"])
    ap.add_argument("--labels", help="labels file (default: <pack>/labels_<split>.jsonl)")
    ap.add_argument("--compare", help="second predictions file for a paired comparison")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--partial", action="store_true",
                    help="dev only: score just the questions present (e.g. a 50-question self-check)")
    a = ap.parse_args()
    if a.partial and a.split != "dev":
        sys.exit("--partial is only allowed on dev; hidden-set submissions must cover every question")
    pack = Path(a.pack)
    qids = {q["qid"] for q in read_jsonl(pack / f"questions_{a.split}.jsonl")}
    doc_ids = {d["doc_id"] for d in read_jsonl(pack / "corpus.jsonl")}
    files = [a.predictions] + ([a.compare] if a.compare else [])
    loaded = {}
    for path in files:
        preds = read_jsonl(path)
        errs = validate(preds, qids, doc_ids, a.partial)
        if errs:
            print(f"INVALID: {path} ({len(errs)} problems)")
            for e in errs[:20]:
                print("  -", e)
            sys.exit(1)
        loaded[path] = {p["qid"]: p for p in preds}
        print(f"valid: {path} ({len(preds)} predictions{' -- PARTIAL, not a full split' if a.partial else ''})")
    if a.validate_only:
        return
    lab_path = Path(a.labels) if a.labels else pack / f"labels_{a.split}.jsonl"
    if not lab_path.exists():
        sys.exit(f"no labels at {lab_path} (hidden labels are held by the TAs)")
    labels = {r["qid"]: r for r in read_jsonl(lab_path)}
    scores = {path: {q: per_question(p, labels[q]) for q, p in preds.items()} for path, preds in loaded.items()}
    main_scores = scores[a.predictions]
    table(list(main_scores.values()), "overall")
    by_type = defaultdict(list)
    for q, m in main_scores.items():
        by_type[labels[q]["type"]].append(m)
    for t in sorted(by_type):
        table(by_type[t], f"type: {t}")
    if a.compare:
        other = scores[a.compare]
        print(f"\npaired difference: {a.predictions} minus {a.compare} (points, 95 % bootstrap interval)")
        for name in main_scores[next(iter(main_scores))]:
            diffs = [main_scores[q][name] - other[q][name] for q in sorted(main_scores) if name in other[q]]
            if diffs:
                lo, hi = bootstrap(diffs)
                print(f"  {name:15s} {100 * sum(diffs) / len(diffs):+6.1f}   [{100 * lo:+5.1f}, {100 * hi:+5.1f}]")


if __name__ == "__main__":
    main()
