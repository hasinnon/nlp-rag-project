# Run-file format: `predictions.jsonl`

One JSON object per line, one line per question of the split (every `qid` exactly once, no extra questions).

| field | type | required | meaning |
|---|---|---|---|
| `qid` | string | yes | question id from `questions_<split>.jsonl` |
| `answer` | string | yes | short final answer (an entity, date, place, number, or `yes`/`no`); `""` if you abstain |
| `retrieved` | list of strings | yes | doc ids of `corpus.jsonl`, **ranked best first**, no duplicates; at least 10 recommended (metrics use the top 2, 5 and 10). Use `[]` for a system without retrieval (closed-book). |
| `cited` | list of strings | no | doc ids your answer is based on |
| `abstained` | bool | no | `true` if the system declined to answer |

Example line:

```json
{"qid": "q0007", "answer": "Steven Spielberg", "retrieved": ["p01234", "p00871", "p03002"], "cited": ["p01234"], "abstained": false}
```

No other fields are allowed. Check your file with `python score.py <file> --pack course_pack_v1 --split dev --validate-only`.
Put the file in its own folder together with `run_config.yaml` (see `example_run_config.yaml`).
