import argparse
import json
from pathlib import Path

from .chunking import chunk_doc, load_docs
from .fetch import RAW_DIR
from .retrieval import MODELS, build_index, search


def norm(s: str) -> str:
    return " ".join(s.split())


def first_hit_rank(results: list[dict], doc: str, phrase: str):
    """Rank (1-based) of the first retrieved chunk from the right page that contains the phrase."""
    for i, r in enumerate(results, 1):
        if r["meta"]["doc_id"] == doc and norm(phrase) in norm(r["text"]):
            return i
    return None


def summarize(ranks: list) -> dict:
    n = len(ranks)
    out = {f"recall@{k}": sum(1 for r in ranks if r and r <= k) / n for k in (1, 3, 5)}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    return out


def validate(questions: list[dict], chunk_texts: dict[str, list[str]]) -> list[str]:
    """Check every question has its fields, a known page, and a phrase that exists in one of that page's chunks."""
    problems = []
    for i, q in enumerate(questions, 1):
        missing = [k for k in ("q", "doc", "phrase", "type") if k not in q]
        if missing:
            problems.append(f"question {i}: missing {missing}")
            continue
        texts = chunk_texts.get(q["doc"])
        if texts is None:
            problems.append(f"question {i}: unknown doc '{q['doc']}'")
        elif not any(norm(q["phrase"]) in norm(t) for t in texts):
            problems.append(f"question {i}: phrase not found in any chunk of '{q['doc']}': {q['phrase']}")
    return problems


def load_questions(path: str) -> list[dict]:
    questions = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            questions.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}, line {n}: invalid JSON ({e.msg})")
    return questions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e5-small", choices=list(MODELS))
    ap.add_argument("--max-chars", type=int, default=800)
    ap.add_argument("--no-prefix", action="store_true")
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--file", default="eval/questions.jsonl")
    args = ap.parse_args()

    questions = load_questions(args.file)
    chunk_texts = {d["id"]: [c["text"] for c in chunk_doc(d)] for d in load_docs(RAW_DIR)}
    problems = validate(questions, chunk_texts)
    if problems:
        print("\n".join(problems))
        raise SystemExit(f"\n{len(problems)} problem(s) in the questions file. Fix them and run again.")

    col, emb, n_chunks = build_index(args.model, args.max_chars,
                                     with_header=not args.no_header, use_prefix=not args.no_prefix)
    ranks = []
    for q in questions:
        rank = first_hit_rank(search(col, emb, q["q"], 5), q["doc"], q["phrase"])
        ranks.append(rank)
        if rank is None:
            print(f"MISS: {q['q']}")
    print(f"\n{args.model} | chunks={n_chunks} | questions={len(questions)}")
    print({k: round(v, 2) for k, v in summarize(ranks).items()})


if __name__ == "__main__":
    main()