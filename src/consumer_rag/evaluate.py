import argparse
import json
from pathlib import Path

from .chunking import chunk_doc, load_docs
from .fetch import RAW_DIR
from .retrieval import MODELS, build_index, search


def norm(s: str) -> str:
    return " ".join(s.split())


def first_hit_rank(results: list[dict], doc=None, phrase=None, relevant=None):
    """Return the rank of the first relevant retrieved chunk."""

    if relevant is None:
        relevant = [{"doc": doc, "phrase": phrase}]

    for i, result in enumerate(results, 1):
        for target in relevant:
            if (
                result["meta"]["doc_id"] == target["doc"]
                and norm(target["phrase"]) in norm(result["text"])
            ):
                return i

    return None


def summarize(ranks: list) -> dict:
    n = len(ranks)
    out = {f"recall@{k}": sum(1 for r in ranks if r and r <= k) / n for k in (1, 3, 5)}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    return out


def validate(questions: list[dict], chunk_texts: dict[str, list[str]]) -> list[str]:
    """Validate questions and verify that each relevant answer exists in the indexed chunks."""
    problems = []

    for i, q in enumerate(questions, 1):
        # Every question must have a question and a type
        missing = [k for k in ("q", "type") if k not in q]

        if missing:
            problems.append(f"question {i}: missing {missing}")
            continue

        # Support both the old and new question formats
        if "relevant" in q:
            targets = q["relevant"]
        elif "doc" in q and "phrase" in q:
            targets = [{"doc": q["doc"], "phrase": q["phrase"]}]
        else:
            problems.append(f"question {i}: missing 'relevant' or 'doc'/'phrase'")
            continue

        if not isinstance(targets, list) or not targets:
            problems.append(f"question {i}: 'relevant' must be a non-empty list")
            continue

        for target in targets:
            if not isinstance(target, dict) or not all(
                isinstance(target.get(k), str) and target[k].strip()
                for k in ("doc", "phrase")
            ):
                problems.append(f"question {i}: invalid relevant target")
                continue

            doc = target["doc"]
            phrase = target["phrase"]

            texts = chunk_texts.get(doc)

            if texts is None:
                problems.append(f"question {i}: unknown doc '{doc}'")
            elif not any(norm(phrase) in norm(t) for t in texts):
                problems.append(
                    f"question {i}: phrase not found in any chunk of '{doc}': {phrase}"
                )

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
        rank = first_hit_rank(
        search(col, emb, q["q"], 5),
        relevant=q.get("relevant", [{"doc": q.get("doc"), "phrase": q.get("phrase")}]))
        ranks.append(rank)
        if rank is None:
            print(f"MISS: {q['q']}")
    print(f"\n{args.model} | chunks={n_chunks} | questions={len(questions)}")
    print({k: round(v, 2) for k, v in summarize(ranks).items()})


if __name__ == "__main__":
    main()