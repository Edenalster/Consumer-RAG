import argparse

from .retrieval import MODELS, build_index, search


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--model", default="e5-small", choices=list(MODELS))
    ap.add_argument("--max-chars", type=int, default=800)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--no-prefix", action="store_true")
    args = ap.parse_args()

    col, emb, n = build_index(args.model, args.max_chars, use_prefix=not args.no_prefix)
    print(f"{n} chunks indexed with {args.model}\n")
    for i, r in enumerate(search(col, emb, args.query, args.k), 1):
        m = r["meta"]
        body = r["text"].split("\n", 1)[-1][:200].replace("\n", " ")
        print(f"{i}. [{r['distance']:.3f}] {m['title']} | {m['heading']}\n   {body}\n")


if __name__ == "__main__":
    main()