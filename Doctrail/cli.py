import argparse
from Doctrail.index import load_chunks, build_bm25
from Doctrail.retrieve import bm25_search
from Doctrail.generate import build_rag_prompt, generate_answer_ollama


def main():
    parser = argparse.ArgumentParser(description="DocTrail CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-command: search
    search_parser = subparsers.add_parser("search", help="BM25 lexical search")
    search_parser.add_argument("query", type=str, help="Search query string")
    search_parser.add_argument("-k", type=int, default=5, help="Number of results")

    # Sub-command: ask
    ask_parser = subparsers.add_parser("ask", help="End-to-end RAG answer generation")
    ask_parser.add_argument("query", type=str, help="Question to ask")
    ask_parser.add_argument("-k", type=int, default=5, help="Number of context chunks to retrieve")
    ask_parser.add_argument("--model", type=str, default="llama3.2:1b", help="Ollama model name")

    args = parser.parse_args()

    chunks = load_chunks("data/processed/chunks.jsonl")
    if not chunks:
        print("No chunks found in data/processed/chunks.jsonl. Run ingest first.")
        return

    bm25 = build_bm25(chunks)

    if args.command == "search":
        results = bm25_search(args.query, chunks, bm25, k=args.k)
        if not results:
            print(f"No results found for query: '{args.query}'")
            return

        print(f"\n--- Top {len(results)} Results for: '{args.query}' ---\n")
        for i, res in enumerate(results, start=1):
            print(f"[{i}] Score: {res['score']:.4f} | File: {res['rel_path']} (Page {res['page']})")
            print(f"    Chunk ID: {res['chunk_id']}")
            print(f"    Snippet : {res['snippet']}...")
            print()

    elif args.command == "ask":
        # 1. Retrieve top-k context chunks
        results = bm25_search(args.query, chunks, bm25, k=args.k)
        if not results:
            print(f"No relevant context found in documents for: '{args.query}'")
            return

        # 2. Extract full text from matched chunks
        chunk_lookup = {c["chunk_id"]: c for c in chunks}
        context_chunks = [chunk_lookup[r["chunk_id"]] for r in results if r["chunk_id"] in chunk_lookup]

        # 3. Assemble RAG prompt
        prompt = build_rag_prompt(args.query, context_chunks)

        print(f"\n[Retrieved {len(context_chunks)} chunks for context]")
        for r in results:
            print(f" - {r['rel_path']} (Page {r['page']}, ID: {r['chunk_id']})")

        print("\n[Generating answer...]\n")
        answer = generate_answer_ollama(prompt, model=args.model)
        print(answer)
        print()


if __name__ == "__main__":
    main()