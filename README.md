# DocTrail

DocTrail is a small, local experimental document ingestion and search pipeline built from scratch without high-level RAG frameworks like LangChain or LlamaIndex.

It processes a local corpus of 5 documents into 254 deterministic text and table chunks, supports character-level bigram tokenization for mixed English and Japanese text, and provides BM25 lexical search and local LLM generation via CLI and Python functions.

---

## Corpus & Processed State

The current evaluation dataset consists of 5 real-world PDFs placed in `data/docs/`:

- `attention_en.pdf` (academic layout with tables and formulas)
- `bitcoin_en.pdf` (standard text layout)
- `bitcoin_jp.pdf` (Japanese text)
- `tesla_financial_en.pdf` (corporate financial report)
- `toyota_financial_jp.pdf` (Japanese corporate financial report)

Running ingestion produces a deterministic `manifest.json` tracking 254 chunks with SHA-256 file fingerprints and saves chunk data to `data/processed/chunks.jsonl`.

---

## What Is Implemented

- **Deterministic Ingestion & Chunking (`Doctrail/ingest.py`, `Doctrail/chunk.py`):**
  - Fixed-window chunking (800 characters with 100-character overlap) and rule-based heading chunking.
  - Deterministic chunk IDs using truncated SHA-256 hashes of document path, page number, chunker name, and content.
  - Basic geometric line-based table extraction via PyMuPDF (`page.find_tables()`) converted to Markdown tables.
  - Bounding-box exclusion using block center-points to prevent extracted table content from being duplicated into normal prose chunks.

- **Multilingual Tokenizer (`Doctrail/tokenizer.py`):**
  - Whitespace splitting for Latin script combined with contiguous character-level bigrams for CJK (Japanese) characters without external morphological dictionary dependencies.

- **Lexical Search (`Doctrail/index.py`, `Doctrail/retrieve.py`):**
  - BM25Okapi inverted index construction over chunk text using `rank_bm25`.
  - Scoring, filtering of non-positive scores, and top-$k$ snippet retrieval.

- **Grounded Answer Generation (`Doctrail/generate.py`):**
  - Assembles prompt with strictly cited retrieved chunk metadata (file, page, chunk ID).
  - Framework-free inference calling local Ollama (`/api/generate`) via Python's standard library `urllib`.

- **CLI (`Doctrail/cli.py`):**
  - Commands to execute BM25 lexical searches (`search`) and run end-to-end grounded generation (`ask`).

- **Test Suite (`tests/`):**
  - Automated tests verifying SHA-256 determinism, in-memory BM25 retrieval, and end-to-end integration matching English ("attention heads") and Japanese ("二重支払い") queries to their source documents.

---

## What Is NOT Implemented

- **Dense Vector Search:** Dense embeddings, model loading (`multilingual-e5-small`), passage/query prefixes, and vector similarity search are not yet implemented.
- **Reciprocal Rank Fusion (RRF):** Rank fusion logic combining dense and sparse passes is not yet implemented. Only working code and test suite is present.
- **Evaluation Pipeline:** No formal evaluation metrics (Precision@k, Recall@k, MRR, or faithfulness benchmarks).
- **User Permissions & Security:** No multi-tenant authentication, access control lists, or document security boundaries.

---

## Known Limitations

1. **No OCR:** Text extraction relies purely on programmatic PDF text layers. Scanned pages, diagrams, and raster image text produce zero chunks.
2. **Reading Order on Multi-Column Layouts:** Blocks are ingested according to the PDF stream extraction order without geometric multi-column sorting, which can cause split-column text to interleave.
3. **Imperfect Table Extraction:** Only tables with detected border lines are parsed reliably; borderless or merged-cell tables break. The midpoint/center-point heuristic for table bounding boxes can occasionally misclassify border-adjacent text blocks.
4. **Lexical-Only Retrieval:** BM25 matches exact surface forms and bigrams. It cannot resolve synonyms, paraphrasing, or conceptual matches without dense retrieval.
5. **Character-Based Windows vs. Token Limits:** Chunk boundaries are defined by character counts, not subword tokenizer tokens. When dense models like `multilingual-e5-small` (512-token context limit) are added, larger chunks risk silent token truncation.
6. **Noisy Heading Detection:** Font flag and size heuristics can be fooled by stylized layouts (e.g., wide letter-spaced title headers in the Tesla PDF).

---

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Ollama locally so grounded generation can call the local model backend:
   ```bash
   ollama serve
   ```

   If you have not downloaded a model yet, pull one before running generation queries:
   ```bash
   ollama pull llama3.2:1b
   ```

4. Build the processed index from the bundled PDF corpus:
   ```bash
   python -m Doctrail.ingest
   ```

5. Run the test suite:
   ```bash
   pytest -q
   ```

6. Try a search or question-answer command:
   ```bash
   python -m Doctrail.cli search "attention heads"
   python -m Doctrail.cli ask "how is hash value calculated in bitcoin network?"
   python -m Doctrail.cli ask "トヨタとは何ですか?"
   ```

---

### Made by [Ayush Saksena](https://ayush-saksena.vercel.app/)

---