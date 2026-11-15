1. No OCR: scanned pages and image text produce no chunks.
2. Multi-column pages aren’t reordered, so columns can interleave.
3. Table extraction is imperfect (merged cells and borderless tables break), and the center-point rule can wrongly include or exclude a block.
4. BM25 is lexical not semantic.
5. Chunk limits are in characters, not model tokens, and multilingual-e5-small has a 512-token limit, so long chunks risk truncation once you add embeddings.