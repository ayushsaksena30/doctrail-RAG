import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
import yaml

from Doctrail.chunk import (
    Chunk,
    create_chunk,
    create_chunk_by_heading,
    id_generator,
)

def table_to_markdown(table: Any) -> str:
    data= table.extract()
    if not data or len(data) < 1:
        return ""

    table_headers= [str(c or "").strip() for c in data[0]]
    md = ["| " + " | ".join(table_headers) + " |"]
    md.append("| " + " | ".join(["---"] * len(table_headers)) + " |")

    for row in data[1:]:
        clean_row=[str(c or "").replace("\n", " ").strip() for c in row]
        md.append("| " + " | ".join(clean_row) + " |")

    return "\n".join(md)

def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk :=f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()

def _is_inside_any_bbox(bbox: tuple, table_bboxes: list) -> bool:
    """Checks whether the center point of bbox falls inside any table bbox."""
    bx = (bbox[0] + bbox[2]) / 2
    by = (bbox[1] + bbox[3]) / 2
    for t_bbox in table_bboxes:
        if t_bbox[0] <= bx <= t_bbox[2] and t_bbox[1] <= by <= t_bbox[3]:
            return True
    return False

def ingest_corpus(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    chunker_name = cfg.get("chunker", "fixed")
    fixed_size = cfg.get("fixed_chunk_size", 800)
    fixed_overlap = cfg.get("fixed_chunk_overlap", 100)
    heading_max = cfg.get("heading_max_size", 1200)

    pdf_files = sorted(list(data_dir.rglob("*.pdf")))
    all_chunks: List[Chunk] = []
    manifest_files: List[Dict[str, Any]] = []

    for pdf_path in pdf_files:
        rel_path = str(pdf_path.relative_to(data_dir)).replace("\\", "/")
        folder = pdf_path.parent.name
        f_hash = file_sha256(pdf_path)

        doc = fitz.open(pdf_path)
        doc_chunk_count = 0

        for page_idx in range(len(doc)):
            page_num = page_idx + 1  # 1-based int
            page = doc[page_idx]

            # 1. Extract tables and save bboxes to prevent duplicate text extraction
            tabs = page.find_tables()
            if tabs is None:
                tabs = []

            table_bboxes = []
            
            for tab in tabs:
                md_tab = table_to_markdown(tab)
                if md_tab.strip():
                    table_bboxes.append(tab.bbox)  # Only exclude if successfully extracted
                    t_id = id_generator(rel_path, page_num, chunker_name, md_tab)
                    t_chunk = Chunk(
                        chunk_id=t_id,
                        rel_path=rel_path,
                        folder=folder,
                        page=page_num,
                        chunker=chunker_name,
                        chunk_type="table",
                        text=md_tab,
                    )
                    all_chunks.append(t_chunk)
                    doc_chunk_count += 1

            # 2. Extract text excluding blocks located inside table bboxes
            page_dict = page.get_text("dict")
            non_table_blocks = []
            for b in page_dict.get("blocks", []):
                if b.get("type") == 0:
                    if not _is_inside_any_bbox(b["bbox"], table_bboxes):
                        non_table_blocks.append(b)

            if chunker_name == "fixed":
                # Build page text solely from non-table blocks
                text_segments = []
                for b in non_table_blocks:
                    b_text = " ".join(
                        span.get("text", "").strip()
                        for line in b.get("lines", [])
                        for span in line.get("spans", [])
                        if span.get("text", "").strip()
                    ).strip()
                    if b_text:
                        text_segments.append(b_text)
                clean_page_text = "\n\n".join(text_segments)

                text_chunks = create_chunk(
                    clean_page_text, rel_path, folder, page_num, fixed_size, fixed_overlap
                )
            elif chunker_name == "heading":
                text_chunks = create_chunk_by_heading(
                    non_table_blocks, rel_path, folder, page_num, heading_max
                )
            else:
                raise ValueError(f"Unknown chunker: {chunker_name}")

            all_chunks.extend(text_chunks)
            doc_chunk_count += len(text_chunks)

        doc.close()
        manifest_files.append({
            "rel_path": rel_path,
            "sha256": f_hash,
            "chunks": doc_chunk_count,
        })

    # Save chunks.jsonl
    chunks_file = output_dir / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    # Save manifest.json
    manifest = {
        "chunker": chunker_name,
        "total_chunks": len(all_chunks),
        "files": manifest_files,
    }
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


if __name__ == "__main__":
    ingest_corpus()