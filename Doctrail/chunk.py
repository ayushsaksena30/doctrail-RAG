from dataclasses import dataclass
import hashlib
import statistics
from typing import Optional

@dataclass
class Chunk:
    chunk_id: str
    rel_path: str
    folder: str
    page: int
    chunker: str
    chunk_type: str
    text: str
    heading_path: Optional[str]= None

def id_generator(rel_path: str, page: int, chunker: str, text:str) -> str:
    delimiter='|'
    join_str= delimiter.join([rel_path, str(page), chunker, text])
    encode_bytes = join_str.encode("utf-8")
    sha256_hash = hashlib.sha256(encode_bytes).hexdigest()
    return sha256_hash[:12]

def create_chunk(text: str, rel_path: str, folder: str, page: int, chunk_size: int, overlap: int) -> list[Chunk]:
    all_chunks = []
    step = max(1, chunk_size-overlap)
    
    for i in range(0,len(text), step):
        substring= text[i:i+chunk_size].strip()
        if substring:
            chunk_i_id = id_generator(rel_path, page, "fixed", substring)
            chunk_i=Chunk(
                chunk_id= chunk_i_id,
                rel_path= rel_path,
                folder= folder,
                page= page,
                chunker= "fixed",
                chunk_type= "text",
                text= substring,
            )

            all_chunks.append(chunk_i)

        if i+chunk_size >=len(text):
            break

    return all_chunks

def create_chunk_by_heading(block: list[dict], rel_path: str, folder: str, page: int, max_size: int=1200) -> list[Chunk]:
    #1 calculate baseline font size
    font_sizes = []

    for b in block:
        if b.get("type") == 0:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()

                    if text:
                         font_sizes.append(span.get("size", 10.0))

    if font_sizes:
        baseline_font_size = statistics.median(font_sizes)
    else:
        baseline_font_size = 10.0

    #2 iterate over again and find out headings
    all_chunks = []
    current_body = []
    current_heading = ""

    def flush_section(heading: str, body_lines: list[str]) -> list[Chunk]:
        body_text = " ".join(body_lines).strip()
        if not body_text and not heading:
            return []
        
        full_text = f"{heading}\n{body_text}".strip() if heading else body_text
        flushed = []

        for start in range(0, len(full_text), max_size):
            segment = full_text[start : start + max_size].strip()
            if segment:
                chunk_id = id_generator(rel_path, page, "heading", segment)
                flushed.append(
                    Chunk(
                        chunk_id=chunk_id,
                        rel_path=rel_path,
                        folder=folder,
                        page=page,
                        chunker="heading",
                        chunk_type="text",
                        text=segment,
                        heading_path=heading or None,
                    )
                )
        return flushed

    for b in block:
        if b.get("type") != 0:
            continue

        for line in b.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            line_text = " ".join(
                s.get("text", "").strip()
                for s in spans
                if s.get("text", "").strip()
            ).strip()
            if not line_text:
                continue

            first_span = spans[0]
            is_heading = (
                first_span.get("size", 0.0) >= baseline_font_size * 1.25
                or bool(first_span.get("flags", 0) & 16)
            ) and len(line_text) < 150

            if is_heading:
                all_chunks.extend(flush_section(current_heading, current_body))
                current_body = []
                current_heading = line_text
            else:
                current_body.append(line_text)

    all_chunks.extend(flush_section(current_heading, current_body))

    return all_chunks