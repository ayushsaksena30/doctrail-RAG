import json
from pathlib import Path
import fitz
import pytest
from Doctrail.ingest import ingest_corpus


@pytest.fixture
def test_env(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    out_dir = tmp_path / "processed"
    cfg_path = tmp_path / "config.yaml"

    # Create a PDF with text AND a clear table with borders
    pdf_path = docs_dir / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()

    # Normal text outside the table
    page.insert_text((50, 50), "Outside prose paragraph before table.")

    # Table drawing (rectangles with cell borders)
    # 2 rows x 2 cols
    r1_c1 = fitz.Rect(50, 100, 150, 130)
    r1_c2 = fitz.Rect(150, 100, 250, 130)
    r2_c1 = fitz.Rect(50, 130, 150, 160)
    r2_c2 = fitz.Rect(150, 130, 250, 160)

    for r in [r1_c1, r1_c2, r2_c1, r2_c2]:
        page.draw_rect(r, color=(0, 0, 0), width=1)

    page.insert_text((55, 120), "MetricHeader")
    page.insert_text((155, 120), "ValueHeader")
    page.insert_text((55, 150), "UniqueRevenueCell")
    page.insert_text((155, 150), "$100M")

    page.insert_text((50, 200), "Outside prose paragraph after table.")
    doc.save(str(pdf_path))
    doc.close()

    docs_dir_str = str(docs_dir).replace("\\", "/")
    out_dir_str = str(out_dir).replace("\\", "/")

    cfg_path.write_text(f"""
        data_dir: "{docs_dir_str}"
        output_dir: "{out_dir_str}"
        chunker: "fixed"
        fixed_chunk_size: 400
        fixed_chunk_overlap: 50
        heading_max_size: 800
        """)
    return cfg_path, out_dir


def test_determinism_and_types(test_env):
    cfg_path, out_dir = test_env

    # Run 1
    m1 = ingest_corpus(str(cfg_path))
    lines1 = (out_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()

    # Run 2
    m2 = ingest_corpus(str(cfg_path))
    lines2 = (out_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()

    assert m1 == m2
    assert lines1 == lines2

    chunks = [json.loads(line) for line in lines1]
    assert len(chunks) > 0

    for c in chunks:
        assert isinstance(c["page"], int)
        assert c["page"] >= 1
        assert len(c["chunk_id"]) == 12

def test_table_text_appears_once(test_env):
    cfg_path, out_dir = test_env
    ingest_corpus(str(cfg_path))
    rows = [json.loads(l) for l in (out_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(r["text"].count("UniqueRevenueCell") for r in rows) == 1
