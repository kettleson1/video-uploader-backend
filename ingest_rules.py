# ingest_rules.py
from __future__ import annotations

import os
import glob
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from openai import OpenAI

# ---------- Env / Setup ----------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RULES_DIR = os.getenv("RULES_DIR", "rules")  # directory of *.txt files to ingest
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536 dims

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
client = OpenAI(api_key=OPENAI_API_KEY)


# ---------- DB bootstrap ----------
DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rules (
  id BIGSERIAL PRIMARY KEY,
  title   TEXT,
  section TEXT,
  body    TEXT NOT NULL,
  embedding vector(1536)
);

-- for upsert by natural key
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE schemaname = 'public' AND indexname = 'uq_rules_title_section'
  ) THEN
    EXECUTE 'CREATE UNIQUE INDEX uq_rules_title_section ON rules (title, section)';
  END IF;
END $$;
"""

# Upsert (if conflict on title+section, update body & embedding)
UPSERT = text("""
INSERT INTO rules (title, section, body, embedding)
VALUES (:title, :section, :body, (:embedding)::vector)
ON CONFLICT (title, section)
DO UPDATE SET
  body = EXCLUDED.body,
  embedding = EXCLUDED.embedding
""")

# ---------- Helpers ----------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch embed texts; returns list of 1536-d float vectors."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def load_rule_files() -> list[tuple[str, str]]:
    """
    Returns list of (filename_without_ext, file_contents)
    for all *.txt in RULES_DIR.
    """
    files = sorted(glob.glob(str(Path(RULES_DIR) / "*.txt")))
    out: list[tuple[str, str]] = []
    for fp in files:
        try:
            text_content = Path(fp).read_text(encoding="utf-8").strip()
            if text_content:
                out.append((Path(fp).stem, text_content))
        except Exception as e:
            print(f"⚠️  Skipping {fp}: {e}")
    return out

def chunk_text(body: str, max_chars: int = 2000) -> list[str]:
    """
    Simple chunker to keep bodies from being too long for embedding.
    """
    body = " ".join(body.split())  # collapse whitespace
    if len(body) <= max_chars:
        return [body]
    chunks = []
    i = 0
    while i < len(body):
        chunks.append(body[i:i+max_chars])
        i += max_chars
    return chunks


# ---------- Main ----------
def main():
    # 1) Ensure schema
    with engine.begin() as conn:
        conn.execute(text(DDL))
    print("✅ Schema ready (extension/table/index).")

    # 2) Load rule files
    pairs = load_rule_files()
    if not pairs:
        print(f"⚠️  No rule files found in '{RULES_DIR}/'. Add *.txt files and retry.")
        return

    total_inserted = 0
    for title, body in pairs:
        chunks = chunk_text(body)
        vecs = embed_texts(chunks)

        # Upsert each chunk with a numbered section label
        with engine.begin() as conn:
            for i, (chunk, vec) in enumerate(zip(chunks, vecs), start=1):
                params = {
                    "title": title,
                    "section": f"chunk-{i}",
                    "body": chunk,
                    # Bind as JSON text then cast to ::vector in SQL
                    "embedding": json.dumps(vec),
                }
                conn.execute(UPSERT, params)
                total_inserted += 1

        print(f"✔️  Upserted {len(chunks)} chunk(s) for '{title}'")

    print(f"🎉 Ingest complete. Total chunks upserted: {total_inserted}")


if __name__ == "__main__":
    main()