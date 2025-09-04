# ingest_rules.py
import os
import glob
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from openai import OpenAI

# -------------------------------------------------------------------
# Env / clients
# -------------------------------------------------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("ERROR: DATABASE_URL is not set in your environment or .env")

OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")  # 3072-D
client = OpenAI()

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

# -------------------------------------------------------------------
# DB bootstrap (vector ext, table, dedupe, unique index)
# -------------------------------------------------------------------
DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rules (
  id BIGSERIAL PRIMARY KEY,
  title   TEXT,
  section TEXT,
  body    TEXT NOT NULL,
  embedding vector(3072)  -- <- enforce 3072-D
);

-- Remove duplicates before creating the unique index
DELETE FROM rules a
USING rules b
WHERE a.ctid < b.ctid
  AND a.title = b.title
  AND a.section = b.section;

-- Create the unique index if it doesn't exist
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

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _embed_text(text_in: str) -> List[float]:
    """Return a list of floats for the embedding. Defaults to 3072-D model."""
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text_in)
    vec = resp.data[0].embedding
    return [float(x) for x in vec]

def _vec_literal(vec: List[float]) -> str:
    """Format a pgvector literal like: [0.123,0.456,...]"""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def upsert_rule_chunk(title: str, section: str, body: str) -> None:
    """Insert or update a single rule chunk with its embedding."""
    emb = _embed_text(body)
    # (Optional) sanity log once per run
    # print(f"Embedded with {OPENAI_EMBED_MODEL}, dim={len(emb)} for {title}:{section}")

    vec = _vec_literal(emb)
    sql = text("""
        INSERT INTO rules (title, section, body, embedding)
        VALUES (:title, :section, :body, (:vec)::vector)
        ON CONFLICT (title, section) DO UPDATE
        SET body = EXCLUDED.body,
            embedding = EXCLUDED.embedding
    """)

    with engine.begin() as conn:
        conn.execute(sql, {"title": title, "section": section, "body": body, "vec": vec})

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    # Initialize DB objects
    with engine.begin() as conn:
        conn.execute(text(DDL))

    rules_dir = Path("rules")
    rules_dir.mkdir(parents=True, exist_ok=True)

    # Ingest every *.txt in /rules
    files = sorted(glob.glob(str(rules_dir / "*.txt")))
    if not files:
        print("No rule files found in ./rules. Add *.txt files and rerun.")
        return

    inserted = 0
    for fpath in files:
        title = Path(fpath).name  # e.g., holding.txt
        with open(fpath, "r", encoding="utf-8") as f:
            body = f.read().strip()

        # simple one-chunk strategy; extend to multi-chunk if needed
        upsert_rule_chunk(title=title, section="chunk-0", body=body)
        print(f"Upserted {title}:chunk-0")
        inserted += 1

    print(f"✅ Ingest complete. Total chunks upserted: {inserted}")

if __name__ == "__main__":
    main()