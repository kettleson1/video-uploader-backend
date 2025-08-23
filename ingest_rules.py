# ingest_rules.py
import os
import glob
from typing import List

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from openai import OpenAI

# 1) Load env
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DB_URL:
    raise SystemExit("DATABASE_URL missing in .env")
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY missing in .env")

# 2) Init clients
client = OpenAI(api_key=OPENAI_API_KEY)
engine = create_engine(DB_URL)

# 3) Simple chunker (keeps paragraphs reasonably sized)
def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    parts, cur, count = [], [], 0
    for line in text.splitlines():
        if count + len(line) + 1 > max_chars and cur:
            parts.append("\n".join(cur)); cur=[]; count=0
        cur.append(line)
        count += len(line) + 1
    if cur:
        parts.append("\n".join(cur))
    return [p.strip() for p in parts if p.strip()]

def embed_texts(texts: List[str]) -> List[List[float]]:
    # text-embedding-3-small => 1536-dim vectors
    res = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in res.data]

def upsert_rule_chunks(title: str, section_prefix: str, body: str):
    chunks = chunk_text(body, max_chars=1200)
    if not chunks:
        return 0
    embeddings = embed_texts(chunks)
    assert len(chunks) == len(embeddings)

    inserted = 0
    with engine.begin() as conn:
        for i, (ch, emb) in enumerate(zip(chunks, embeddings)):
            conn.execute(
                text("""
                    INSERT INTO rules (title, section, body, embedding)
                    VALUES (:title, :section, :body, :embedding)
                """),
                {
                    "title": title,
                    "section": f"{section_prefix}-{i}",
                    "body": ch,
                    "embedding": emb,  # pgvector accepts Python lists
                },
            )
            inserted += 1
    return inserted

def main():
    rules_dir = os.path.join(os.path.dirname(__file__), "rules")
    paths = sorted(glob.glob(os.path.join(rules_dir, "*.txt")) + glob.glob(os.path.join(rules_dir, "*.md")))
    if not paths:
        print("No files found in rules/. Add .txt or .md files and rerun.")
        return

    total = 0
    for p in paths:
        title = os.path.basename(p)
        with open(p, "r", encoding="utf-8") as f:
            body = f.read()
        inserted = upsert_rule_chunks(title=title, section_prefix="chunk", body=body)
        print(f"Inserted {inserted} chunks from {title}")
        total += inserted

    print(f"✅ Ingest complete. Total chunks: {total}")

if __name__ == "__main__":
    main()