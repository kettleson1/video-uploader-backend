import os
from glob import glob
from sqlalchemy import create_engine, text
from openai import OpenAI

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert DATABASE_URL, "DATABASE_URL not set"
assert OPENAI_API_KEY, "OPENAI_API_KEY not set"

EMBED_MODEL = "text-embedding-3-large"  # 3072-D
EMBED_DIM   = 3072

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
client = OpenAI(api_key=OPENAI_API_KEY)

def _embed_text(txt: str) -> list[float]:
    txt = (txt or "").strip()
    if not txt:
        return [0.0] * EMBED_DIM
    vec = client.embeddings.create(model=EMBED_MODEL, input=txt).data[0].embedding
    if len(vec) != EMBED_DIM:
        if len(vec) > EMBED_DIM:
            vec = vec[:EMBED_DIM]
        else:
            vec = vec + [0.0] * (EMBED_DIM - len(vec))
    return vec

def upsert_rule_chunks(title: str, section_prefix: str, body: str) -> int:
    """
    Splits body into simple paragraphs, embeds each, and UPSERTS into rules(title, section, body, embedding).
    Your table must be:
      id bigserial PK,
      title text,
      section text,
      body text NOT NULL,
      embedding vector(3072)
    """
    # very simple chunking by blank lines
    chunks = [p.strip() for p in body.split("\n\n") if p.strip()]
    inserted = 0

    with engine.begin() as conn:
        for idx, ch in enumerate(chunks, start=1):
            emb = _embed_text(ch)
            qvec_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"

            sql = text("""
                INSERT INTO rules (title, section, body, embedding)
                VALUES (:title, :sec, :body, (:emb)::vector)
            """)
            conn.execute(sql, {"title": title, "sec": f"{section_prefix}{idx}", "body": ch, "emb": qvec_literal})
            inserted += 1

    return inserted

def main():
    rules_dir = os.path.join(os.getcwd(), "rules")
    files = sorted(glob(os.path.join(rules_dir, "*.txt")))
    if not files:
        print("No rule files found in ./rules/*.txt")
        return
    total = 0
    for fp in files:
        title = os.path.splitext(os.path.basename(fp))[0]
        with open(fp, "r", encoding="utf-8") as f:
            body = f.read()
        n = upsert_rule_chunks(title=title, section_prefix="chunk", body=body)
        total += n
        print(f"Inserted {n} chunks from {title}.txt")
    print(f"✅ Ingest complete. Total chunks: {total}")

if __name__ == "__main__":
    main()