# ingest_rules.py
import os, glob
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from openai import OpenAI

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not DB_URL or not OPENAI_API_KEY:
    raise SystemExit("Set DATABASE_URL and OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_API_KEY)
engine = create_engine(DB_URL)

def chunk(text, max_chars=1200):
    parts, cur, count = [], [], 0
    for line in text.splitlines():
        if count + len(line) + 1 > max_chars and cur:
            parts.append("\n".join(cur)); cur=[]; count=0
        cur.append(line); count += len(line) + 1
    if cur: parts.append("\n".join(cur))
    return parts

def embed(texts):
    res = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in res.data]

with engine.begin() as conn:
    for path in glob.glob("rules/*.txt") + glob.glob("rules/*.md"):
        title = os.path.basename(path)
        body = open(path, "r", encoding="utf-8").read()
        for i, ch in enumerate(chunk(body)):
            emb = embed([ch])[0]
            conn.execute(
                text("INSERT INTO rules(title, section, body, embedding) VALUES (:t, :s, :b, :e)"),
                {"t": title, "s": f"chunk-{i}", "b": ch, "e": emb},
            )
print("✅ Ingest complete")