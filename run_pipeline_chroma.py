import os, sys, time
from pathlib import Path
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    print('ERROR: export OPENROUTER_API_KEY="sk-or-..."')
    sys.exit(1)
from src.legal_chunks import build_structured_chunks
parsed_dir = Path("data/parsed")
documents = []
for path in sorted(parsed_dir.glob("*.txt")):
    text = path.read_text(encoding="utf-8").strip()
    if text:
        documents.append({"source": path.name, "text": text})
if not documents:
    print("ERROR: data/parsed/ is empty")
    sys.exit(1)
all_chunks = []
for doc in documents:
    all_chunks.extend(build_structured_chunks(text=doc["text"], source=doc["source"], max_chars=1200, min_chars=350))
print(f"chunks: {len(all_chunks)}")
from openai import OpenAI
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
chunk_texts = [c.get("text", "") for c in all_chunks]
all_embeddings = []
BATCH = 50
for i in range(0, len(chunk_texts), BATCH):
    batch = chunk_texts[i:i+BATCH]
    print(f"  embedding batch {i//BATCH+1}...")
    resp = client.embeddings.create(model="openai/text-embedding-3-small", input=batch)
    all_embeddings.extend([item.embedding for item in resp.data])
    if i+BATCH < len(chunk_texts): time.sleep(1)
print(f"embeddings: {len(all_embeddings)}, dim: {len(all_embeddings[0])}")
import chromadb
db = chromadb.PersistentClient(path="./chroma_db")
try: db.delete_collection("eu_ai_act")
except: pass
col = db.create_collection("eu_ai_act")
ids = [f"chunk_{i}" for i in range(len(all_chunks))]
metas = [{"source":str(c.get("source","")),"canonical_citation":str(c.get("canonical_citation","")),"article_number":str(c.get("article_number","")),"annex_ref":str(c.get("annex_ref","")),"recital_ref":str(c.get("recital_ref",""))} for c in all_chunks]
for i in range(0, len(ids), BATCH):
    col.add(ids=ids[i:i+BATCH], embeddings=all_embeddings[i:i+BATCH], documents=chunk_texts[i:i+BATCH], metadatas=metas[i:i+BATCH])
print(f"stored: {col.count()} records")
query = "What are the obligations for providers of high-risk AI systems?"
qr = client.embeddings.create(model="openai/text-embedding-3-small", input=query)
results = col.query(query_embeddings=[qr.data[0].embedding], n_results=3)
print(f"\nTest: {query}")
for j,(doc,meta) in enumerate(zip(results['documents'][0],results['metadatas'][0])):
    print(f"  [{j+1}] {meta.get('canonical_citation','N/A')}: {doc[:120]}...")
print("\nDONE")
