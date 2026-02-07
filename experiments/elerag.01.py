import sys
import json
import subprocess
import spacy
import requests
import csv
import numpy as np
from sentence_transformers import SentenceTransformer

csv.field_size_limit(sys.maxsize)

# --- CONFIGURATION ---
# ON MAC: Keep these paths.
# ON PI:  Change BITNET_EXEC to "/home/pi/elerag/llama.cpp/llama-cli"
BITNET_EXEC = "/Users/henrystiglitz/BitNet/build/bin/llama-cli" 
BITNET_MODEL = "ggml-model-i2_s.gguf" 
MEMORY_FILE = "pi_memory.json"
ENTITY_CACHE_FILE = "entity_cache.json"

# Load Models
print("Loading system...")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Spacy model not found. Downloading...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

embed_model = SentenceTransformer('all-MiniLM-L6-v2') 

# --- HELPER FUNCTIONS ---
ENTITY_CACHE = {}
try:
    with open(ENTITY_CACHE_FILE, 'r') as f: ENTITY_CACHE = json.load(f)
except: pass

def save_cache():
    with open(ENTITY_CACHE_FILE, 'w') as f: json.dump(ENTITY_CACHE, f)

def get_wikidata_id(text):
    if text in ENTITY_CACHE: return ENTITY_CACHE[text]
    try:
        url = "https://www.wikidata.org/w/api.php"
        params = {"action": "wbsearchentities", "language": "en", "format": "json", "search": text, "limit": 1}
        r = requests.get(url, params=params, timeout=1) 
        data = r.json()
        qid = data['search'][0]['id'] if data.get('search') else None
        ENTITY_CACHE[text] = qid
        return qid
    except: return None

def cosine_sim(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

# --- INGESTION ---
def ingest_file(filepath):
    print(f"Reading {filepath}...")
    chunks = []
    if filepath.endswith('.csv'):
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Student / Logic Test Format
                if row.get('Fact'):
                    combined = f"[{row.get('Unit','General')} - {row.get('Subtopic','Info')}] {row.get('Fact','')}"
                    if len(combined) > 10: chunks.append(combined)
                # Fallback for other CSVs
                else:
                    text = row.get('text') or row.get('content') or row.get('body') or ""
                    if len(text) > 20: chunks.append(text)
    else:
        with open(filepath, 'r') as f: chunks = [p for p in f.read().split('\n\n') if len(p) > 20]

    print(f"Embedding {len(chunks)} items...")
    vectors = embed_model.encode(chunks, show_progress_bar=True)
    
    memory_data = []
    use_entities = len(chunks) < 500 
    
    for i, chunk in enumerate(chunks):
        entities = []
        if use_entities:
            # Simple Entity Extraction
            entities = list(set(filter(None, [get_wikidata_id(e.text) for e in nlp(chunk).ents if e.label_ in ["PERSON","ORG","GPE"]])))
        memory_data.append({"id": i, "text": chunk, "vector": vectors[i].tolist(), "entities": entities})
        if i % 100 == 0: print(f"Processed {i}/{len(chunks)}...")

    with open(MEMORY_FILE, 'w') as f: json.dump(memory_data, f)
    if use_entities: save_cache()
    print("Ingestion Complete.")

# --- RETRIEVAL & QUERY ---
def query_system(query):
    try: 
        with open(MEMORY_FILE, 'r') as f: memory = json.load(f)
    except: return print("Error: Memory file not found. Run 'ingest' first.")

    print("Thinking...")
    query_vec = embed_model.encode([query])[0]
    
    # 1. Retrieve
    dense = sorted([(d['id'], cosine_sim(query_vec, d['vector'])) for d in memory], key=lambda x:x[1], reverse=True)[:15]
    rrf = {id: 1/(60+i) for i, (id, _) in enumerate(dense)}
    
    q_ents = list(filter(None, [get_wikidata_id(e.text) for e in nlp(query).ents]))
    if q_ents:
        ent_match = sorted([(d['id'], len(set(d['entities']) & set(q_ents))) for d in memory], key=lambda x:x[1], reverse=True)[:15]
        for i, (id, _) in enumerate(ent_match): rrf[id] = rrf.get(id, 0) + 1/(60+i)

    top_ids = sorted(rrf, key=rrf.get, reverse=True)[:3]
    context = "\n".join([memory[i]['text'] for i in top_ids])

    # 2. PROMPT
    prompt = (
        f"### Context:\n{context}\n\n"
        f"### Instruction:\n"
        f"You are a database. Answer the user question in one short sentence using ONLY the context above.\n"
        f"If the answer is not in the context, say 'Data not available'.\n"
        f"Question: {query}\n\n"
        f"### Response:"
    )
    
    # 3. Generate
    cmd = [BITNET_EXEC, "-m", BITNET_MODEL, "-p", prompt, "-n", "64", "-c", "2048", "--temp", "0", "-r", "###"]
    
    try:
        # Changed stderr to PIPE so we can see errors if it fails
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        
        # 4. SCORCHED EARTH CLEANING
        if "### Response:" in res.stdout:
            ans = res.stdout.split("### Response:")[-1].strip()
            # Stop at the first period to prevent looping
            if "." in ans: ans = ans.split(".")[0] + "."
            
            # Kill List for other artifacts
            for char in ["\n", "(", "`", "[", "Response:", "Answer:"]:
                if char in ans: ans = ans.split(char)[0]
                
            print("\n--- Answer ---")
            print(ans.strip())
            print("--------------\n")
        else:
            # Fallback: Print whatever came out, or the error if empty
            if res.stdout.strip():
                print(f"Raw Output: {res.stdout.strip()}")
            elif res.stderr.strip():
                print(f"Error from Model: {res.stderr.strip()}")
            else:
                print("Model returned no output.")
                
    except Exception as e: print(f"Python Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "ingest": ingest_file(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "query": query_system(" ".join(sys.argv[2:]))
    else: print("Usage: uv run elerag.py [ingest file.csv | query 'Question']")