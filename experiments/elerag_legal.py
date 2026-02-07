import sys
import json
import subprocess
import spacy
import requests
import csv
import numpy as np
import os
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer

# 1. INCREASE CSV LIMIT for massive legal emails
csv.field_size_limit(sys.maxsize)

# --- CONFIGURATION ---
# ON MAC: Keep these paths.
# ON PI:  Change BITNET_EXEC to "/home/pi/elerag/llama.cpp/llama-cli"
BITNET_EXEC = "/Users/henrystiglitz/BitNet/build/bin/llama-cli" 
BITNET_MODEL = "ggml-model-i2_s.gguf" 
MEMORY_FILE = "pi_memory.json"
ENTITY_CACHE_FILE = "entity_cache.json"
REPORT_FILE = "evidence_report.txt"

# Load Models
print("Loading system...")
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

def clean_and_date_email(raw_text):
    """
    Extracts the Date, strips the massive header block, and returns clean text prepended with [DATE].
    """
    # 1. Extract Date (Look for "Date: ...")
    date_match = re.search(r'Date:\s+(.*?)(\r\n|\n)', raw_text)
    date_str = ""
    if date_match:
        # Try to parse specific Enron format "Wed, 14 Mar 2001" or just keep the raw string
        try:
            raw_date = date_match.group(1).split(" (")[0] # Remove timezone comment like (PST)
            dt = datetime.strptime(raw_date.strip(), "%a, %d %b %Y %H:%M:%S %z") # E.g., Wed, 14 Mar 2001...
            date_str = f"[{dt.strftime('%Y-%m-%d')}] "
        except:
            pass # Keep empty if parsing fails

    # 2. Strip Headers (Everything before X-FileName or Subject)
    # Strategy: Find the last common Enron header tag and cut there.
    split_match = re.search(r'(X-FileName:|X-Folder:|Subject:).*?(\r\n|\n)', raw_text, re.IGNORECASE)
    
    body_text = raw_text
    if split_match:
        # Find the end of that line, then skip until we hit a blank line (start of body)
        header_end_idx = split_match.end()
        # Look for the next double newline which usually signals body start
        body_start = raw_text.find('\n\n', header_end_idx)
        if body_start != -1:
            body_text = raw_text[body_start:].strip()
        else:
            body_text = raw_text[header_end_idx:].strip()
            
    # 3. Collapse Forwarded Junk
    # Replace big blocks of "----- Forwarded by..." with a marker
    body_text = re.sub(r'-+\s?Forwarded by.*?-+\n', '\n[...Forwarded Email Chain...]\n', body_text, flags=re.DOTALL)
    
    # 4. Collapse whitespace
    clean_body = " ".join(body_text.split())
    
    return f"{date_str}{clean_body}"

# --- INGESTION (ROBUST V2) ---
def ingest_file(filepath):
    print(f"Reading {filepath}...")
    chunks = []
    seen_hashes = set() # For deduplication

    if filepath.endswith('.csv'):
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Universal Column Selector
                raw = row.get('message') or row.get('body') or row.get('text') or row.get('Fact') or row.get('content')
                
                if not raw or len(raw) < 20: continue

                # CLEAN & EXTRACT DATE
                clean_text = clean_and_date_email(raw)
                
                # DEDUPLICATION
                text_hash = hash(clean_text)
                if text_hash in seen_hashes:
                    continue # Skip duplicate
                seen_hashes.add(text_hash)

                # Append (Increased limit to 5000 chars for 16GB RAM assumption)
                chunks.append(clean_text[:5000]) 
    else:
        with open(filepath, 'r') as f: chunks = [p for p in f.read().split('\n\n') if len(p) > 20]

    print(f"Embedding {len(chunks)} unique items...")
    vectors = embed_model.encode(chunks, show_progress_bar=True)
    
    memory_data = []
    use_entities = len(chunks) < 500 
    
    for i, chunk in enumerate(chunks):
        entities = []
        if use_entities:
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
    except: return print("Run 'ingest' first.")

    print("Thinking...")
    query_vec = embed_model.encode([query])[0]
    
    # 1. Retrieve (Increased pool size)
    dense = sorted([(d['id'], cosine_sim(query_vec, d['vector'])) for d in memory], key=lambda x:x[1], reverse=True)[:25]
    rrf = {id: 1/(60+i) for i, (id, _) in enumerate(dense)}
    
    q_ents = list(filter(None, [get_wikidata_id(e.text) for e in nlp(query).ents]))
    if q_ents:
        ent_match = sorted([(d['id'], len(set(d['entities']) & set(q_ents))) for d in memory], key=lambda x:x[1], reverse=True)[:25]
        for i, (id, _) in enumerate(ent_match): rrf[id] = rrf.get(id, 0) + 1/(60+i)

    # INCREASED CONTEXT: Get top 5 results instead of 3
    top_ids = sorted(rrf, key=rrf.get, reverse=True)[:5]
    top_results = [memory[i]['text'] for i in top_ids]

    # --- EVIDENCE EXPORT ---
    print(f"\n--- 💾 SAVING FULL DISCOVERY TO {REPORT_FILE} ---")
    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write(f"LEGAL DISCOVERY REPORT\n")
        report.write(f"Query: {query}\n")
        report.write(f"Timestamp: {datetime.now()}\n")
        report.write("="*60 + "\n\n")
        
        for i, result in enumerate(top_results, 1):
            header = f"--- EVIDENCE ITEM #{i} ---"
            report.write(f"{header}\n{result}\n\n")
            
            # Print snippet to console
            # Adjusted slice to 0:200 since headers are gone now!
            print(f"[{i}] {result[:200].replace(chr(10), ' ')}...")
            
    print("------------------------------------------------\n")

    context = "\n".join(top_results)

    # 2. PROMPT
    prompt = (
        f"### Context:\n{context}\n\n"
        f"### Instruction:\n"
        f"You are a legal assistant. Answer the user question in one short sentence using ONLY the context above.\n"
        f"If the answer is not in the context, say 'Data not available'.\n"
        f"Question: {query}\n\n"
        f"### Response:"
    )
    
    # 3. Generate (Increased context window to 4096 to handle larger chunks)
    cmd = [BITNET_EXEC, "-m", BITNET_MODEL, "-p", prompt, "-n", "64", "-c", "4096", "--temp", "0", "-r", "###"]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8', errors='ignore')
        
        # 4. SCORCHED EARTH CLEANING
        if "### Response:" in res.stdout:
            ans = res.stdout.split("### Response:")[-1].strip()
            if "." in ans: ans = ans.split(".")[0] + "."
            for char in ["\n", "(", "`", "[", "Response:", "Answer:"]:
                if char in ans: ans = ans.split(char)[0]
                
            print("\n--- Answer ---")
            print(ans.strip())
            print("--------------\n")
            
            # Append answer to report
            with open(REPORT_FILE, "a", encoding="utf-8") as report:
                report.write(f"--- AI SUMMARY ---\n{ans.strip()}\n")
                
        else:
            print(res.stdout)
    except Exception as e: print(e)

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "ingest": ingest_file(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "query": query_system(" ".join(sys.argv[2:]))
    else: print("Usage: uv run elerag.py [ingest file.csv | query 'Question']")