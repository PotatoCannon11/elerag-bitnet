import sys
import json
import subprocess
import spacy
import requests
import csv
import numpy as np
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
# ON MAC: Keep these paths.
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

def get_wikidata_id(text, context_sentence=None):
    if text in ENTITY_CACHE: return ENTITY_CACHE[text]
    
    try:
        url = "https://www.wikidata.org/w/api.php"
        # Wikidata requires a User-Agent header to allow the request
        headers = {"User-Agent": "ELERAG_Project/1.0 (contact: admin@example.com)"}
        
        params = {
            "action": "wbsearchentities",
            "language": "en",
            "format": "json",
            "search": text,
            "limit": 5
        }
        
        r = requests.get(url, params=params, headers=headers, timeout=5)
        data = r.json()
        
        candidates = data.get('search', [])
        if not candidates: return None

        # --- Semantic Disambiguation ---
        query_text = context_sentence if context_sentence else text
        cand_descs = [c.get('description', c['label']) for c in candidates]
        
        # Reuse existing model to find best match
        query_vec = embed_model.encode([query_text])[0]
        cand_vecs = embed_model.encode(cand_descs)
        
        scores = [cosine_sim(query_vec, cv) for cv in cand_vecs]
        best_idx = np.argmax(scores)
        best_qid = candidates[best_idx]['id']
        
        ENTITY_CACHE[text] = best_qid
        return best_qid

    except Exception as e:
        # Print detailed error if it's not a simple not-found
        print(f"Wikidata error for '{text}': {e}")
        return None

def extract_entities(text):
    doc = nlp(text)
    entities = []
    relevant_labels = ["PERSON", "ORG", "GPE", "DATE", "LAW", "PRODUCT"]
    
    for ent in doc.ents:
        if ent.label_ in relevant_labels:
            if ent.label_ in ["DATE", "PRODUCT"]:
                entities.append(("text", ent.text.lower()))
            else:
                # PASS THE SENTENCE CONTEXT HERE
                qid = get_wikidata_id(ent.text, context_sentence=ent.sent.text)
                if qid: entities.append(("wiki", qid))
    return list(set(entities))

def smart_chunk_text(text, chunk_size=300, overlap=50):
    """Smart Segmentation (Paper Requirement)"""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Create overlap
            overlap_sents = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len < overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s.split())
                else: break
            current_chunk = overlap_sents
            current_len = overlap_len
        
        current_chunk.append(sent)
        current_len += sent_len
    
    if current_chunk: chunks.append(" ".join(current_chunk))
    return chunks

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
                # Combine columns to ensure context isn't lost
                text = f"[{row.get('Unit','')} - {row.get('Subtopic','')}] {row.get('Fact','')}"
                if len(text) > 20: chunks.append(text)
    else:
        with open(filepath, 'r') as f: 
            content = f.read()
            chunks = smart_chunk_text(content)

    print(f"Embedding {len(chunks)} items...")
    vectors = embed_model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    
    memory_data = []
    for i, chunk in enumerate(chunks):
        entities = extract_entities(chunk)
        memory_data.append({
            "id": i, 
            "text": chunk, 
            "vector": vectors[i].tolist(), 
            "entities": entities
        })

    with open(MEMORY_FILE, 'w') as f: json.dump(memory_data, f)
    save_cache()
    print("Ingestion Complete.")

# --- RETRIEVAL & QUERY ---
def query_system(query):
    try:
        with open(MEMORY_FILE, 'r') as f: memory = json.load(f)
    except: return print("Run 'ingest' first.")

    print("Thinking...")
    
    # 1. Query Expansion (Concept Search)
    doc = nlp(query)
    keywords = [t.text for t in doc if not t.is_stop and t.is_alpha]
    variations = [query]
    if len(keywords) > 2: variations.append(" ".join(keywords))
    
    query_vec = np.mean(embed_model.encode(variations, normalize_embeddings=True), axis=0)
    
    # 2. RRF Fusion (Paper Requirement)
    dense_hits = sorted([(d['id'], cosine_sim(query_vec, d['vector'])) for d in memory], key=lambda x:x[1], reverse=True)[:15]
    rrf_scores = {doc_id: 1/(60+r) for r, (doc_id, _) in enumerate(dense_hits)}
    
    q_ents = extract_entities(query)
    if q_ents:
        entity_hits = []
        for doc in memory:
            # Fix: JSON loads entities as lists, which breaks sets. Convert back to tuples.
            doc_ents_set = {tuple(e) for e in doc['entities']}
            matches = len(doc_ents_set.intersection(set(q_ents)))
            if matches > 0: entity_hits.append((doc['id'], matches))
        
        for r, (doc_id, _) in enumerate(sorted(entity_hits, key=lambda x:x[1], reverse=True)[:15]):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1/(60+r)

    # 3. Diversity Check (Dedup)
    top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:10]
    final_ids = [top_ids[0]]
    for cand_id in top_ids[1:]:
        is_diverse = True
        for sel_id in final_ids:
            if cosine_sim(memory[cand_id]['vector'], memory[sel_id]['vector']) > 0.85:
                is_diverse = False; break
        if is_diverse and len(final_ids) < 3: final_ids.append(cand_id)

    context = "\n".join([memory[i]['text'] for i in final_ids])

    # 4. Generate & Clean
    prompt = (
        f"### Context:\n{context}\n\n"
        f"### Instruction:\n"
        f"Based strictly on the context above, answer this: {query}\n"
        f"Answer in one short sentence. Do not cite sources.\n\n"
        f"### Response:"
    )
    
    cmd = [
        BITNET_EXEC, "-m", BITNET_MODEL, "-p", prompt,
        "-n", "128", "-c", "2048", "--temp", "0", "-r", "###"
    ]
    
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, 
            text=True, encoding='utf-8', errors='ignore'
        )
        
        full_output = result.stdout
        if "### Response:" in full_output:
            answer = full_output.split("### Response:")[-1].strip()
            if "." in answer: answer = answer.split(".")[0] + "."
            for char in ["\n", "(", "`", "["]:
                if char in answer: answer = answer.split(char)[0]
            print("\n--- Answer ---\n" + answer.strip() + "\n--------------\n")
        else:
            print(full_output)

    except Exception as e: print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "ingest": ingest_file(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "query": query_system(" ".join(sys.argv[2:]))
    else: print("Usage: uv run elerag_improved.py [ingest file.csv | query 'Question']")
