import os
import re
import pandas as pd
from pypdf import PdfReader

# --- CONFIGURATION ---
PDF_SOURCE_FOLDER = "raw_pdfs"
# CHANGED: This now writes to a separate file, protecting your original work
OUTPUT_CSV = "ep_dump.csv" 
MIN_SENTENCE_LENGTH = 25

def clean_text(text):
    # Fixes broken lines common in PDFs
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_facts(text):
    # Splits text into individual sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    clean_facts = []
    for s in sentences:
        s = s.strip()
        if len(s) > MIN_SENTENCE_LENGTH:
            clean_facts.append(s)
    return clean_facts

def process_pdfs():
    all_new_facts = []
    
    # 1. Setup Folder
    if not os.path.exists(PDF_SOURCE_FOLDER):
        os.makedirs(PDF_SOURCE_FOLDER)
        print(f"[Setup] Created folder '{PDF_SOURCE_FOLDER}'. Put PDF files here!")
        return

    files = [f for f in os.listdir(PDF_SOURCE_FOLDER) if f.endswith(".pdf")]
    
    if not files:
        print(f"[Error] No PDFs found in '{PDF_SOURCE_FOLDER}'.")
        return

    print(f"[Start] Found {len(files)} PDFs. Extracting text...")

    # 2. Extract Text
    for filename in files:
        filepath = os.path.join(PDF_SOURCE_FOLDER, filename)
        print(f" -> Reading: {filename}...")
        
        try:
            reader = PdfReader(filepath)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
                
            cleaned_text = clean_text(full_text)
            facts = split_into_facts(cleaned_text)
            
            all_new_facts.extend(facts)
            print(f"    Extracted {len(facts)} facts.")
            
        except Exception as e:
            print(f"    ERROR reading {filename}: {e}")

    # 3. Save to NEW CSV
    if all_new_facts:
        print(f"[Saving] Writing {len(all_new_facts)} facts to '{OUTPUT_CSV}'...")
        
        # Create new DataFrame
        df_final = pd.DataFrame(all_new_facts, columns=["Fact"])

        # Remove Exact Duplicates
        df_final = df_final.drop_duplicates().reset_index(drop=True)
        
        # Save
        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"[Success] Done! You now have a new database file: {OUTPUT_CSV}")
        
    else:
        print("[Warning] No valid text found in PDFs.")

if __name__ == "__main__":
    process_pdfs()
