# ELERAG-BitNet: Entity-Linked RAG on 1-Bit Edge Hardware

![System Status](https://img.shields.io/badge/Status-Research_Prototype-blue) ![Hardware](https://img.shields.io/badge/Hardware-1--Bit_Quantization-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Abstract
**ELERAG-BitNet** is a specialized Retrieval-Augmented Generation (RAG) system designed to run on low-resource hardware without sacrificing factual accuracy. 

By integrating the **Entity Linking Enhanced RAG (ELERAG)** framework with **1-bit Transformer architecture (BitNet b1.58)**, this system addresses two critical challenges in modern AI:
1.  **Hallucination in Specialized Domains:** Uses Wikidata entity disambiguation to prevent terminological confusion (e.g., distinguishing "Mercury" the element from "Mercury" the planet).
2.  **Edge Deployability:** Replaces FP16 matrix multiplications with 1-bit integer additions, significantly reducing energy consumption and memory bandwidth.

## ⚡ Key Features
* **Hybrid Retrieval:** Implements **Reciprocal Rank Fusion (RRF)** to combine dense vector search (`all-MiniLM-L6-v2`) with symbolic Entity Linking.
* **1-Bit Inference:** Optimized for `BitNet b1.58` (via `llama.cpp` GGUF) to enable high-speed generation on consumer CPUs.
* **Zero-Shot Disambiguation:** Dynamically links entities using a semantic-similarity + popularity hybrid score.

## 🛠️ Installation

### 1. Environment Setup
This project uses `uv` for dependency management (recommended), but standard pip works as well.
```bash
# Option A: Using pip
pip install -r requirements.txt

# Option B: Using uv
uv sync
```

### 2. Download the Model
*Note: Large model files are excluded from this repository.*
1.  Download the **BitNet b1.58** GGUF model (e.g., `bitnet_b1_58-3B-quantized.gguf`) from HuggingFace.
2.  Create a folder named `models/` in the root directory.
3.  Place the `.gguf` file inside `models/`.

## 🚀 Usage

### 1. Ingest Data (Build the Index)
Process the raw data into a hybrid vector/entity store.
```bash
# Uses the sample data provided in the repo
python main.py ingest data/textbook_high_quality.csv
```

### 2. Query the System
Run the full RAG pipeline (Retrieval -> Re-ranking -> 1-bit Generation).
```bash
python main.py query "What is the atomic weight of Mercury?"
```

## 📊 Methodology
This system follows a three-stage pipeline:

1.  **Smart Segmentation & Linking:** Input text is chunked with overlap. Entities are extracted using **Spacy** and linked to Wikidata IDs.
2.  **RRF Re-ranking:** * *Dense Score:* Cosine similarity via SentenceTransformers.
    * *Entity Score:* Jaccard similarity of linked Wikidata IDs.
    * *Fusion:* $Score = \frac{1}{k + rank_{dense}} + \frac{1}{k + rank_{entity}}$
3.  **1-Bit Generation:** The re-ranked context is fed to BitNet, which generates the answer using significantly lower energy per token than standard LLMs.

## 📂 Repository Structure
* `src/`: Core logic for ELERAG retrieval and BitNet inference.
* `data/`: High-quality benchmark datasets.
* `scripts/`: Utilities for corpus generation and PDF ingestion.
* `experiments/`: Ablation studies and domain-specific tests (Legal, Education).

## 📜 References
This work is an implementation and extension of the following papers:
* **ELERAG:** Granata et al., "Enhancing Retrieval-Augmented Generation with Entity Linking" (2025).
* **BitNet:** Wang et al., "BitNet: Scaling 1-bit Transformers for Large Language Models" (2023).

---
*Developed as a research implementation for low-resource factual QA.*
