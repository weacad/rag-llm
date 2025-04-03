
# DS4300 Practical 02 - RAG Pipeline Testing & Optimization

This project explores the design and evaluation a local Retrieval-Augmented Generation system that allows a user to query the collective DS4300 notes from members of our team. The goal was to benchmark different RAG pipeline configurations to help generate high-quality answers to potential exam questions.

## Project Objectives
- Build a functional RAG system using course materials (PDFs, slides, notes)
- Benchmark different pipeline parameters: vector DBs, LLMs, embedding models, chunk sizes, and overlap
- Use real data to assess speed, memory usage, and response quality

## Experimentation Strategy
We tested and recorded results for multiple dimensions:
- **Chunk Size**: 200, 300, 500, 1000, 1500, 2000
- **Overlap**: 0, 50, 100
- **Embedding Models**:
  - `nomic-embed-text` (local via Ollama)
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `sentence-transformers/all-mpnet-base-v2`
- **LLMs**:
  - `mistral` (Ollama)
  - `llama2` (Ollama)
- **Vector DBs**:
  - Redis
  - FAISS
  - Chroma

Each config was tested multiple prompts such as:

```
What is a transaction?
Why does the CAP principle not make sense when applied to a single-node MongoDB 
instance?
Describe the differences between horizontal and vertical scaling
What is disk-based indexing and why is it important for database systems?
```

Evaluated for:
- Ingest time (seconds)
- Memory usage (MB)
- Search time (seconds)
- Response clarity & correctness

## How to Run

### 1. **Clone the Repository**

```bash
git clone https://github.com/weacad/rag-llm.git
cd rag-llm/src
```

---

### 2. **Create a Python Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate  
```

---

### 3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### 4. **Start Ollama (for local LLMs + embeddings)**

Make sure you have [Ollama](https://ollama.com) installed and running:

```bash
ollama run mistral
ollama run llama2
ollama pull nomic-embed-text
```

---

### 5. **Add Your Data**

Place any course `.pdf` files into the `../data/` folder.

---

### 6. **Run a Test Pipeline**

You can run the full ingestion + search + evaluation using `driver.py`.

#### Modify `utils/pipeline_tests.json` to define configs:
```json
[
  {
    "name": "Redis-Mistral-Test",
    "vector_db": "redis",
    "chunk_size": 300,
    "overlap": 50,
    "embedding_model": "nomic-embed-text",
    "llm_model": "mistral"
  }
]
```

Then run (from `/src`):

```bash
python -m driver
```

The output will include:
- Ingest time
- Memory usage
- Search time
- LLM response
- Saved to `results.csv`

---

### 7. **Use Interactive Search Mode**

You can test each vector DB in a live CLI:

1. First ingest your data into a given DB:
```bash
python -m ingest.ingest_redis   # for Redis
python -m ingest.ingest_faiss   # for FAISS
python -m ingest.ingest_chroma  # for Chroma
```
2. Run the live CLI to live prompt the RAG system

```bash
python -m search.search_redis   # for Redis
python -m search.search_faiss   # for FAISS
python -m search.search_chroma  # for Chroma
```

Type in a query like:

```
What is a transaction?
```

And see the RAG-based LLM response using retrieved chunks.

---



## Contributors
- Caden Weaver 

---

> Northeastern University – DS4300 Practical 02 – March-April 2025
