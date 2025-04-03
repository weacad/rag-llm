import os
import faiss
import numpy as np
import pickle
import ollama
from utils.prompt import prompt_template
from sentence_transformers import SentenceTransformer
import sys
import time

# Pull config from environment
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LLM_MODEL", "mistral:latest")

# FAISS configuration
FAISS_DB_DIR = "faiss_db"
INDEX_PATH = os.path.join(FAISS_DB_DIR, "index")
DOCSTORE_PATH = os.path.join(FAISS_DB_DIR, "docs.pkl")

st_model = None


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list:
    global st_model

    if model.startswith("sentence-transformers/"):
        if st_model is None:
            st_model = SentenceTransformer(model)
        return st_model.encode(text).tolist()
    else:
        return ollama.embeddings(model=model, prompt=text)["embedding"]


def search_embeddings(query, top_k=3):
    query_embedding = np.array(get_embedding(query), dtype=np.float32).reshape(1, -1)
    index = faiss.read_index(INDEX_PATH)

    with open(DOCSTORE_PATH, "rb") as f:
        docstore = pickle.load(f)

    distances, indices = index.search(query_embedding, top_k)
    top_results = []

    for i, idx in enumerate(indices[0]):
        metadata = docstore[idx]
        top_results.append(
            {
                "file": metadata.get("file", "Unknown file"),
                "page": metadata.get("page", "Unknown page"),
                "chunk": metadata.get("chunk", ""),
                "similarity": float(distances[0][i]),
            }
        )

    for result in top_results:
        print(f"---> File: {result['file']}, Page: {result['page']}, Chunk Preview: {result['chunk'][:100]}")

    return top_results


def generate_rag_response(query, context_results):
    context_str = "\n".join(
        [
            f"From {r['file']} (page {r['page']}, chunk {r['chunk'][:100]}) with similarity {r['similarity']:.2f}"
            for r in context_results
        ]
    )

    print(f"\n--- Context ---\n{context_str}\n")

    prompt = prompt_template.format(context=context_str, question=query)
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]


def interactive_search():
    print("🔍 FAISS RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        sys.stdout.flush()
        time.sleep(0.1)
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        context_results = search_embeddings(query)
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    interactive_search()
