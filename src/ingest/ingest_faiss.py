import os
import re
import fitz
import faiss
import numpy as np
import ollama
import pickle
from sentence_transformers import SentenceTransformer


# Configuration
FAISS_DB_DIR = "faiss_db"
INDEX_PATH = os.path.join(FAISS_DB_DIR, "index")
DOCSTORE_PATH = os.path.join(FAISS_DB_DIR, "docs.pkl")

# Ensure output directory exists
os.makedirs(FAISS_DB_DIR, exist_ok=True)


# Clear the FAISS DB folder (index + metadata)
def clear_faiss_store():
    for filename in [INDEX_PATH, DOCSTORE_PATH]:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Removed: {filename}")


st_model = None


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    global st_model

    if model.startswith("sentence-transformers/"):
        if st_model is None:
            st_model = SentenceTransformer(model)
        return st_model.encode(text).tolist()
    else:
        return ollama.embeddings(model=model, prompt=text)["embedding"]


# Store the embedding to FAISS index


def store_faiss_index(vectors, metadata):
    vectors = np.vstack(vectors)
    vector_dim = vectors.shape[1]
    index = faiss.IndexFlatL2(vector_dim)
    index.add(vectors)

    index = faiss.IndexFlatL2(vector_dim)
    index.add(np.vstack(vectors))

    os.makedirs(FAISS_DB_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(DOCSTORE_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("FAISS index and doc metadata stored.")


# Extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return [(page_num, page.get_text()) for page_num, page in enumerate(doc)]


# Split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]


# Preprocess text
def preprocess_text(text):
    text = re.sub(r"(Mark Fontenot, PhD.*?Northeastern University)", "", text, flags=re.DOTALL)
    text = re.sub(r"DS 4300.*?\n", "", text)
    text = re.sub(r"(Page \d+|^\s*\d+\s*$)", "", text, flags=re.MULTILINE)
    text = re.sub(r"(mailto:.*?@.*?\.\w+)", "", text)
    text = re.sub(r"\u2022|\u25CF|-\u25CB ", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    return text.strip()


# Process all PDF files in a given directory and store to FAISS
def process_pdfs_to_faiss(data_dir, chunk_size, overlap, embedding_model):
    vectors = []
    metadata = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            for page_num, text in extract_text_from_pdf(pdf_path):
                cleaned = preprocess_text(text)
                chunks = split_text_into_chunks(cleaned, chunk_size, overlap)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = np.array(get_embedding(chunk, embedding_model), dtype=np.float32)
                    vectors.append(embedding)
                    metadata.append({"file": file_name, "page": page_num, "chunk": chunk, "chunk_index": chunk_index})
            print(f"-----> Processed {file_name}")
    store_faiss_index(vectors, metadata)


def ingest_faiss(chunk_size, overlap, embedding_model):
    clear_faiss_store()
    process_pdfs_to_faiss("../data/", chunk_size, overlap, embedding_model)


def main():
    ingest_faiss


if __name__ == "__main__":
    main()
