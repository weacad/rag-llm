import ollama
import numpy as np
import os
import fitz
import re
import chromadb
from sentence_transformers import SentenceTransformer


# Chroma connection, we will save to folder in files
chroma_client = chromadb.PersistentClient(path="./chroma_db")

COLLECTION_NAME = "ds4300_chroma"


# used to clear the chroma vector store
def clear_chroma_store():

    # list all collections
    collections = chroma_client.list_collections()

    # remove existing collection if it exists
    if COLLECTION_NAME in collections:
        print(f"Found collection and deleting")
        chroma_client.delete_collection(name=COLLECTION_NAME)
    else:
        print(f"Did not find collection")

    # Create new collection to populate
    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    print(f"Created new collection: '{COLLECTION_NAME}'")

    return collection


# Generate an embedding
st_model = None


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    global st_model

    if model.startswith("sentence-transformers/"):
        if st_model is None:
            st_model = SentenceTransformer(model)
        return st_model.encode(text).tolist()
    else:
        return ollama.embeddings(model=model, prompt=text)["embedding"]


# Store the embedding in Chroma
def store_embedding(collection, file: str, page: str, chunk_index: int, chunk: str, embedding: list):
    doc_id = f"{file}_page_{page}_chunk_{chunk_index}"
    collection.add(ids=[doc_id], documents=[chunk], embeddings=[embedding], metadatas=[{"file": file, "page": page}])
    print(f"Stored embedding for: {file} page {page} chunk {chunk_index}")


# Extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# Split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Preprocess text
def preprocess_text(text):
    text = re.sub(r"(Mark Fontenot, PhD.*?Northeastern University)", "", text, flags=re.DOTALL)
    text = re.sub(r"DS 4300.*?\n", "", text)
    text = re.sub(r"(Page \d+|^\s*\d+\s*$)", "", text, flags=re.MULTILINE)
    text = re.sub(r"(mailto:.*?@.*?\.\w+)", "", text)
    text = re.sub(r"•|●|-○ ", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    return text.strip()


# Process all PDF files in a given directory
def process_pdfs(collection, data_dir, chunk_size=300, overlap=50, embedding_model="nomic-embed-text"):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page:
                cleaned_text = preprocess_text(text)
                chunks = split_text_into_chunks(cleaned_text, chunk_size, overlap)

                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, model=embedding_model)
                    store_embedding(
                        collection,
                        file_name,
                        page=str(page_num),
                        chunk_index=chunk_index,
                        chunk=chunk,
                        embedding=embedding,
                    )

            print(f" -----> Processed {file_name}")


# Query Chroma (vector similarity search)
def query_chroma(collection, query_text: str, top_k=5):
    embedding = get_embedding(query_text)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    print("\nQuery Results:\n")
    for doc, score, metadata in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
        print(f"File: {metadata['file']} | Page: {metadata['page']}")
        print(f"Chunk: {doc[:100]}...")
        print(f"Distance: {score}\n")


def ingest_chroma(chunk_size, overlap, embedding_model):
    # Clear store and recreate collection
    collection = clear_chroma_store()

    # Ingest PDFs into Chroma
    process_pdfs(collection, "../data/", chunk_size, overlap, embedding_model)

    print("\n--- Done processing PDFs ---\n")

    print("\n--- Your database has been created at ../chroma_db. ---\n")

    # # Query Chroma after ingestion
    # query_chroma(collection, "What is the capital of France?")


def main():
    ingest_chroma()


if __name__ == "__main__":
    main()
