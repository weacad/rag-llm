import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
from sentence_transformers import SentenceTransformer
import re


# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")


# Create an HNSW index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


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


# store the embedding in Redis
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {chunk}")


# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def preprocess_text(text):
    """Preprocess text by removing patterns and cleaning up formatting."""

    # Remove pattern/consistent text
    text = re.sub(r"(Mark Fontenot, PhD.*?Northeastern University)", "", text, flags=re.DOTALL)
    text = re.sub(r"DS 4300.*?\n", "", text)
    text = re.sub(r"(Page \d+|^\s*\d+\s*$)", "", text, flags=re.MULTILINE)
    text = re.sub(r"(mailto:.*?@.*?\.\w+)", "", text)

    # Remove bullet points and special characters (line breaks)
    text = re.sub(r"•|●|-○ ", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    # Whitespace
    text = text.strip()

    return text


# Process all PDF files in a given directory
def process_pdfs(data_dir, chunk_size=300, overlap=50, embedding_model="nomic-embed-text"):
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
                        file=file_name,
                        page=str(page_num),
                        chunk=chunk,
                        embedding=embedding,
                    )

            print(f" -----> Processed {file_name}")


def query_redis(query_text: str):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    query_text = "Efficient search in vector databases"
    embedding = get_embedding(query_text)
    res = redis_client.ft(INDEX_NAME).search(q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()})
    # print(res.docs)

    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")


# replace main with this so i can call it when needed
def ingest_redis(chunk_size=300, overlap=50, embedding_model="nomic-embed-text"):
    clear_redis_store()
    create_hnsw_index()
    process_pdfs("../data/", chunk_size, overlap, embedding_model)
    print("\n--- Ingested: Redis ---\n")


def main():
    ingest_redis()


if __name__ == "__main__":
    main()
