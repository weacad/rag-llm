import chromadb
import ollama
from utils.prompt import prompt_template
from sentence_transformers import SentenceTransformer


chroma_client = chromadb.PersistentClient(path="./chroma_db")

COLLECTION_NAME = "ds4300_chroma"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:latest"


collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


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
    """Search embeddings from Chroma vector database."""
    query_embedding = get_embedding(query)

    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Transform results into the expected format
        top_results = [
            {
                "file": metadata.get("file", "Unknown file"),
                "page": metadata.get("page", "Unknown page"),
                "chunk": doc,
                "similarity": distance,
            }
            for doc, metadata, distance in zip(documents, metadatas, distances)
        ]

        # Print results for debugging
        for result in top_results:
            print(f"---> File: {result['file']}, Page: {result['page']}, Chunk Preview: {result['chunk'][:100]}")

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results):
    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk preview: {result.get('chunk', 'Unknown chunk')[:100]}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    print(f"\n--- Context ---\n{context_str}\n")

    prompt = prompt_template.format(context=context_str, question=query)

    # Generate response using Ollama
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    print("🔍 Chroma RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search Chroma collection for similar embeddings
        context_results = search_embeddings(query)

        # Generate RAG response from Ollama
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    interactive_search()
