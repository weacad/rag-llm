import json
import time
import subprocess
import psutil
import os
import csv
from ingest.ingest_redis import ingest_redis
from ingest.ingest_chroma import ingest_chroma
from ingest.ingest_faiss import ingest_faiss


def measure_memory():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # MB


def run_pipeline(config):
    print(f"\nRunning config: {config['name']}")

    # start timer and zero memory
    start_mem = measure_memory()
    start_time = time.time()

    # map json tex to ingest function
    ingest_funcs = {
        "faiss": ingest_faiss,
        "redis": ingest_redis,
        "chroma": ingest_chroma,
    }

    # each db type entails different python command to run
    search_cmds = {
        "faiss": ["python", "-m", "search.search_faiss"],
        "redis": ["python", "-m", "search.search_redis"],
        "chroma": ["python", "-m", "search.search_chroma"],
    }

    vector_db = config["vector_db"]

    # get params from config
    chunk_size = config.get("chunk_size", "300")
    overlap = config.get("overlap", "50")
    embedding_model = config.get("embedding_model", "nomic-embed-text")

    # Run ingest (runs imported function from individual python files)
    ingest_funcs[vector_db](chunk_size, overlap, embedding_model)
    ingest_time = time.time() - start_time
    ingest_mem = measure_memory() - start_mem
    print(f"Ingest time: {ingest_time:.3f}s, Memory used: {ingest_mem:.3f}MB")

    # Don't capture memory for search as OLLama does everything p sure
    search_start = time.time()

    # set environmental variables here
    llm_model = config.get("llm_model", "mistral")
    env = os.environ.copy()
    env["LLM_MODEL"] = llm_model
    env["EMBEDDING_MODEL"] = embedding_model

    question = "Why does the CAP principle not make sense when applied to a single-node MongoDB instance? "

    question += "\nexit\n"

    result = subprocess.run(
        search_cmds[vector_db], input=question, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env
    )
    search_time = time.time() - search_start
    print(f"Search time: {search_time:.2f}s")

    # Extract LLM response
    response = result.stdout.split("--- Response ---")[-1].strip()
    # response = response_sections[1].strip() if len(response_sections) > 1 else "REPONSE ERROR"
    # response = result.stdout

    print(response)

    # Return data
    return {
        "name": config["name"],
        "vector_db": vector_db,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": embedding_model,
        "llm": llm_model,
        "ingest_time": round(ingest_time, 2),
        "ingest_memory": round(ingest_mem, 2),
        "search_time": round(search_time, 2),
        "question": question[:-26],
        # remove next line text and truncate response to 500 characters
        "response": response.replace("Enter your search query:", ""),
    }


def write_results(row, write_header=False):
    with open("results.csv", "a", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    with open("utils/pipeline_tests.json", "r") as file:
        configs = json.load(file)

    # Write header only if file doesn't exist
    write_header = not os.path.exists("results.csv")

    for i, config in enumerate(configs):
        row = run_pipeline(config)
        write_results(row, write_header=(write_header and i == 0))

    print("----------------------------")
    print("     Testing Complete    ")
    print("----------------------------")


if __name__ == "__main__":
    main()
