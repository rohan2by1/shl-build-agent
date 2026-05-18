import json
import chromadb

def run_one_time_embedding():
    print("Starting offline embedding process...")
    
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        client.delete_collection(name="shl_assessments")
    except Exception:
        pass
        
    collection = client.create_collection(name="shl_assessments")
    
    with open("data/shl_full_catalog.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    metadatas = []
    ids = []

    for i, item in enumerate(data):
        doc_text = (
            f"Title: {item.get('title')}\n"
            f"Description: {item.get('description')}\n"
            f"Job Levels: {', '.join(item.get('job_levels', []))}\n"
            f"Test Types: {', '.join(item.get('test_types', []))}\n"
        )
        documents.append(doc_text)
        
        metadatas.append({
            "name": item.get("title", ""),
            "url": item.get("url", ""),
            "test_type": item.get("test_type_codes", [""])[0] if item.get("test_type_codes") else "Unknown"
        })
        ids.append(f"doc_{i}")

    print(f"Embedding {len(documents)} documents. This may take a moment...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print("✅ Embeddings generated and saved to ./chroma_db")

if __name__ == "__main__":
    run_one_time_embedding()