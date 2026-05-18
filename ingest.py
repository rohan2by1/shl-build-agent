import json
import chromadb

# The official data uses full strings, but the API response schema requires the 1-letter codes.
# This map translates them before they go into the vector database.
TEST_TYPE_MAP = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S"
}

def run_one_time_embedding():
    print("Starting offline embedding process with official dataset...")
    
    # 1. Initialize ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        client.delete_collection(name="shl_assessments")
    except Exception:
        pass
        
    collection = client.create_collection(name="shl_assessments")
    
    # 2. Load the official JSON (ensure you replace your scraped file with this official one)
    with open("data/shl_product_catalog.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    metadatas = []
    ids = []

    for item in data:
        name = item.get('name', '')
        desc = item.get('description', '')
        job_levels = ", ".join(item.get('job_levels', []))
        keys_list = item.get('keys', [])
        keys_str = ", ".join(keys_list)
        duration = item.get('duration', '')
        languages = ", ".join(item.get('languages', []))

        # Build a rich context document for the Vector Search to match against
        doc_text = (
            f"Title: {name}\n"
            f"Description: {desc}\n"
            f"Job Levels: {job_levels}\n"
            f"Categories: {keys_str}\n"
            f"Duration: {duration}\n"
            f"Languages: {languages}\n"
        )
        documents.append(doc_text)
        
        # Figure out the primary 1-letter test code. 
        # If an item has multiple keys, we take the first one and map it.
        primary_key = keys_list[0] if keys_list else ""
        test_type_code = TEST_TYPE_MAP.get(primary_key, primary_key[:1].upper() if primary_key else "U")

        # Standardize the metadata so the agent doesn't have to guess
        metadatas.append({
            "name": name,
            "url": item.get('link', ''), # Map 'link' to 'url'
            "test_type": test_type_code
        })
        
        # Use the official entity_id as the Chroma document ID
        ids.append(str(item.get('entity_id')))

    print(f"Embedding {len(documents)} official documents. This may take a moment...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print("✅ Embeddings generated and saved to ./chroma_db")

if __name__ == "__main__":
    run_one_time_embedding()