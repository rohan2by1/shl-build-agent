import json
import chromadb
from typing import List, Dict

class CatalogRetriever:
    def __init__(self, data_path: str = "data/shl_full_catalog.json"):
        self.chroma_client = chromadb.Client()
        self.collection_name = "shl_assessments"
        
        # Reset collection if it exists to ensure fresh data
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except:
            pass
            
        self.collection = self.chroma_client.create_collection(name=self.collection_name)
        self._load_data(data_path)

    def _load_data(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []
        metadatas = []
        ids = []

        for i, item in enumerate(data):
            # Create a rich text document for semantic search to grasp
            doc = (
                f"Title: {item.get('title')}\n"
                f"Description: {item.get('description')}\n"
                f"Job Levels: {', '.join(item.get('job_levels', []))}\n"
                f"Test Types: {', '.join(item.get('test_types', []))}\n"
            )
            documents.append(doc)
            
            # Store structured data to return later
            metadatas.append({
                "name": item.get("title", ""),
                "url": item.get("url", ""),
                "test_type": item.get("test_type_codes", [""])[0] if item.get("test_type_codes") else "Unknown"
            })
            ids.append(f"doc_{i}")

        # Add to Chroma (Chroma will automatically generate embeddings using its default model)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Loaded {len(documents)} assessments into Vector DB.")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results back into a clean list of dictionaries
        formatted_results = []
        if results and results['metadatas']:
            for meta, doc in zip(results['metadatas'][0], results['documents'][0]):
                meta['document_context'] = doc
                formatted_results.append(meta)
        return formatted_results