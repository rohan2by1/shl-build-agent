import chromadb
from typing import List, Dict

class CatalogRetriever:
    def __init__(self):
        # Connect to the pre-computed embeddings saved on disk by ingest.py
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

        try:
            self.collection = self.chroma_client.get_collection(
                name="shl_assessments"
            )
        except Exception:
            raise RuntimeError(
                "Vector collection missing. Run ingestion first."
            )

        print("Connected to existing Vector DB.")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        formatted_results = []

        if results and results['metadatas']:
            for meta, doc in zip(
                results['metadatas'][0],
                results['documents'][0]
            ):
                meta['document_context'] = doc
                formatted_results.append(meta)

        return formatted_results