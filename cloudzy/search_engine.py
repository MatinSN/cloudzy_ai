"""FAISS-based semantic search engine using ID-mapped index"""
import faiss
import numpy as np
from typing import List, Tuple
import os


class SearchEngine:
    """FAISS-based search engine for image embeddings"""

    def __init__(self, dim: int = 1024, index_path: str = "faiss_index.bin"):
        self.dim = dim
        self.index_path = index_path

        # Load existing index or create a new one
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            base_index = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIDMap(base_index)

    def add_embedding(self, photo_id: int, embedding: np.ndarray) -> None:
        """
        Add an embedding to the index.

        Args:
            photo_id: Unique photo identifier
            embedding: 1D numpy array of shape (dim,)
        """
        # Ensure embedding is float32 and correct shape
        embedding = embedding.astype(np.float32).reshape(1, -1)

        # Add embedding with its ID
        self.index.add_with_ids(embedding, np.array([photo_id], dtype=np.int64))

        # Save index to disk
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: 1D numpy array of shape (dim,)
            top_k: Number of results to return

        Returns:
            List of (photo_id, distance) tuples with distance <= 0.5
        """
        self.load()

        if self.index.ntotal == 0:
            return []

        # Ensure query is float32 and correct shape
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # Search in FAISS index
        distances, ids = self.index.search(query_embedding, top_k)

        # Filter invalid and distant results
        results = [
            (int(photo_id), float(distance))
            for photo_id, distance in zip(ids[0], distances[0])
            if photo_id != -1 and distance <= 0.5
        ]

        return results

    def save(self) -> None:
        """Save FAISS index to disk"""
        faiss.write_index(self.index, self.index_path)

    def load(self) -> None:
        """Load FAISS index from disk"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            # Recreate empty ID-mapped index if missing
            base_index = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIDMap(base_index)

    def get_stats(self) -> dict:
        """Get index statistics"""
        return {
            "total_embeddings": self.index.ntotal,
            "dimension": self.dim,
            "index_type": type(self.index).__name__,
        }
