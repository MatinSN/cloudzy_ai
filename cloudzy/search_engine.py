"""FAISS-based semantic search engine"""
import faiss
import numpy as np
from typing import List, Tuple, Optional
import os
import pickle


class SearchEngine:
    """FAISS-based search engine for image embeddings"""
    
    def __init__(self, dim: int = 1024, index_path: str = "faiss_index.bin"):
        self.dim = dim
        self.index_path = index_path
        self.id_map: List[int] = []  # Map FAISS indices to photo IDs
        
        # Load existing index or create new one
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(dim)
    
    def add_embedding(self, photo_id: int, embedding: np.ndarray) -> None:
        """
        Add an embedding to the index.
        
        Args:
            photo_id: Unique photo identifier
            embedding: 1D numpy array of shape (dim,)
        """
        # Ensure embedding is float32 and correct shape
        embedding = embedding.astype(np.float32).reshape(1, -1)
        
        # Add to FAISS index
        self.index.add(embedding)
        
        # Track photo ID
        self.id_map.append(photo_id)
        
        # Save index to disk
        self.save()
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: 1D numpy array of shape (dim,)
            top_k: Number of results to return

        Returns:
            List of (photo_id, distance) tuples with distance <= 0.4
        """

        self.load()

        if self.index.ntotal == 0:
            return []

        # Ensure query is float32 and correct shape
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        # Map back to photo IDs and filter distances > 0.4
        results = [
            (self.id_map[int(idx)], float(distance))
            for distance, idx in zip(distances[0], indices[0])
            if distance <= 0.5
        ]

        return results

    def save(self) -> None:
        """Save index and id_map to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".ids", "wb") as f:
            pickle.dump(self.id_map, f)

    def load(self) -> None:
        """Load index and id_map from disk"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.index_path + ".ids"):
            with open(self.index_path + ".ids", "rb") as f:
                self.id_map = pickle.load(f)
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        return {
            "total_embeddings": self.index.ntotal,
            "dimension": self.dim,
            "id_map_size": len(self.id_map)
        }