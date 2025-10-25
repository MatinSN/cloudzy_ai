"""FAISS-based semantic search engine using ID-mapped index"""
import faiss
import numpy as np
from typing import List, Tuple
import os
import random


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

    def create_albums(self, top_k: int = 5, distance_threshold: float = 0.3, album_size: int = 5) -> List[List[int]]:
        """
        Group similar images into albums (clusters).
        
        Returns exactly top_k albums, each containing up to album_size similar photos.
        Photos are marked as visited to avoid duplicate albums.
        Only includes photos within the distance threshold.
        
        Args:
            top_k: Number of albums to return
            distance_threshold: Maximum distance to consider photos as similar (default 0.3)
            album_size: How many similar photos to search for per album (default 5)
            
        Returns:
            List of top_k albums, each album is a list of photo_ids (randomized order each call)
        """
        from cloudzy.database import SessionLocal
        from cloudzy.models import Photo
        from sqlmodel import select
        
        self.load()
        if self.index.ntotal == 0:
            return []

        # Get all photo IDs from FAISS index
        id_map = self.index.id_map
        all_ids = [id_map.at(i) for i in range(id_map.size())]
        
        # Shuffle for randomization - different albums each call
        random.shuffle(all_ids)

        visited = set()
        albums = []

        for photo_id in all_ids:
            # Stop if we have enough albums
            if len(albums) >= top_k:
                break
            
            # Skip if already in an album
            if photo_id in visited:
                continue
            
            # Get embedding from database
            session = SessionLocal()
            try:
                photo = session.exec(select(Photo).where(Photo.id == photo_id)).first()
                if not photo:
                    continue
                
                embedding = photo.get_embedding()
                if not embedding:
                    continue
                
                # Search for similar images
                query_embedding = np.array(embedding).reshape(1, -1).astype(np.float32)
                distances, ids = self.index.search(query_embedding, album_size)
                
                # Build album: collect similar photos that haven't been visited and are within threshold
                album = []
                for pid, distance in zip(ids[0], distances[0]):
                    if pid != -1 and pid not in visited and distance <= distance_threshold:
                        album.append(int(pid))
                        visited.add(pid)
                
                # Add album if it has at least 1 photo
                if album:
                    albums.append(album)
                    
            finally:
                session.close()
        
        return albums

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

  