"""FAISS-based semantic search engine using ID-mapped index"""
import faiss
import numpy as np
from typing import List, Tuple
import os
import random


class SearchEngine:
    """FAISS-based search engine for image embeddings"""

    def __init__(self, dim: int = 4096, index_path: str = "faiss_index.bin"):
        self.dim = dim
        self.index_path = index_path

        # Load existing index or create a new one
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            base_index = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIDMap(base_index)

    def create_albums(self, top_k: int = 5, distance_threshold: float = 1.5, album_size: int = 5) -> List[List[int]]:
        """
        Group similar images into albums (clusters).
        
        Returns exactly top_k albums, each containing up to album_size similar photos.
        Photos are marked as visited to avoid duplicate albums.
        Only includes photos within the distance threshold.
        
        OPTIMIZATIONS:
        - Batch retrieves all photos in ONE database query (not per-photo)
        - Caches embeddings in memory during execution
        - Single session for all DB operations
        
        Args:
            top_k: Number of albums to return
            distance_threshold: Maximum distance to consider photos as similar (default 1.0 for normalized embeddings)
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

        # ✅ OPTIMIZATION 1: Batch retrieve all photos in ONE query
        session = SessionLocal()
        try:
            # Fetch all photos at once, not in a loop
            photos_query = session.exec(select(Photo).where(Photo.id.in_(all_ids))).all()
            # ✅ OPTIMIZATION 2: Cache embeddings in memory
            embedding_cache = {}
            for photo in photos_query:
                embedding = photo.get_embedding()
                if embedding:
                    embedding_cache[photo.id] = embedding
        finally:
            session.close()

        visited = set()
        albums = []

        for photo_id in all_ids:
            # Stop if we have enough albums
            if len(albums) >= top_k:
                break
            
            # Skip if already in an album
            if photo_id in visited:
                continue
            
            # Skip if no embedding cached
            if photo_id not in embedding_cache:
                continue
            
            # Get embedding from cache (not DB)
            embedding = embedding_cache[photo_id]
            
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
        
        return albums

    def create_albums_kmeans(self, top_k: int = 5, seed: int = 42) -> List[List[int]]:
        """
        Group similar images into albums using FAISS k-means clustering.
        
        This is a BETTER approach than nearest-neighbor grouping:
        - Uses true k-means clustering instead of ad-hoc neighbor search
        - All photos get assigned to a cluster (no "orphans")
        - Deterministic results for same seed
        - Much faster for large datasets
        
        Args:
            top_k: Number of clusters (albums) to create
            seed: Random seed for reproducibility
            
        Returns:
            List of top_k albums, each album is a list of photo_ids
        """
        self.load()
        if self.index.ntotal < top_k:
            return []

        # Get all photo IDs from FAISS index
        id_map = self.index.id_map
        all_ids = np.array([id_map.at(i) for i in range(id_map.size())], dtype=np.int64)
        
        # Get all embeddings from the underlying index (IndexIDMap wraps the actual index)
        underlying_index = faiss.downcast_index(self.index.index)
        all_embeddings = underlying_index.reconstruct_n(0, self.index.ntotal).astype(np.float32)
        
        # ✅ Run k-means clustering
        kmeans = faiss.Kmeans(
            d=self.dim,
            k=top_k,
            niter=20,
            verbose=False,
            seed=seed
        )
        kmeans.train(all_embeddings)
        
        # Assign each embedding to nearest cluster
        distances, cluster_assignments = kmeans.index.search(all_embeddings, 1)
        
        # Group photos by cluster
        albums = [[] for _ in range(top_k)]
        for photo_id, cluster_id in zip(all_ids, cluster_assignments.flatten()):
            albums[cluster_id].append(int(photo_id))
        
        # Remove empty albums and return
        return [album for album in albums if album]

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
            List of (photo_id, distance) tuples with distance <= 1.0 (normalized embeddings)
        """
        self.load()

        if self.index.ntotal == 0:
            return []

        # Ensure query is float32 and correct shape
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # Search in FAISS index
        distances, ids = self.index.search(query_embedding, top_k)

        print(distances)

        # Filter invalid and distant results
        # With normalized embeddings, L2 distance range is 0-2, threshold of 1.0 works well
        results = [
            (int(photo_id), float(distance))
            for photo_id, distance in zip(ids[0], distances[0])
            if photo_id != -1 and distance <= 1.5
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

  