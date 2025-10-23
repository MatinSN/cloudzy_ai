"""AI utilities for generating tags, captions, and embeddings"""
import numpy as np
from typing import List, Tuple
import random


def generate_tags(filename: str) -> List[str]:
    """
    Generate tags for an image based on filename.
    In production, this would use CLIP or similar models.
    Currently using placeholder logic.
    """
    # Extract meaningful words from filename
    name_parts = filename.lower().replace("_", " ").replace("-", " ").split()
    name_parts = [p.replace(".jpg", "").replace(".png", "").replace(".jpeg", "") 
                  for p in name_parts if p]
    
    # Common image tags for demo
    common_tags = [
        "photo", "image", "landscape", "portrait", "nature", "architecture",
        "people", "animal", "food", "object", "abstract", "text", "sunset",
        "mountain", "beach", "forest", "urban", "indoor", "outdoor"
    ]
    
    # Select random subset of common tags + filename parts
    tags = list(set(name_parts[:2] + random.sample(common_tags, min(3, len(common_tags)))))
    return tags[:5]  # Return up to 5 tags


def generate_caption(filename: str, tags: List[str]) -> str:
    """
    Generate a caption for an image.
    In production, this would use BLIP or similar models.
    Currently using placeholder logic.
    """
    caption_templates = [
        "A beautiful {tag} photograph",
        "Captured moment: {tag}",
        "Scenic view of {tag}",
        "Amazing {tag} scene",
        "Photography: {tag} collection",
    ]
    
    tag = tags[0] if tags else "image"
    template = random.choice(caption_templates)
    return template.format(tag=tag)


def generate_embedding(filename: str, tags: List[str], caption: str) -> np.ndarray:
    """
    Generate a 512-dimensional embedding for semantic search.
    In production, this would use CLIP or similar models.
    Currently using placeholder random embeddings (reproducible from filename).
    """
    # Create a reproducible random embedding based on filename
    # In production: use CLIP or similar to generate real embeddings
    random.seed(hash(filename) % (2**32))
    embedding = np.random.randn(512).astype(np.float32)
    # Normalize to unit vector
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def generate_filename_embedding(filename: str) -> np.ndarray:
    """
    Generate a deterministic embedding from filename for testing.
    Ensures same filename always gets same embedding.
    """
    random.seed(hash(filename) % (2**32))
    embedding = np.random.randn(512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding