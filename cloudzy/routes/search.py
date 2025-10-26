"""Semantic search endpoint using FAISS"""
from fastapi import APIRouter, Query, Depends, HTTPException
from sqlmodel import Session, select
import numpy as np

from cloudzy.database import get_session
from cloudzy.models import Photo
from cloudzy.schemas import SearchResponse, SearchResult
from cloudzy.search_engine import SearchEngine
# from cloudzy.ai_utils import generate_filename_embedding
from cloudzy.ai_utils import  ImageEmbeddingGenerator
import os

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
async def search_photos(
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results"),
    session: Session = Depends(get_session),
):
    """
    Semantic search endpoint using FAISS.

    Args:
        q: Search query (used to generate embedding)
        top_k: Number of results to return (max 50)

    Returns: List of similar photos
    """

    generator = ImageEmbeddingGenerator()
    query_embedding = generator._embed_text(q)

    search_engine = SearchEngine()
    search_results = search_engine.search(query_embedding, top_k=top_k)

    if not search_results:
        return SearchResponse(
            query=q,
            results=[],
            total_results=0,
        )

    APP_DOMAIN = os.getenv("APP_DOMAIN")
    result_objects = []

    for photo_id, distance in search_results:
        statement = select(Photo).where(Photo.id == photo_id)
        photo = session.exec(statement).first()

        if photo:
            result_objects.append(
                SearchResult(
                    photo_id=photo.id,
                    filename=photo.filename,
                    image_url=f"{APP_DOMAIN}uploads/{photo.filename}",
                    tags=photo.get_tags(),
                    caption=photo.caption,
                    distance=distance,
                )
            )

    return SearchResponse(
        query=q,
        results=result_objects,
        total_results=len(result_objects),
    )


# @router.post("/search/image-to-image")
# async def image_to_image_search(
#     reference_photo_id: int = Query(..., description="Reference photo ID"),
#     top_k: int = Query(5, ge=1, le=50),
#     session: Session = Depends(get_session),
# ):
#     """
#     Find similar images to a reference photo (image-to-image search).
    
#     Args:
#         reference_photo_id: ID of the reference photo
#         top_k: Number of similar results
    
#     Returns: Similar photos
#     """
#     # Get reference photo
#     statement = select(Photo).where(Photo.id == reference_photo_id)
#     reference_photo = session.exec(statement).first()
    
#     if not reference_photo:
#         raise HTTPException(status_code=404, detail=f"Photo {reference_photo_id} not found")
    
#     # Get reference embedding
#     reference_embedding = reference_photo.get_embedding()
#     if not reference_embedding:
#         raise HTTPException(status_code=400, detail="Photo has no embedding")
    
#     # Search in FAISS
#     search_engine = SearchEngine()
#     search_results = search_engine.search(
#         np.array(reference_embedding, dtype=np.float32),
#         top_k=top_k + 1  # +1 to skip the reference photo itself
#     )
    
#     # Build results (skip first result which is the reference photo itself)
#     result_objects = []
#     for photo_id, distance in search_results[1:]:  # Skip first result
#         statement = select(Photo).where(Photo.id == photo_id)
#         photo = session.exec(statement).first()
        
#         if photo:
#             result_objects.append(
#                 SearchResult(
#                     photo_id=photo.id,
#                     filename=photo.filename,
#                     tags=photo.get_tags(),
#                     caption=photo.caption,
#                     distance=distance,
#                 )
#             )
    
#     return SearchResponse(
#         query=f"Similar to photo {reference_photo_id}",
#         results=result_objects[:top_k],
#         total_results=len(result_objects),
#     )