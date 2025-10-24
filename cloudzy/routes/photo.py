"""Photo retrieval endpoints"""
from fastapi import APIRouter, Depends, HTTPException,Query
from sqlmodel import Session, select
import numpy as np

from cloudzy.database import get_session
from cloudzy.models import Photo
from cloudzy.schemas import PhotoDetailResponse,AlbumsResponse,PhotoItem,AlbumItem
from cloudzy.search_engine import SearchEngine
from cloudzy.ai_utils import TextSummarizer
import os

router = APIRouter(tags=["photos"])


@router.get("/photo/{photo_id}", response_model=PhotoDetailResponse)
async def get_photo(
    photo_id: int,
    session: Session = Depends(get_session),
):
    """
    Get photo metadata by ID.
    
    Returns: Photo metadata including tags, caption, embedding info
    """
    statement = select(Photo).where(Photo.id == photo_id)
    photo = session.exec(statement).first()
    
    if not photo:
        raise HTTPException(status_code=404, detail=f"Photo {photo_id} not found")
    
    APP_DOMAIN = os.getenv("APP_DOMAIN")
    
    return PhotoDetailResponse(
        id=photo.id,
        filename=photo.filename,
        image_url = f"{APP_DOMAIN}uploads/{photo.filename}",
        tags=photo.get_tags(),
        caption=photo.caption,
        embedding=photo.get_embedding(),
        created_at=photo.created_at,
    )


@router.get("/photos", response_model=list[PhotoDetailResponse])
async def list_photos(
    skip: int = 0,
    limit: int = 10,
    session: Session = Depends(get_session),
):
    """
    List all photos with pagination.
    
    Args:
        skip: Number of photos to skip (pagination)
        limit: Max photos to return (default 10)
    
    Returns: List of photo metadata
    """
    if limit > 100:
        limit = 100  # Cap limit at 100
    
    statement = select(Photo).offset(skip).limit(limit)
    photos = session.exec(statement).all()

    APP_DOMAIN = os.getenv("APP_DOMAIN")
    
    return [
        PhotoDetailResponse(
            id=photo.id,
            filename=photo.filename,
            image_url = f"{APP_DOMAIN}uploads/{photo.filename}",
            tags=photo.get_tags(),
            caption=photo.caption,
            embedding=photo.get_embedding(),
            created_at=photo.created_at,
        )
        for photo in photos
    ]


@router.get("/albums", response_model=AlbumsResponse)
async def get_albums(
    top_k: int = Query(5, ge=2, le=50),
    session: Session = Depends(get_session),
):
    """
    Create albums of semantically similar photos.
    """

    search_engine = SearchEngine()
    albums_ids = search_engine.create_albums(top_k=top_k)
    APP_DOMAIN = os.getenv("APP_DOMAIN") or "http://127.0.0.1:8000/"
    summarizer = TextSummarizer()

    albums_response = []

    for album_ids in albums_ids:
        # Query all photos in this album in one go
        statement = select(Photo).where(Photo.id.in_(album_ids))
        photos = session.exec(statement).all()

        # Build a dict for fast lookup
        photo_lookup = {photo.id: photo for photo in photos}

        album_photos = []
        album_descriptions = []  # Collect captions and tags for summary
        
        for pid in album_ids:
            photo = photo_lookup.get(pid)
            if not photo:
                continue

            # Find distance from FAISS search
            embedding = photo.get_embedding()
            if not embedding:
                continue
            
            query_embedding = np.array(embedding).astype(np.float32).reshape(1, -1)
            distances, ids = search_engine.index.search(query_embedding, top_k)
            distance_val = next((d for i, d in zip(ids[0], distances[0]) if i == pid), 0.0)

            album_photos.append(
                PhotoItem(
                    photo_id=photo.id,
                    filename=photo.filename,
                    image_url=f"{APP_DOMAIN}uploads/{photo.filename}",
                    tags=photo.get_tags(),
                    caption=photo.caption,
                    distance=float(distance_val),
                )
            )
            
            # Collect descriptions for album summary
            if photo.caption:
                album_descriptions.append(photo.caption)
            tags = photo.get_tags()
            if tags:
                album_descriptions.append(" ".join(tags))

        # Generate album summary from compiled descriptions
        combined_description = " ".join(album_descriptions)
        album_summary = summarizer.summarize(combined_description)
        
        albums_response.append(
            AlbumItem(album_summary=album_summary, album=album_photos)
        )

    return albums_response