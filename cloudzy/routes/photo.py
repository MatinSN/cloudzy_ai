"""Photo retrieval endpoints"""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from cloudzy.database import get_session
from cloudzy.models import Photo
from cloudzy.schemas import PhotoDetailResponse

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
    
    return PhotoDetailResponse(
        id=photo.id,
        filename=photo.filename,
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
    
    return [
        PhotoDetailResponse(
            id=photo.id,
            filename=photo.filename,
            tags=photo.get_tags(),
            caption=photo.caption,
            embedding=photo.get_embedding(),
            created_at=photo.created_at,
        )
        for photo in photos
    ]