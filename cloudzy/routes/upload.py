"""Upload endpoint for photos"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlmodel import Session
from pathlib import Path
import numpy as np

from cloudzy.database import get_session
from cloudzy.models import Photo
from cloudzy.schemas import UploadResponse
from cloudzy.utils.file_utils import save_uploaded_file
from cloudzy.ai_utils import generate_tags, generate_caption, generate_embedding
from cloudzy.search_engine import SearchEngine

router = APIRouter(tags=["photos"])

# Allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def validate_image_file(filename: str) -> bool:
    """Check if file has valid image extension"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@router.post("/upload", response_model=UploadResponse)
async def upload_photo(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload a photo and analyze it with AI.
    
    - Validates file type
    - Saves file to disk
    - Generates tags, caption, and embedding
    - Stores metadata in database
    - Indexes embedding in FAISS
    
    Returns: Photo metadata with ID
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not validate_image_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Read file content
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    
    # Save file to disk
    saved_filename = save_uploaded_file(content, file.filename)
    filepath = f"uploads/{saved_filename}"
    
    # Generate AI analysis
    tags = generate_tags(file.filename)
    caption = generate_caption(file.filename, tags)
    embedding = generate_embedding(file.filename, tags, caption)
    
    # Create photo record
    photo = Photo(
        filename=saved_filename,
        filepath=filepath,
        caption=caption,
    )
    photo.set_tags(tags)
    photo.set_embedding(embedding.tolist())
    
    # Save to database
    session.add(photo)
    session.commit()
    session.refresh(photo)
    
    # Index in FAISS (in background if needed)
    search_engine = SearchEngine()
    search_engine.add_embedding(photo.id, embedding)
    
    return UploadResponse(
        id=photo.id,
        filename=saved_filename,
        tags=tags,
        caption=caption,
        message=f"Photo uploaded successfully with ID {photo.id}"
    )