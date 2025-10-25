"""Upload endpoint for photos"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlmodel import Session
from pathlib import Path
import numpy as np

from cloudzy.database import get_session
from cloudzy.models import Photo
from cloudzy.schemas import UploadResponse
from cloudzy.utils.file_utils import save_uploaded_file
from cloudzy.ai_utils import  ImageEmbeddingGenerator
from cloudzy.search_engine import SearchEngine

from cloudzy.agents.image_analyzer import ImageDescriber
from cloudzy.agents.image_analyzer_2 import ImageAnalyzerAgent
from cloudzy.utils.file_upload_service import ImgBBUploader


import os

router = APIRouter(tags=["photos"])

# Allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


result = {
  "tags": [
    "tiger",
    "wildlife",
    "predator",
    "forest",
    "golden hour",
    "nature",
    "animal",
    "walking",
    "orange",
    "striped"
  ],
  "description": "A majestic tiger strides forward with purpose through a dry, golden-hued forest. Its powerful body and distinctive orange-and-black striped coat are clearly visible as it moves along a dirt path. The background is softly blurred, emphasizing the tiger's presence and creating a sense of depth. Warm sunlight bathes the scene, highlighting the texture of its fur and the surrounding dry grass and trees. The tiger's intense gaze is fixed ahead, conveying both power and focus. This image captures the raw beauty and untamed spirit of this apex predator in its natural habitat during what appears to be the golden hour.",
  "caption": "A tiger walks confidently through a sun-drenched forest at golden hour."
}

# result = {
#   "tags": [
#     "woman",
#     "photography",
#     "camera",
#     "smiling",
#     "car",
#     "travel",
#     "outdoors",
#     "film",
#     "plaid",
#     "window"
#   ],
#   "description": "A cheerful woman with long brown hair is leaning out of a car window, holding a vintage-style film camera up to her eye. She's wearing a red, white, and blue plaid shirt and has a bright, joyful smile. The background is softly blurred with green trees and an overcast sky, suggesting a scenic road trip. The warm lighting highlights her face and the leather strap of the camera. The composition captures a candid, adventurous moment of travel and photography.",
#   "caption": "Smiling woman taking photos from a car window on a scenic road trip."
# }


def validate_image_file(filename: str) -> bool:
    """Check if file has valid image extension"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def process_image_in_background(photo_id: int, filepath: str):
    """
    Background task to:
    - Analyze image metadata (primary method using local file)
    - Fallback to ImgBB upload + ImageDescriber if metadata analysis fails
    - Generate embedding
    - Update database record
    - Index embedding in FAISS
    """
    from cloudzy.database import SessionLocal
    from sqlmodel import select
    import time
    
    try:
        result = None
        
        # --- Verify file exists and is readable ---
        file_path = Path(filepath)
        max_retries = 5
        retry_count = 0
        while not file_path.exists() and retry_count < max_retries:
            print(f"[Background] Waiting for file {filepath} to be written (attempt {retry_count + 1}/{max_retries})...")
            time.sleep(0.5)
            retry_count += 1
        
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found at {filepath} after {max_retries} retries")
        
        # --- Primary method: Analyze metadata from local filepath ---
        try:
            print(f"[Background] Analyzing image metadata locally for photo {photo_id}...")
            analyzer = ImageAnalyzerAgent()
            result = analyzer.analyze_image_metadata(filepath)
            print(f"[Background] Successfully extracted metadata for photo {photo_id}")
        except Exception as metadata_error:
            print(f"[Background] Metadata analysis failed for photo {photo_id}: {metadata_error}")
            print(f"[Background] Falling back to ImgBB upload + ImageDescriber...")
            
            # --- Fallback method: Upload to ImgBB and use ImageDescriber ---
            try:
                uploader = ImgBBUploader(expiration=600)
                image_url = uploader.upload(filepath)
                print(f"[Background] Image {photo_id} uploaded to ImgBB: {image_url}")
                
                describer = ImageDescriber()
                print(f"[Background] Processing image {photo_id} with ImageDescriber...")
                result = describer.describe_image(image_url)
                print(f"[Background] Successfully described image using ImageDescriber")
            except Exception as fallback_error:
                raise Exception(f"Both metadata analysis and ImageDescriber failed - Primary: {str(metadata_error)}, Fallback: {str(fallback_error)}")

        tags = result.get("tags", [])
        caption = result.get("caption", "")
        description = result.get("description", "")

        generator = ImageEmbeddingGenerator()
        embedding = generator.generate_embedding(tags, description, caption)

        # Use a fresh session for background task
        session = SessionLocal()
        try:
            photo = session.exec(select(Photo).where(Photo.id == photo_id)).first()
            if photo:
                photo.caption = caption
                photo.set_tags(tags)
                photo.set_embedding(embedding.tolist())
                session.add(photo)
                session.commit()
                print(f"[Background] Photo {photo_id} updated with embedding")
            else:
                print(f"[Background] Photo {photo_id} not found in database")
        finally:
            session.close()
        
        # Index in FAISS
        search_engine = SearchEngine()
        search_engine.add_embedding(photo_id, embedding)
        print(f"[Background] Photo {photo_id} indexed in FAISS")

    except Exception as e:
        print(f"[Background Task] Error processing image {photo_id}: {e}")
        import traceback
        traceback.print_exc()


@router.post("/upload", response_model=UploadResponse)
async def upload_photo(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    background_tasks: BackgroundTasks = None,
):
    # --- Validate and save file ---
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not validate_image_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    
    saved_filename = save_uploaded_file(content, file.filename)
    filepath = f"uploads/{saved_filename}"

    APP_DOMAIN = os.getenv("APP_DOMAIN")
    image_local_url = f"{APP_DOMAIN}uploads/{saved_filename}"

    # --- Save photo immediately with empty caption/tags ---
    photo = Photo(
        filename=saved_filename,
        filepath=filepath,
        caption="",  # empty for now
    )
    session.add(photo)
    session.commit()
    session.refresh(photo)

    # --- Schedule background task (includes ImgBB upload) ---
    if background_tasks:
        background_tasks.add_task(
            process_image_in_background,
            photo_id=photo.id,
            filepath=filepath
        )

    return UploadResponse(
        id=photo.id,
        filename=saved_filename,
        image_url=image_local_url,
        message=f"Photo uploaded successfully with ID {photo.id}. AI processing is running in the background."
    )