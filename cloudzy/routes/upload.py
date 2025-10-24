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


    APP_DOMAIN = os.getenv("APP_DOMAIN")

    image_url = f"{APP_DOMAIN}uploads/{saved_filename}"

    try:

        print("image_url is",image_url)
        
        describer = ImageDescriber()
        # result = describer.describe_image("https://userx2000-cloudzy-ai-challenge.hf.space/uploads/img_1_20251024_064435_667.jpg")
        # result = describer.describe_image("https://userx2000-cloudzy-ai-challenge.hf.space/uploads/img_2_20251024_082115_102.jpeg")
        result = describer.describe_image(image_url)
       
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    # Generate AI analysis
    tags = result.get("tags", [])
    caption = result.get("caption", "")
    description = result.get("description", "")

    

    generator = ImageEmbeddingGenerator()
    embedding = generator.generate_embedding(tags, description, caption)

    # np.save("embedding_2.npy", embedding)
    # embedding = np.load("embedding_2.npy") 
    
    # Create photo record
    photo = Photo(
        filename=saved_filename,
        filepath=filepath,
        caption=caption,
    )
    photo.set_tags(tags)
    # photo.set_embedding(embedding.tolist())
    
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
        image_url= image_url,
        tags=tags,
        caption=caption,
        message=f"Photo uploaded successfully with ID {photo.id}"
    )