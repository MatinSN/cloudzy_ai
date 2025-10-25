"""Generate endpoint for creating similar images"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import os

from cloudzy.agents.image_analyzer_2 import ImageAnalyzerAgent
from cloudzy.inference_models.text_to_image import TextToImageGenerator
from cloudzy.utils.file_utils import save_uploaded_file
from cloudzy.schemas import GenerateImageResponse

router = APIRouter(tags=["generate"])

# Allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def validate_image_file(filename: str) -> bool:
    """Check if file has valid image extension"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@router.post("/generate-similar-image", response_model=GenerateImageResponse)
async def generate_similar_image(
    file: UploadFile = File(...),
):
    """
    Generate a similar image from an input image.
    
    This endpoint:
    1. Takes an image as input
    2. Analyzes the image to get a description using ImageAnalyzerAgent
    3. Uses the description to generate a new image via TextToImageGenerator
    4. Returns the URL of the generated image
    
    Args:
        file: The input image file
        
    Returns:
        GenerateImageResponse with the generated image URL and description
    """
    # --- Validate file ---
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
    
    # --- Save uploaded file temporarily ---
    try:
        saved_filename = save_uploaded_file(content, file.filename)
        filepath = Path(__file__).parent.parent.parent / "uploads" / saved_filename
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # --- Step 1: Analyze image and get description ---
    try:
        analyzer = ImageAnalyzerAgent()
        description = analyzer.retrieve_similar_images(filepath)
        print(f"Generated description: {description}")
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to analyze image: {str(e)}"
        )
    
    # --- Step 2: Generate image from description ---
    try:
        generator = TextToImageGenerator()
        generated_image_url = generator.generate(description)
        print(f"Generated image URL: {generated_image_url}")
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate image: {str(e)}"
        )
    
    return GenerateImageResponse(
        description=description,
        generated_image_url=generated_image_url,
        message="Similar image generated successfully"
    )