"""File handling utilities"""
import os
import shutil
from pathlib import Path
from datetime import datetime


UPLOAD_DIR = "uploads"


def ensure_upload_dir():
    """Ensure uploads directory exists"""
    Path(UPLOAD_DIR).mkdir(exist_ok=True)


def save_uploaded_file(file_content: bytes, original_filename: str) -> str:
    """
    Save uploaded file with timestamp to ensure uniqueness.
    
    Args:
        file_content: File bytes
        original_filename: Original filename
    
    Returns:
        Saved filename
    """
    ensure_upload_dir()
    
    # Generate unique filename with timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    name, ext = os.path.splitext(original_filename)
    saved_filename = f"{name}_{timestamp}{ext}"
    
    filepath = os.path.join(UPLOAD_DIR, saved_filename)
    
    # Write file
    with open(filepath, "wb") as f:
        f.write(file_content)
    
    return saved_filename


def get_file_path(filename: str) -> str:
    """Get full path for a saved file"""
    return os.path.join(UPLOAD_DIR, filename)


def file_exists(filename: str) -> bool:
    """Check if a saved file exists"""
    return os.path.exists(get_file_path(filename))


def delete_file(filename: str) -> bool:
    """Delete a saved file"""
    filepath = get_file_path(filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False