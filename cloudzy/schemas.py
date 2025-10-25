"""Pydantic response schemas"""
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class PhotoResponse(BaseModel):
    """Response model for photo metadata"""
    id: int
    filename: str
    image_url: str
    tags: List[str]
    caption: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class PhotoDetailResponse(PhotoResponse):
    """Detailed photo response with embedding info"""
    embedding: Optional[List[float]] = None



class SearchResult(BaseModel):
    """Search result with similarity score"""
    photo_id: int
    filename: str
    image_url: str
    tags: List[str]
    caption: str
    distance: float  # L2 distance (lower is more similar)
    
    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    """Response for search endpoint"""
    query: str
    results: List[SearchResult]
    total_results: int


class UploadResponse(BaseModel):
    """Response after uploading a photo"""
    id: int
    filename: str
    image_url: str
    # tags: List[str]
    # caption: str
    message: str


class PhotoItem(BaseModel):
    photo_id: int
    filename: str
    image_url: str
    tags: List[str]
    caption: str
    distance: float

class AlbumItem(BaseModel):
    album_summary: str
    album: List[PhotoItem]

AlbumsResponse = List[AlbumItem]


class GenerateImageResponse(BaseModel):
    """Response for generating a similar image"""
    description: str
    generated_image_url: str
    message: str