"""SQLModel database models"""
from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
import json


class Photo(SQLModel, table=True):
    """Photo metadata model"""
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(index=True)
    filepath: str  # Full path to stored image
    tags: str = Field(default="[]")  # JSON string of tags
    caption: str = Field(default="")
    description: str = Field(default="")
    embedding: Optional[str] = Field(default=None)  # JSON string of embedding vector
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_tags(self) -> list[str]:
        """Parse tags from JSON string"""
        try:
            return json.loads(self.tags)
        except:
            return []
    
    def set_tags(self, tags: list[str]):
        """Store tags as JSON string"""
        self.tags = json.dumps(tags)
    
    def get_embedding(self) -> Optional[list[float]]:
        """Parse embedding from JSON string"""
        try:
            if self.embedding:
                return json.loads(self.embedding)
        except:
            pass
        return None
    
    def set_embedding(self, embedding: list[float]):
        """Store embedding as JSON string"""
        self.embedding = json.dumps(embedding)