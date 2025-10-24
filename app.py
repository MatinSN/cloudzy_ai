"""FastAPI application entry point"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from cloudzy.database import create_db_and_tables
from cloudzy.routes import upload, photo, search
from cloudzy.search_engine import SearchEngine
import os

# Initialize search engine at startup
search_engine = None
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle - startup and shutdown"""
    # Startup
    print("🚀 Starting Cloudzy AI service...")
    create_db_and_tables()
    
    # Initialize search engine
    global search_engine
    search_engine = SearchEngine()
    stats = search_engine.get_stats()
    print(f"📊 FAISS Index loaded: {stats}")
    print("✅ Application ready!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Cloudzy AI service...")


# Create FastAPI app
app = FastAPI(
    title="Cloudzy AI",
    description="Cloud photo management with AI tagging, captioning, and semantic search",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router)
app.include_router(photo.router)
app.include_router(search.router)

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static file serving
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/", tags=["info"])
async def root():
    """Root endpoint - API info"""
    return {
        "service": "Cloudzy AI",
        "version": "1.0.0",
        "description": "Cloud photo management with AI tagging, captioning, and semantic search",
        "endpoints": {
            "upload": "POST /upload - Upload a photo",
            "get_photo": "GET /photo/{id} - Get photo metadata",
            "list_photos": "GET /photos - List all photos",
            "search": "GET /search?q=... - Semantic search",
            "image_to_image": "POST /search/image-to-image - Similar images",
            "docs": "/docs - Interactive API documentation",
        }
    }


@app.get("/health", tags=["info"])
async def health_check():
    """Health check endpoint"""
    global search_engine
    stats = search_engine.get_stats() if search_engine else {}
    
    return {
        "status": "healthy",
        "service": "Cloudzy AI",
        "search_engine": stats,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )