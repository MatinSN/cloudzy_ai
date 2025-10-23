# 🧭 Cloudzy AI - Cloud Photo Management Service

A FastAPI-based cloud photo management service with AI tagging, captioning, and semantic search using FAISS.

## 🎯 Features

- **Photo Upload** - Upload images with automatic metadata generation
- **AI Analysis** - Automatic tag and caption generation
- **Semantic Search** - FAISS-powered similarity search on embeddings
- **Image-to-Image Search** - Find similar photos to a reference image
- **RESTful API** - Full REST API with automatic documentation
- **Docker Support** - Production-ready Docker and Docker Compose setup

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **Database**: SQLModel + SQLite (PostgreSQL ready)
- **Search Engine**: FAISS (Fast Approximate Nearest Neighbors)
- **Image Processing**: Pillow
- **ORM**: SQLModel
- **API Documentation**: Swagger/OpenAPI

## 📋 Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional)
- 2GB+ RAM for FAISS index

## ⚙️ Installation

### Local Development

1. **Clone and setup**
```bash
cd image_embedder
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create uploads directory**
```bash
mkdir -p uploads
```

4. **Run the server**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at `http://localhost:8000`

### Docker

```bash
# Build and run
docker compose up --build

# Run in background
docker compose up -d

# View logs
docker compose logs -f cloudzy_api

# Stop
docker compose down
```

## 🚀 API Endpoints

### Upload Photo
```bash
POST /upload
Content-Type: multipart/form-data

# Returns:
{
  "id": 1,
  "filename": "photo_20231023_120000.jpg",
  "tags": ["nature", "landscape", "mountain"],
  "caption": "A beautiful nature photograph",
  "message": "Photo uploaded successfully with ID 1"
}
```

### Get Photo Metadata
```bash
GET /photo/{id}

# Returns:
{
  "id": 1,
  "filename": "photo_20231023_120000.jpg",
  "tags": ["nature", "landscape"],
  "caption": "A beautiful landscape",
  "embedding": [0.123, -0.456, ...],  # 512-dim vector
  "created_at": "2023-10-23T12:00:00"
}
```

### List All Photos
```bash
GET /photos?skip=0&limit=10

# Returns: List of photo objects with pagination
```

### Semantic Search
```bash
GET /search?q=mountain&top_k=5

# Returns:
{
  "query": "mountain",
  "results": [
    {
      "photo_id": 1,
      "filename": "photo_1.jpg",
      "tags": ["nature", "mountain"],
      "caption": "Mountain landscape",
      "distance": 0.123
    },
    ...
  ],
  "total_results": 5
}
```

### Image-to-Image Search
```bash
POST /search/image-to-image?reference_photo_id=1&top_k=5

# Returns similar photos to reference photo 1
```

### Health Check
```bash
GET /health

# Returns service status and FAISS index stats
```

## 📚 API Documentation

**Interactive Docs (Swagger UI)**:
```
http://localhost:8000/docs
```

**Alternative Docs (ReDoc)**:
```
http://localhost:8000/redoc
```

## 🗂️ Project Structure

```
image_embedder/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry point
│   ├── database.py              # SQLModel engine + session
│   ├── models.py                # Photo database model
│   ├── schemas.py               # Pydantic response models
│   ├── ai_utils.py              # AI generation (tags, captions, embeddings)
│   ├── search_engine.py         # FAISS index manager
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── upload.py            # POST /upload endpoint
│   │   ├── photo.py             # GET /photo/:id and /photos endpoints
│   │   └── search.py            # GET /search and image-to-image endpoints
│   │
│   └── utils/
│       ├── __init__.py
│       └── file_utils.py        # File saving and management
│
├── uploads/                     # Stored images (created at runtime)
├── faiss_index.bin              # FAISS index file (created at runtime)
├── photos.db                    # SQLite database (created at runtime)
│
├── requirements.txt             # Python dependencies
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🔄 Development Workflow

### Test Upload
```bash
# Use curl
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:8000/upload

# Or use Python
import requests
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": f}
    )
    print(response.json())
```

### Test Search
```bash
# Query-based search
curl "http://localhost:8000/search?q=tree&top_k=5"

# Image-to-image search
curl -X POST "http://localhost:8000/search/image-to-image?reference_photo_id=1&top_k=5"
```

### View Database
```bash
# Install sqlite3 CLI and view database
sqlite3 photos.db
> .tables
> SELECT * FROM photo;
> .quit
```

## 🧠 AI Features (Placeholder Phase)

Currently, AI functions use placeholder implementations:

- **Tags**: Generated from filename patterns + random selection from common tags
- **Captions**: Template-based generation from tags
- **Embeddings**: Deterministic random vectors (reproducible from filename)

### Upgrade Path (Production)

1. **CLIP Integration** (Recommended)
   - Zero-shot image understanding
   - Excellent for tagging and search
   - ~1-2 sec per image on GPU

2. **BLIP Integration** (Alternative)
   - Visual question answering
   - Better captions
   - ~2-3 sec per image on GPU

3. **Fine-tuned Models**
   - Train on domain-specific data
   - Improved accuracy
   - Higher latency/complexity

## 📊 Performance Considerations

- **FAISS Index**: Supports millions of embeddings
- **Database**: SQLite suitable for 100k+ photos; PostgreSQL for larger scale
- **Embeddings**: 512-dim vectors (adjustable)
- **Search**: <100ms for 100k+ embeddings on CPU

## 🚨 Troubleshooting

### FAISS Installation Issues
```bash
# If faiss-cpu fails, try:
pip install faiss-cpu==1.7.4 --no-cache-dir
```

### SQLite Lock Error
```bash
# Restart the application or remove locked database
rm photos.db
```

### Docker Build Issues
```bash
# Rebuild without cache
docker compose build --no-cache
```

## 🔐 Security Notes

- ⚠️ Currently no authentication - add for production
- ⚠️ CORS allows all origins - restrict for production
- ⚠️ File upload validation needed - add size limits
- ⚠️ Use PostgreSQL + proper secrets management for production

## 📝 Next Steps

1. ✅ Core backend working
2. ⬜ Add authentication (JWT)
3. ⬜ Implement real AI models (CLIP/BLIP)
4. ⬜ Add background job processing (Celery)
5. ⬜ Frontend dashboard
6. ⬜ Production deployment (Railway/AWS)

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please test thoroughly before submitting.

---

**Questions?** Check the interactive docs at `/docs` or review the code comments.