from transformers import AutoModel, AutoProcessor
from PIL import Image
import requests
import numpy as np
import torch
from io import BytesIO

# Load model and processor directly
model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)

texts = ["Woman taking pictures on a road trip.", "delicious fruits glowing under sunlight"]
# Process and encode text
text_inputs = processor(text=texts, return_tensors="pt", padding=True)
with torch.no_grad():
    text_embeddings = model.get_text_features(**text_inputs)
text_embeddings = text_embeddings.cpu().numpy()
print("Text embeddings shape:", text_embeddings.shape)

image_paths = [
    "/Users/komeilfathi/Documents/hf_deploy_test/cloudzy_ai_challenge/uploads/img_1_20251026_014959_886.jpg",
    "/Users/komeilfathi/Documents/hf_deploy_test/cloudzy_ai_challenge/uploads/img_9_20251024_185602_319.webp"
]
images = []
for path in image_paths:
    try:
        img = Image.open(path).convert("RGB")
        images.append(img)
        print(f"✓ Loaded image from {path}")
    except Exception as e:
        print(f"✗ Failed to load image from {path}: {e}")

# Process and encode images
if images:
    image_inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        image_embeddings = model.get_image_features(**image_inputs)
    image_embeddings = image_embeddings.cpu().numpy()
    print("Image embeddings shape:", image_embeddings.shape)
else:
    print("⚠ No images loaded successfully")
    image_embeddings = np.array([])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if len(image_embeddings) > 0:
    for i, t_emb in enumerate(text_embeddings):
        for j, i_emb in enumerate(image_embeddings):
            sim = cosine_similarity(t_emb, i_emb)
            print(f"Similarity between text {i} and image {j}: {sim:.4f}")
else:
    print("No images to compare similarity with")
