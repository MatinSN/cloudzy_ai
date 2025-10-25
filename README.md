---
title: Cloudzy AI Photo Management
emoji: 📸
colorFrom: blue
colorTo: purple
sdk: docker
pinned: true
---
Cloudzy AI Photo Management
Cloudzy is a cloud-based photo management service with AI-powered analysis and semantic search. It allows users to upload photos, automatically analyzes them using AI models, and provides intelligent features such as semantic search and creative AI-generated insights.

🚀 Objective
Build a photo management service with the following requirements:

Photo Upload and Storage

POST /upload: Receive a photo, store it, and generate a unique ID.

GET /photo/:id: Return metadata including upload date, file name, and AI-generated tags and captions.

AI-Powered Analysis

Each uploaded photo must use a vision model (local or API-based) to produce:

Tags (at least 5 relevant)

Caption (short descriptive sentence)

Embedding vector (for semantic search)

Semantic Search

GET /search?q=... : Return photos ranked by semantic similarity using embeddings.

Mandatory AI-Powered Smart Feature

Include at least one creative AI-powered feature, e.g.:

Album generation

Daily summaries

Emotion/facial recognition

AI Usage Report

Document where and how AI was used

Provide prompts and model inputs

Explain how model outputs were refined

📂 Example Workflow
Upload a photo via /upload.

The system generates:

Tags and captions using AI

Embeddings for semantic search

Retrieve photo metadata via /photo/:id.

Search for photos with /search?q=<query> to find semantically similar images.

Enjoy AI-powered smart features like automatic album generation or summaries.

🤖 AI Usage
This project integrates AI in multiple stages:

Tagging and Captioning: Vision model automatically labels photos and generates descriptive captions.

Embedding Generation: Produces vector representations for semantic search.

Creative AI Features: Enhances user experience, e.g., grouping similar photos into albums or summarizing daily uploads.

📌 Notes
The service is designed for mandatory AI usage—all main features leverage AI models.

This README uses Hugging Face Spaces front matter to define metadata, including color, emoji, and SDK type.

