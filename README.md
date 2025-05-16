# 🔧 RAG News Chatbot – Backend

This is the **backend** for a full-stack RAG-powered chatbot, built using **FastAPI**. It answers user queries over a news corpus using a Retrieval-Augmented Generation pipeline.

---

## 🛠 Tech Stack

- **FastAPI** – High-performance Python web framework
- **Redis** – In-memory cache for chat history
- **ChromaDB** – Vector store for document embeddings
- **Jina Embeddings** – Free open-source text embeddings
- **Gemini API** – Google’s LLM for final response generation

---

## 🔍 Features

- RESTful chat API
- Session handling via `uuid` and Redis
- Ingest & embed ~50 news articles
- Retrieve top-k context using vector search
- Generate response using Gemini API
- Session reset & chat history APIs

---

## 📦 Setup Instructions

### 1. Clone the repo & navigate

```bash
cd backend
```
### 2. Create virtual environment
```bash

python -m venv venv

source venv/bin/activate
```
### 3. Install dependencies
```bash

pip install -r requirements.txt
```
### 4. Start server
```bash

python3 index.py
```
## 🧠 RAG Flow
1. Ingest 40 news data

2. Embed text chunks using Jina

3. Store vectors in ChromaDB

4. retrieve top-k chunks

5. Send to Gemini API for generation

## 📂 API Endpoints

| Method | Endpoint                     | Description                  |
|--------|------------------------------|------------------------------|
| GET    | `/api/chat/history/{id}`     | Fetch chat history by ID     |
| POST   | `/api/chat`                  | Send user message            |
| POST   | `/api/chat/reset/{id}`       | Reset session & history      |


