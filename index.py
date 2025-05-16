import os
import requests
from flask import Flask, request, jsonify
from chromadb.config import Settings
from chromadb import PersistentClient
import feedparser
from flask_cors import CORS

app=Flask(__name__)
CORS(app, resources={r"/api/*": {"origins":"https://frontend-eosin-mu-95.vercel.app/"}})

RSS_FEED='https://feeds.bbci.co.uk/news/rss.xml'
MAX_ARTICLES=50
JINA_API_URL="https://api.jina.ai/v1/embeddings"
JINA_API_KEY="jina_5349f83703b34e73acae61efdcf0ab07rDpSn3-UMwL_VORzHKnz8Z8Kk9XY"
GEMINI_API_KEY="AIzaSyBCxmzOS-kscYYROsPdzgMx_zCeNM7Cpss"

persist_dir = './chroma_persist'
client = PersistentClient(path="./chroma_db")


collection_name='news_articles'
collection = client.get_or_create_collection(name=collection_name)


def fetch_rss_articles():
    feed=feedparser.parse(RSS_FEED)
    return feed.entries[:MAX_ARTICLES]


def get_embedding(text: str):
    print(JINA_API_URL)
    headers = {
        'Authorization': f'Bearer {JINA_API_KEY}',
        'Content-Type': 'application/json'
    }
    try:
        response=requests.post(
            JINA_API_URL,
            json={
                "input": [text],
                "model": "jina-embeddings-v2-base-en"
            },
            headers=headers
        )
        response.raise_for_status()
        data=response.json()
        return data['data'][0]['embedding'] 
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def ingest_articles():
    articles = fetch_rss_articles()

    ids = []
    embeddings = []
    metadatas = []

    for article in articles:
        text = article.get('summary') or article.get('title') or ''
        embedding = get_embedding(text)
        if not embedding:
            print(f"Skipping article (embedding failed): {article.get('title')}")
            continue
        # print("embedded", article.get('title'))

        article_id = article.get('id') or article.get('link') or f"{int(time.time())}-{os.urandom(4).hex()}"
        ids.append(article_id)
        embeddings.append(embedding)
        metadatas.append({
            "title": article.get('title'),
            "url": article.get('link'),
            "snippet": text
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Ingested {len(ids)} articles into Chroma.")


def retrieve_relevant_chunks(query: str, top_k=5):
    query_embedding = get_embedding(query)
    # print(query)
    if not query_embedding:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    # print(results)
    return [meta.get('snippet') or meta.get('title') for meta in results['metadatas'][0]]


def query_gemini(prompt: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Sorry, something went wrong with the AI response."


def build_prompt(context_chunks, user_query):
    return f"""You are a helpful news assistant. Use the context below to answer:

Context:
{chr(10).join(context_chunks)}

User: {user_query}"""


memory_store = {}


@app.route('/api/chat', methods=['POST'])
def chat():
    
    data = request.json
    session_id = data.get('sessionId')
    message = data.get('message')
    # print("data")

    if not session_id or not message:
        return jsonify({"error": "Missing sessionId or message"}), 400
    
    # print(message)

    memory_store.setdefault(session_id, []).append({"role": "user", "message": message})

    chunks = retrieve_relevant_chunks(message)
    # print(chunks)
    prompt = build_prompt(chunks, message)
    # print(prompt)
    answer = query_gemini(prompt)
    # print(answer)

    memory_store[session_id].append({"role": "bot", "message": answer})

    return jsonify({"answer": answer})


@app.route('/api/chat/history/<session_id>', methods=['GET'])
def chat_history(session_id):
    return jsonify(memory_store.get(session_id, []))


@app.route('/api/chat/reset/<session_id>', methods=['POST'])
def reset_session(session_id):
    memory_store.pop(session_id, None)
    return jsonify({"message": "Session reset."})


if __name__ == '__main__':
    print("Starting server and ingesting articles...")
    ingest_articles()
    app.run(host='0.0.0.0', port=5001)
