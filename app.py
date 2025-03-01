import os
import re
import requests
import logging
import textwrap
import json
import uuid
import redis
from flask import Flask, request, jsonify
from docx import Document
from PIL import Image, ImageEnhance
import pytesseract
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pinecone import Pinecone, ServerlessSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# ENVIRONMENT VARIABLES
# ==============================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")
REDIS_URL = os.environ.get("REDIS_URL")

if not all([OPENAI_API_KEY, PINECONE_API_KEY, XAI_API_KEY, REDIS_URL]):
    logger.error("Missing required environment variables")
    raise EnvironmentError("Set OPENAI_API_KEY, PINECONE_API_KEY, XAI_API_KEY, and REDIS_URL")

# ==============================
# REDIS SETUP
# ==============================
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# ==============================
# FLASK SETUP
# ==============================
app = Flask(__name__)
CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["1000 per day", "100 per hour"],
    storage_uri=REDIS_URL
)

# ==============================
# PINECONE INITIALIZATION
# ==============================
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "dokie-document-embeddings"

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
pinecone_index = pc.Index(INDEX_NAME)

# ==============================
# HELPER FUNCTIONS
# ==============================
def generate_embeddings(text: str):
    """Generates an embedding vector using OpenAI API"""
    try:
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"input": text, "model": "text-embedding-ada-002"}
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None

def retrieve_relevant_knowledge(user_id: str, query: str, top_k: int = 3) -> str:
    """Fetches the most relevant document information based on user query"""
    embedding = generate_embeddings(query)
    if not embedding:
        return ""

    result = pinecone_index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=user_id
    )

    if not result.get("matches"):
        return ""

    matches = [m["metadata"]["text"] for m in result["matches"]]
    return "\n".join(matches)

# ==============================
# CHAT HISTORY API
# ==============================
@app.route("/get_conversation", methods=["POST"])
def get_conversation():
    """Retrieve full conversation history for a claim"""
    data = request.get_json()
    claim_id = data.get("claimID")
    user_id = data.get("userID")

    if not claim_id or not user_id:
        return jsonify({"error": "claimID and userID required"}), 400

    if not redis_client.exists(f"claim:{claim_id}"):
        return jsonify({"error": "Invalid claimID"}), 400

    stored_user_id = redis_client.hget(f"claim:{claim_id}", "user_id")
    if stored_user_id != user_id:
        return jsonify({"error": "Unauthorized: claimID does not belong to this user"}), 403

    conversation = redis_client.hget(f"claim:{claim_id}", "conversation")
    return jsonify({"conversation": json.loads(conversation)})

# ==============================
# CHAT SEARCH API (WITH HISTORY)
# ==============================
@app.route("/search", methods=["POST"])
def search():
    """Handles user queries and references full chat history"""
    data = request.get_json()
    claim_id = data.get("claimID")
    query = data.get("query")
    user_id = data.get("userID")

    if not claim_id or not query or not user_id:
        return jsonify({"error": "claimID, query, and userID required"}), 400

    if not redis_client.exists(f"claim:{claim_id}"):
        return jsonify({"error": "Invalid claimID. Start a conversation first."}), 400

    stored_user_id = redis_client.hget(f"claim:{claim_id}", "user_id")
    if stored_user_id != user_id:
        return jsonify({"error": "Unauthorized: claimID does not belong to this user"}), 403

    # Load full conversation history
    conversation = json.loads(redis_client.hget(f"claim:{claim_id}", "conversation"))

    # Append user's new query
    conversation.append({"role": "user", "content": query})

    # Retrieve relevant knowledge
    knowledge = retrieve_relevant_knowledge(user_id, query)
    if knowledge:
        conversation.append({"role": "system", "content": f"Relevant knowledge from your permanent documents:\n{knowledge}"})

    # Generate AI response
    answer = "This is a placeholder response from the AI model."
    
    # Append AI response
    conversation.append({"role": "assistant", "content": answer})

    # Save updated conversation
    redis_client.hset(f"claim:{claim_id}", "conversation", json.dumps(conversation))

    return jsonify({"answer": answer, "claimID": claim_id, "conversation": conversation})

# ==============================
# START A NEW CONVERSATION
# ==============================
@app.route("/start_conversation", methods=["POST"])
def start_conversation():
    """Initialize a new conversation for a claim"""
    data = request.get_json()
    user_id = data.get("userID")
    if not user_id:
        return jsonify({"error": "userID required"}), 400

    claim_id = str(uuid.uuid4())

    system_prompt = (
        "You are Dokie, an AI assistant specializing in insurance claims. "
        "Your goal is to help users navigate their claims, analyze documents, and maximize compensation. "
        "You track full conversation history to avoid repeating advice and progress logically."
    )

    redis_client.hset(f"claim:{claim_id}", mapping={
        "user_id": user_id,
        "conversation": json.dumps([{"role": "system", "content": system_prompt}])
    })
    redis_client.sadd(f"user:{user_id}:claims", claim_id)

    return jsonify({"message": "Conversation started.", "claimID": claim_id})

# ==============================
# FINALIZE A CLAIM
# ==============================
@app.route("/finalize_claim", methods=["POST"])
def finalize_claim():
    """Mark a claim as complete and summarize the conversation"""
    data = request.get_json()
    claim_id = data.get("claimID")
    user_id = data.get("userID")

    if not claim_id or not user_id:
        return jsonify({"error": "claimID and userID required"}), 400

    if not redis_client.exists(f"claim:{claim_id}"):
        return jsonify({"error": "Invalid claimID"}), 400

    conversation = json.loads(redis_client.hget(f"claim:{claim_id}", "conversation"))

    return jsonify({
        "message": "Claim finalized. Here is your conversation summary.",
        "conversation_summary": conversation
    })

# ==============================
# FLASK APP START
# ==============================
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
