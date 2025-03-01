import os
import re
import requests
import logging
import textwrap
import json
import uuid
import redis
from flask import Flask, request, jsonify, send_from_directory
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

UPLOAD_FOLDER = "uploads"
EPHEMERAL_FOLDER = "ephemeral_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EPHEMERAL_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["EPHEMERAL_FOLDER"] = EPHEMERAL_FOLDER

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
# xAI CHAT CLIENT WRAPPER
# ==============================
class XAIChatClient:
    def __init__(self, api_key, base_url="https://api.x.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-2-latest",
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.2,
            "stream": False
        }
        try:
            logger.info(f"Sending xAI request with payload: {json.dumps(payload, indent=2)}")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            logger.info(f"xAI response: {response.text}")
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            error_msg = f"xAI API error: {str(e)}"
            if hasattr(e.response, 'text'):
                error_msg += f" - Response: {e.response.text}"
            logger.error(error_msg)
            return f"Error: Could not process your request due to an issue with the AI service ({str(e)}). Please try again later."

xai_chat_client = XAIChatClient(api_key=XAI_API_KEY)

# ==============================
# CRAFTSMAN RECOMMENDATIONS SETUP
# ==============================
def load_craftsmen():
    try:
        with open('craftsmen.json', 'r') as f:
            data = json.load(f)
            logger.info(f"Loaded craftsmen.json: {data}")
            return data
    except Exception as e:
        logger.error(f"Failed to load craftsmen.json: {e}")
        return []

CRAFTSMEN = load_craftsmen()
CRAFTSMEN_JSON = json.dumps(CRAFTSMEN)

# ==============================
# HELPER FUNCTIONS
# ==============================
def generate_embeddings(text: str):
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

def chunk_text(text: str, chunk_size: int = 1500) -> list:
    return textwrap.wrap(text, width=chunk_size)

def store_embeddings(user_id: str, text: str):
    if not text.strip():
        return
    chunks = chunk_text(text)
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = generate_embeddings(chunk)
        if embedding:
            vector_id = f"{user_id}-{hash(chunk)}-{i}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {"user_id": user_id, "text": chunk}
            })
    if vectors:
        pinecone_index.upsert(vectors=vectors, namespace=user_id)

def retrieve_relevant_knowledge(user_id: str, query: str, top_k: int = 3) -> str:
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

def extract_text_from_docx(docx_path: str):
    try:
        doc = Document(docx_path)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        return None

def extract_text_from_image(image_path: str):
    try:
        img = Image.open(image_path).convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        return pytesseract.image_to_string(img, lang="eng").strip()
    except Exception as e:
        logger.error(f"Image OCR failed: {e}")
        return None

def extract_text_from_pdf(pdf_path: str):
    text = ""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        if text.strip():
            return text.strip()
        logger.warning("PyMuPDF empty text. Trying PyPDF2.")
    except Exception as e:
        logger.error(f"PyMuPDF failed: {e}")
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        logger.error(f"PyPDF2 also failed: {e}")
    return None

# ==============================
# FLASK ROUTES
# ==============================
@app.route("/")
def index():
    return "Welcome to Dokie!"

@app.route("/start_conversation", methods=["POST"])
def start_conversation():
    data = request.get_json()
    user_id = data.get("userID")
    if not user_id:
        return jsonify({"error": "userID required"}), 400

    claim_id = str(uuid.uuid4())
    
    system_prompt = (
        "You are Dokie, an empathetic and highly logical AI assistant helping users with insurance claims. "
        "Respond in the same language as the user's query, using natural, conversational, and empathetic phrasing to make the user feel understood and guided. "
        "You have a permanent document library in Pinecone for coverage details and ephemeral memory for claim evidence (photos, etc.). "
        "Your primary goal is to help users claim compensation by analyzing their document library and providing the best approach for a successful claim. "
        "Prioritize the user's immediate query or concern, always responding directly and logically. "
        "If the query is vague or ambiguous (e.g., 'I noticed it now'), interpret 'now' as 'today' unless the user specifies otherwise, and ask clarifying, "
        "open-ended questions (e.g., 'What did you notice today? Can you describe the issue in more detail?') before offering advice or gathering details. "
        "Avoid making assumptions unless the user explicitly states the problem (e.g., don’t assume a leak unless mentioned). "
        "If the query implies a problem (e.g., 'small leak', 'roof fallen in'), offer practical, one-time advice to minimize risk or damage (e.g., 'Try to turn off the water supply'), "
        "then proceed logically with claim details (date, location, description) one question at a time, building on the user’s responses without redundancy or repetitive phrases like "
        "'För att kunna hjälpa dig på bästa sätt med din försäkringsclaim, behöver jag lite mer information.' Instead, use varied, natural phrases such as "
        "'Let’s get the details to help with your claim,' or 'I’m here to assist—can you share more?' "
        "For severe issues (e.g., 'roof fallen in', 'flooding'), suggest a craftsman type (e.g., 'roofing', 'plumbing', 'general') immediately "
        "with up to 3 recommendations from this list: " + CRAFTSMEN_JSON + ", formatted as '- Name (Rating: X/5, Contact: email)' "
        "in the user's language, using a natural statement (not a question). For minor issues (e.g., 'small leak'), after offering risk mitigation advice and gathering sufficient details "
        "(date, location, description), ask explicitly, 'Would you like a recommendation for a craftsman (e.g., plumber, roofer, general contractor) to help with this issue?' "
        "Provide recommendations only if the user confirms they need one, using the list provided. Craftsman suggestions are secondary to claim assistance. "
        "Track the conversation history in Redis to avoid repeating advice, questions, or phrasing unnecessarily. Use logical, context-aware reasoning to ensure each response feels natural, "
        "empathetic, and progresses the conversation smoothly. Your ultimate goal: maximize the user's claim success with a smart, intuitive, and human-like conversation flow."
    )

    redis_client.hset(f"claim:{claim_id}", mapping={
        "user_id": user_id,
        "conversation": json.dumps([{"role": "system", "content": system_prompt}]),
        "claim_details": json.dumps({"date": None, "location": None, "description": None, "ephemeral_docs": []})
    })
    redis_client.sadd(f"user:{user_id}:claims", claim_id)
    
    logger.info(f"Started new conversation with claimID: {claim_id} for userID: {user_id}")
    return jsonify({"message": "Conversation started.", "claimID": claim_id}), 200

@app.route("/get_user_claims", methods=["POST"])
def get_user_claims():
    data = request.get_json()
    user_id = data.get("userID")
    if not user_id:
        return jsonify({"error": "userID required"}), 400

    claim_ids = redis_client.smembers(f"user:{user_id}:claims") or []
    return jsonify({"claimIDs": list(claim_ids)}), 200

@app.route("/userclaimidtest")
def user_claimid_test():
    return send_from_directory('static', 'userclaimidtest.html')

@app.route("/upload_library", methods=["POST"])
def upload_library():
    if "file" not in request.files or "userID" not in request.form:
        return jsonify({"error": "file and userID required"}), 400

    file_obj = request.files["file"]
    user_id = request.form["userID"]

    file_path = os.path.join(UPLOAD_FOLDER, file_obj.filename)
    file_obj.save(file_path)

    ext = os.path.splitext(file_obj.filename)[1].lower()
    if ext == ".docx":
        text = extract_text_from_docx(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        text = extract_text_from_image(file_path)
    elif ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    else:
        os.remove(file_path)
        return jsonify({"error": "Unsupported file type"}), 400

    os.remove(file_path)
    if not text or not text.strip():
        return jsonify({"error": "Could not extract text"}), 400

    store_embeddings(user_id, text)
    return jsonify({"message": "Document embedded in your permanent library."}), 200

@app.route("/upload_ephemeral", methods=["POST"])
def upload_ephemeral():
    if "file" not in request.files or "claimID" not in request.form or "userID" not in request.form:
        return jsonify({"error": "file, claimID, and userID required"}), 400

    file_obj = request.files["file"]
    claim_id = request.form["claimID"]
    user_id = request.form["userID"]

    if not redis_client.exists(f"claim:{claim_id}"):
        return jsonify({"error": "Invalid claimID. Start a conversation first."}), 400
    
    stored_user_id = redis_client.hget(f"claim:{claim_id}", "user_id")
    if stored_user_id != user_id:
        return jsonify({"error": "Unauthorized: claimID does not belong to this user"}), 403

    ephemeral_path = os.path.join(EPHEMERAL_FOLDER, file_obj.filename)
    file_obj.save(ephemeral_path)

    ext = os.path.splitext(file_obj.filename)[1].lower()
    extracted_text = None
    if ext == ".docx":
        extracted_text = extract_text_from_docx(ephemeral_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        extracted_text = extract_text_from_image(ephemeral_path)
    elif ext == ".pdf":
        extracted_text = extract_text_from_pdf(ephemeral_path)

    claim_details = json.loads(redis_client.hget(f"claim:{claim_id}", "claim_details"))
    claim_details["ephemeral_docs"].append({
        "filename": file_obj.filename,
        "file_path": ephemeral_path,
        "extracted_text": extracted_text or ""
    })
    redis_client.hset(f"claim:{claim_id}", "claim_details", json.dumps(claim_details))

    return jsonify({"message": "Ephemeral file stored for this claim."}), 200

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    logger.info(f"Received /search request with data: {json.dumps(data)}")  # Debug log
    claim_id = data.get("claimID")
    query = data.get("query")
    user_id = data.get("userID")

    if not claim_id or not query or not user_id:
        logger.error(f"Missing required fields in /search request: claimID={claim_id}, query={query}, userID={user_id}")
        return jsonify({"error": "claimID, query, and userID required"}), 400

    if not redis_client.exists(f"claim:{claim_id}"):
        logger.error(f"Invalid claimID: {claim_id}")
        return jsonify({"error": "Invalid claimID. Start a conversation first."}), 400

    stored_user_id = redis_client.hget(f"claim:{claim_id}", "user_id")
    if stored_user_id != user_id:
        logger.error(f"Unauthorized access attempt: claimID={claim_id}, userID={user_id}, stored_user_id={stored_user_id}")
        return jsonify({"error": "Unauthorized: claimID does not belong to this user"}), 403

    conversation = json.loads(redis_client.hget(f"claim:{claim_id}", "conversation"))
    claim_details = json.loads(redis_client.hget(f"claim:{claim_id}", "claim_details"))

    details_summary = (
        f"Current claim details: Date: {claim_details['date']}, "
        f"Location: {claim_details['location']}, "
        f"Description: {claim_details['description']}"
    )
    conversation.append({"role": "system", "content": details_summary})
    conversation.append({"role": "user", "content": query})

    knowledge = retrieve_relevant_knowledge(user_id, query)
    if knowledge:
        conversation.append({
            "role": "system",
            "content": f"Relevant knowledge from your permanent documents:\n{knowledge}"
        })

    answer = xai_chat_client.chat(conversation)

    if "datum" in query.lower() or "date" in query.lower() or "när" in query.lower() or "when" in query.lower():
        claim_details["date"] = query
    elif "var" in query.lower() or "where" in query.lower():
        claim_details["location"] = query
    elif "beskriv" in query.lower() or "describe" in query.lower() or "vad" in query.lower() or "what" in query.lower():
        claim_details["description"] = query

    redis_client.hset(f"claim:{claim_id}", "conversation", json.dumps(conversation))
    redis_client.hset(f"claim:{claim_id}", "claim_details", json.dumps(claim_details))
    logger.info(f"Updated claim details for claimID {claim_id}: {claim_details}")

    return jsonify({"answer": answer, "claimID": claim_id, "conversation": json.dumps(conversation)}), 200

@app.route("/finalize_claim", methods=["POST"])
def finalize_claim():
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

    claim_details = json.loads(redis_client.hget(f"claim:{claim_id}", "claim_details"))
    attached_files = [doc["filename"] for doc in claim_details["ephemeral_docs"]]
    message = (
        "Claim finalized. The following ephemeral documents have been attached:\n"
        f"{', '.join(attached_files)}\n\n"
        "Permanent library references are included in the claim text. Good luck!"
    )

    return jsonify({"message": message, "claimID": claim_id}), 200

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
