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
    
    # Refined system prompt emphasizing step-by-step questioning and confirmation of known details.
    system_prompt = (
        "You are Dokie, a highly empathetic and logical AI assistant who acts as both an advisor and a case manager for insurance claims. "
        "Your role is to help clients navigate the insurance claims process from start to finish, ensuring they receive fair compensation. "
        "Ask one question at a time so the client can answer easily. When you already have information from the client's document library—such as their address—do not ask for it again. Instead, confirm it by saying something like, 'I have your address as [address]. Is that correct?' "
        "Listen carefully to the client's situation and gather all necessary details about the damage, loss, or issue they face. "
        "Review their insurance policy to determine what is covered, and prepare a strong, comprehensive claim with all required documentation. "
        "Communicate with the insurance company, respond to their requests, and negotiate on behalf of the client to maximize compensation. "
        "If repairs are needed—for example, in cases of home damage or car repairs—help the client find a reliable and qualified craftsman, ensuring repair estimates align with insurance coverage. "
        "Respond in the same language as the client (e.g., if they write in Swedish, reply in Swedish using natural expressions like 'Jag förstår, det låter jobbigt—kan du berätta mer?' or 'Låt oss lösa detta tillsammans—vad är nästa steg?'). "
        "Review the *entire conversation history* stored in Redis for each claimID to avoid repetition. Progress logically based on what has already been discussed, ensuring that each question builds on previous answers without redundancy. "
        "If a query is vague (e.g., 'I noticed it now'), interpret 'now' as 'today' unless specified otherwise, and ask clarifying, open-ended questions before offering advice. "
        "Act as the client's trusted advocate throughout the insurance claim process."
    )

    redis_client.hset(f"claim:{claim_id}", mapping={
        "user_id": user_id,
        "conversation": json.dumps([{"role": "system", "content": system_prompt}]),
        "claim_details": json.dumps({"date": None, "location": None, "description": None, "ephemeral_docs": []})
    })
    redis_client.sadd(f"user:{user_id}:claims", claim_id)
    
    logger.info(f"Started new conversation with claimID: {claim_id} for userID: {user_id}")
    return jsonify({"message": "Conversation started.", "claimID": claim_id})

@app.route("/get_user_claims", methods=["POST"])
def get_user_claims():
    data = request.get_json()
    user_id = data.get("userID")
    if not user_id:
        return jsonify({"error": "userID required"}), 400

    claim_ids = redis_client.smembers(f"user:{user_id}:claims") or []
    return jsonify({"claimIDs": list(claim_ids)})

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
    return jsonify({"message": "Document embedded in your permanent library."})

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

    return jsonify({"message": "Ephemeral file stored for this claim."})

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    logger.info(f"Received /search request with data: {json.dumps(data)}")
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

    # Load full conversation history
    conversation = json.loads(redis_client.hget(f"claim:{claim_id}", "conversation"))
    claim_details = json.loads(redis_client.hget(f"claim:{claim_id}", "claim_details"))

    # Append a summary of current claim details for context
    details_summary = (
        f"Current claim details: Date: {claim_details['date']}, "
        f"Location: {claim_details['location']}, "
        f"Description: {claim_details['description']}"
    )
    conversation.append({"role": "system", "content": details_summary})
    conversation.append({"role": "user", "content": query})

    # Retrieve relevant knowledge from Pinecone
    knowledge = retrieve_relevant_knowledge(user_id, query)
    if knowledge:
        conversation.append({"role": "system", "content": f"Relevant knowledge from your permanent documents:\n{knowledge}"})

    # Generate response using the full conversation history
    answer = xai_chat_client.chat(conversation)

    # Update claim details based on query, but avoid overwriting details already set.
    if ("datum" in query.lower() or "date" in query.lower() or "när" in query.lower() or "when" in query.lower()) and not claim_details["date"]:
        claim_details["date"] = query
    elif ("var" in query.lower() or "where" in query.lower()):
        # Only update location if not already set; otherwise, expect Dokie to confirm the existing address.
        if not claim_details["location"]:
            claim_details["location"] = query
    elif ("beskriv" in query.lower() or "describe" in query.lower() or "vad" in query.lower() or "what" in query.lower()) and not claim_details["description"]:
        claim_details["description"] = query

    # Append Dokie's response to conversation history
    conversation.append({"role": "assistant", "content": answer})

    # Save updated conversation and claim details to Redis
    redis_client.hset(f"claim:{claim_id}", "conversation", json.dumps(conversation))
    redis_client.hset(f"claim:{claim_id}", "claim_details", json.dumps(claim_details))
    logger.info(f"Updated claim details for claimID {claim_id}: {claim_details}")

    return jsonify({"answer": answer, "claimID": claim_id, "conversation": json.dumps(conversation)})

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

    return jsonify({"message": message, "claimID": claim_id})

# ==============================
# NEW ENDPOINT: Retrieve Full Conversation History
# ==============================
@app.route("/get_conversation", methods=["POST"])
def get_conversation():
    data = request.get_json()
    claim_id = data.get("claimID")
    user_id = data.get("userID")
    if not claim_id or not user_id:
        return jsonify({"error": "claimID and userID required"}), 400
    if not redis_client.exists(f"claim:{claim_id}"):
        return jsonify({"error": "Invalid claimID."}), 400
    stored_user_id = redis_client.hget(f"claim:{claim_id}", "user_id")
    if stored_user_id != user_id:
        return jsonify({"error": "Unauthorized: claimID does not belong to this user"}), 403
    conversation = json.loads(redis_client.hget(f"claim:{claim_id}", "conversation"))
    return jsonify({"conversation": conversation})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
