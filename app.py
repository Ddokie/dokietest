import os
import re
import requests
import logging
import textwrap
import json
import uuid
import redis
import threading
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
EXTERNAL_SERVICE_URL = os.environ.get("EXTERNAL_SERVICE_URL")  # For external processing

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
            return f"Error: Could not process your request due to an issue with the AI service. Please try again later."

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

# Updated to accept an optional category parameter.
def store_embeddings(user_id: str, text: str, category: str = None):
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
                "metadata": {"user_id": user_id, "text": chunk, "category": category}
            })
    if vectors:
        pinecone_index.upsert(vectors=vectors, namespace=user_id)

# Updated to use a filter if a category is provided.
def retrieve_relevant_knowledge(user_id: str, query: str, top_k: int = 3, category: str = None) -> str:
    embedding = generate_embeddings(query)
    if not embedding:
        return ""
    filter = {"category": category} if category else None
    result = pinecone_index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=user_id,
        filter=filter
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

def generate_detailed_summary(claim_id: str, conversation: list, claim_details: dict) -> str:
    summary_lines = []
    summary_lines.append(f"Detailed Claim Report for Claim ID: {claim_id}")
    summary_lines.append("-" * 60)
    summary_lines.append("Policyholder and Claim Details:")
    summary_lines.append(f"  Full Name: {claim_details.get('full_name')}")
    summary_lines.append(f"  Policy Number: {claim_details.get('policy_number')}")
    summary_lines.append(f"  Phone: {claim_details.get('phone')}")
    summary_lines.append(f"  Email: {claim_details.get('email')}")
    summary_lines.append(f"  Claim Type: {claim_details.get('claim_type')}")
    summary_lines.append(f"  Date & Time of Incident: {claim_details.get('date_time')}")
    summary_lines.append(f"  Location: {claim_details.get('location')}")
    summary_lines.append(f"  Incident Description: {claim_details.get('incident_description')}")
    summary_lines.append(f"  Policy Category: {claim_details.get('category')}")
    summary_lines.append("")
    
    ephemeral_docs = claim_details.get("ephemeral_docs", [])
    if ephemeral_docs:
        summary_lines.append("Uploaded Documents:")
        for doc in ephemeral_docs:
            summary_lines.append(f"  - Filename: {doc.get('filename')}")
            extracted = doc.get("extracted_text")
            if extracted:
                summary_lines.append(f"    Extracted Text: {extracted[:200]}{'...' if len(extracted) > 200 else ''}")
        summary_lines.append("")
    
    summary_lines.append("Full Conversation History:")
    for msg in conversation:
        role = msg.get("role")
        content = msg.get("content")
        summary_lines.append(f"  [{role.upper()}] {content}")
    summary_lines.append("")
    summary_lines.append("End of Report")
    
    return "\n".join(summary_lines)

def send_summary_to_external(summary: str, claim_id: str):
    if not EXTERNAL_SERVICE_URL:
        logger.info("No external service URL provided. Skipping summary send.")
        return
    payload = {"claimID": claim_id, "summary": summary}
    try:
        response = requests.post(EXTERNAL_SERVICE_URL, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"Detailed summary sent to external service for claimID {claim_id}")
    except Exception as e:
        logger.error(f"Failed to send detailed summary to external service: {e}")

def is_claim_complete(claim_details: dict) -> bool:
    required_fields = ["full_name", "policy_number", "phone", "email",
                       "claim_type", "date_time", "location", "incident_description"]
    for field in required_fields:
        if not claim_details.get(field):
            return False
    if not claim_details.get("ephemeral_docs"):
        return False
    return True

def is_negative_response(query: str) -> bool:
    negative_responses = {"no", "nope", "nothing", "none", "that's all", "no thanks", "no thank you"}
    return query.strip().lower() in negative_responses

# NEW HELPER: Update missing fields from the document library (via Pinecone) using the claim's category.
def update_claim_details_from_documents(user_id: str, claim_details: dict) -> dict:
    category = claim_details.get("category")
    field_queries = {
        "full_name": "full name",
        "policy_number": "policy number",
        "phone": "phone number",
        "email": "email address",
        "claim_type": "type of claim",
        "date_time": "date and time of incident",
        "location": "location of incident",
        "incident_description": "brief description of incident"
    }
    for field, query in field_queries.items():
        if not claim_details.get(field):
            result = retrieve_relevant_knowledge(user_id, query, category=category)
            if result:
                claim_details[field] = result.splitlines()[0].strip()
                logger.info(f"Updated field '{field}' from documents: {claim_details[field]}")
    return claim_details

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
    
    # English system prompt instructing Dokie to confirm already known data.
    system_prompt = (
        "You are Dokie, a highly empathetic and logical AI assistant who helps policyholders with their insurance claims. "
        "Your task is to guide the customer through the entire process and ensure they receive proper compensation. "
        "Ask one question at a time. If you already have information from the customer's document library (e.g., full name, address, etc.), "
        "confirm that information instead of asking for it again. The information to be gathered includes:\n"
        "  1. Policyholder Information: Full Name, Policy Number, Phone, Email\n"
        "  2. Claim Details: Type of Claim (Property, Vehicle, Medical, Other), Date & Time of Incident, Location, Brief description of the incident\n"
        "  3. Required Documents: Photos/Videos of damage, Invoices/Receipts, and Police/Incident report (if applicable).\n"
        "Once all necessary information is collected, ask if there is any additional information. If the customer replies 'No', "
        "send a detailed report to the external service."
    )
    
    initial_details = {
        "full_name": None,
        "policy_number": None,
        "phone": None,
        "email": None,
        "claim_type": None,
        "date_time": None,
        "location": None,
        "incident_description": None,
        "category": None,  # New field to indicate the insurance policy category.
        "ephemeral_docs": [],
        "summary_sent": False
    }
    
    redis_client.hset(f"claim:{claim_id}", mapping={
        "user_id": user_id,
        "conversation": json.dumps([{"role": "system", "content": system_prompt}]),
        "claim_details": json.dumps(initial_details)
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

# ------------------
# Updated upload endpoints now accept a "category" field.
# ------------------
@app.route("/upload_library", methods=["POST"])
def upload_library():
    if "file" not in request.files or "userID" not in request.form:
        return jsonify({"error": "file and userID required"}), 400
    file_obj = request.files["file"]
    user_id = request.form["userID"]
    category = request.form.get("category")  # New: category for this document
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
    store_embeddings(user_id, text, category=category)
    return jsonify({"message": "Document embedded in your permanent library."})

@app.route("/upload_ephemeral", methods=["POST"])
def upload_ephemeral():
    if "file" not in request.files or "claimID" not in request.form or "userID" not in request.form:
        return jsonify({"error": "file, claimID, and userID required"}), 400
    file_obj = request.files["file"]
    claim_id = request.form["claimID"]
    user_id = request.form["userID"]
    category = request.form.get("category")  # Optionally include category
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
        "extracted_text": extracted_text or "",
        "category": category  # Save category for this document if provided
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
        return jsonify({"error": "claimID, query, and userID required"}), 400
    if not redis_client.exists(f"claim:{claim_id}"):
        return jsonify({"error": "Invalid claimID. Start a conversation first."}), 400
    stored_user_id = redis_client.hget(f"claim:{claim_id}", "user_id")
    if stored_user_id != user_id:
        return jsonify({"error": "Unauthorized: claimID does not belong to this user"}), 403

    # Load conversation and claim details
    conversation = json.loads(redis_client.hget(f"claim:{claim_id}", "conversation"))
    claim_details = json.loads(redis_client.hget(f"claim:{claim_id}", "claim_details"))
    
    # Update claim details from document library using the specified category.
    claim_details = update_claim_details_from_documents(user_id, claim_details)
    
    # Check if a category is already set; if not, try to infer from the query.
    if not claim_details.get("category"):
        if "home" in query.lower():
            claim_details["category"] = "home"
            answer = "I detected that you might be referring to your home insurance. Is that correct?"
        elif "car" in query.lower() or "vehicle" in query.lower():
            claim_details["category"] = "vehicle"
            answer = "I detected that you might be referring to your vehicle insurance. Is that correct?"
        else:
            answer = None
    else:
        answer = None

    # Append user's query to conversation
    conversation.append({"role": "user", "content": query})

    # If the query relates to the full name, confirm the known data.
    if "name" in query.lower() or "full name" in query.lower():
        if claim_details.get("full_name"):
            answer = f"I see we already have your full name as {claim_details['full_name']}. Is that correct?"
        else:
            answer = "Could you please provide your full name?"
    
    # If answer hasn't been set by the above category detection or field confirmation, generate a response.
    if not answer:
        knowledge = retrieve_relevant_knowledge(user_id, query, category=claim_details.get("category"))
        if knowledge:
            conversation.append({"role": "system", "content": f"Relevant knowledge:\n{knowledge}"})
        answer = xai_chat_client.chat(conversation)
    
    conversation.append({"role": "assistant", "content": answer})
    
    # Save updated conversation and claim details
    redis_client.hset(f"claim:{claim_id}", "conversation", json.dumps(conversation))
    redis_client.hset(f"claim:{claim_id}", "claim_details", json.dumps(claim_details))
    logger.info(f"Updated claim details for claimID {claim_id}: {claim_details}")
    
    # If the claim is complete, ask if additional info is available.
    if is_claim_complete(claim_details):
        if is_negative_response(query):
            if not claim_details.get("summary_sent"):
                summary = generate_detailed_summary(claim_id, conversation, claim_details)
                threading.Thread(target=send_summary_to_external, args=(summary, claim_id)).start()
                claim_details["summary_sent"] = True
                redis_client.hset(f"claim:{claim_id}", "claim_details", json.dumps(claim_details))
                answer += "\n\nYour claim is complete and has been finalized."
        else:
            answer += "\n\nWe have all the necessary information for your claim. Do you have any additional details to add? (If not, please reply with 'No')."
    
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
    conversation = json.loads(redis_client.hget(f"claim:{claim_id}", "conversation"))
    attached_files = [doc["filename"] for doc in claim_details.get("ephemeral_docs", [])]
    message = (
        "Claim finalized. The following documents have been attached:\n"
        f"{', '.join(attached_files)}\n\n"
        "Permanent library references are included in the claim report. Good luck!"
    )
    if not claim_details.get("summary_sent"):
        summary = generate_detailed_summary(claim_id, conversation, claim_details)
        threading.Thread(target=send_summary_to_external, args=(summary, claim_id)).start()
        claim_details["summary_sent"] = True
        redis_client.hset(f"claim:{claim_id}", "claim_details", json.dumps(claim_details))
    return jsonify({"message": message, "claimID": claim_id})

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
