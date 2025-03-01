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
EXTERNAL_SERVICE_URL = os.environ.get("EXTERNAL_SERVICE_URL")  # URL for external processing service

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

def generate_detailed_summary(claim_id: str, conversation: list, claim_details: dict) -> str:
    """
    Generate a detailed report including:
      - Claim ID
      - All claim details (Policyholder Info & Claim Details)
      - Information about uploaded documents (with a snippet of extracted text)
      - Full conversation history
    """
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
    summary_lines.append("")
    
    # Uploaded Documents
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
    """Send the detailed summary to the external service, if configured."""
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
    """
    Determine if all required fields are present.
    Required fields include:
      - Policyholder Information: full_name, policy_number, phone, email
      - Claim Details: claim_type, date_time, location, incident_description
      - At least one uploaded document
    """
    required_fields = ["full_name", "policy_number", "phone", "email",
                       "claim_type", "date_time", "location", "incident_description"]
    for field in required_fields:
        if not claim_details.get(field):
            return False
    if not claim_details.get("ephemeral_docs"):
        return False
    return True

def is_negative_response(query: str) -> bool:
    """Determine if the user's query indicates no further information."""
    negative_responses = {"no", "nope", "nothing", "none", "that's all", "no thanks", "no thank you"}
    return query.strip().lower() in negative_responses

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
    
    # System prompt instructs Dokie to gather required info step-by-step.
    system_prompt = (
        "You are Dokie, a highly empathetic and logical AI assistant who acts as both an advisor and case manager for insurance claims. "
        "Help clients navigate the entire process and ensure they receive fair compensation. Ask one question at a time. "
        "If you already have information from documents (e.g., an address), confirm it instead of asking again. "
        "Gather the following information if not already available:\n"
        "  1. Policyholder Information: Full Name, Policy Number, Phone, Email\n"
        "  2. Claim Details: Type of Claim (Property/Vehicle/Medical/Other), Date & Time of Incident, Location, Brief Description of Incident\n"
        "  3. Required Documents: Photos/Videos of Damage, Invoices/Receipts, Police/Incident Report if applicable.\n"
        "Once all required fields are collected, ask the client if there is any additional information. "
        "If the client replies with a negative response (e.g., 'No'), then finalize the claim by sending a detailed report to the external service."
    )
    
    # Initialize claim details with all required fields as None
    initial_details = {
        "full_name": None,
        "policy_number": None,
        "phone": None,
        "email": None,
        "claim_type": None,
        "date_time": None,
        "location": None,
        "incident_description": None,
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
        return jsonify({"error": "claimID, query, and userID required"}), 400

    if not redis_client.exists(f"claim:{claim_id}"):
        return jsonify({"error": "Invalid claimID. Start a conversation first."}), 400

    stored_user_id = redis_client.hget(f"claim:{claim_id}", "user_id")
    if stored_user_id != user_id:
        return jsonify({"error": "Unauthorized: claimID does not belong to this user"}), 403

    # Load conversation history and claim details
    conversation = json.loads(redis_client.hget(f"claim:{claim_id}", "conversation"))
    claim_details = json.loads(redis_client.hget(f"claim:{claim_id}", "claim_details"))

    # Append the user's query to the conversation
    conversation.append({"role": "user", "content": query})

    # Optionally, retrieve additional context from the permanent document library
    knowledge = retrieve_relevant_knowledge(user_id, query)
    if knowledge:
        conversation.append({"role": "system", "content": f"Relevant knowledge:\n{knowledge}"})

    # Generate Dokie's response based on the updated conversation
    answer = xai_chat_client.chat(conversation)
    conversation.append({"role": "assistant", "content": answer})

    # (In practice, you would update claim_details fields based on the query here,
    # e.g., if query mentions "full name", update claim_details["full_name"], etc.
    # This logic should ensure that if a field is already filled (extracted from documents),
    # it is not overwritten.)

    # Save the updated conversation and claim details
    redis_client.hset(f"claim:{claim_id}", "conversation", json.dumps(conversation))
    redis_client.hset(f"claim:{claim_id}", "claim_details", json.dumps(claim_details))
    logger.info(f"Updated claim details for claimID {claim_id}: {claim_details}")

    # If the claim is complete, Dokie asks if there is additional information.
    if is_claim_complete(claim_details):
        if is_negative_response(query):
            # User indicates no further info; finalize claim and send summary
            if not claim_details.get("summary_sent"):
                summary = generate_detailed_summary(claim_id, conversation, claim_details)
                threading.Thread(target=send_summary_to_external, args=(summary, claim_id)).start()
                claim_details["summary_sent"] = True
                redis_client.hset(f"claim:{claim_id}", "claim_details", json.dumps(claim_details))
                answer += "\n\nYour claim is complete and has been finalized."
        else:
            # If claim is complete but user response is not negative, ask for additional info.
            answer += "\n\nWe have received all the necessary information for your claim. Do you have any additional details to add? (If not, please reply with 'No')."

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
        "Permanent library references are included in the claim text. Good luck!"
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
