import os
import re
import requests
import logging
import textwrap
import json
import uuid  # For generating unique claimIDs
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

if not all([OPENAI_API_KEY, PINECONE_API_KEY, XAI_API_KEY]):
    logger.error("Missing required environment variables")
    raise EnvironmentError("Set OPENAI_API_KEY, PINECONE_API_KEY, and XAI_API_KEY")

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
    storage_uri=REDIS_URL if REDIS_URL else "memory://"
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
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            logger.error(f"xAI API error: {e}")
            return "Sorry, I encountered an error. Please try again."

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
# GLOBAL USER CONTEXT
# ==============================
user_context = {}  # Maps claimID to conversation and claim details

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

    # Generate a unique claimID
    claim_id = str(uuid.uuid4())
    
    # System prompt with updated priorities
    system_prompt = (
        "You are Dokie, an empathetic AI assisting users with insurance claims. "
        "Respond in the same language as the user's query. "
        "You have a permanent document library in Pinecone for coverage details. "
        "For ephemeral claim evidence (photos, etc.), store them in ephemeral memory. "
        "Your primary goal is to help users claim compensation by analyzing their document library "
        "and providing the best approach for a successful claim. Prioritize responding to the user's immediate query or concern. "
        "If the query implies a problem (e.g., 'small leak', 'roof fallen in'), first offer practical advice to minimize risk or damage, "
        "then gather claim details (date, location, description) one question at a time if needed for the claim. "
        "For severe issues (e.g., 'roof fallen in'), suggest a craftsman type (e.g., 'roofing', 'plumbing', 'general') immediately "
        "with up to 3 recommendations from this list: " + CRAFTSMEN_JSON + ", formatted as '- Name (Rating: X/5, Contact: email)' "
        "in the user's language, using a natural statement (not a question). For minor issues (e.g., 'small leak'), suggest craftsmen "
        "only after offering risk mitigation advice and gathering sufficient details. Craftsman suggestions are secondary to claim assistance. "
        "Adapt your response based on conversation history and claim details. Your ultimate goal: maximize the user's claim success "
        "with a natural, empathetic, and context-aware conversation flow."
    )

    user_context[claim_id] = {
        "user_id": user_id,
        "conversation": [{"role": "system", "content": system_prompt}],
        "claim_details": {
            "date": None,
            "location": None,
            "description": None,
            "ephemeral_docs": []
        }
    }
    return jsonify({"message": "Conversation started.", "claimID": claim_id}), 200

@app.route("/upload_library", methods=["POST"])
def upload_library():
    if "file" not in request.files or "userID" not in request.form:
        return jsonify({"error": "file and userID required"}), 400

    file_obj = request.files["file"]
    user_id = request.form["userID"]

    # Initialize minimal context if userID isn't tied to a claim yet
    if not any(ctx["user_id"] == user_id for ctx in user_context.values()):
        claim_id = str(uuid.uuid4())
        user_context[claim_id] = {
            "user_id": user_id,
            "conversation": [],
            "claim_details": {
                "date": None,
                "location": None,
                "description": None,
                "ephemeral_docs": []
            }
        }

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
    if "file" not in request.files or "claimID" not in request.form:
        return jsonify({"error": "file and claimID required"}), 400

    file_obj = request.files["file"]
    claim_id = request.form["claimID"]

    if claim_id not in user_context:
        return jsonify({"error": "Invalid claimID. Start a conversation first."}), 400

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

    user_context[claim_id]["claim_details"]["ephemeral_docs"].append({
        "filename": file_obj.filename,
        "file_path": ephemeral_path,
        "extracted_text": extracted_text or ""
    })

    return jsonify({"message": "Ephemeral file stored for this claim."}), 200

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    claim_id = data.get("claimID")
    query = data.get("query")

    if not claim_id or not query:
        return jsonify({"error": "claimID and query required"}), 400

    if claim_id not in user_context:
        return jsonify({"error": "Invalid claimID. Start a conversation first."}), 400

    conversation = user_context[claim_id]["conversation"]
    claim_details = user_context[claim_id]["claim_details"]
    user_id = user_context[claim_id]["user_id"]

    # Provide current claim details as context
    details_summary = (
        f"Current claim details: Date: {claim_details['date']}, "
        f"Location: {claim_details['location']}, "
        f"Description: {claim_details['description']}"
    )
    conversation.append({"role": "system", "content": details_summary})
    conversation.append({"role": "user", "content": query})

    # Retrieve relevant knowledge from permanent library
    knowledge = retrieve_relevant_knowledge(user_id, query)
    if knowledge:
        conversation.append({
            "role": "system",
            "content": f"Relevant knowledge from your permanent documents:\n{knowledge}"
        })

    # Get xAI's response
    answer = xai_chat_client.chat(conversation)

    # Update claim details based on user response (basic parsing)
    if any(word in query.lower() for word in ["datum", "date", "när", "when"]):
        claim_details["date"] = query
    elif any(word in query.lower() for word in ["var", "where"]):
        claim_details["location"] = query
    elif any(word in query.lower() for word in ["beskriv", "describe", "vad", "what"]):
        claim_details["description"] = query

    conversation.append({"role": "assistant", "content": answer})
    logger.info(f"Updated claim details for claimID {claim_id}: {claim_details}")

    return jsonify({"answer": answer, "claimID": claim_id}), 200

@app.route("/finalize_claim", methods=["POST"])
def finalize_claim():
    data = request.get_json()
    claim_id = data.get("claimID")
    if not claim_id or claim_id not in user_context:
        return jsonify({"error": "Invalid or missing claimID"}), 400

    ephemeral_docs = user_context[claim_id]["claim_details"]["ephemeral_docs"]
    attached_files = [doc["filename"] for doc in ephemeral_docs]
    message = (
        "Claim finalized. The following ephemeral documents have been attached:\n"
        f"{', '.join(attached_files)}\n\n"
        "Permanent library references are included in the claim text. Good luck!"
    )

    return jsonify({"message": message, "claimID": claim_id}), 200

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
