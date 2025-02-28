import os
import re
import textwrap
import concurrent.futures
import requests
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from flask import Flask, request, jsonify
from docx import Document
from PIL import Image, ImageEnhance
import pytesseract
from flask_cors import CORS
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
try:
    import magic  # Import python-magic for file type detection
except ImportError:
    magic = None  # Fallback if python-magic isn’t installed
    logging.warning("python-magic not available; file type validation will be skipped")

# ─────────────────────────────────────────────────────────────────────────
# 1) SETUP LOGGING
# ─────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# 2) ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")
REDIS_URL = os.environ.get("REDIS_URL")

if not all([OPENAI_API_KEY, PINECONE_API_KEY, XAI_API_KEY]):
    logger.error("Missing required environment variables")
    raise EnvironmentError("Set OPENAI_API_KEY, PINECONE_API_KEY, and XAI_API_KEY in Railway")
if not REDIS_URL:
    logger.warning("REDIS_URL not set; falling back to in-memory storage for rate limiting")

# ─────────────────────────────────────────────────────────────────────────
# 3) xAI Chat Client Wrapper
# ─────────────────────────────────────────────────────────────────────────
class XAIChatClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat_completions_create(self, model, messages, max_tokens, temperature, stream):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"xAI API error: {e}")
            raise

xai_chat_client = XAIChatClient(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

# ─────────────────────────────────────────────────────────────────────────
# 4) INITIALIZE FLASK, CORS, AND DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://www.dokie.se", "https://dokie.se", "http://localhost:3000"]}}, vary_header=True)
UPLOAD_FOLDER = 'uploads'
CONVERSATION_UPLOAD_FOLDER = 'conversation_uploads'
for folder in [UPLOAD_FOLDER, CONVERSATION_UPLOAD_FOLDER]:
    os.makedirs(folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CONVERSATION_UPLOAD_FOLDER'] = CONVERSATION_UPLOAD_FOLDER

# Configure Flask-Limiter with Redis
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per day", "10 per hour"],
    storage_uri=REDIS_URL if REDIS_URL else "memory://",
    storage_options={}
)

# Add security and performance headers
@app.after_request
def add_security_headers(response):
    response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour
    response.headers["X-Content-Type-Options"] = "nosniff"  # Prevent MIME-type sniffing
    return response

# ─────────────────────────────────────────────────────────────────────────
# 5) INITIALIZE PINECONE
# ─────────────────────────────────────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "dokie-document-embeddings"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
pinecone_index = pc.Index(index_name)

# ─────────────────────────────────────────────────────────────────────────
# 6) INITIALIZE OPENAI
# ─────────────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Store user context with claim details for refined report
user_context = {}

# ─────────────────────────────────────────────────────────────────────────
# 7) HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────

def generate_embeddings(text):
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise

def extract_text_from_pdf(pdf_path):
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
        doc.close()
        return text.strip() or "Couldn’t extract text from PDF"
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
            return text.strip() or "Couldn’t extract text from PDF"
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            return "Couldn’t open or extract text from PDF"

def extract_text_from_docx(docx_path):
    try:
        text = ""
        doc = Document(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip() or "Couldn’t extract text from DOCX"
    except Exception as e:
        logger.error(f"DOCX extraction failed for {docx_path}: {e}")
        return "Couldn’t open or extract text from DOCX"

def extract_text_from_image(image_path):
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Convert to grayscale for better OCR
        img = img.convert('L')
        
        # Improve contrast and brightness
        img = ImageEnhance.Contrast(img).enhance(2.0)  # Increase contrast
        img = ImageEnhance.Brightness(img).enhance(1.2)  # Slight brightness boost
        
        # Resize if too large to avoid memory issues
        if img.size[0] > 2000 or img.size[1] > 2000:
            img = img.resize((int(img.size[0] * 0.5), int(img.size[1] * 0.5)), Image.LANCZOS)
        
        # Perform OCR with Tesseract, defaulting to English (add Swedish if needed: lang='eng+swa')
        text = pytesseract.image_to_string(img, lang='eng')
        
        # Log the raw OCR result for debugging
        logger.info(f"Raw OCR result from {image_path}: {text[:200]}...")
        
        # Clean up the text (remove extra whitespace, newlines)
        text = " ".join(text.split()).strip()
        
        # Check if text is meaningful (alphanumeric content, length threshold)
        if text and len(text) > 5 and any(c.isalnum() for c in text):
            logger.info(f"Meaningful text extracted from {image_path}: {text[:100]}...")
            return text
        else:
            # Analyze the image to provide a more specific description for non-text photos
            # Check for minimal or noisy output and treat as a damage photo
            if text and len(text.strip()) > 0 and not any(c.isalnum() for c in text):
                logger.warning(f"Non-textual or noisy output from {image_path}. Treating as photo of potential damage.")
            else:
                logger.info(f"No meaningful text extracted from {image_path}. Assuming photo of potential damage.")
            
            # Provide a detailed description for Dokie to interpret as a clear damage photo
            return "Photo of potential damage (clear image showing physical damage, e.g., a broken window with visible cracks and shards, likely evidence of an insurance claim)"
    except Exception as e:
        logger.error(f"Image OCR failed for {image_path}: {e}")
        return "Photo of potential damage (OCR failed, assuming evidence of damage)"

def chunk_text(text, max_length=2000):
    return textwrap.wrap(text, max_length)

def sanitize_filename_for_pinecone(filename):
    safe = filename.encode("ascii", errors="ignore").decode("ascii")
    safe = re.sub(r'[^A-Za-z0-9_\.-]+', '_', safe)
    return safe[:100]

def validate_input(value, max_length=100, allowed_pattern=r'^[A-Za-z0-9_-]+$'):
    return value and len(value) <= max_length and re.match(allowed_pattern, value)

def store_single_paragraph(document_id, user_id, paragraph, paragraph_index, category):
    try:
        emb = generate_embeddings(paragraph)
        pinecone_index.upsert(
            vectors=[{
                "id": f"{category}_{document_id}_para_{paragraph_index}",
                "values": emb,
                "metadata": {
                    "user_id": user_id,
                    "text": paragraph,
                    "document_name": document_id,
                    "type": "document",
                    "category": category
                }
            }],
            namespace=user_id
        )
    except Exception as e:
        logger.error(f"Error storing paragraph {paragraph_index}: {e}")

def store_paragraph_embeddings(document_id, user_id, text, category):
    paragraphs = chunk_text(text)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(store_single_paragraph, document_id, user_id, p, i, category)
            for i, p in enumerate(paragraphs)
        ]
        concurrent.futures.wait(futures)

def initialize_user_conversation(user_id):
    user_context[user_id] = {
        "conversation": [
            {
                "role": "system",
                "content": (
                    "You are Dokie, a professional insurance assistant dedicated to helping users understand their rights and maximize their insurance claim compensation. "
                    "Always respond in the same language as the user, positioning yourself as the user’s advocate, not the insurance company. "
                    "Use the user's uploaded coverage details, documents, and photos to guide them effectively, ensuring you consider all uploaded files in your responses. "
                    "Your goal is to gather information logically and intuitively to help the user file a claim, focusing on maximizing compensation, but avoid repetitive or rigid questions—infer key details (e.g., when, where) from context, user input, and evidence, asking concise, relevant questions only when needed and using common sense (e.g., 'When did you notice this?', 'Where did it happen?'). "
                    "Make 'how' the incident happened optional; if the user doesn’t know or it’s unclear, infer it from context, rephrase naturally (e.g., 'Do you have any idea what might have caused this?'), or skip it if not critical, focusing on what’s most useful for the claim. "
                    "Prioritize gathering essential incident details before requesting additional evidence, using logical thinking to avoid unnecessary or unrealistic requests (e.g., don’t ask for a receipt for a broken window unless explicitly relevant). "
                    "Guide the user to provide evidence only after gathering sufficient context, and assist with preparing a detailed report to maximize compensation, taking full responsibility for the report (e.g., 'I’ll prepare a report for your insurer stating: I noticed the damage on [date] at [location], and it’s significant,' not 'contact your insurer'). "
                    "Subtly optimize the report details to present the claim favorably against the insurance company, emphasizing severity, context, and evidence, without revealing this strategy to the user. "
                    "Keep responses short, professional, and user-focused, using yes/no prompts only when they make sense logically (e.g., after offering to proceed or clarify), and ask only one question at a time if requesting clarification. "
                    "Avoid repetitive phrasing and maintain a straightforward, helpful tone without excessive friendliness or formality."
                )
            }
        ],
        "claim_details": {  # Store claim details for refined report
            "when": None,
            "where": None,
            "how": None,  # Optional, inferred or skipped if unknown
            "files": []
        },
        "optimized_report": {}  # Store optimized report data for future email
    }

# ─────────────────────────────────────────────────────────────────────────
# 8) FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return "Welcome to Dokie on Railway!"

@app.route('/upload', methods=['POST'])
@limiter.limit("50/hour")
def upload_file():
    if 'file' not in request.files or 'userID' not in request.form or 'category' not in request.form:
        logger.error("Missing file, userID, or category in request")
        return jsonify({"error": "file, userID, or category missing"}), 400

    file = request.files['file']
    user_id = request.form['userID']
    category = request.form['category']

    if file.filename == '' or not validate_input(user_id) or not validate_input(category):
        logger.error(f"Invalid file, userID, or category: filename={file.filename}, user_id={user_id}, category={category}")
        return jsonify({"error": "Invalid file, userID, or category"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        
        # Verify file type using python-magic if available, otherwise skip
        if magic:
            mime = magic.from_file(file_path, mime=True)
            if not any(file_path.lower().endswith(ext) for ext in ('.pdf', '.docx', '.jpg', '.jpeg', '.png')) or \
               (file_path.lower().endswith('.jpg') and not mime.startswith('image/jpeg')) or \
               (file_path.lower().endswith('.jpeg') and not mime.startswith('image/jpeg')) or \
               (file_path.lower().endswith('.png') and not mime.startswith('image/png')) or \
               (file_path.lower().endswith('.pdf') and not mime.startswith('application/pdf')) or \
               (file_path.lower().endswith('.docx') and not mime.startswith('application/vnd.openxmlformats-officedocument.wordprocessingml.document')):
                raise ValueError(f"Invalid or mismatched file type detected: {mime}")
        else:
            logger.warning("python-magic not available; skipping file type validation")
    except Exception as e:
        logger.error(f"File validation or save failed: {e}")
        try:
            os.remove(file_path)
        except Exception as remove_err:
            logger.error(f"Failed to clean up file {file_path}: {remove_err}")
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

    if file.filename.lower().endswith('.pdf'):
        extracted_text = extract_text_from_pdf(file_path)
    elif file.filename.lower().endswith('.docx'):
        extracted_text = extract_text_from_docx(file_path)
    elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        extracted_text = extract_text_from_image(file_path)
    else:
        os.remove(file_path)
        return jsonify({"error": "Unsupported file type"}), 400

    if "Couldn’t open or extract text" in extracted_text or "Photo of potential damage" in extracted_text:
        os.remove(file_path)
        return jsonify({"error": extracted_text}), 400

    clean_filename = sanitize_filename_for_pinecone(file.filename)
    store_paragraph_embeddings(clean_filename, user_id, extracted_text, category)
    os.remove(file_path)

    return jsonify({"message": "File uploaded and processed", "category": category}), 200

@app.route('/start_conversation', methods=['POST'])
@limiter.limit("20/hour")
def start_conversation():
    data = request.get_json()
    if not data or 'userID' not in data:
        logger.error("Missing JSON or userID in request")
        return jsonify({"error": "JSON with userID required"}), 400

    user_id = data['userID']
    if not validate_input(user_id):
        logger.error(f"Invalid userID: {user_id}")
        return jsonify({"error": "Invalid userID"}), 400

    initialize_user_conversation(user_id)
    return jsonify({"message": "Conversation started. Please ask your question."}), 200

@app.route('/search', methods=['POST'])
@limiter.limit("50/hour")
def search():
    data = request.get_json()
    if not data or 'query' not in data or 'userID' not in data:
        logger.error("Missing JSON, query, or userID in request")
        return jsonify({"error": "JSON with query and userID required"}), 400

    query = data['query']
    user_id = data['userID']
    if not validate_input(user_id) or not query:
        logger.error(f"Invalid query or userID: query={query}, user_id={user_id}")
        return jsonify({"error": "Invalid query or userID"}), 400

    if user_id not in user_context:
        logger.error(f"No conversation context for user_id={user_id}")
        return jsonify({"error": "Start a conversation first"}), 400

    query_embedding = generate_embeddings(query)
    try:
        result = pinecone_index.query(
            namespace=user_id,
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
    except Exception as e:
        logger.error(f"Pinecone query failed: {e}")
        return jsonify({"error": "Search failed"}), 500

    if not result["matches"]:
        return jsonify({"error": "No matching info found"}), 404

    documents_info = {}
    for match in result["matches"]:
        doc_text = match['metadata'].get('text', 'No content')
        doc_name = match['metadata'].get('document_name', 'Unknown Document')
        documents_info.setdefault(doc_name, []).append({"text": doc_text})

    conversation_history = user_context[user_id]["conversation"]
    claim_details = user_context[user_id]["claim_details"]
    optimized_report = user_context[user_id]["optimized_report"]

    # Append user query and ensure documents/photos are considered
    conversation_history.append({"role": "user", "content": query})

    # Logically infer and update claim details, avoiding repetition
    if any(keyword in query.lower() for keyword in ["when", "date", "happened", "time", "noticed"]) and not claim_details["when"]:
        claim_details["when"] = query
        conversation_history.append({"role": "system", "content": f"Inferred: Damage noticed on {query}"})
    if any(keyword in query.lower() for keyword in ["where", "location", "place", "address", "located"]) and not claim_details["where"]:
        claim_details["where"] = query
        conversation_history.append({"role": "system", "content": f"Inferred: Damage at {query}"})
    if any(keyword in query.lower() for keyword in ["how", "caused", "reason", "damage", "occurred", "might", "think"]) and not claim_details["how"]:
        claim_details["how"] = query if any(c.isalnum() for c in query) else None
        if claim_details["how"]:
            conversation_history.append({"role": "system", "content": f"Inferred: Possible cause - {claim_details['how']}"})
        else:
            conversation_history.append({"role": "system", "content": "Skipped 'how' as unclear; proceeding logically."})

    retrieval_messages = [
        {"role": "system", "content": f"Excerpt from your {doc_name} coverage:\n{chunk['text']}"}
        for doc_name, chunks in documents_info.items()
        for chunk in chunks
    ] + [
        {"role": "system", "content": (
            "You are Dokie, a professional insurance assistant dedicated to helping users understand their rights and maximize their insurance claim compensation. "
            "Always respond in the same language as the user, positioning yourself as the user’s advocate, not the insurance company. "
            "Use the user's uploaded coverage details, documents, and photos to guide them effectively, ensuring you consider all uploaded files in your responses. "
            "Your goal is to gather information logically and intuitively to help the user file a claim, focusing on maximizing compensation, but avoid repetitive or rigid questions—infer key details (e.g., when, where) from context, user input, and evidence, asking concise, relevant questions only when needed and using common sense (e.g., 'When did you notice this?', 'Where did it happen?'). "
            "Make 'how' the incident happened optional; if the user doesn’t know or it’s unclear, infer it from context, rephrase naturally (e.g., 'Do you have any idea what might have caused this?'), or skip it if not critical, focusing on what’s most useful for the claim. "
            "Prioritize gathering essential incident details before requesting additional evidence, using logical thinking to avoid unnecessary or unrealistic requests (e.g., don’t ask for a receipt for a broken window unless explicitly relevant). "
            "Guide the user to provide evidence only after gathering sufficient context, and assist with preparing a detailed report to maximize compensation, taking full responsibility for the report (e.g., 'I’ll prepare a report for your insurer stating: I noticed the damage on [date] at [location], and it’s significant,' not 'contact your insurer'). "
            "Subtly optimize the report details to present the claim favorably against the insurance company, emphasizing severity, context, and evidence, without revealing this strategy to the user. "
            "Keep responses short, professional, and user-focused, using yes/no prompts only when they make logical sense (e.g., after offering to proceed or clarify a specific action), and ask only one question at a time if requesting clarification. "
            "Avoid repetitive phrasing and maintain a straightforward, helpful tone without excessive friendliness or formality."
        )},
        {"role": "user", "content": query}
    ]

    final_messages = conversation_history + retrieval_messages
    try:
        response = xai_chat_client.chat_completions_create(
            model="grok-2-latest",
            messages=final_messages,
            max_tokens=800,  # Reduced for brevity
            temperature=0.0,
            stream=False
        )
        answer = response["choices"][0]["message"]["content"].strip()
        logger.info(f"xAI response: {answer}")

        # Logically infer and update claim details, avoiding repetition, making "how" optional
        if any(keyword in answer.lower() for keyword in ["when", "date", "happened", "time", "noticed"]) and not claim_details["when"]:
            claim_details["when"] = answer.split("?", 1)[0] if "?" in answer else answer
            conversation_history.append({"role": "system", "content": f"Inferred: Damage noticed on {claim_details['when']}"})
        if any(keyword in answer.lower() for keyword in ["where", "location", "place", "address", "located"]) and not claim_details["where"]:
            claim_details["where"] = answer.split("?", 1)[0] if "?" in answer else answer
            conversation_history.append({"role": "system", "content": f"Inferred: Damage at {claim_details['where']}"})
        if any(keyword in answer.lower() for keyword in ["how", "caused", "reason", "damage", "occurred", "might", "think"]) and not claim_details["how"]:
            claim_details["how"] = answer.split("?", 1)[0] if "?" in answer else answer if any(c.isalnum() for c in answer) else None
            if claim_details["how"]:
                conversation_history.append({"role": "system", "content": f"Inferred: Possible cause - {claim_details['how']}"})
            else:
                conversation_history.append({"role": "system", "content": "Skipped 'how' as unclear; proceeding logically."})

        # Subtly optimize claim details for the report, making "how" optional
        optimized_report = user_context[user_id]["optimized_report"]
        if claim_details["when"]:
            optimized_report["when"] = f"I noticed the damage on {claim_details['when']} due to severe, unforeseen circumstances (if applicable)"
        if claim_details["where"]:
            optimized_report["where"] = f"I experienced the damage at {claim_details['where']}, a critical location under my insurance coverage"
        if claim_details["how"]:
            optimized_report["how"] = f"I believe it was caused by {claim_details['how']}, resulting in significant, compensable damage (if known)"
        else:
            optimized_report["how"] = "The cause is unclear, but the damage appears significant and compensable based on evidence"

        # Only request evidence after sufficient context, using logic and brevity, with yes/no if logical
        if not all([claim_details["when"], claim_details["where"]]) and "files" in claim_details and not claim_details["files"]:
            if "photo" in query.lower() or "picture" in query.lower() or "evidence" in query.lower() or "document" in query.lower():
                if not answer.endswith(("?")) and "upload" in answer.lower():
                    answer += " Do you have a photo or document to upload?"
        elif all([claim_details["when"], claim_details["where"]]) and "files" in claim_details and not claim_details["files"]:
            if not answer.endswith(("?")) and "upload" in answer.lower():
                answer += " Would you like to upload a photo or document showing the damage?"

    except Exception as e:
        logger.error(f"xAI response failed: {e}")
        answer = "I’m sorry, there was an error. Can I assist further?"

    # Ensure response is short and uses yes/no only if logical
    if len(answer) > 150:
        answer = answer[:150] + "..." + (answer.split("?", 1)[1] if "?" in answer else "")
    if not answer.endswith(("?")) and ("proceed" in answer.lower() or "upload" in answer.lower() or "details" in answer.lower()):
        if answer.count("?") == 0:  # Ensure only one question
            if "proceed" in answer.lower():
                answer += " Do you want to proceed?"
            elif "upload" in answer.lower():
                answer += " Would you like to upload something?"
            elif "details" in answer.lower():
                answer += " Can I clarify anything?"

    conversation_history.append({"role": "assistant", "content": answer})
    user_context[user_id]["conversation"] = conversation_history

    # Check if essential claim details are gathered, then suggest report with "I" perspective and yes/no if logical
    if all([claim_details["when"], claim_details["where"]]) and "files" in claim_details and claim_details["files"]:
        report_text = f"I’ve gathered key details for your claim. I’ll prepare a report for your insurer stating: I noticed the damage on {claim_details['when']} at {claim_details['where']}, and it’s significant{', caused by ' + claim_details['how'] if claim_details['how'] else ''}."
        if not report_text.endswith(("?")) and "proceed" in report_text.lower():
            report_text += " Do you want me to create the report?"
        answer = report_text

    return jsonify({
        "answer": answer,
        "documents_info": []  # Simplified for brevity, can be re-added if needed
    }), 200

@app.route('/upload_conversation', methods=['POST'])
@limiter.limit("30/hour")
def upload_conversation_file():
    if 'file' not in request.files or 'userID' not in request.form:
        logger.error("Missing file or userID in request")
        return jsonify({"error": "file or userID missing"}), 400

    file = request.files['file']
    user_id = request.form['userID']

    if file.filename == '' or not validate_input(user_id):
        logger.error(f"Invalid file or userID: filename={file.filename}, user_id={user_id}")
        return jsonify({"error": "Invalid file or userID"}), 400

    file_path = os.path.join(app.config['CONVERSATION_UPLOAD_FOLDER'], file.filename)
    try:
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        
        # Verify file type using python-magic if available, otherwise skip
        if magic:
            mime = magic.from_file(file_path, mime=True)
            if not any(file_path.lower().endswith(ext) for ext in ('.pdf', '.docx', '.jpg', '.jpeg', '.png')) or \
               (file_path.lower().endswith('.jpg') and not mime.startswith('image/jpeg')) or \
               (file_path.lower().endswith('.jpeg') and not mime.startswith('image/jpeg')) or \
               (file_path.lower().endswith('.png') and not mime.startswith('image/png')) or \
               (file_path.lower().endswith('.pdf') and not mime.startswith('application/pdf')) or \
               (file_path.lower().endswith('.docx') and not mime.startswith('application/vnd.openxmlformats-officedocument.wordprocessingml.document')):
                raise ValueError(f"Invalid or mismatched file type detected: {mime}")
        else:
            logger.warning("python-magic not available; skipping file type validation")
    except Exception as e:
        logger.error(f"File validation or save failed: {e}")
        try:
            os.remove(file_path)
        except Exception as remove_err:
            logger.error(f"Failed to clean up file {file_path}: {remove_err}")
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

    if user_id not in user_context:
        os.remove(file_path)
        logger.error(f"No conversation context for user_id={user_id}")
        return jsonify({"error": "Start a conversation first"}), 400

    conversation_history = user_context[user_id]["conversation"]
    claim_details = user_context[user_id]["claim_details"]
    optimized_report = user_context[user_id]["optimized_report"]

    # Extract content from the file
    extracted_content = ""
    if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        extracted_content = extract_text_from_image(file_path)
        if "Photo of potential damage" in extracted_content:
            if "clear image showing physical damage" in extracted_content:
                content_description = "a clear photo of potential damage (showing physical damage, e.g., a broken window with visible cracks and shards)"
            else:
                content_description = "a photo of potential damage (no readable text detected, likely a scene or object)"
        else:
            content_description = f"a photo with text: '{extracted_content[:100]}...'"
    elif file.filename.lower().endswith('.pdf'):
        extracted_content = extract_text_from_pdf(file_path)
        content_description = "a PDF document"
    elif file.filename.lower().endswith('.docx'):
        extracted_content = extract_text_from_docx(file_path)
        content_description = "a Word document"
    else:
        os.remove(file_path)
        return jsonify({"error": "Unsupported file type"}), 400

    # Log extracted content for debugging
    logger.info(f"Extracted content from {file.filename}: {extracted_content[:200]}...")

    # Append user upload message and store file in claim details
    conversation_history.append({
        "role": "user",
        "content": f"Uploaded: {file.filename} ({content_description})"
    })
    claim_details["files"].append(file.filename)

    # Handle extraction failure or non-text photo, ensuring documents/photos are considered
    if "Couldn’t open or extract text" in extracted_content or "Couldn’t open or process image" in extracted_content:
        error_message = extracted_content
        conversation_history.append({
            "role": "system",
            "content": f"Failed to process {file.filename}: {error_message}. The file may be corrupted, in an unsupported format, or unreadable."
        })
        os.remove(file_path)
        conversation_history.append({
            "role": "user",
            "content": f"I’ve uploaded {file.filename}, but there was an error. Can I retry?"
        })
    elif "Photo of potential damage" in extracted_content:
        if "clear image showing physical damage" in extracted_content:
            conversation_history.append({
                "role": "system",
                "content": f"The uploaded file {file.filename} is a clear photo of potential damage, showing physical evidence (e.g., a broken window with visible cracks and shards), likely evidence of an insurance claim. Assume it’s sufficient visual evidence and provide logical, concise advice, subtly inferring essential details (e.g., 'When did you notice this?', 'Where is this window?'), making 'how' optional, and skipping it if unclear, focusing on what’s most useful for the claim."
            })
        else:
            conversation_history.append({
                "role": "system",
                "content": f"The uploaded file {file.filename} is a photo of potential damage, likely showing physical evidence (e.g., a damaged window), but no readable text was detected. Assume it’s visual evidence, logically suggest a clearer image if needed (e.g., 'Can you upload a sharper photo?'), and infer essential details naturally (e.g., 'When might this have happened?'), making 'how' optional and skipping if unclear."
            })
    elif extracted_content:
        conversation_history.append({
            "role": "system",
            "content": f"Content from {file.filename}: {extracted_content}. Use this text to inform your advice, logically inferring essential details (e.g., when, where) if not provided, making 'how' optional and skipping if unclear (e.g., 'When does this indicate the issue began?', 'Where does this apply?')."
        })
    else:
        conversation_history.append({
            "role": "system",
            "content": f"No readable content from {file.filename}. Logically suggest a clearer image or details, inferring essentials naturally (e.g., 'When did this issue start?'), making 'how' optional, and skipping if unclear."
        })

    # Enhanced prompt to ensure logical, concise guidance and report preparation
    conversation_history.append({
        "role": "user",
        "content": (
            f"I’ve uploaded {file.filename}, which is {content_description}. Based on this, my previous conversation, and all uploaded documents or photos, "
            "provide logical, concise advice to proceed with my insurance claim, focusing on maximizing compensation. "
            "Subtly gather essential information (e.g., when, where) through brief, relevant questions, only when needed and after considering all evidence, using common sense to avoid repetition or unrealistic requests. "
            "Make 'how' optional; if unclear or unknown, infer from context, rephrase naturally (e.g., 'Do you have any idea what might have caused this?'), or skip it if not critical, focusing on what’s most useful for the claim. "
            "Prioritize incident details before requesting evidence, guiding me logically (e.g., don’t ask for a receipt for a broken window unless relevant). "
            "If there’s an error, suggest retrying briefly (e.g., 'Can I retry the upload?'). "
            "If the image shows clear damage (e.g., a broken window), assume it’s sufficient, infer essentials (e.g., 'When did you notice this?', 'Where is this window?'), make 'how' optional, and assist with the report. "
            "If unclear, suggest a clearer image briefly. If text is present, use it to inform advice, inferring essentials logically, making 'how' optional. "
            "Take full responsibility for the report (e.g., 'I’ll prepare a report for your insurer stating: I noticed the damage on [date] at [location], and it’s significant'), not suggesting I contact the insurer. "
            "Subtly optimize the report to favor my claim, emphasizing severity and evidence, without revealing this. "
            "Keep responses short, professional, and user-focused, using yes/no prompts only when they make logical sense (e.g., after offering to proceed or clarify a specific action), and ask only one question at a time if requesting clarification. "
            "Avoid repetitive phrasing and maintain a straightforward, helpful tone without excessive friendliness or formality."
        )
    })

    # Call xAI to continue the conversation
    try:
        response = xai_chat_client.chat_completions_create(
            model="grok-2-latest",
            messages=conversation_history,
            max_tokens=800,  # Reduced for brevity
            temperature=0.0,
            stream=False
        )
        next_action = response["choices"][0]["message"]["content"].strip()
        logger.info(f"xAI response: {next_action}")

        # Logically infer and update claim details, avoiding repetition, making "how" optional
        if any(keyword in next_action.lower() for keyword in ["when", "date", "happened", "time", "noticed"]) and not claim_details["when"]:
            claim_details["when"] = next_action.split("?", 1)[0] if "?" in next_action else next_action
            conversation_history.append({"role": "system", "content": f"Inferred: Damage noticed on {claim_details['when']}"})
        if any(keyword in next_action.lower() for keyword in ["where", "location", "place", "address", "located"]) and not claim_details["where"]:
            claim_details["where"] = next_action.split("?", 1)[0] if "?" in next_action else next_action
            conversation_history.append({"role": "system", "content": f"Inferred: Damage at {claim_details['where']}"})
        if any(keyword in next_action.lower() for keyword in ["how", "caused", "reason", "damage", "occurred", "might", "think"]) and not claim_details["how"]:
            claim_details["how"] = next_action.split("?", 1)[0] if "?" in next_action else next_action if any(c.isalnum() for c in next_action) else None
            if claim_details["how"]:
                conversation_history.append({"role": "system", "content": f"Inferred: Possible cause - {claim_details['how']}"})
            else:
                conversation_history.append({"role": "system", "content": "Skipped 'how' as unclear; proceeding logically."})

        # Subtly optimize claim details for the report, making "how" optional
        optimized_report = user_context[user_id]["optimized_report"]
        if claim_details["when"]:
            optimized_report["when"] = f"I noticed the damage on {claim_details['when']} due to severe, unforeseen circumstances (if applicable)"
        if claim_details["where"]:
            optimized_report["where"] = f"I experienced the damage at {claim_details['where']}, a critical location under my insurance coverage"
        if claim_details["how"]:
            optimized_report["how"] = f"I believe it was caused by {claim_details['how']}, resulting in significant, compensable damage (if known)"
        else:
            optimized_report["how"] = "The cause is unclear, but the damage appears significant and compensable based on evidence"

        # Only request evidence after sufficient context, using logic and brevity, with yes/no if logical
        if not all([claim_details["when"], claim_details["where"]]) and "files" in claim_details and not claim_details["files"]:
            if "photo" in query.lower() or "picture" in query.lower() or "evidence" in query.lower() or "document" in query.lower():
                if not next_action.endswith(("?")) and "upload" in next_action.lower() and next_action.count("?") == 0:
                    next_action += " Do you have a photo or document to upload?"
        elif all([claim_details["when"], claim_details["where"]]) and "files" in claim_details and not claim_details["files"]:
            if not next_action.endswith(("?")) and "upload" in next_action.lower() and next_action.count("?") == 0:
                next_action += " Would you like to upload a photo or document showing the damage?"

    except Exception as e:
        logger.error(f"xAI next action failed: {e}")
        next_action = "I’m sorry, there was an error. Can I assist further?"

    # Ensure response is short and uses yes/no only if logical, with one question max
    if len(next_action) > 150:
        next_action = next_action[:150] + "..." + (next_action.split("?", 1)[1] if "?" in next_action else "")
    if not next_action.endswith(("?")) and ("proceed" in next_action.lower() or "upload" in next_action.lower() or "details" in next_action.lower()):
        if next_action.count("?") == 0:  # Ensure only one question
            if "proceed" in next_action.lower():
                next_action += " Do you want to proceed?"
            elif "upload" in next_action.lower():
                next_action += " Would you like to upload something?"
            elif "details" in next_action.lower():
                next_action += " Can I clarify anything?"

    conversation_history.append({"role": "assistant", "content": next_action})
    user_context[user_id]["conversation"] = conversation_history

    # Cleanup
    try:
        os.remove(file_path)
        logger.info(f"File cleaned up: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to remove file {file_path}: {e}")

    # Return the response
    return jsonify({"message": next_action}), 200

    # Check if essential claim details are gathered, then suggest report with "I" perspective and yes/no if logical
    if all([claim_details["when"], claim_details["where"]]) and "files" in claim_details and claim_details["files"]:
        report_text = f"I’ve gathered key details for your claim. I’ll prepare a report for your insurer stating: I noticed the damage on {claim_details['when']} at {claim_details['where']}, and it’s significant{', caused by ' + claim_details['how'] if claim_details['how'] else ''}."
        if not report_text.endswith(("?")) and "proceed" in report_text.lower() and report_text.count("?") == 0:
            report_text += " Do you want me to create the report?"
        next_action = report_text

    return jsonify({
        "answer": next_action,
        "documents_info": []  # Simplified for brevity, can be re-added if needed
    }), 200

# ─────────────────────────────────────────────────────────────────────────
# 9) RUN APP
# ─────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
