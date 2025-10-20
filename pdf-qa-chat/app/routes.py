import os
import traceback
from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename
from .pdf_utils import extract_text_from_pdf, split_into_chunks
from .rag import embed_text, save_vector_store, load_vector_store, answer_question

bp = Blueprint("main", __name__)

# Global memory (stores embeddings and chunks in RAM)
VECTOR_INDEX = None
CHUNKS = None

# Allowed file types
ALLOWED_EXTENSIONS = {".pdf"}

def allowed_file(filename):
    """Check if file is a PDF."""
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@bp.route("/upload", methods=["POST"])
def upload_pdf():
    """Handle PDF upload, extract content, embed, and store vectors."""
    global VECTOR_INDEX, CHUNKS

    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Only PDF files are supported"}), 400

    try:
        # Save the file
        upload_folder = current_app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_folder, exist_ok=True)
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(upload_folder, filename)
        file.save(pdf_path)

        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        if not text:
            return jsonify({"status": "error", "message": "Could not extract text from PDF"}), 400

        # Split text into manageable chunks
        chunks = split_into_chunks(text)
        if not chunks:
            return jsonify({"status": "error", "message": "PDF content is too short or invalid"}), 400

        # Embed text and store vectors
        vector_folder = current_app.config["VECTOR_FOLDER"]
        os.makedirs(vector_folder, exist_ok=True)
        embeddings, CHUNKS = embed_text(chunks)
        save_vector_store(embeddings, CHUNKS, vector_folder)

        # Load vectors into memory
        VECTOR_INDEX, CHUNKS = load_vector_store(vector_folder)

        return jsonify({
            "status": "success",
            "message": f"PDF processed successfully with {len(CHUNKS)} chunks."
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Processing error: {str(e)}"}), 500


@bp.route("/ask", methods=["POST"])
def ask_question():
    """Answer questions using RAG (Retrieval-Augmented Generation)."""
    global VECTOR_INDEX, CHUNKS

    if VECTOR_INDEX is None or CHUNKS is None:
        return jsonify({"status": "error", "message": "Please upload a PDF first"}), 400

    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"status": "error", "message": "Question cannot be empty"}), 400

    try:
        answer = answer_question(question, VECTOR_INDEX, CHUNKS)
        return jsonify({"status": "success", "answer": answer})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Error generating answer: {str(e)}"}), 500

