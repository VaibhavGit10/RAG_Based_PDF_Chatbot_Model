import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev")
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
    VECTOR_FOLDER = os.path.join(os.path.dirname(__file__), "vectorstore")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
