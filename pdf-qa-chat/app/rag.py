import os
import faiss
import json
import numpy as np
import onnxruntime as ort
import requests
from typing import List, Tuple
from tokenizers import Tokenizer
from openai import OpenAI
from .config import Config

# -----------------------------
# Groq client for Chat
# -----------------------------
client = OpenAI(
    api_key=Config.OPENAI_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
CHAT_MODEL = "llama-3.1-8b-instant"

# -----------------------------
# ONNX Embedding model path
# -----------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "minilm-onnx")
MODEL_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
TOKENIZER_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

SEP = "\n====chunk====\n"
INDEX_FILENAME = "index.faiss"
DOCS_FILENAME = "documents.txt"

def _download(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return
    print(f"⬇️ Downloading {url} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def _ensure_onnx_and_tokenizer():
    print("🔎 Checking ONNX model & tokenizer...")
    _download(MODEL_URL, MODEL_PATH)
    _download(TOKENIZER_URL, TOKENIZER_PATH)

def _load_embedder():
    _ensure_onnx_and_tokenizer()
    print("🧠 Loading ONNX model (CPU only)...")
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print("🔤 Loading tokenizer...")
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    return sess, tok

_EMBED_SESSION = None
_TOKENIZER = None

def _get_embed_session():
    global _EMBED_SESSION, _TOKENIZER
    if _EMBED_SESSION is None:
        _EMBED_SESSION, _TOKENIZER = _load_embedder()
    return _EMBED_SESSION, _TOKENIZER

def _tokenize_batch(texts: List[str], tokenizer: Tokenizer, max_len: int = 256):
    encodings = tokenizer.encode_batch(texts)
    input_ids, attention_mask, token_type_ids = [], [], []

    for enc in encodings:
        ids = enc.ids[:max_len]
        mask = enc.attention_mask[:max_len]
        types = [0] * len(ids)

        pad_len = max_len - len(ids)
        ids += [0] * pad_len
        mask += [0] * pad_len
        types += [0] * pad_len

        input_ids.append(ids)
        attention_mask.append(mask)
        token_type_ids.append(types)

    return (
        np.array(input_ids, dtype=np.int64),
        np.array(attention_mask, dtype=np.int64),
        np.array(token_type_ids, dtype=np.int64)
    )

def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    embeddings = last_hidden_state * mask
    sum_embeddings = np.sum(embeddings, axis=1)
    lengths = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / lengths

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm

def _encode(texts: List[str]) -> np.ndarray:
    sess, tokenizer = _get_embed_session()
    input_ids, attention_mask, token_type_ids = _tokenize_batch(texts, tokenizer)

    feed = {}
    for inp in sess.get_inputs():
        name = inp.name
        if "input_ids" in name:
            feed[name] = input_ids
        elif "attention_mask" in name:
            feed[name] = attention_mask
        elif "token_type_ids" in name:
            feed[name] = token_type_ids

    outputs = sess.run(None, feed)
    out = outputs[0]
    if out.ndim == 3:
        out = _mean_pool(out, attention_mask)
    return _l2_normalize(out.astype(np.float32))

def embed_text(texts: List[str]) -> Tuple[np.ndarray, List[str]]:
    print("🔄 Generating ONNX embeddings...")
    embeddings = _encode(texts)
    return embeddings, texts

def save_vector_store(embeddings: np.ndarray, texts: List[str], folder: str):
    os.makedirs(folder, exist_ok=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(folder, INDEX_FILENAME))

    with open(os.path.join(folder, DOCS_FILENAME), "w", encoding="utf-8") as f:
        f.write(SEP.join(texts))

def load_vector_store(folder: str):
    index = faiss.read_index(os.path.join(folder, INDEX_FILENAME))
    with open(os.path.join(folder, DOCS_FILENAME), "r", encoding="utf-8") as f:
        texts = f.read().split(SEP)
    return index, texts

def _retrieve(query: str, index, texts: List[str], k: int = 4) -> List[str]:
    query_emb = _encode([query])
    distances, indices = index.search(query_emb, k)
    return [texts[i] for i in indices[0] if 0 <= i < len(texts)]

def answer_question(question: str, index, texts: List[str], top_k: int = 4) -> str:
    contexts = _retrieve(question, index, texts, k=top_k)
    if not contexts:
        return "I couldn't find relevant information in the document."

    context_blob = "\n\n---\n\n".join(contexts)
    prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not found in the document, say "I don't know."

Context:
{context_blob}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Answer only from the context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

