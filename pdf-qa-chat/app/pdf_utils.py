import pdfplumber
import re

def extract_text_from_pdf(file_path: str) -> str:
    """Extract plain text from a PDF using pdfplumber with basic fallbacks."""
    parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if not txt:
                # fallback: join words if extract_text() fails
                words = page.extract_words() or []
                txt = " ".join(w["text"] for w in words)
            parts.append(txt)
    full = "\n".join(parts)
    # normalize whitespace
    full = re.sub(r"[ \t]+", " ", full)
    full = re.sub(r"\n{3,}", "\n\n", full).strip()
    return full

def split_into_chunks(text: str, max_chars: int = 1500, overlap: int = 200):
    """
    Simple, robust chunking by sentences with overlap.
    max_chars ~ roughly ~500-600 tokens per chunk (enough context, faster embeddings).
    """
    # Split into sentences-ish
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""

    for s in sentences:
        if not s.strip():
            continue
        if len(cur) + len(s) + 1 <= max_chars:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                chunks.append(cur)
            # start new chunk with overlap from previous tail
            if overlap and chunks:
                tail = chunks[-1][-overlap:]
                cur = (tail + " " + s).strip()
            else:
                cur = s.strip()

    if cur:
        chunks.append(cur)

    # safety: drop empty or tiny chunks
    chunks = [c for c in chunks if len(c) > 20]
    return chunks

