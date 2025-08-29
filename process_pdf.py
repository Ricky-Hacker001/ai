import fitz  # PyMuPDF
import re
import json
import faiss
import numpy as np
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of a PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    print(f"[*] Extracting text from {len(doc)} pages in '{pdf_path.name}'...")
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        pages.append({"page": i + 1, "text": text})
    return pages

def chunk_text(pages):
    """Splits the combined text into meaningful chunks based on headings."""
    combined_text = "\n\n".join([f"[PAGE:{p['page']}]\n{p['text']}" for p in pages])
    
    # A robust regex to find headings (e.g., "Chapter 1", "1. Introduction", "MAIN HEADING")
    heading_regex = re.compile(
        r"(?m)^(Chapter\s+\d+|CHAPTER\s+\d+|[0-9]+\.\s+[A-Z][^\n]{0,80}|[A-Z][A-Z ]{6,})\s*$"
    )
    matches = list(heading_regex.finditer(combined_text))
    print(f"[*] Found {len(matches)} potential headings. Chunking text...")

    chunks = []
    if not matches:
        print("[!] No headings found. Using fallback sliding window chunker.")
        return chunk_text_sliding(combined_text)

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(combined_text)
        chunk_text = combined_text[start:end].strip()
        
        page_marker = re.search(r"\[PAGE:(\d+)\]", chunk_text)
        page_num = int(page_marker.group(1)) if page_marker else None
        
        heading_line = match.group(0).strip()
        
        # Clean the text
        cleaned_text = re.sub(r"\[PAGE:\d+\]\n?", "", chunk_text).strip()
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

        if len(cleaned_text) > 100: # Filter out very short chunks
            chunks.append({
                "chunk_id": len(chunks),
                "heading": heading_line,
                "page": page_num,
                "text": cleaned_text
            })
    return chunks

def chunk_text_sliding(text, max_words=300, overlap=60):
    """A fallback chunker that uses a sliding window."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + max_words]
        chunk_text = " ".join(chunk_words)
        
        page_marker = re.search(r"\[PAGE:(\d+)\]", chunk_text)
        page_num = int(page_marker.group(1)) if page_marker else None

        cleaned_text = re.sub(r"\[PAGE:\d+\]\s*", "", chunk_text).strip()
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

        if len(cleaned_text) > 100:
            chunks.append({
                "chunk_id": len(chunks),
                "heading": f"Chunk starting at word {i}",
                "page": page_num,
                "text": cleaned_text
            })
        i += max_words - overlap
    return chunks

def create_embeddings_and_index(chunks, model):
    """Creates vector embeddings and a FAISS index."""
    print("[*] Creating text embeddings... (This may take a while)")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Using Inner Product for cosine similarity
    index.add(embeddings)
    
    print(f"[*] Created FAISS index with {index.ntotal} vectors.")
    return index, texts

def main():
    parser = argparse.ArgumentParser(description="Process a PDF to create a searchable question-generation index.")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to the PDF file to process.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save the index and metadata.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    output_dir = Path(args.output_dir)

    if not pdf_path.exists():
        print(f"[ERROR] PDF file not found at: {pdf_path}")
        return

    output_dir.mkdir(exist_ok=True)

    # 1. Process PDF
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pages)

    if not chunks:
        print("[ERROR] No valid text chunks could be created. Exiting.")
        return

    # 2. Create Embeddings and Index
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    index, texts = create_embeddings_and_index(chunks, embedding_model)

    # 3. Save Artifacts
    meta = [{"chunk_id": c["chunk_id"], "page": c["page"], "heading": c["heading"]} for c in chunks]
    
    faiss.write_index(index, str(output_dir / "chunks_index.faiss"))
    
    with open(output_dir / "chunks_meta.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "texts": texts}, f, ensure_ascii=False)

    print(f"\n[SUCCESS] Processing complete. Index and metadata saved in '{output_dir}' directory.")

if __name__ == "__main__":
    main()