import faiss
import json
import numpy as np
import torch
import re
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration ---
INDEX_DIR = "data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Upgraded to a more powerful model for better accuracy
GENERATION_MODEL_NAME = "google/flan-t5-large"

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)  # Allow frontend to call the backend

# --- Load Models and Data (once, on startup) ---
print("[*] Loading models and data. This may take a moment...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Using device: {device}")

try:
    index_path = Path(INDEX_DIR)
    index = faiss.read_index(str(index_path / "chunks_index.faiss"))
    with open(index_path / "chunks_meta.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data["meta"]
    texts = data["texts"]
    print("[*] FAISS index and metadata loaded successfully.")
except FileNotFoundError:
    print(f"[ERROR] Index not found in '{INDEX_DIR}'. Please run 'process_pdf.py' first.")
    index, meta, texts = None, None, None

embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"[*] Loading generation model '{GENERATION_MODEL_NAME}'... (This may take some time and memory)")
gen_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME).to(device)
print("[*] All models loaded.")


# --- Prompt Template (Revised Structure) ---
REVISED_PROMPT = """You are a strict exam-writer. ONLY use facts from the Context provided. Do NOT invent facts.

Follow this example format exactly:
B1) Who was assassinated in Sarajevo in 1914? — Answer: Archduke Franz Ferdinand
M1) Explain why imperial rivalry increased before WWI. — Answer: European powers competed for colonies and raw materials.
A1) Analyse how the Treaty of Versailles destabilised Europe. — Answer: Reparations and territorial losses fostered resentment and economic hardship.

Context:
{context}

Based *only* on the context above, generate 12 NEW questions (B1–B4, M1–M4, A1–A4) in the same format as the example.
"""

# --- Helper Functions ---
def clean_text(text):
    """Removes common noise and artifacts from extracted PDF text."""
    # Remove .indd filenames and similar patterns
    text = re.sub(r'\S+\.indd\s*\d*', '', text)
    # Remove dates in DD-MM-YYYY format
    text = re.sub(r'\d{2}-\d{2}-\d{4}', '', text)
    # Remove times in HH:MM:SS format
    text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def retrieve_context(topic_text, top_k=4):
    """Retrieves the most relevant text chunks for a given topic."""
    q_emb = embed_model.encode(topic_text, convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(np.array([q_emb]), top_k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx != -1:
            results.append({"score": float(score), "text": texts[idx], **meta[idx]})
    return results

def build_prompt(context_chunks):
    """Builds the final prompt by injecting the retrieved context."""
    # The context is now built from cleaned text
    context = "\n\n".join([c["text"] for c in context_chunks])
    if len(context) > 10000:
        context = context[:10000]
    return REVISED_PROMPT.format(context=context)

def parse_output(generated_text):
    """
    Parses the model's raw output into a structured list of Q&A.
    """
    qas = []
    question_pattern = re.compile(r'^(B|M|A)\s?(\d+)\)?\.?\s*(.*)', re.IGNORECASE)

    for line in generated_text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = re.split(r'\s*(?:—|-|–)\s*Answer:\s*', line, maxsplit=1, flags=re.IGNORECASE)

        if len(parts) == 2:
            question_part, answer = parts
            answer = answer.strip()
            
            match = question_pattern.match(question_part)
            if match:
                difficulty_map = {"B": "Basic", "M": "Moderate", "A": "Advanced"}
                q_type = match.group(1).upper()
                q_id = match.group(2)
                question = match.group(3).strip()

                if question and answer:
                    qas.append({
                        "id": f"{q_type}{q_id}",
                        "difficulty": difficulty_map[q_type],
                        "question": question,
                        "answer": answer
                    })
    return qas

# --- API Endpoint ---
@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if not all([index, meta, texts]):
        return jsonify({"error": "Backend is not ready. Index file not found."}), 500

    data = request.json
    topic = data.get("topic", "")

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    print(f"\n[*] Received new request for topic: '{topic}'")

    # 1. Retrieve and Clean Prompt
    context_chunks = retrieve_context(topic)
    
    if not context_chunks:
        return jsonify({"questions": [], "error": "Could not find relevant context for this topic in the PDF."})
    
    # Clean the text of each retrieved chunk
    for chunk in context_chunks:
        chunk["text"] = clean_text(chunk["text"])

    prompt = build_prompt(context_chunks)

    # 2. Generate Questions
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = gen_model.generate(
        **inputs, max_length=768, num_beams=6, early_stopping=True
    )
    generated_text = gen_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # 3. Parse and Return
    parsed_qas = parse_output(generated_text)
    
    if not parsed_qas:
        print("\n[!] The model did not generate any parsable questions. Raw output below:\n")
        print(generated_text)

    return jsonify({"questions": parsed_qas})

if __name__ == "__main__":
    print("[SUCCESS] Backend server is ready to accept requests.")
    app.run(host="0.0.0.0", port=5000, debug=False)

