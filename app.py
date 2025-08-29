import faiss
import json
import numpy as np
import torch
import re
import os
import google.generativeai as genai
# Import the specific exception for rate limiting
from google.api_core import exceptions
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

# --- Configuration ---
INDEX_DIR = "data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Configure Gemini API ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    print("[*] Gemini API key configured successfully.")
except KeyError:
    print("[ERROR] GOOGLE_API_KEY environment variable not set.")
    print("         Please set it by running 'export GOOGLE_API_KEY=...'")
    exit()

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- Load Models and Data (once, on startup) ---
print("[*] Loading local models and data. This may take a moment...")
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

# --- [THE FIX] Switched to the model with a more generous free tier ---
gen_model = genai.GenerativeModel('gemini-1.5-flash-latest')
print("[*] Gemini 1.5 Flash model initialized.")
print("[*] All models loaded.")


FEW_SHOT_PROMPT = """Based *only* on the Context below, generate exactly 12 exam questions with answers.

Strictly follow the format of the examples below. Generate 4 Basic (B), 4 Moderate (M), and 4 Advanced (A) questions.

--- EXAMPLES ---
B1) Who was assassinated in Sarajevo in 1914? — Answer: Archduke Franz Ferdinand
M1) Explain the significance of the Schlieffen Plan. — Answer: It was Germany's strategy for a two-front war, aiming to defeat France quickly before Russia could mobilize.
A1) Analyse the primary reasons for the failure of the League of Nations. — Answer: Key weaknesses included the absence of major powers like the USA, lack of an army, and decisions requiring unanimous consent.
--- END EXAMPLES ---

Context:
{context}

--- QUESTIONS ---
"""

# --- Helper functions (no changes needed here) ---
def clean_text(text):
    text = re.sub(r'\S+\.indd\s*\d*', '', text)
    text = re.sub(r'\d{2}-\d{2}-\d{4}', '', text)
    text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def retrieve_context(topic_text, top_k=4):
    q_emb = embed_model.encode(topic_text, convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(np.array([q_emb]), top_k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx != -1:
            results.append({"score": float(score), "text": texts[idx], **meta[idx]})
    return results

def build_prompt(context_chunks):
    context = "\n\n".join([c["text"] for c in context_chunks])
    if len(context) > 10000:
        context = context[:10000]
    return FEW_SHOT_PROMPT.format(context=context)

def parse_output(generated_text):
    qas = []
    pattern = re.compile(
        r'^(B|M|A)\s?(\d+)\)?\.?\s*(.*?)\s*(?:—|–|-)\s*(?:Answer:)?\s*(.*)',
        re.IGNORECASE | re.MULTILINE
    )
    matches = pattern.finditer(generated_text)
    for match in matches:
        q_type, q_id, question, answer = [s.strip() for s in match.groups()]
        if question and answer:
            difficulty_map = {"B": "Basic", "M": "Moderate", "A": "Advanced"}
            qas.append({
                "id": f"{q_type.upper()}{q_id}",
                "difficulty": difficulty_map[q_type.upper()],
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

    context_chunks = retrieve_context(topic)
    if not context_chunks:
        return jsonify({"questions": [], "error": "Could not find relevant context for this topic in the PDF."})

    for chunk in context_chunks:
        chunk["text"] = clean_text(chunk["text"])

    prompt = build_prompt(context_chunks)

    try:
        print("[*] Sending request to Gemini API...")
        response = gen_model.generate_content(prompt)
        generated_text = response.text
    # --- [IMPROVEMENT] Catch the specific rate limit error ---
    except exceptions.ResourceExhausted as e:
        print(f"[ERROR] Rate limit exceeded: {e}")
        return jsonify({"error": "Rate limit exceeded. Please wait a minute and try again."}), 429
    except Exception as e:
        print(f"[ERROR] An unexpected API error occurred: {e}")
        return jsonify({"error": f"An unexpected API error occurred: {e}"}), 500

    parsed_qas = parse_output(generated_text)

    if not parsed_qas:
        print("\n[!] The model did not generate any parsable questions. Raw output below:\n")
        print(generated_text)

    return jsonify({"questions": parsed_qas})

if __name__ == "__main__":
    print("[SUCCESS] Backend server is ready to accept requests.")
    app.run(host="0.0.0.0", port=5000, debug=False)