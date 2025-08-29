import faiss
import json
import numpy as np
import torch
import re
import os
import google.generativeai as genai
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
    exit()

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- Load Models and Data ---
print("[*] Loading local models and data...")
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
    index, meta, texts = None, None, None

embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
gen_model = genai.GenerativeModel('gemini-1.5-flash-latest')
print("[*] Gemini 1.5 Flash model initialized.")
print("[*] All models loaded.")

# --- Prompt for MCQs ---
FEW_SHOT_MCQ_PROMPT = """Based *only* on the Context, generate exactly 12 Multiple Choice Questions (MCQs) with four options each.
Clearly mark the correct answer. Separate each MCQ block with '---'.

--- EXAMPLE ---
Q: Who was assassinated in Sarajevo in 1914, sparking WWI?
A) Tsar Nicholas II
B) Archduke Franz Ferdinand
C) Kaiser Wilhelm II
D) Winston Churchill
Correct: B) Archduke Franz Ferdinand
---
Q: What was the primary goal of the Schlieffen Plan?
A) To achieve a quick victory against Russia
B) To establish a naval blockade of Britain
C) To avoid a two-front war by defeating France quickly
D) To encourage the USA to join the war
Correct: C) To avoid a two-front war by defeating France quickly
---

Context:
{context}

--- QUESTIONS ---
"""

# --- Helper Functions ---
def clean_text(text):
    text = re.sub(r'\S+\.indd\s*\d*', '', text)
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
    if len(context) > 15000:
      context = context[:15000]
    return FEW_SHOT_MCQ_PROMPT.format(context=context)

# --- Parser for MCQ Format ---
def parse_mcq_output(generated_text):
    mcqs = []
    question_blocks = generated_text.strip().split('---')
    
    difficulty_levels = ["Basic"] * 4 + ["Moderate"] * 4 + ["Advanced"] * 4
    ids = [f"B{i}" for i in range(1, 5)] + [f"M{i}" for i in range(1, 5)] + [f"A{i}" for i in range(1, 5)]

    for i, block in enumerate(question_blocks):
        if not block.strip() or i >= 12:
            continue

        question_match = re.search(r'Q:\s*(.*)', block, re.IGNORECASE)
        
        # --- [THE FIX] ---
        # Added ^ to anchor the search to the start of a line and re.MULTILINE flag.
        # This prevents the "Correct:" line from being matched as an option.
        options_matches = re.findall(r'^([A-D])\)\s*(.*)', block, re.MULTILINE)
        
        correct_match = re.search(r'Correct:\s*([A-D])\)\s*(.*)', block, re.IGNORECASE)

        if question_match and len(options_matches) == 4 and correct_match:
            question = question_match.group(1).strip()
            options = [opt[1].strip() for opt in options_matches]
            correct_answer = correct_match.group(2).strip()

            mcqs.append({
                "id": ids[i],
                "difficulty": difficulty_levels[i],
                "question": question,
                "options": options,
                "answer": correct_answer
            })
    return mcqs

# --- API Endpoint ---
@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if not all([index, meta, texts]):
        return jsonify({"error": "Backend is not ready."}), 500

    data = request.json
    topic = data.get("topic", "")
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    print(f"\n[*] Received new request for topic: '{topic}'")

    context_chunks = retrieve_context(topic)
    if not context_chunks:
        return jsonify({"mcqs": [], "error": "Could not find relevant context for this topic."})

    for chunk in context_chunks:
        chunk["text"] = clean_text(chunk["text"])

    prompt = build_prompt(context_chunks)

    try:
        print("[*] Sending request to Gemini API...")
        response = gen_model.generate_content(prompt)
        generated_text = response.text
    except exceptions.ResourceExhausted as e:
        print(f"[ERROR] Rate limit exceeded: {e}")
        return jsonify({"error": "Rate limit exceeded. Please wait a minute and try again."}), 429
    except Exception as e:
        print(f"[ERROR] An unexpected API error occurred: {e}")
        return jsonify({"error": f"An unexpected API error occurred: {e}"}), 500

    parsed_mcqs = parse_mcq_output(generated_text)
    
    if not parsed_mcqs:
        print("\n[!] The model did not generate any parsable MCQs. Raw output below:\n")
        print(generated_text)
    else:
        print(f"\n[*] Successfully parsed {len(parsed_mcqs)} MCQs.")

    return jsonify({"mcqs": parsed_mcqs})

if __name__ == "__main__":
    print("[SUCCESS] Backend server is ready to accept requests.")
    app.run(host="0.0.0.0", port=5000, debug=False)