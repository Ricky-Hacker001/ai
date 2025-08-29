import faiss
import json
import numpy as np
import argparse
import torch
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# The refined prompt template with few-shot examples for better results
FEWSHOT_PROMPT = """You are a strict exam-writer. ONLY use facts from the Context. Do NOT invent facts.

Context:
{context}

Example (format to follow exactly):
B1) Who was assassinated in Sarajevo in 1914? — Answer: Archduke Franz Ferdinand
M1) Explain why imperial rivalry increased before WWI. — Answer: European powers competed for colonies and raw materials.
A1) Analyse how the Treaty of Versailles destabilised Europe. — Answer: Reparations and territorial losses fostered resentment and economic hardship.

Now, generate 12 NEW questions (B1–B4, M1–M4, A1–A4) in the SAME format, using ONLY the Context provided.
"""

def load_index_and_meta(index_dir):
    """Loads the FAISS index, metadata, and text chunks."""
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "chunks_index.faiss"))
    with open(index_dir / "chunks_meta.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return index, data["meta"], data["texts"]

def retrieve_context(topic_text, embed_model, index, texts, meta, top_k=4):
    """Retrieves the most relevant text chunks for a given topic."""
    q_emb = embed_model.encode(topic_text, convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(np.array([q_emb]), top_k)
    
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx != -1: # FAISS can return -1 if index is smaller than top_k
            results.append({
                "score": float(score),
                "text": texts[idx],
                **meta[idx]
            })
    return results

def build_prompt(context_chunks):
    """Builds the final prompt by injecting the retrieved context."""
    context = "\n\n".join([c["text"] for c in context_chunks])
    # Truncate context if it's too long for the model
    if len(context) > 10000:
        context = context[:10000]
    return FEWSHOT_PROMPT.format(context=context)

def parse_output(generated_text):
    """Parses the model's raw output into a structured list of Q&A."""
    pattern = re.compile(r'^(B|M|A)(\d+)\)\s*(.*?)\s*—\s*Answer:\s*(.*)$', re.UNICODE)
    qas = []
    for line in generated_text.splitlines():
        match = pattern.match(line.strip())
        if match:
            difficulty_map = {"B": "Basic", "M": "Moderate", "A": "Advanced"}
            q_type, q_id, question, answer = match.groups()
            qas.append({
                "id": f"{q_type}{q_id}",
                "difficulty": difficulty_map[q_type],
                "question": question.strip(),
                "answer": answer.strip()
            })
    return qas

def main():
    parser = argparse.ArgumentParser(description="Generate questions from a processed PDF index.")
    parser.add_argument("--topic_query", type=str, required=True, help="The topic to generate questions about.")
    parser.add_argument("--index_dir", type=str, default="data", help="Directory containing the FAISS index and metadata.")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base", help="Hugging Face model to use for generation.")
    parser.add_argument("--top_k", type=int, default=4, help="Number of text chunks to retrieve for context.")
    args = parser.parse_args()

    # --- 1. Load Models and Data ---
    print("[*] Loading models and data...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}")

    try:
        index, meta, texts = load_index_and_meta(args.index_dir)
    except FileNotFoundError:
        print(f"[ERROR] Index not found in '{args.index_dir}'. Did you run 'process_pdf.py' first?")
        return

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    gen_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    # --- 2. Retrieve and Build Prompt ---
    print(f"[*] Retrieving context for topic: '{args.topic_query}'")
    context_chunks = retrieve_context(args.topic_query, embed_model, index, texts, meta, args.top_k)
    
    if not context_chunks:
        print("[ERROR] Could not retrieve any relevant context for the given topic. Try a different query.")
        return
        
    prompt = build_prompt(context_chunks)

    # --- 3. Generate Questions ---
    print("[*] Generating questions... (This can take a moment)")
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    outputs = gen_model.generate(
        **inputs,
        max_length=768,
        num_beams=6,
        early_stopping=True
    )
    generated_text = gen_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # --- 4. Parse and Display Output ---
    parsed_qas = parse_output(generated_text)
    
    print("\n" + "="*20 + " GENERATED QUESTIONS " + "="*20)
    if not parsed_qas:
        print("\n[!] The model did not generate any parsable questions. Raw output below:\n")
        print(generated_text)
    else:
        for qa in parsed_qas:
            print(f"\n[{qa['difficulty']} - {qa['id']}]")
            print(f"  Q: {qa['question']}")
            print(f"  A: {qa['answer']}")
    print("\n" + "="*59)

if __name__ == "__main__":
    main()