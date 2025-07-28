from pdf_loader import load_all_pdfs
from persona_parser import encode_query,model
from section_ranker import rank_segments
from sub_section_analyzer import extract_best_span
from generate_output import generate_output
from persona_detector import detect_persona_job # ✅ Added

import re
import json
import os
from datetime import datetime

# ==== CONFIG ====
PDF_DIR = "data"
OUTPUT_FILE = "challenge1b_output.json"
TOP_K = 10

# ==== PIPELINE ====
segments = load_all_pdfs(PDF_DIR)

# Ensure metadata includes all PDFs in input folder
all_input_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])

# ==== ✅ Step 1: Dynamically detect persona and job ====
sample_texts = [s["text"] for s in segments[:10]]
persona, job = detect_persona_job(sample_texts)

# ==== Step 2: Embed query ====
query_embedding = encode_query(persona, job)

# ==== Step 3: Rank segments ====
scored_segments = rank_segments(segments, query_embedding, model, top_k=9999)

# Top K globally relevant
top_k_global = scored_segments[:TOP_K]

# Ensure 1 best segment per document is included
best_by_doc = {}
for seg in scored_segments:
    fname = seg["document"]
    if fname not in best_by_doc:
        best_by_doc[fname] = seg

# Merge top-K and one-per-doc (avoid duplicates)
final_segments = {id(s): s for s in top_k_global}
for s in best_by_doc.values():
    final_segments[id(s)] = s

# Sort and assign importance
final_sorted = sorted(final_segments.values(), key=lambda x: x["score"], reverse=True)
for i, seg in enumerate(final_sorted):
    seg["importance_rank"] = i + 1
    seg["page_number"] = seg.pop("page") + 1

# Refined sub-spans
refined_spans = [
    extract_best_span(s["text"], query_embedding, model) for s in final_sorted
]





def extract_short_title(text: str) -> str:
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    candidates = []

    for line in lines:
        # Ignore overly short or long lines
        if len(line) < 5 or len(line) > 80:
            continue
        # Ignore steps or procedural lines
        if re.match(r'^\d+\.', line) or re.match(r'^step\s+\d+', line.lower()):
            continue
        if re.search(r'\b(click|press|submit|type|select|enter|fill|add|check|choose|heat|cook)\b', line.lower()):
            continue
        # Accept lines with 2+ capitalized words as probable headings
        if sum(1 for w in line.split() if w[0].isupper()) >= 2:
            candidates.append(line)

    if candidates:
        return candidates[0].rstrip(":")

    # Fallback: short, clean starting sentence
    fallback = re.split(r'[.?!]', text.strip())[0]
    return fallback.strip()[:80]





# Assign clean dynamic section titles
for seg in final_sorted:
    raw_text = seg.get("text", "")
    seg["section_title"] = extract_short_title(raw_text).strip()


refined_spans = [s.replace('\n', ' ').strip() for s in refined_spans]

# Metadata
metadata = {
    "persona": persona,
    "job_to_be_done": job,
    "input_documents": all_input_files,
    "timestamp": str(datetime.now())
}

# Output
output_json = generate_output(metadata, final_sorted, refined_spans)

# Save to file
os.makedirs("output", exist_ok=True)
with open(os.path.join("output", OUTPUT_FILE), "w", encoding="utf-8") as f:
    f.write(json.dumps(output_json, indent=2, ensure_ascii=False))

print(f"✅ Output saved to output/{OUTPUT_FILE}")
