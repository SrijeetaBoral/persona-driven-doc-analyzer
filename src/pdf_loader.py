import fitz  # PyMuPDF
import os
from persona_parser import encode_text  # Reuse your existing model’s encoder

def extract_segments_from_pdf(file_path, file_name):
    doc = fitz.open(file_path)
    segments = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")

        for block in blocks:
            text = block[4].strip()
            if text:
                embedding = encode_text(text)  # Compute embedding here
                segments.append({
                    "document": file_name,
                    "page": page_num,               # Use 0-based internally, will be adjusted later
                    "text": text,
                    "embedding": embedding
                })

    return segments

def load_all_pdfs(pdf_dir):
    all_segments = []
    for file in os.listdir(pdf_dir):
        if file.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, file)
            segments = extract_segments_from_pdf(path, file)
            all_segments.extend(segments)
    return all_segments
