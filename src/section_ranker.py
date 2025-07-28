from sentence_transformers import util
import re

def is_potentially_relevant(text: str) -> bool:
    if len(text) < 30:
        return False
    if re.match(r"^(Figure|Table|Supplementary)", text, re.IGNORECASE):
        return False
    if re.search(r"\d{4}.*(et al\.|http|bioinformatics|doi|vol|issue)", text, re.IGNORECASE):
        return False
    if re.search(r"(Department of|University|Lab|Faculty)", text):
        return False
    return True

def is_high_priority(text: str, keywords: list) -> bool:
    return any(kw.lower() in text.lower() for kw in keywords)

def rank_segments(segments, query_embedding, model, top_k=10):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    scored = []
    for seg in segments:
        score = cosine_similarity([query_embedding], [seg["embedding"]])[0][0]

        # ✅ Use text field since section_title isn't available yet
        penalty = 0.15 if seg["text"].strip().startswith("•") else 0.0
        length_bonus = 0.05 if len(seg["text"].split()) > 4 else 0.0

        seg["score"] = score - penalty + length_bonus
        scored.append(seg)

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
