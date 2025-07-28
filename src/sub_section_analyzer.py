from sentence_transformers import util

def extract_best_span(text: str, query_embedding, model, window_size: int = 3) -> str:
    """
    Selects the most relevant chunk of up to `window_size` consecutive lines based on semantic similarity.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    spans = []

    for i in range(len(lines)):
        span = " ".join(lines[i:i + window_size])
        spans.append((span, model.similarity(query_embedding, model.encode(span))))

    if not spans:
        return text[:200]

    best_span = max(spans, key=lambda x: x[1])[0]
    return best_span.strip()
