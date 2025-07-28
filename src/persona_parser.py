from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def encode_query(persona: str, job: str):
    full_prompt = f"Persona: {persona}\nJob to be done: {job}"
    return model.encode(full_prompt, normalize_embeddings=True)

def encode_text(text: str):
    return model.encode(text, normalize_embeddings=True)
