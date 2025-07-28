# src/persona_detector.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample personas & jobs (can expand later)
PREDEFINED_PROFILES = [
    {
        "persona": "Travel Planner",
        "job": "Plan a trip of 4 days for a group of 10 college friends."
    },
    {
        "persona": "HR professional",
        "job": "Create and manage fillable forms for onboarding and compliance."
    },
    {
        "persona": "Food Contractor",
        "job": "Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items."
    },
    {
        "persona": "PhD Researcher",
        "job": "Extract related work and methods section from recent academic publications."
    }
]

def detect_persona_job(sample_texts):
    input_embedding = model.encode(" ".join(sample_texts[:300]), normalize_embeddings=True)

    best_score = -1
    best_match = None

    for profile in PREDEFINED_PROFILES:
        query = f"Persona: {profile['persona']}\nJob: {profile['job']}"
        profile_embedding = model.encode(query, normalize_embeddings=True)
        score = cosine_similarity([input_embedding], [profile_embedding])[0][0]
        if score > best_score:
            best_score = score
            best_match = profile

    return best_match["persona"], best_match["job"]
