from datetime import datetime

def clean_title(text):
    # Remove overly long paragraph-like sentences from section_title
    lines = text.strip().split('\n')
    cleaned = lines[0]
    if len(cleaned.split()) > 20:
        # If too long, try to extract title-like substring
        cleaned = cleaned[:150].rsplit('.', 1)[0]  # Cut after last full stop
    return cleaned.strip("•:- ").replace('\n', ' ')

def generate_output(metadata, top_sections, refined_spans):
    seen = set()
    extracted_sections = []
    sub_section_analysis = []

    for s, refined_text in zip(top_sections, refined_spans):
        # Deduplicate based on document + page_number
        key = (s["document"], s["page_number"])
        if key in seen:
            continue
        seen.add(key)

        # Clean/abstract titles
        title = clean_title(s.get("section_title", ""))

        extracted_sections.append({
            "document": s["document"],
            "page_number": s["page_number"],
            "section_title": title,
            "importance_rank": s["importance_rank"]
        })

        sub_section_analysis.append({
            "document": s["document"],
            "page_number": s["page_number"],
            "refined_text": refined_text
        })

    return {
        "metadata": {
            "input_documents": list({s['document'] for s in top_sections}),
            "persona": metadata["persona"],
            "job_to_be_done": metadata["job_to_be_done"],
            "timestamp": str(datetime.now())
        },
        "extracted_sections": extracted_sections,
        "sub_section_analysis": sub_section_analysis
    }
