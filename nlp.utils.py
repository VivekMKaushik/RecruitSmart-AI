import spacy
nlp = spacy.load("en_core_web_sm")

def parse_resume_text(text):
    doc = nlp(text)
    skills = set()
    edu = []
    exp = []
    # Simple: collect ORG, PERSON, DATE etc. and keywords as skill candidates
    for ent in doc.ents:
        if ent.label_ in ("ORG", "GPE", "PERSON", "DATE", "EDUCATION"): 
            # use as needed
            pass
    # Use patterns or regex to detect sections like 'Skills', 'Experience'
    return {"skills": list(skills), "education": edu, "experience": exp}
