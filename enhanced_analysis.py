# enhanced_analysis.py
from enhanced_parsing import parse_resume_sections, parse_job_description
from embeddings_utils import compute_semantic_similarity
fuzz = None
try:
    # Prefer rapidfuzz (faster, pure-python wheel available)
    from rapidfuzz import fuzz as _rf_fuzz
    fuzz = _rf_fuzz
except Exception:
    # Try to import fuzzywuzzy dynamically at runtime to avoid static LSP errors
    try:
        import importlib
        _fw = importlib.import_module('fuzzywuzzy.fuzz')
        fuzz = _fw
    except Exception:
        # Lightweight fallback for partial_ratio when neither package is installed
        class _SimpleFuzz:
            @staticmethod
            def partial_ratio(a, b):
                if not a or not b:
                    return 0
                a = a.lower()
                b = b.lower()
                # exact substring heuristic
                if a in b or b in a:
                    return 100
                # token overlap heuristic
                a_tokens = set(a.split())
                b_tokens = set(b.split())
                if not a_tokens or not b_tokens:
                    return 0
                overlap = len(a_tokens & b_tokens)
                # normalize by smaller token count to approximate partial match
                score = int(100 * overlap / min(len(a_tokens), len(b_tokens)))
                return min(100, max(0, score))

        fuzz = _SimpleFuzz()
import numpy as np

import spacy
nlp = spacy.load("en_core_web_sm")

def hard_match_skills(resume_skills, jd_must_have, jd_good_to_have, threshold=80):
    """Perform fuzzy matching of skills"""
    must_have_matches = []
    good_to_have_matches = []
    missing_must_have = []
    missing_good_to_have = []
    
    # Check must-have skills
    for jd_skill in jd_must_have:
        matched = False
        for resume_skill in resume_skills:
            # Fuzzy match with threshold
            if fuzz.partial_ratio(jd_skill.lower(), resume_skill.lower()) >= threshold:
                must_have_matches.append(jd_skill)
                matched = True
                break
        
        if not matched:
            missing_must_have.append(jd_skill)
    
    # Check good-to-have skills
    for jd_skill in jd_good_to_have:
        matched = False
        for resume_skill in resume_skills:
            # Fuzzy match with threshold
            if fuzz.partial_ratio(jd_skill.lower(), resume_skill.lower()) >= threshold:
                good_to_have_matches.append(jd_skill)
                matched = True
                break
        
        if not matched:
            missing_good_to_have.append(jd_skill)
    
    # Calculate scores
    must_have_score = (len(must_have_matches) / len(jd_must_have)) * 100 if jd_must_have else 0
    good_to_have_score = (len(good_to_have_matches) / len(jd_good_to_have)) * 50 if jd_good_to_have else 0
    
    return {
        'hard_match_score': min(must_have_score + good_to_have_score, 100),
        'must_have_matches': must_have_matches,
        'good_to_have_matches': good_to_have_matches,
        'missing_must_have': missing_must_have,
        'missing_good_to_have': missing_good_to_have
    }

def analyze_resume_jd_comprehensive(resume_text, jd_text):
    """Comprehensive analysis of resume against job description"""
    # Parse resume and JD
    resume_data = parse_resume_sections(resume_text)
    jd_data = parse_job_description(jd_text)
    
    # Extract skills from resume (combine skills and technologies from experience)
    resume_skills = resume_data.get('skills', [])
    
    # Add technologies from experience section
    experience_text = " ".join(resume_data.get('experience', []))
    doc = nlp(experience_text)
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT'] and ent.text not in resume_skills:
            resume_skills.append(ent.text)
    
    # Perform hard matching
    hard_match_result = hard_match_skills(
        resume_skills, 
        jd_data.get('must_have_skills', []), 
        jd_data.get('good_to_have_skills', [])
    )
    
    # Perform semantic matching
    semantic_score = compute_semantic_similarity(resume_text, jd_text)
    
    # Calculate final score (weighted average)
    hard_score = hard_match_result['hard_match_score']
    final_score = round((hard_score * 0.6) + (semantic_score * 0.4), 2)
    
    # Determine verdict
    if final_score >= 80:
        verdict = "High"
    elif final_score >= 60:
        verdict = "Medium"
    else:
        verdict = "Low"
    
    # Generate suggestions
    missing_skills = hard_match_result['missing_must_have'] + hard_match_result['missing_good_to_have']
    suggestions = generate_suggestions(resume_text, jd_text, missing_skills)
    
    return {
        'score': final_score,
        'verdict': verdict,
        'hard_match_score': hard_score,
        'semantic_match_score': semantic_score,
        'missing_skills': missing_skills,
        'suggestions': suggestions,
        'resume_data': resume_data,
        'jd_data': jd_data
    }

def generate_suggestions(resume_text, jd_text, missing_skills):
    """Generate improvement suggestions using LLM"""
    from gemini_utils import get_gemini_response
    
    prompt = f"""
    Based on the resume and job description, provide specific suggestions for the candidate to improve their match.
    
    Job Description: {jd_text[:1000]}
    Missing Skills: {', '.join(missing_skills)}
    
    Provide 3-5 actionable suggestions including specific skills to learn, projects to undertake, or certifications to pursue.
    Keep the response concise and focused.
    """
    
    return get_gemini_response(prompt, resume_text[:2000], "Generate improvement suggestions")