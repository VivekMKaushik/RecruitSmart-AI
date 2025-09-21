from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from embeddings_utils import compute_semantic_similarity
import numpy as np
import re


def hard_match_score(resume_text, jd_text, must_have_skills, good_to_have_skills):
    """Calculate hard match score based on keyword presence"""
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    
    # Check must-have skills
    must_have_matches = 0
    for skill in must_have_skills:
        if skill.lower() in resume_lower:
            must_have_matches += 1
    
    # Check good-to-have skills
    good_to_have_matches = 0
    for skill in good_to_have_skills:
        if skill.lower() in resume_lower:
            good_to_have_matches += 1
    
    # Calculate scores
    must_have_score = (must_have_matches / len(must_have_skills)) * 100 if must_have_skills else 0
    good_to_have_score = (good_to_have_matches / len(good_to_have_skills)) * 50 if good_to_have_skills else 0
    
    return min(must_have_score + good_to_have_score, 100)


def semantic_match_score(resume_text, jd_text):
    """Calculate semantic similarity score"""
    return compute_semantic_similarity(resume_text, jd_text)


def calculate_final_score(hard_score, semantic_score, weights=(0.6, 0.4)):
    """Calculate weighted final score"""
    return round((hard_score * weights[0]) + (semantic_score * weights[1]), 2)


def get_verdict(score):
    """Get suitability verdict based on score"""
    if score >= 80:
        return "High"
    elif score >= 60:
        return "Medium"
    else:
        return "Low"


def find_missing_elements(resume_text, must_have_skills, good_to_have_skills):
    """Find missing skills and qualifications"""
    resume_lower = resume_text.lower()
    
    missing_must_have = []
    for skill in must_have_skills:
        if skill.lower() not in resume_lower:
            missing_must_have.append(skill)
    
    missing_good_to_have = []
    for skill in good_to_have_skills:
        if skill.lower() not in resume_lower:
            missing_good_to_have.append(skill)
    
    return missing_must_have, missing_good_to_have


def generate_suggestions(resume_text, jd_text, missing_skills):
    """Generate improvement suggestions using LLM"""
    prompt = f"""
    Based on the resume and job description, provide specific suggestions for the candidate to improve their match.
    
    Resume: {resume_text[:1000]}...
    Job Description: {jd_text[:1000]}...
    Missing Skills: {', '.join(missing_skills)}
    
    Provide 3-5 actionable suggestions including specific skills to learn, projects to undertake, or certifications to pursue.
    """
    
    # Use your existing Gemini function
    from gemini_utils import get_gemini_response
    return get_gemini_response(prompt, "", "Generate improvement suggestions")
