# parsing_utils.py
import re
import spacy
from PyPDF2 import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nlp = spacy.load("en_core_web_sm")

def extract_text_from_file(uploaded_file):
    """Extract text from PDF or DOCX files"""
    if uploaded_file.type == 'application/pdf':
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        doc = docx.Document(uploaded_file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    return ""

def parse_resume_sections(text):
    """Parse resume into structured sections"""
    sections = {
        'skills': [],
        'education': [],
        'experience': [],
        'projects': [],
        'certifications': []
    }
    
    # Simple section detection using keywords
    lines = text.split('\n')
    current_section = None
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if any(keyword in line_lower for keyword in ['skills', 'technical skills', 'competencies']):
            current_section = 'skills'
        elif any(keyword in line_lower for keyword in ['education', 'academic', 'qualifications']):
            current_section = 'education'
        elif any(keyword in line_lower for keyword in ['experience', 'work history', 'employment']):
            current_section = 'experience'
        elif any(keyword in line_lower for keyword in ['projects', 'portfolio']):
            current_section = 'projects'
        elif any(keyword in line_lower for keyword in ['certifications', 'certificates']):
            current_section = 'certifications'
        elif current_section and line.strip():
            sections[current_section].append(line.strip())
    
    return sections

def extract_skills_from_jd(jd_text):
    """Extract skills from job description using NLP"""
    doc = nlp(jd_text)
    skills = []
    
    # Look for technical skills patterns
    for sent in doc.sents:
        for token in sent:
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                # Simple heuristic for technical terms
                if len(token.text) > 3 and token.text.isalpha():
                    skills.append(token.text.lower())
    
    return list(set(skills))

def categorize_skills(jd_text, must_have_keywords, good_to_have_keywords):
    """Categorize skills into must-have and good-to-have"""
    all_skills = extract_skills_from_jd(jd_text)
    
    must_have = []
    good_to_have = []
    
    for skill in all_skills:
        if any(keyword.lower() in skill.lower() for keyword in must_have_keywords):
            must_have.append(skill)
        elif any(keyword.lower() in skill.lower() for keyword in good_to_have_keywords):
            good_to_have.append(skill)
    
    return must_have, good_to_have