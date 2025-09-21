# enhanced_parsing.py
import re
import spacy
from PyPDF2 import PdfReader
import docx
import json

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model is not available, download it
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extract_text_from_file(uploaded_file):
    """Extract text from PDF or DOCX files with error handling"""
    try:
        if uploaded_file.type == 'application/pdf':
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            doc = docx.Document(uploaded_file)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        elif uploaded_file.type == 'text/plain':
            return uploaded_file.read().decode("utf-8")
            
        return ""
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def standardize_resume_format(text):
    """Remove headers/footers and normalize resume sections"""
    # Remove common header/footer patterns
    patterns_to_remove = [
        r'\n{3,}',  # Multiple newlines
        r'Page \d+ of \d+',  # Page numbers
        r'©.*\d{4}',  # Copyright notices
        r'Confidential|Proprietary',  # Confidential markers
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Normalize section headings
    section_mappings = {
        r'(work\s*history|employment\s*history|experience)': 'EXPERIENCE',
        r'(education|academic\s*background|qualifications)': 'EDUCATION',
        r'(skills|technical\s*skills|competencies)': 'SKILLS',
        r'(projects|project\s*experience)': 'PROJECTS',
        r'(certifications|certificate)': 'CERTIFICATIONS',
        r'(achievements|accomplishments)': 'ACHIEVEMENTS'
    }
    
    for pattern, replacement in section_mappings.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def parse_resume_sections(text):
    """Parse resume into structured sections"""
    standardized_text = standardize_resume_format(text)
    sections = {
        'contact': {},
        'summary': '',
        'skills': [],
        'education': [],
        'experience': [],
        'projects': [],
        'certifications': [],
        'achievements': []
    }
    
    # Extract contact information
    doc = nlp(standardized_text)
    for ent in doc.ents:
        if ent.label_ == 'EMAIL':
            sections['contact']['email'] = ent.text
        elif ent.label_ == 'PHONE':
            sections['contact']['phone'] = ent.text
    
    # Simple section detection using keywords
    lines = standardized_text.split('\n')
    current_section = None
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if any(keyword in line_lower for keyword in ['summary', 'objective', 'profile']):
            current_section = 'summary'
        elif any(keyword in line_lower for keyword in ['skills', 'technical skills', 'competencies']):
            current_section = 'skills'
        elif any(keyword in line_lower for keyword in ['education', 'academic', 'qualifications']):
            current_section = 'education'
        elif any(keyword in line_lower for keyword in ['experience', 'work history', 'employment']):
            current_section = 'experience'
        elif any(keyword in line_lower for keyword in ['projects', 'portfolio']):
            current_section = 'projects'
        elif any(keyword in line_lower for keyword in ['certifications', 'certificates']):
            current_section = 'certifications'
        elif any(keyword in line_lower for keyword in ['achievements', 'accomplishments']):
            current_section = 'achievements'
        elif current_section and line.strip():
            if current_section == 'summary':
                sections[current_section] += line.strip() + " "
            else:
                sections[current_section].append(line.strip())
    
    return sections

def parse_job_description(jd_text):
    """Extract structured information from job description"""
    doc = nlp(jd_text)
    
    # Extract role title (often at the beginning)
    role_title = ""
    first_sentence = next(doc.sents, None)
    if first_sentence:
        role_title = first_sentence.text.split('\n')[0].strip()
    
    # Extract skills using pattern matching
    skills = set()
    must_have_keywords = ['required', 'must have', 'essential', 'mandatory']
    good_to_have_keywords = ['preferred', 'nice to have', 'desired', 'plus']
    
    must_have_skills = []
    good_to_have_skills = []
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        
        # Check for must-have skills
        if any(keyword in sent_text for keyword in must_have_keywords):
            for token in sent:
                if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and len(token.text) > 3:
                    must_have_skills.append(token.text)
        
        # Check for good-to-have skills
        elif any(keyword in sent_text for keyword in good_to_have_keywords):
            for token in sent:
                if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and len(token.text) > 3:
                    good_to_have_skills.append(token.text)
    
    # Extract qualifications
    qualifications = []
    qualification_keywords = ['degree', 'bachelor', 'master', 'phd', 'diploma', 'certification']
    
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in qualification_keywords):
            qualifications.append(sent.text.strip())
    
    return {
        'role_title': role_title,
        'must_have_skills': list(set(must_have_skills)),
        'good_to_have_skills': list(set(good_to_have_skills)),
        'qualifications': qualifications
    }