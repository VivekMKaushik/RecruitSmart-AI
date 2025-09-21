import os
# Disable Streamlit file watcher early to avoid scanning torch internals which can raise
# runtime errors during import in some environments (see Streamlit watcher + torch issue).
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_FILE_WATCHER", "false")
import streamlit as st
import time
import pandas as pd
from datetime import datetime
import io
import json
import inspect
import google.generativeai as genai
from PyPDF2 import PdfReader
import docx as _docx
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import func  # Added import for func
from contextlib import contextmanager
try:
    import fitz  # type: ignore # PyMuPDF
except ImportError:
    fitz = None
import pdfplumber
import spacy
import nltk
# --- Stub for analyze_resume (used in resume comparison) ---
def analyze_resume(resume_filename, job_role):
    """Analyze a stored resume file against a job description.

    Parameters
    - resume_filename: filename stored on disk (searches resumes/, uploads/, cwd)
    - job_role: either a job description title to look up in DB or plain JD text

    Returns a dict with keys: hard_match, soft_match, ats_score, missing_keywords, verdict, details
    """
    # Resolve job description text: try DB lookup by title, otherwise treat job_role as JD text
    jd_text = ""
    try:
        session = db.SessionLocal()
        if job_role:
            try:
                jd_row = session.query(db.JobDescription).filter(db.JobDescription.title == job_role).order_by(db.JobDescription.created_at.desc()).first()
            except Exception:
                jd_row = None
            if jd_row and getattr(jd_row, 'description', None):
                jd_text = jd_row.description
    except Exception:
        jd_text = ""
    finally:
        try:
            session.close()
        except Exception:
            pass

    if not jd_text:
        # Treat job_role as the job description text if it's not a title in DB
        jd_text = job_role or ""

    # Load resume file from storage
    uploaded = load_resume_file_from_storage(resume_filename)
    if uploaded is None:
        return {
            'error': f'Resume file not found: {resume_filename}',
            'hard_match': 0,
            'soft_match': 0,
            'ats_score': 0,
            'missing_keywords': [],
            'verdict': 'Unknown'
        }

    # Extract text
    try:
        resume_text, ftype = extract_text_from_file(uploaded)
    except Exception as e:
        return {
            'error': f'Failed to extract resume text: {e}',
            'hard_match': 0,
            'soft_match': 0,
            'ats_score': 0,
            'missing_keywords': [],
            'verdict': 'Unknown'
        }

    # If there's no JD text, return minimal info
    if not jd_text.strip():
        # Basic skill extraction and counts
        resume_skills = extract_skills(resume_text)
        return {
            'hard_match': 0,
            'soft_match': 0,
            'ats_score': 0,
            'missing_keywords': [],
            'verdict': 'Unknown',
            'resume_skills': resume_skills
        }

    # Perform comprehensive analysis using existing helper
    try:
        analysis = analyze_resume_jd_comprehensive(resume_text, jd_text)
    except Exception as e:
        return {
            'error': f'Analysis failed: {e}',
            'hard_match': 0,
            'soft_match': 0,
            'ats_score': 0,
            'missing_keywords': [],
            'verdict': 'Unknown'
        }

    # Map comprehensive analysis to expected return keys
    result = {
        'hard_match': float(analysis.get('hard_match_score', analysis.get('hard_match', 0))),
        'soft_match': float(analysis.get('semantic_match_score', analysis.get('soft_match', 0))),
        'ats_score': float(analysis.get('score', 0)),
        'missing_keywords': analysis.get('missing_skills', analysis.get('missing_keywords', [])),
        'verdict': analysis.get('verdict', 'Unknown'),
        'details': analysis,
    }

    return result

# --- Stub for save_resume_comparison (used in resume comparison) ---
def save_resume_comparison(app_id, resume, ats_score, hard_match, soft_match, missing, verdict, status):
    """Persist resume comparison results into the resume_analyses table.

    Strategy:
    - Find the Application by `app_id` to read contact and job_role.
    - Find or create a Resume record (by filename).
    - Find a JobRequirement by title matching the application's job_role (optional).
    - Insert a ResumeAnalysis row with scores and metadata.
    """
    session = None
    try:
        session = db.SessionLocal()

        # Try to fetch the application to obtain context
        app_row = None
        try:
            app_row = session.query(db.Application).filter(db.Application.id == app_id).first()
        except Exception:
            app_row = None

        # Ensure there's a Resume row for this filename
        resume_row = None
        try:
            resume_row = session.query(db.Resume).filter(db.Resume.filename == resume).first()
        except Exception:
            resume_row = None

        if resume_row is None:
            resume_row = db.Resume(
                filename=resume,
                parsed={},
                uploaded_at=datetime.utcnow(),
                candidate_email=getattr(app_row, 'email', None) if app_row else None,
                candidate_name=getattr(app_row, 'name', None) if app_row else None,
                candidate_phone=getattr(app_row, 'phone', None) if app_row else None,
            )
            session.add(resume_row)
            session.commit()
            session.refresh(resume_row)

        # Attempt to resolve a JobRequirement (jd) by the application's job_role
        jd_obj = None
        jd_id = None
        job_role_title = getattr(app_row, 'job_role', None) if app_row else None
        if job_role_title:
            try:
                jd_obj = session.query(db.JobRequirement).filter(db.JobRequirement.title == job_role_title).first()
                if jd_obj:
                    jd_id = jd_obj.id
            except Exception:
                jd_obj = None
                jd_id = None

        # Insert ResumeAnalysis
        try:
            ra = db.ResumeAnalysis(
                resume_id=resume_row.id,
                jd_id=jd_id,
                relevance_score=int(round(float(ats_score))) if ats_score is not None else None,
                hard_match_score=int(round(float(hard_match))) if hard_match is not None else None,
                semantic_match_score=int(round(float(soft_match))) if soft_match is not None else None,
                missing_skills=missing if isinstance(missing, (list, dict)) else [missing] if missing else [],
                missing_qualifications=[],
                verdict=str(verdict) if verdict is not None else None,
                suggestions=None,
                created_at=datetime.utcnow(),
            )
            session.add(ra)
            session.commit()
            session.refresh(ra)
            return ra.id
        except Exception as e:
            # If insertion fails, raise or log
            st.warning(f"Failed to save resume analysis: {e}")
            session.rollback()
            return None

    except Exception as e_outer:
        try:
            if session:
                session.rollback()
        except Exception:
            pass
        st.warning(f"Error in save_resume_comparison: {e_outer}")
        return None
    finally:
        try:
            if session:
                session.close()
        except Exception:
            pass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment from .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Disable file watching for torch to prevent errors
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# Must be the first Streamlit command in the script
st.set_page_config(
    page_title="ATS + Career Guidance & AI Chat", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "# AI-Powered ATS & Career Guidance System"
    }
)

# Load NLP models
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

# Download NLTK data if not already present
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Import local modules with error handling
try:
    import nlp_utils
    from embeddings import semantic_score
    from scoring import compute_final_score
    import db
    from email_utils import send_feedback_email
    from enhanced_scoring import (
        hard_match_score,
        semantic_match_score,
        calculate_final_score,
        get_verdict,
        find_missing_elements,
        generate_suggestions,
    )
except ImportError as e:
    st.error(f"Error importing required modules: {e}")
    st.info("Please ensure all required modules are installed and available.")

# -----------------------------
# Configuration
# -----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        st.sidebar.success("✅ Gemini API configured successfully")
    except Exception as e:
        st.sidebar.warning(f"Could not configure Gemini client: {e}")
else:
    st.sidebar.warning("⚠️ Gemini API key not found. Set GOOGLE_API_KEY environment variable.")

# -----------------------------
# Enhanced File processing with multiple libraries
# -----------------------------
def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file using multiple libraries for better accuracy"""
    try:
        raw = uploaded_file.read()
        uploaded_file.seek(0)  # Reset stream
        
        content_type = getattr(uploaded_file, "type", "") or ""
        name = getattr(uploaded_file, "name", "") or ""

        # PDF extraction with multiple methods
        if "pdf" in content_type or name.lower().endswith(".pdf"):
            text = ""
            
            # Method 1: PyMuPDF (fitz)
            try:
                doc = fitz.open(stream=raw, filetype="pdf")
                for page in doc:
                    text += page.get_text() + "\n"
                if text.strip():
                    return text, "pdf"
            except Exception as e:
                st.warning(f"PyMuPDF extraction failed: {e}")
            
            # Method 2: pdfplumber
            try:
                with pdfplumber.open(io.BytesIO(raw)) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    if text.strip():
                        return text, "pdf"
            except Exception as e:
                st.warning(f"pdfplumber extraction failed: {e}")
            
            # Method 3: pdfplumber (additional fallback)
            try:
                # Try opening the bytes with pdfplumber as a robust final fallback
                with pdfplumber.open(io.BytesIO(raw)) as pdf_fallback:
                    text = ""
                    for page in pdf_fallback.pages:
                        text += page.extract_text() or ""
                    if text.strip():
                        return text, "pdf"
            except Exception as e:
                st.warning(f"pdfplumber extraction failed (fallback): {e}")
            
            return "", "pdf"

        # DOCX extraction
        if "word" in content_type or name.lower().endswith((".docx", ".doc")):
            try:
                doc = _docx.Document(io.BytesIO(raw))
                return "\n".join([p.text for p in doc.paragraphs]), "docx"
            except Exception as e:
                st.warning(f"DOCX extraction failed: {e}")
                return "", "docx"

        # Text file
        if "text" in content_type or name.lower().endswith(".txt"):
            return raw.decode("utf-8", errors="replace"), "text"

        # Fallback: try to decode as text
        try:
            return raw.decode("utf-8", errors="replace"), "text"
        except Exception:
            return "", "unknown"

    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return "", "unknown"

# -----------------------------
# NLP and Analysis Functions
# -----------------------------
def extract_entities(text):
    """Extract entities from text using spaCy"""
    if not nlp or not text:
        return []
    
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "GPE", "DATE", "PRODUCT", "SKILL", "TECH"]:
            entities.append((ent.text, ent.label_))
    return entities

def extract_skills(text):
    """Extract skills from text using pattern matching and NLP"""
    # Common tech skills patterns
    tech_skills = [
        "python", "java", "javascript", "sql", "html", "css", "react", "angular",
        "vue", "node", "express", "django", "flask", "fastapi", "pandas", "numpy",
        "tensorflow", "pytorch", "machine learning", "deep learning", "ai",
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git"
    ]
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in tech_skills:
        if skill in text_lower:
            found_skills.append(skill.title())
    
    return list(set(found_skills))

def calculate_similarity(text1, text2):
    """Calculate semantic similarity between two texts"""
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]

def analyze_resume_jd_comprehensive(resume_text, jd_text):
    """Comprehensive analysis of resume against job description"""
    # Hard keyword matching
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    
    # Calculate matches
    matched_skills = set(jd_skills) & set(resume_skills)
    missing_skills = set(jd_skills) - set(resume_skills)
    
    # Calculate scores
    hard_match = len(matched_skills) / len(jd_skills) * 100 if jd_skills else 0
    semantic_match = calculate_similarity(resume_text, jd_text) * 100
    
    # Weighted final score
    final_score = (hard_match * 0.6) + (semantic_match * 0.4)
    
    # Generate verdict
    if final_score >= 80:
        verdict = "High"
    elif final_score >= 60:
        verdict = "Medium"
    else:
        verdict = "Low"
    
    # Generate suggestions
    suggestions = []
    if missing_skills:
        suggestions.append(f"Add missing skills: {', '.join(missing_skills)}")
    if final_score < 80:
        suggestions.append("Improve semantic alignment with job description")
    if len(resume_skills) < 5:
        suggestions.append("Consider adding more technical skills to your resume")
    
    return {
        'score': round(final_score, 1),
        'verdict': verdict,
        'hard_match_score': round(hard_match, 1),
        'semantic_match_score': round(semantic_match, 1),
        'matched_skills': list(matched_skills),
        'missing_skills': list(missing_skills),
        'suggestions': ' '.join(suggestions),
        'resume_skills_count': len(resume_skills),
        'jd_skills_count': len(jd_skills)
    }

# -----------------------------
# Database Functions
# -----------------------------
@contextmanager
def get_db_session():
    """Contextmanager that yields a SQLAlchemy session and ensures it is closed.

    Use as:
        with get_db_session() as session:
            session.query(...)
    """
    session = None
    try:
        session = db.SessionLocal()
        yield session
    finally:
        try:
            if session:
                session.close()
        except Exception:
            pass


def setup_database():
    """Initialize database connection and ensure tables exist"""
    try:
        # Get database URL from environment or use default
        DATABASE_URL = os.getenv("DATABASE_URL")
        if not DATABASE_URL:
            # Construct default PostgreSQL URL
            DATABASE_URL = (
                f"postgresql://{os.getenv('POSTGRES_USER', 'pguser')}:"
                f"{os.getenv('POSTGRES_PASSWORD', 'pgpass')}@"
                f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
                f"{os.getenv('POSTGRES_PORT', '5432')}/"
                f"{os.getenv('POSTGRES_DB', 'resumes')}"
            )
        
        # Create SQLAlchemy engine
        engine = create_engine(DATABASE_URL, echo=False)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        
        print(f"✅ Successfully connected to database: {DATABASE_URL}")
        
        # Run database migrations first
        from db import migrate_database_schema
        try:
            migrate_database_schema()
        except Exception:
            # Non-fatal if migration helper fails; continue to create tables
            pass
        
        # Create all tables if they don't exist
        from db import Base
        Base.metadata.create_all(bind=engine)
        
        # Check if tables have data
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        print("📊 Database tables:", table_names)
        
        # Create session factory
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        return engine, SessionLocal
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        # Fallback to SQLite for development
        try:
            sqlite_url = "sqlite:///./resumes.db"
            engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            
            from db import Base
            Base.metadata.create_all(bind=engine)
            
            print(f"⚠️ Using SQLite fallback: {sqlite_url}")
            return engine, SessionLocal
        except Exception as sqlite_error:
            print(f"❌ SQLite fallback also failed: {sqlite_error}")
            raise

def get_all_applications():
    """Retrieve all applications from the database"""
    try:
        session = db.SessionLocal()
        applications = session.query(db.Application).order_by(db.Application.created_at.desc()).all()
        return applications
    except Exception as e:
        st.error(f"Error retrieving applications: {str(e)}")
        return []
    finally:
        session.close()

def load_resume_file_from_storage(filename):
    """Search common upload folders for `filename` and return a file-like BytesIO with
    `.name` and `.type` attributes compatible with `extract_text_from_file`.
    Returns None if not found.
    """
    search_dirs = [
        os.path.join(os.getcwd(), 'resumes'),
        os.path.join(os.getcwd(), 'uploads'),
        os.getcwd(),
    ]

    for d in search_dirs:
        try:
            full = os.path.join(d, filename)
        except Exception:
            continue

        if os.path.exists(full) and os.path.isfile(full):
            try:
                with open(full, 'rb') as f:
                    data = f.read()
                bio = io.BytesIO(data)
                # attach small compatibility attributes
                bio.name = filename
                ext = filename.lower().split('.')[-1]
                if ext == 'pdf':
                    bio.type = 'application/pdf'
                elif ext in ('docx', 'doc'):
                    bio.type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                else:
                    bio.type = 'text/plain'
                bio.seek(0)
                return bio
            except Exception:
                continue

    return None

def get_applications_by_job_role(job_role=None):
    """Retrieve applications filtered by job role"""
    try:
        session = db.SessionLocal()
        if job_role and job_role != "All":
            applications = session.query(db.Application).filter(
                db.Application.job_role == job_role
            ).order_by(db.Application.created_at.desc()).all()
        else:
            applications = session.query(db.Application).order_by(
                db.Application.created_at.desc()
            ).all()
        return applications
    except Exception as e:
        st.error(f"Error retrieving applications: {str(e)}")
        return []
    finally:
        session.close()

def delete_application(application_id):
    """Delete an application from the database"""
    try:
        session = db.SessionLocal()
        application = session.query(db.Application).filter(db.Application.id == application_id).first()
        if application:
            session.delete(application)
            session.commit()
            return True, "Application deleted successfully"
        return False, "Application not found"
    except Exception as e:
        session.rollback()
        return False, f"Error deleting application: {str(e)}"
    finally:
        session.close()

def get_job_roles_from_db():
    """Get all job roles from job_descriptions table"""
    try:
        session = db.SessionLocal()
        job_roles = session.query(db.JobDescription.title).distinct().all()
        return [role[0] for role in job_roles] if job_roles else []
    except Exception as e:
        st.error(f"Error retrieving job roles: {str(e)}")
        return []
    finally:
        session.close()

def save_job_description(title, description):
    """Save job description to database"""
    try:
        session = db.SessionLocal()
        jd = db.JobDescription(
            title=title,
            description=description,
            created_at=datetime.utcnow()
        )
        session.add(jd)
        session.commit()
        session.refresh(jd)
        return True, f"Job description saved with ID: {jd.id}"
    except Exception as e:
        return False, f"Error saving job description: {str(e)}"
    finally:
        session.close()

def get_job_descriptions():
    """Get all job descriptions from database"""
    try:
        session = db.SessionLocal()
        jds = session.query(db.JobDescription).order_by(db.JobDescription.created_at.desc()).all()
        return jds
    except Exception as e:
        st.error(f"Error retrieving job descriptions: {str(e)}")
        return []
    finally:
        session.close()


def ensure_resume_analysis_data():
    """Ensure resume analysis data exists and is properly formatted"""
    try:
        session = SessionLocal()
        try:
            # Check if we have any resume analyses
            analysis_count = session.query(db.ResumeAnalysis).count()

            if analysis_count == 0:
                # Get some applications (limit to 3 as requested)
                applications = session.query(db.Application).limit(3).all()
                if not applications:
                    # Nothing to seed; be silent to avoid noisy UI messages
                    return

                created_count = 0

                for app in applications:
                    try:
                        # Analyze the resume
                        analysis_result = analyze_resume(app.resume_filename, app.job_role)

                        # Save the analysis
                        analysis_id = save_resume_comparison(
                            app_id=app.id,
                            resume=app.resume_filename,
                            ats_score=analysis_result.get('ats_score', 0),
                            hard_match=analysis_result.get('hard_match', 0),
                            soft_match=analysis_result.get('soft_match', 0),
                            missing=analysis_result.get('missing_keywords', []),
                            verdict=analysis_result.get('verdict', 'Medium'),
                            status="New"
                        )

                        if analysis_id:
                            created_count += 1

                    except Exception as e:
                        # Log a non-fatal warning but continue; don't surface noisy UI warnings
                        try:
                            st.debug if hasattr(st, 'debug') else lambda *a, **k: None
                        except Exception:
                            pass
                        continue

                # Only show a success banner when we actually created analyses
                if created_count > 0:
                    try:
                        st.success(f"Generated {created_count} resume analyses")
                    except Exception:
                        pass
        finally:
            try:
                session.close()
            except Exception:
                pass

    except Exception as e:
        try:
            st.error(f"Error ensuring resume analysis data: {str(e)}")
        except Exception:
            pass

def delete_job_description(jd_id):
    """Delete a job description from the database"""
    try:
        session = db.SessionLocal()
        jd = session.query(db.JobDescription).filter(db.JobDescription.id == jd_id).first()
        if jd:
            session.delete(jd)
            session.commit()
            return True, "Job description deleted successfully"
        return False, "Job description not found"
    except Exception as e:
        session.rollback()
        return False, f"Error deleting job description: {str(e)}"
    finally:
        session.close()

# -----------------------------
# Gemini Functions
# -----------------------------
def get_gemini_response(input_text, pdf_text, prompt, model_name="gemini-1.5-flash"):
    """Get response from Gemini AI"""
    if not GOOGLE_API_KEY:
        return "LLM not configured. Set GOOGLE_API_KEY to enable this feature."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([input_text, pdf_text, prompt])
        return response.text
    except Exception as e:
        return f"Error calling Gemini: {str(e)}"

def simple_chat_with_ai(user_message, history=[]):
    """Simple chat function with Gemini AI"""
    try:
        if not GOOGLE_API_KEY:
            return ("LLM not configured. Set GOOGLE_API_KEY to enable chat.", history)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        gemini_history = []
        
        for msg in history:
            if msg.get("role") == "user":
                gemini_history.append({"role": "user", "parts": [msg.get("content", "")]})
            else:
                gemini_history.append({"role": "model", "parts": [msg.get("content", "")]})
        
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(user_message)
        
        updated_history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.text}
        ]
        
        return response.text, updated_history
    except Exception as e:
        return f"Error: {str(e)}", history

# -----------------------------
# Application Management
# -----------------------------
def save_application_to_db(name, phone, email, gender, job_role, resume_file):
    """Save job application to database and store resume on disk.

    Expects `resume_file` to be a file-like object from Streamlit's uploader
    (it should provide `.name` and either `.getbuffer()` or `.read()`).
    The function writes the file to `resumes/` and stores only the filename in
    the `Application` DB record.
    """
    session = None
    try:
        session = db.SessionLocal()

        # Ensure folder exists
        os.makedirs("resumes", exist_ok=True)

        # Save resume file physically (support getbuffer() or read())
        resume_name = getattr(resume_file, "name", "resume")
        resume_path = os.path.join("resumes", resume_name)
        with open(resume_path, "wb") as f:
            try:
                f.write(resume_file.getbuffer())
            except Exception:
                try:
                    resume_file.seek(0)
                except Exception:
                    pass
                f.write(resume_file.read())

        # Save DB record (only filename, not path)
        application = db.Application(
            name=name,
            phone=phone,
            email=email,
            gender=gender,
            job_role=job_role,
            resume_filename=resume_name
        )
        session.add(application)
        session.commit()
        session.refresh(application)

        return True, f"Application saved with ID: {application.id}"
    except Exception as e:
        if session is not None:
            try:
                session.rollback()
            except Exception:
                pass
        return False, f"Error saving application: {str(e)}"
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

# -----------------------------
# UI helper: Metric card
# -----------------------------
def create_metric_card(title, value, delta=None, help_text=None):
    """Small wrapper to render a metric with an optional caption and delta."""
    if help_text:
        st.caption(help_text)
    # Streamlit's st.metric expects title, value, delta
    try:
        st.metric(title, value, delta=delta)
    except Exception:
        # Fallback: simple write if metric fails for any reason
        st.write(f"{title}: {value}" + (f" ({delta})" if delta else ""))
def main():
    # Initialize database
    global engine, SessionLocal
    try:
        engine, SessionLocal = setup_database()
    except Exception:
        # fallback: use existing db.SessionLocal if setup_database does not return as expected
        try:
            SessionLocal = db.SessionLocal
            engine = db.engine
        except Exception:
            pass

    # Ensure we have some analysis data (safe to call multiple times)
    try:
        ensure_resume_analysis_data()
    except Exception:
        pass

    # Sidebar Configuration
    with st.sidebar:
        st.title("🤖 RecruitSmart  AI")
        st.markdown("---")
        
        # Dashboard Selection
        dashboard_option = st.radio(
            "Select Dashboard:",
            ["👤 Job Seeker", "🏢 Company/Placement"],
            index=0,
            help="Choose between job seeker tools or company/placement team tools"
        )
        
        st.markdown("---")
        
        # Quick Actions
        st.subheader("Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.show_analytics = True
        
        st.markdown("---")
        
        # System Status
        st.subheader("System Status")
        if GOOGLE_API_KEY:
            st.success("✅ Gemini API: Connected")
        else:
            st.error("❌ Gemini API: Not Configured")
        
        try:
            db.SessionLocal()
            st.success("✅ Database: Connected")
        except Exception:
            st.error("❌ Database: Connection Failed")
        
        # Feature indicators
        st.markdown("---")
        st.subheader("Enabled Features")
        st.success("✅ PDF/DOCX Text Extraction")
        st.success("✅ NLP Entity Recognition")
        st.success("✅ Semantic Similarity")
        st.success("✅ Keyword Matching")
        st.success("✅ AI Chat Assistant")
    
    # Main Content Area
    if dashboard_option == "👤 Job Seeker":
        render_job_seeker_dashboard()
    else:
        render_company_dashboard()

def render_job_seeker_dashboard():
    """Render the job seeker dashboard"""
    st.title("👤 Job Seeker Dashboard")
    
    # Create tabs for different functionalities - REORDERED as requested
    tab1, tab2, tab3, tab4 = st.tabs([
        "💼 Job Application", 
        "📝 ATS Resume Checker", 
        "📊 Career Insights",
        "🤖 AI Career Chat"
    ])
    
    with tab1:
        st.header("💼 Apply for a Job")
        
        # Get job roles from database
        job_roles = get_job_roles_from_db()
        
        with st.form("job_application_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name*", placeholder="Enter your full name")
                phone = st.text_input("Phone Number*", placeholder="+91 97123-4567")
                email = st.text_input("Email Address*", placeholder="your.email@example.com")
            
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
                job_role = st.selectbox("Job Role/Position*", options=job_roles if job_roles else [""], 
                                      placeholder="Select job role")
                resume_file = st.file_uploader("Upload Resume*", type=["pdf", "docx", "txt"])
            
            submitted = st.form_submit_button("🚀 Submit Application", type="primary", use_container_width=True)
            
            if submitted:
                if all([name, phone, email, job_role, resume_file]):
                    with st.spinner("Submitting your application..."):
                        success, message = save_application_to_db(
                            name, phone, email, gender, job_role, resume_file
                        )
                        
                        if success:
                            st.success("✅ Application submitted successfully!")
                            st.balloons()
                            
                            # Auto-analyze the resume
                            try:
                                resume_text, _ = extract_text_from_file(resume_file)
                                analysis_result = analyze_resume_jd_comprehensive(resume_text, "")
                                
                                with st.expander("📊 Quick Resume Analysis"):
                                    st.metric("Resume Score", f"{analysis_result['score']}%")
                                    st.write(f"**Skills Found:** {analysis_result.get('resume_skills_count', 0)}")
                                    
                                    if analysis_result.get('resume_skills_count', 0) < 5:
                                        st.warning("Consider adding more skills to your resume")
                                    else:
                                        st.success("Good number of skills detected!")
                                        
                            except Exception as e:
                                st.warning(f"Could not analyze resume: {e}")
                                
                        else:
                            st.error(f"❌ {message}")
                else:
                    st.warning("⚠️ Please fill all required fields")
    
    with tab2:
        st.header("📝 ATS Resume Score Checker")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Your Resume", 
                type=["pdf", "docx", "txt"],
                help="Supported formats: PDF, DOCX, TXT"
            )
            
            # Get job roles from database for selection
            job_roles = get_job_roles_from_db()
            selected_job_role = st.selectbox(
                "Select Job Role",
                options=[""] + job_roles if job_roles else [""],
                help="Select the job role you want to compare against"
            )
            
            job_description = st.text_area(
                "Paste Job Description (or select job role above)",
                height=200,
                placeholder="Copy and paste the job description you're applying for...",
                help="The more detailed the job description, the better the analysis"
            )
            
            if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
                if uploaded_file and (job_description or selected_job_role):
                    with st.spinner("Analyzing your resume..."):
                        try:
                            resume_text, _ = extract_text_from_file(uploaded_file)
                            
                            # If job role is selected but no description provided, try to get from DB
                            if selected_job_role and not job_description:
                                try:
                                    session = db.SessionLocal()
                                    jd = session.query(db.JobDescription).filter(
                                        db.JobDescription.title == selected_job_role
                                    ).first()
                                    if jd:
                                        job_description = jd.description
                                    else:
                                        st.warning("No job description found for selected role. Please paste a job description.")
                                        return
                                except Exception as e:
                                    st.warning(f"Could not retrieve job description: {e}")
                                    return
                            
                            analysis_result = analyze_resume_jd_comprehensive(resume_text, job_description)
                            
                            # Display results
                            st.success("Analysis Complete!")
                            
                            # Score cards
                            score_col1, score_col2, score_col3 = st.columns(3)
                            
                            with score_col1:
                                create_metric_card("Overall Score", f"{analysis_result['score']}%")
                            
                            with score_col2:
                                create_metric_card("Keyword Match", f"{analysis_result.get('hard_match_score', 0)}%")
                            
                            with score_col3:
                                create_metric_card("Semantic Match", f"{analysis_result.get('semantic_match_score', 0)}%")
                            
                            # Verdict
                            verdict = analysis_result.get('verdict', 'Medium')
                            if verdict == 'High':
                                st.success(f"🎉 **{verdict} Match!** Your resume is well-aligned with the job requirements.")
                            elif verdict == 'Medium':
                                st.warning(f"⚠️ **{verdict} Match.** Your resume has some alignment but could be improved.")
                            else:
                                st.error(f"❌ **{verdict} Match.** Your resume needs significant improvements.")
                            
                            # Skills analysis
                            # Removed Matched Skills, Missing Skills, and Improvement Suggestions expanders from student dashboard
                            
                            # Detailed analysis
                            with st.expander("📊 Detailed Analysis"):
                                st.write(f"**Resume Skills Found:** {analysis_result.get('resume_skills_count', 0)}")
                                st.write(f"**JD Skills Required:** {analysis_result.get('jd_skills_count', 0)}")
                                st.write(f"**Skills Match Rate:** {analysis_result.get('hard_match_score', 0)}%")
                            
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                else:
                    st.warning("Please upload your resume and provide/select a job description")
        
        with col2:
            st.info("💡 **Tips for Better Results:**")
            st.write("• Use PDF format for best compatibility")
            st.write("• Include detailed job descriptions")
            st.write("• Ensure your resume is up-to-date")
            st.write("• Highlight relevant skills and experience")
            st.write("• Use industry-standard keywords")
    
    with tab3:
        st.header("📊 Career Insights")
        
        # Career analytics placeholder
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card("Applications Sent", "12", "+3 this week", "Total job applications submitted")
        
        with col2:
            create_metric_card("Response Rate", "42%", "+8%", "Percentage of applications that received responses")
        
        with col3:
            create_metric_card("Interview Rate", "25%", "+5%", "Percentage of applications that led to interviews")
        
        # Skills assessment
        st.subheader("Skills Assessment")
        
        skills = {
            "Python": 85,
            "SQL": 70,
            "Machine Learning": 60,
            "Communication": 75,
            "Project Management": 65
        }
        
        for skill, score in skills.items():
            st.write(f"**{skill}**")
            st.progress(score / 100)
        
        # Job market insights
        st.subheader("Job Market Insights")
        
        insights_data = {
            "Role": ["Data Analyst", "Software Engineer", "Data Scientist", "Product Manager"],
            "Demand": ["High", "Very High", "High", "Medium"],
            "Avg Salary": ["$75K", "$95K", "$105K", "$90K"],
            "Growth": ["+15%", "+20%", "+25%", "+18%"]
        }
        
        insights_df = pd.DataFrame(insights_data)
        st.dataframe(insights_df, use_container_width=True, hide_index=True)
    
    with tab4:
        st.header("🤖 AI Career Assistant")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about careers, resumes, or job search..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, updated_history = simple_chat_with_ai(prompt, st.session_state.chat_history)
                    st.markdown(response)
                    st.session_state.chat_history = updated_history

def safe_save_resume_comparison(**data):
    """
    Call the project's save_resume_comparison(...) safely even if its parameter names differ.
    Strategy:
      1. Try direct keyword call.
      2. Retry by mapping resume_filename -> common alternatives.
      3. As a last resort, build a positional args list from the target function signature.
    Raises the original exception if all attempts fail.
    """
    if "save_resume_comparison" not in globals():
        raise RuntimeError("save_resume_comparison function not found in globals()")

    fn = save_resume_comparison

    # 1) Try direct keyword call first
    try:
        return fn(**data)
    except TypeError as e_initial:
        # We'll try alternatives below
        pass

    # 2) Common alternative names mapping for resume filename and missing keywords
    alt_mappings = [
        ("resume_filename", "resume"),
        ("resume_filename", "resume_file"),
        ("resume_filename", "resume_name"),
        ("missing_keywords", "missing"),
        ("missing_keywords", "missing_kw"),
    ]

    for src, tgt in alt_mappings:
        if src in data and tgt not in data:
            newdata = data.copy()
            newdata[tgt] = newdata.pop(src)
            try:
                return fn(**newdata)
            except TypeError:
                # try next mapping
                continue

    # 3) Last resort: call by positional args inferred from function signature
    try:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        positional_args = []
        for pname in param_names:
            # prefer exact match in data, else fallbacks for resume fields
            if pname in data:
                positional_args.append(data[pname])
            elif pname in ("resume", "resume_filename", "resume_file", "resume_name"):
                # pick whichever is available
                for candidate in ("resume", "resume_filename", "resume_file", "resume_name"):
                    if candidate in data:
                        positional_args.append(data[candidate])
                        break
                else:
                    positional_args.append(None)
            elif pname in ("missing", "missing_keywords", "missing_kw"):
                for candidate in ("missing_keywords", "missing", "missing_kw"):
                    if candidate in data:
                        positional_args.append(data[candidate])
                        break
                else:
                    positional_args.append(None)
            else:
                # generic fallback: use data.get(pname)
                positional_args.append(data.get(pname))
        return fn(*positional_args)
    except Exception as e_final:
        # Raise a clearer error that includes both TypeError and last attempt exception
        raise RuntimeError(
            "safe_save_resume_comparison failed to call save_resume_comparison. "
            f"Initial TypeError and final exception: {e_final}"
        ) from e_final


def render_company_dashboard():
    """Render the company/placement team dashboard"""
    st.title("🏢 Company & Placement Dashboard")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["📋 Applications", "📊 Analytics"])
    
    with tab1:
        st.header("📋 Job Applications Management")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            # Get job roles from database for filtering
            job_roles = get_job_roles_from_db()
            job_filter = st.selectbox("Filter by Job Role", ["All"] + job_roles)
        
        with col2:
            status_filter = st.selectbox("Filter by Status", ["All", "New", "Reviewed", "Rejected", "Hired"])
        
        # Applications list
        applications = get_applications_by_job_role(job_filter if job_filter != "All" else None)
        
        if applications:
            for app in applications:
                # Remove the 'key' parameter from st.expander() to fix the error
                with st.expander(f"{app.name} - {app.job_role} ({app.created_at.strftime('%Y-%m-%d')})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Email:** {app.email}")
                        st.write(f"**Phone:** {app.phone}")
                        st.write(f"**Gender:** {app.gender}")
                    
                    with col2:
                        st.write(f"**Applied:** {app.created_at.strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Resume:** {app.resume_filename}")
                        
                        # Action buttons
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button("📄 View Resume", key=f"view_{app.id}"):
                                resume_bio = load_resume_file_from_storage(app.resume_filename)
                                if resume_bio:
                                    resume_text, _ = extract_text_from_file(resume_bio)
                                    st.text_area("Resume Preview", resume_text[:2000], height=300)
                                    st.download_button(
                                        "📥 Download Resume",
                                        data=resume_bio.getvalue(),
                                        file_name=app.resume_filename,
                                        mime="application/pdf"
                                    )
                                else:
                                    st.error(f"Resume file {app.resume_filename} not found.")
                        with col_btn2:
                            if st.button("🗑️ Delete", key=f"delete_{app.id}"):
                                success, message = delete_application(app.id)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
            
            # Export functionality
            st.download_button(
                label="📥 Export to CSV",
                data=pd.DataFrame([{
                    'Name': app.name,
                    'Email': app.email,
                    'Phone': app.phone,
                    'Job Role': app.job_role,
                    'Applied Date': app.created_at
                } for app in applications]).to_csv(index=False),
                file_name="applications_export.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No applications found.")
        
        # -------------------------------
        # Job Descriptions Management Section
        # -------------------------------
        st.markdown("---")
        st.subheader("📝 Job Descriptions Management")
        
        # Add New Job Role
        with st.expander("➕ Add New Job Role", expanded=False):
            new_jd_title = st.text_input("Job Title*", key="new_jd_title")
            new_jd_description = st.text_area("Job Description*", height=150, key="new_jd_desc", 
                                            placeholder="Enter detailed job description including requirements, responsibilities, and skills needed...")
            
            if st.button("💾 Save Job Role", key="save_jd"):
                if new_jd_title and new_jd_description:
                    success, message = save_job_description(new_jd_title, new_jd_description)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please fill in both job title and description")
        
        # Manage Existing Job Roles
        with st.expander("📋 Manage Existing Job Roles", expanded=False):
            jds = get_job_descriptions()
            
            if jds:
                for jd in jds:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{jd.title}**")
                        st.caption(f"ID: {jd.id} | Created: {jd.created_at.strftime('%Y-%m-%d')}")
                    
                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_jd_{jd.id}"):
                            success, message = delete_job_description(jd.id)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                    
                    # Avoid nested expanders: display description directly instead
                    st.markdown("**Description:**")
                    st.write(jd.description[:1000] + "..." if len(jd.description) > 1000 else jd.description)
                    
                    st.markdown("---")
            else:
                st.info("No job descriptions found. Add your first job role!")
        
        st.markdown("---")
        st.subheader("🔍 Resume Comparison with Job Description")

        # Check if there are job descriptions available
        jds = get_job_descriptions()
        if jds and applications:
            if st.button("Compare Resumes with JD"):
                st.info("Analyzing resumes... Please wait.")
                
                results = []
                for app in applications:
                    try:
                        # 1. Analyze Resume
                        analysis = analyze_resume(app.resume_filename, app.job_role)
                        
                        # 2. Extract metrics - handle potential missing keys
                        hard_match = analysis.get("hard_match", 0)
                        soft_match = analysis.get("soft_match", 0)
                        ats_score = analysis.get("ats_score", 0)
                        missing = analysis.get("missing_keywords", [])
                        verdict = analysis.get("verdict", "Medium")
                        
                        # 3. Decide status
                        status = "Shortlisted" if ats_score >= 70 else "Rejected"
                        
                        # 4. Store results
                        results.append({
                            "Resume": app.resume_filename,
                            "Hard Match %": hard_match,
                            "Soft Match %": soft_match,
                            "ATS Score": ats_score,
                            "Missing": ", ".join(missing[:5]),  # Limit to first 5 for display
                            "Verdict": verdict,
                            "Final Status": status
                        })
                        
                        # 5. Save in DB if function exists
                        try:
                            if 'save_resume_comparison' in globals():
                                save_resume_comparison(
                                    app_id=app.id,
                                    resume=app.resume_filename,
                                    ats_score=ats_score,
                                    hard_match=hard_match,
                                    soft_match=soft_match,
                                    missing=missing,
                                    verdict=verdict,
                                    status=status
                                )
                        except Exception as e:
                            st.warning(f"Could not save comparison for {app.name}: {str(e)}")
                        
                        # 6. Generate and store feedback
                        try:
                            suggestions = analysis.get('suggestions', '')
                            if not isinstance(suggestions, str):
                                suggestions = str(suggestions)
                                
                            feedback_lines = [
                                f"Hello {app.name},",
                                f"\nWe have completed a review of your resume for the role: {app.job_role}.",
                                f"ATS Score: {ats_score}%",
                                f"Verdict: {verdict}",
                            ]
                            
                            if missing:
                                feedback_lines.append(f"Missing/Recommended skills: {', '.join(missing[:5])}")
                            
                            if suggestions:
                                feedback_lines.append(f"Suggestions: {suggestions[:500]}{'...' if len(suggestions) > 500 else ''}")
                            
                            feedback_lines.append("\nBest regards,\nRecruitment Team")
                            feedback_message = "\n".join(feedback_lines)
                            
                            # Store feedback in session state for display
                            if 'applicant_feedback' not in st.session_state:
                                st.session_state.applicant_feedback = {}
                            st.session_state.applicant_feedback[app.id] = feedback_message
                            
                        except Exception as e:
                            st.warning(f"Could not generate feedback for {app.name}: {str(e)}")
                            
                    except Exception as e:
                        st.error(f"Error analyzing {app.name}: {str(e)}")
                        results.append({
                            "Resume": app.resume_filename,
                            "Hard Match %": "Error",
                            "Soft Match %": "Error",
                            "ATS Score": "Error",
                            "Missing": f"Analysis failed: {str(e)}",
                            "Verdict": "Error",
                            "Final Status": "Error"
                        })
                
                if results:
                    st.success("Resume analysis completed!")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Display feedback for selected applicant
                    if results:
                        selected_app = st.selectbox(
                            "Select applicant to view feedback",
                            options=[f"{app.name} - {app.job_role}" for app in applications],
                            key="feedback_selector"
                        )
                        
                        if selected_app:
                            # Find the corresponding application
                            selected_app_obj = None
                            for app in applications:
                                if f"{app.name} - {app.job_role}" == selected_app:
                                    selected_app_obj = app
                                    break
                            
                            if selected_app_obj and selected_app_obj.id in st.session_state.get('applicant_feedback', {}):
                                st.subheader(f"Feedback for {selected_app_obj.name}")
                                st.text_area(
                                    "Feedback Message",
                                    st.session_state.applicant_feedback[selected_app_obj.id],
                                    height=200,
                                    key=f"feedback_{selected_app_obj.id}"
                                )
                                
                                # Email sending option
                                if st.button(f"📧 Send Feedback to {selected_app_obj.email}", key=f"email_{selected_app_obj.id}"):
                                    with st.spinner("Sending email..."):
                                        success, message = send_feedback_email(
                                            student_email=selected_app_obj.email,
                                            student_name=selected_app_obj.name,
                                            job_title=selected_app_obj.job_role,
                                            score=next((item['ATS Score'] for item in results if item['Resume'] == selected_app_obj.resume_filename), 0),
                                            missing_skills=next((item['Missing'] for item in results if item['Resume'] == selected_app_obj.resume_filename), ""),
                                            suggestions=st.session_state.applicant_feedback[selected_app_obj.id],
                                            company_name="Your Company Name"
                                        )
                                        
                                        if success:
                                            st.success("✅ Email sent successfully!")
                                        else:
                                            st.error(f"❌ {message}")
                else:
                    st.warning("No results generated from analysis.")
        elif not jds:
            st.info("No job descriptions found. Please add job descriptions first to use the resume comparison feature.")
        elif not applications:
            st.info("No applications found. Please wait for applicants to submit their resumes.")
    
    with tab2:
        st.header("📊 Analytics Dashboard")
        
        # Get applications data
        applications = get_all_applications()
        
        if applications:
            # Application metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                create_metric_card("Total Applications", len(applications))
            
            with col2:
                recent_apps = [app for app in applications if (datetime.utcnow() - app.created_at).days <= 7]
                create_metric_card("This Week", len(recent_apps), f"+{len(recent_apps)}")
            
            with col3:
                unique_roles = len(set(app.job_role for app in applications))
                create_metric_card("Job Roles", unique_roles)
            
            with col4:
                avg_per_day = len(applications) / max(1, (datetime.utcnow() - min(app.created_at for app in applications)).days)
                create_metric_card("Avg/Day", f"{avg_per_day:.1f}")
            
            # Applications by job role chart
            st.subheader("Applications by Job Role")
            
            try:
                session = db.SessionLocal()
                role_counts = session.query(
                    db.Application.job_role,
                    func.count(db.Application.id)  # Fixed: use func.count instead of db.func.count
                ).group_by(db.Application.job_role).all()
                
                if role_counts:
                    role_df = pd.DataFrame(role_counts, columns=['Job Role', 'Count'])
                    st.bar_chart(role_df.set_index('Job Role'))
                else:
                    st.info("No application data available for charting")
            except Exception as e:
                st.error(f"Error generating chart: {str(e)}")
            finally:
                session.close()
            
            # Applications over time
            st.subheader("Applications Over Time")
            
            try:
                # Get daily application counts
                daily_counts = {}
                for app in applications:
                    date_str = app.created_at.strftime('%Y-%m-%d')
                    daily_counts[date_str] = daily_counts.get(date_str, 0) + 1
                
                if daily_counts:
                    time_df = pd.DataFrame(list(daily_counts.items()), columns=['Date', 'Count'])
                    time_df['Date'] = pd.to_datetime(time_df['Date'])
                    time_df = time_df.sort_values('Date')
                    st.line_chart(time_df.set_index('Date'))
                else:
                    st.info("No time series data available")
            except Exception as e:
                st.error(f"Error generating time series: {str(e)}")
            
            # Gender distribution
            st.subheader("Gender Distribution")
            
            gender_counts = {}
            for app in applications:
                gender_counts[app.gender] = gender_counts.get(app.gender, 0) + 1
            
            if gender_counts:
                gender_df = pd.DataFrame(list(gender_counts.items()), columns=['Gender', 'Count'])
                st.plotly_chart({
                    'data': [{
                        'values': gender_df['Count'],
                        'labels': gender_df['Gender'],
                        'type': 'pie',
                        'hole': 0.4,
                    }],
                    'layout': {
                        'title': 'Applications by Gender'
                    }
                }, use_container_width=True)
            else:
                st.info("No gender data available")
        
        else:
            st.info("No applications data available for analytics")

# -----------------------------
# Application Entry Point
# -----------------------------
if __name__ == "__main__":
    main()