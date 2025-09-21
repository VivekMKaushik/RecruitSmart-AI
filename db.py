# db.py
import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, JSON, DateTime, ForeignKey
)
from sqlalchemy import func
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import streamlit as st
from PyPDF2 import PdfReader
import docx

# ✅ Correct DATABASE_URL setup
DATABASE_URL = os.getenv("DATABASE_URL") or (
    f"postgresql://{os.getenv('POSTGRES_USER','pguser')}:"
    f"{os.getenv('POSTGRES_PASSWORD','pgpass')}@"
    f"{os.getenv('POSTGRES_HOST','localhost')}:" 
    f"{os.getenv('POSTGRES_PORT','5432')}/"
    f"{os.getenv('POSTGRES_DB','resumes')}"
)

# Try to connect to the configured DATABASE_URL; if it fails, fall back to a
# local SQLite file (convenient for development & smoke tests).
print("Attempting to connect to:", DATABASE_URL)

# Prefer a short connect attempt so import won't hang for long
from sqlalchemy.exc import SQLAlchemyError

def _make_engine_with_fallback(url: str):
    try:
        # For Postgres, allow a small connect timeout; SQLAlchemy will pass
        # this through to the driver when appropriate.
        connect_args = {}
        if url.startswith("postgresql"):
            # psycopg2 accepts connect_timeout via query string or connect_args
            connect_args = {"connect_timeout": 5}

        eng = create_engine(url, echo=False, connect_args=connect_args)

        # Try a quick connect; this will raise if DB not reachable.
        conn = eng.connect()
        conn.close()
        print("Connected successfully to:", url)
        return eng
    except SQLAlchemyError as e:
        print("Warning: cannot connect to configured DATABASE_URL. Falling back to SQLite. Error:", e)
        sqlite_url = "sqlite:///./resumes.db"
        # SQLite needs check_same_thread disabled for some threaded use-cases
        eng = create_engine(sqlite_url, echo=False, connect_args={"check_same_thread": False})
        print("Using SQLite fallback at:", sqlite_url)
        return eng

# SQLAlchemy setup with fallback
engine = _make_engine_with_fallback(DATABASE_URL)

# Ensure the connected database has the expected columns on the `resumes` table.
# If the table exists but is missing expected columns (schema mismatch from older
# migrations), fall back to the local SQLite file to keep the app runnable.
def _ensure_resumes_table_compatibility(eng):
    try:
        from sqlalchemy import inspect
        insp = inspect(eng)
        # If table doesn't exist yet, we'll allow create_all to provision it later
        if not insp.has_table('resumes'):
            return eng

        cols = {c['name'] for c in insp.get_columns('resumes')}
        # These columns are expected by the current ORM model. If any are missing,
        # we consider the schema incompatible for safe operation against Postgres.
        expected = {'id', 'filename', 'parsed', 'uploaded_at', 'candidate_email', 'candidate_name', 'candidate_phone'}
        if not expected.issubset(cols):
            print("Warning: resumes table exists but is missing expected columns:", expected - cols)
            print("Falling back to SQLite to avoid runtime SQL errors. Consider migrating the Postgres schema.")
            sqlite_url = "sqlite:///./resumes.db"
            new_eng = create_engine(sqlite_url, echo=False, connect_args={"check_same_thread": False})
            return new_eng
        return eng
    except Exception as e:
        print("Error inspecting database schema; falling back to SQLite. Error:", e)
        sqlite_url = "sqlite:///./resumes.db"
        new_eng = create_engine(sqlite_url, echo=False, connect_args={"check_same_thread": False})
        return new_eng


engine = _ensure_resumes_table_compatibility(engine)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ORM Models
class JobDescription(Base):
    __tablename__ = 'job_descriptions'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True)
    title = Column(String(256))
    description = Column(Text)
    parsed = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Resume(Base):
    __tablename__ = 'resumes'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True)
    filename = Column(String(512))
    parsed = Column(JSON)  # This will store the structured resume data
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    # Contact fields for feedback and filtering
    candidate_email = Column(String(256))
    candidate_name = Column(String(256))
    candidate_phone = Column(String(64))

class Evaluation(Base):
    __tablename__ = 'evaluations'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True)
    resume_id = Column(Integer, ForeignKey('resumes.id'))
    jd_id = Column(Integer, ForeignKey('job_descriptions.id'))
    score = Column(Integer)
    missing = Column(JSON)
    explanation = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    resume = relationship("Resume")
    jd = relationship("JobDescription")


class Application(Base):
    __tablename__ = 'applications'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True)
    name = Column(String(256))
    phone = Column(String(64))
    email = Column(String(256))
    gender = Column(String(32))
    job_role = Column(String(128))
    resume_filename = Column(String(512))
    created_at = Column(DateTime, default=datetime.utcnow)


class JobRequirement(Base):
    __tablename__ = 'job_requirements'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True)
    title = Column(String(256))
    company = Column(String(256))
    description = Column(Text)
    must_have_skills = Column(JSON)  # List of required skills
    good_to_have_skills = Column(JSON)  # List of preferred skills
    qualifications = Column(JSON)  # List of required qualifications
    location = Column(String(128))
    uploaded_by = Column(String(256))
    created_at = Column(DateTime, default=datetime.utcnow)


class ResumeAnalysis(Base):
    __tablename__ = 'resume_analyses'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True)
    resume_id = Column(Integer, ForeignKey('resumes.id'))
    jd_id = Column(Integer, ForeignKey('job_requirements.id'))
    relevance_score = Column(Integer)
    hard_match_score = Column(Integer)
    semantic_match_score = Column(Integer)
    missing_skills = Column(JSON)
    missing_qualifications = Column(JSON)
    verdict = Column(String(50))  # High/Medium/Low
    suggestions = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    # candidate_location column was optional in earlier iterations.
    # If you no longer need filtering by candidate location, comment out
    # or remove the column definition. Keep the DB migration helpers
    # intact if you plan to run migrations separately.
    # candidate_location = Column(String(128))
    
    # Relationships
    resume = relationship("Resume")
    job_requirement = relationship("JobRequirement")


def save_application(db_session, name, phone, email, gender, job_role, resume_filename):
    app = Application(
        name=name,
        phone=phone,
        email=email,
        gender=gender,
        job_role=job_role,
        resume_filename=resume_filename
    )
    db_session.add(app)
    db_session.commit()
    db_session.refresh(app)
    return app


# -----------------------------
# Helper & File functions
# -----------------------------
def get_db_session():
    return SessionLocal()


def get_all_applications():
    """Retrieve all applications from the database"""
    session = get_db_session()
    try:
        applications = session.query(Application).order_by(Application.created_at.desc()).all()
        return applications
    except Exception as e:
        st.error(f"Error retrieving applications: {str(e)}")
        return []
    finally:
        session.close()


def save_application_to_db(name, phone, email, gender, job_role, resume_filename):
    """Save application to database"""
    session = get_db_session()
    try:
        application = Application(
            name=name,
            phone=phone,
            email=email,
            gender=gender,
            job_role=job_role,
            resume_filename=resume_filename
        )
        session.add(application)
        session.commit()
        session.refresh(application)
        return True, f"Application saved with ID: {application.id}"
    except Exception as e:
        return False, f"Error saving application: {str(e)}"
    finally:
        session.close()


def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file (PDF, DOCX, TXT)"""
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
        st.error(f"Error extracting text: {e}")
        return ""


def save_resume(filename, parsed_text):
    """Save resume to database"""
    session = get_db_session()
    try:
        resume = Resume(
            filename=filename,
            parsed={"text": parsed_text},
            uploaded_at=datetime.utcnow()
        )
        session.add(resume)
        session.commit()
        session.refresh(resume)
        return resume.id
    except Exception as e:
        st.error(f"Error saving resume: {e}")
        return None
    finally:
        session.close()

# Function to initialize the database
def init_db():
    Base.metadata.create_all(bind=engine)
    # After creating tables, ensure `resumes` has the expected contact columns.
    _migrate_resumes_add_contact_columns(engine)
    # Ensure resume_analyses has expected columns introduced in newer schema
    # The candidate_location column was removed from the ORM model.
    # If you want to manage DB schema separately, run migrations externally.
    # _migrate_resume_analyses_add_candidate_location(engine)


def _migrate_resumes_add_contact_columns(eng):
    """Add missing contact columns to the `resumes` table if they don't exist.

    This function is intentionally conservative:
    - It inspects the `resumes` table columns and only issues ALTER TABLE
      statements for columns that are missing.
    - It supports both Postgres and SQLite. For SQLite the ALTER TABLE
      will add a new column (SQLite supports simple ADD COLUMN operations).
    - Errors are caught and printed but do not raise, to avoid breaking
      application startup in noisy environments.
    """
    try:
        from sqlalchemy import inspect, text
        insp = inspect(eng)
        if not insp.has_table('resumes'):
            # nothing to do
            return

        existing = {c['name'] for c in insp.get_columns('resumes')}
        required = {
            'candidate_email': "VARCHAR(256)",
            'candidate_name': "VARCHAR(256)",
            'candidate_phone': "VARCHAR(64)"
        }

        missing = [col for col in required.keys() if col not in existing]
        if not missing:
            return

        # Determine dialect to pick appropriate SQL
        dialect = eng.dialect.name.lower()
        with eng.connect() as conn:
            for col in missing:
                col_type = required[col]
                try:
                    if dialect.startswith('postgres'):
                        stmt = text(f"ALTER TABLE resumes ADD COLUMN IF NOT EXISTS {col} {col_type};")
                        conn.execute(stmt)
                    else:
                        # SQLite and other DBs: try a simple ADD COLUMN
                        stmt = text(f"ALTER TABLE resumes ADD COLUMN {col} {col_type};")
                        conn.execute(stmt)
                    print(f"Added missing column '{col}' to resumes table.")
                except Exception as e:
                    print(f"Warning: failed to add column {col} to resumes: {e}")
    except Exception as e:
        print(f"Error during resumes schema migration: {e}")

def get_filtered_applications(job_filter="All", min_score=0, verdict_filter="All"):
    """Get filtered applications from database"""
    session = SessionLocal()
    try:
        query = session.query(
            ResumeAnalysis.id,
            Resume.filename.label('candidate_name'),
            JobRequirement.title.label('job_title'),
            ResumeAnalysis.relevance_score.label('score'),
            ResumeAnalysis.verdict,
            ResumeAnalysis.missing_skills,
            ResumeAnalysis.created_at.label('applied_date')
        ).join(Resume).join(JobRequirement)
        
        if job_filter != "All":
            query = query.filter(JobRequirement.title == job_filter)
        
        if verdict_filter != "All":
            query = query.filter(ResumeAnalysis.verdict == verdict_filter)
        
        query = query.filter(ResumeAnalysis.relevance_score >= min_score)
        
        return query.all()
    except Exception as e:
        st.error(f"Error fetching applications: {str(e)}")
        return []
    finally:
        session.close()

def get_total_applications():
    """Get total number of applications"""
    session = SessionLocal()
    try:
        return session.query(ResumeAnalysis).count()
    except Exception:
        return 0
    finally:
        session.close()

def get_average_score():
    """Get average score of all applications"""
    session = SessionLocal()
    try:
        avg = session.query(func.avg(ResumeAnalysis.relevance_score)).scalar()
        return avg or 0
    except Exception:
        return 0
    finally:
        session.close()


def _migrate_resume_analyses_add_candidate_location(eng):
    """Add missing candidate_location column to resume_analyses if not present.

    This mirrors the conservative approach used for resumes contact columns.
    """
    # No-op: candidate_location column is intentionally not managed by in-app migrations.
    # Keep this function for backward compatibility if you later want to re-enable it.
    return


def migrate_database_schema():
    """Migrate database schema to match current models.

    This function performs a few lightweight, non-destructive migrations such as
    adding missing columns that are required by the current ORM models.
    It uses the module-level `engine` and is safe to call at startup.
    """
    try:
        from sqlalchemy import text
        from sqlalchemy import inspect as sqla_inspect

        # Get database engine
        engine_local = None
        try:
            engine_local = create_engine(DATABASE_URL)
        except Exception:
            # Fallback to module engine if DATABASE_URL-driven engine can't be created
            engine_local = engine

        inspector = sqla_inspect(engine_local)

        # Define expected columns for each table
        expected_columns = {
            'resumes': ['candidate_email', 'candidate_name', 'candidate_phone'],
            # Add other tables and expected columns as needed
        }

        for table_name, columns in expected_columns.items():
            if not inspector.has_table(table_name):
                continue

            existing_columns = [col['name'] for col in inspector.get_columns(table_name)]

            for column in columns:
                if column not in existing_columns:
                    print(f"Adding missing column: {column} to {table_name} table")
                    with engine_local.connect() as conn:
                        # Add the missing column with appropriate type
                        if table_name == 'resume_analyses' and column == 'candidate_location':
                            sql = f"ALTER TABLE {table_name} ADD COLUMN {column} VARCHAR(128)"
                        elif table_name == 'resumes' and column in ['candidate_email', 'candidate_name']:
                            sql = f"ALTER TABLE {table_name} ADD COLUMN {column} VARCHAR(256)"
                        elif table_name == 'resumes' and column == 'candidate_phone':
                            sql = f"ALTER TABLE {table_name} ADD COLUMN {column} VARCHAR(64)"
                        else:
                            # Default to text type for unknown columns
                            sql = f"ALTER TABLE {table_name} ADD COLUMN {column} TEXT"

                        conn.execute(text(sql))
                        print(f"✅ Added {column} column to {table_name}")

        print("✅ Database schema migration completed successfully")

    except Exception as e:
        print(f"❌ Database migration failed: {e}")
"""
Duplicates removed: merged fields into the original `Resume` and `ResumeAnalysis` classes above.
"""