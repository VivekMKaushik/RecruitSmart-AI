# test_db.py
from db import init_db, SessionLocal, Resume, JobDescription, Evaluation

def main():
    # Step 1: Initialize tables
    print("📌 Initializing database...")
    init_db()

    session = SessionLocal()

    try:
        # Step 2: Insert dummy resume
        resume = Resume(filename="resume1.pdf", parsed={"skills": ["Python", "SQL"]})
        session.add(resume)
        session.commit()
        print(f"✅ Inserted Resume with ID: {resume.id}")

        # Step 3: Insert dummy JD
        jd = JobDescription(
            title="Data Analyst",
            description="Looking for SQL + Python",
            parsed={"skills": ["Python", "SQL"]}
        )
        session.add(jd)
        session.commit()
        print(f"✅ Inserted JobDescription with ID: {jd.id}")

        # Step 4: Insert dummy evaluation
        evaluation = Evaluation(
            resume_id=resume.id,
            jd_id=jd.id,
            score=85,
            missing={"skills": ["Tableau"]},
            explanation="Good match but lacks Tableau"
        )
        session.add(evaluation)
        session.commit()
        print(f"✅ Inserted Evaluation with ID: {evaluation.id}")

        # Step 5: Query back
        print("\n📌 Querying data back...")
        for e in session.query(Evaluation).all():
            print(f"Evaluation {e.id}: Score={e.score}, Missing={e.missing}, Explanation={e.explanation}")

    finally:
        session.close()

if __name__ == "__main__":
    main()
