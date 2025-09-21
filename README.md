# RecruitSmart-AI

RecruitSmart-AI is an AI-powered recruitment assistant that helps streamline the hiring process by automating candidate screening, analysis, and shortlisting. Leveraging machine learning and natural language processing (NLP), this project aims to make recruitment faster, smarter, and more data-driven.

![Uploading Screenshot 2025-09-21 205723.png…]()

Table of Contents

Features

Demo

Installation

Usage

Technologies Used

Contributing

License

Features

Resume Parsing: Extracts key information (skills, experience, education) from candidate resumes.

Candidate Scoring: Uses AI/ML algorithms to score candidates based on job requirements.

Automated Shortlisting: Quickly generates a shortlist of top candidates.

Job Matching: Recommends suitable candidates for open positions using semantic analysis.

Interactive Dashboard: Visualizes candidate analytics and performance metrics.

Demo

You can view the live demo here: RecruitSmart-AI Demo

<!-- Optional screenshot -->

Installation
Prerequisites

Python 3.10+

pip

virtualenv (optional but recommended)

Steps

Clone the repository:

git clone https://github.com/VivekMKaushik/RecruitSmart-AI.git
cd RecruitSmart-AI


Create a virtual environment (optional):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt


Run the application:

python app.py


Open your browser and go to http://localhost:5000 (or the port specified).

Usage

Upload candidate resumes (PDF/DOCX).

Define job descriptions and required skills.

Let the AI parse and score candidates automatically.

View shortlisted candidates in the dashboard.

Technologies Used

Backend: Python, Flask/FastAPI (depending on project)

NLP & ML: spaCy, scikit-learn, Pandas, NumPy

Frontend: HTML, CSS, JavaScript (or Streamlit/React if used)

Database: SQLite/PostgreSQL (if applicable)

Contributing

Contributions are welcome! To contribute:

Fork the repository.

Create a new branch (git checkout -b feature-name).

Make your changes and commit (git commit -m 'Add feature').

Push to the branch (git push origin feature-name).

Open a Pull Request.

License

This project is licensed under the MIT License. See the LICENSE
 file for details.

✅ Optional Enhancements:

Add badges for build status, license, and Python version.

Add screenshots of resume upload, scoring dashboard, and analytics.

Provide links to API documentation if any.
