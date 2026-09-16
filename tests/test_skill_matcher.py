from resume_job_description.src.pdf_extractor import extract_text_from_pdf
from resume_job_description.src.preprocessing import preprocess_text
from resume_job_description.src.skill_matcher import extract_skills, compare_skills, calculate_skill_match

# Resume
pdf_path = "data/raw/resume.pdf"

raw_resume = extract_text_from_pdf(pdf_path)
preprocessed_resume = preprocess_text(raw_resume)

# Job Description
job_description = """
We are looking for a Junior Machine Learning Engineer.

Requirements:

Strong programming skills in Python.
Experience with Pandas, NumPy, and Scikit-learn.

Knowledge of machine learning, data preprocessing,
feature engineering, classification, regression,
and model evaluation.

Experience with SQL, FastAPI, REST APIs, and PostgreSQL.

The candidate should also have experience with
Docker, AWS, Kubernetes, PyTorch, and Apache Airflow.

Experience deploying machine learning applications
to cloud environments is preferred.
"""

clean_job_description = preprocess_text(job_description)

# Extract skills
resume_skills = extract_skills(preprocessed_resume)
job_skills = extract_skills(clean_job_description)

# Compare skills
comparison_result = compare_skills(resume_skills, job_skills)

# Calculate skill match percentage
skill_match_percentage = calculate_skill_match(resume_skills, job_skills)

# Display Result
print("\n==== Resume Skills ====\n")

for skill in resume_skills:
    print(f"- {skill}")

print("\n==== Matched Skills ====\n")
for skill in comparison_result["matched"]:
    print(f"- {skill}")

print(f"\n==== Missing Skills ====\n")
for skill in comparison_result["missing"]:
    print(f"- {skill}")

print(f"\n==== Additional Skills ====\n")
for skill in comparison_result["additional"]:
    print(f"- {skill}")

print(f"\n==== Skill Match Percentage ====\n")
print(f"{skill_match_percentage:.2f}% of required job skills are present in the resume.")