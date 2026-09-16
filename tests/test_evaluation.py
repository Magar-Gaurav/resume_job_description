from resume_job_description.src.pdf_extractor import extract_text_from_pdf
from resume_job_description.src.preprocessing import preprocess_text
from resume_job_description.src.text_vectorizer import create_tfidf_vectors
from resume_job_description.src.similarity import calculate_similarity, similarity_to_percentage
from resume_job_description.src.semantic_matcher import (
    load_model,
    calculate_semantic_similarity,
    semantic_similarity_to_percentage,
)
from resume_job_description.src.skill_matcher import (
    extract_skills,
    compare_skills,
    calculate_skill_match,
)
from resume_job_description.src.experience import (
    extract_required_experience,
    extract_candidate_experience,
    calculate_experience_match,
)
from resume_job_description.src.education import calculate_education_match
from resume_job_description.src.scoring import calculate_overall_score


# ============================================================
# Load and preprocess resume
# ============================================================

pdf_path = "data/raw/resume.pdf"

raw_resume = extract_text_from_pdf(pdf_path)
resume_text = preprocess_text(raw_resume)


# ============================================================
# Load the semantic model once, up front
# ============================================================

print("Loading semantic model...")
semantic_model = load_model()


# ============================================================
# Evaluation Job Descriptions
# ============================================================

job_descriptions = {

    "Strong ML Match": """
    We are looking for a Junior Machine Learning Engineer.

    Strong knowledge of Python, Pandas, NumPy, and Scikit-learn
    is required.

    The candidate should have experience with machine learning,
    data preprocessing, exploratory data analysis,
    feature engineering, classification, regression,
    and model evaluation.

    Knowledge of SQL, FastAPI, REST APIs, and PostgreSQL
    is preferred.
    """,

    "Moderate ML Match": """
    We are looking for a Data Science Intern.

    The candidate should know Python, Pandas, NumPy,
    statistics, data preprocessing, and machine learning.

    Experience with SQL and data analysis is preferred.

    Knowledge of TensorFlow and Docker would be an advantage.
    """,

    "Poor ML Match": """
    We are looking for a Frontend Web Developer.

    The candidate should have strong experience with HTML,
    CSS, JavaScript, React, TypeScript, and responsive
    web application development.

    Experience with Next.js, Tailwind CSS, and frontend
    testing is preferred.
    """,

    "Backend Match": """
    We are looking for a Backend Developer.

    The candidate should have experience with Python,
    FastAPI, REST APIs, SQL, PostgreSQL, Git, and
    database development.

    Experience with Docker, AWS, and Kubernetes
    is preferred.
    """,

    "AI / GenAI Match": """
    We are looking for an AI Engineer.

    The candidate should have knowledge of Python,
    machine learning, LLMs, Generative AI, RAG,
    prompt engineering, and vector databases.

    Experience with FastAPI and PostgreSQL is preferred.
    """
}


# ============================================================
# Evaluate each job
# ============================================================

print("\n" + "=" * 70)
print("              RESUME-JOB MATCH EVALUATION")
print("=" * 70)


for job_name, job_description in job_descriptions.items():

    # --------------------------------------------------------
    # Preprocess job description
    # --------------------------------------------------------

    clean_job_description = preprocess_text(job_description)

    # --------------------------------------------------------
    # TF-IDF cosine similarity
    # --------------------------------------------------------

    resume_vector, job_vector, vectorizer = create_tfidf_vectors(
        resume_text,
        clean_job_description
    )

    cosine_score = calculate_similarity(
        resume_vector,
        job_vector
    )

    text_similarity = similarity_to_percentage(
        cosine_score
    )

    # --------------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------------

    semantic_score = calculate_semantic_similarity(
        resume_text,
        clean_job_description,
        semantic_model,
    )

    semantic_similarity = semantic_similarity_to_percentage(
        semantic_score
    )

    # --------------------------------------------------------
    # Skill extraction
    # --------------------------------------------------------

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(clean_job_description)

    # --------------------------------------------------------
    # Skill comparison
    # --------------------------------------------------------

    comparison = compare_skills(
        resume_skills,
        job_skills
    )

    skill_match = calculate_skill_match(
        resume_skills,
        job_skills
    )

    # --------------------------------------------------------
    # Experience match
    # --------------------------------------------------------

    required_experience = extract_required_experience(
        clean_job_description
    )

    candidate_experience = extract_candidate_experience(
        resume_text
    )

    experience_match = calculate_experience_match(
        candidate_experience,
        required_experience
    )

    # --------------------------------------------------------
    # Education match
    # --------------------------------------------------------

    education_match, candidate_education, required_education = (
        calculate_education_match(
            resume_text,
            clean_job_description
        )
    )

    # --------------------------------------------------------
    # Overall score
    # --------------------------------------------------------

    overall_score = calculate_overall_score(
        text_similarity,
        semantic_similarity,
        skill_match,
        experience_match,
        education_match
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(f"JOB: {job_name}")
    print("-" * 70)

    print(f"Text Similarity     : {text_similarity:.2f}%")
    print(f"Semantic Similarity : {semantic_similarity:.2f}%")
    print(f"Skill Match         : {skill_match:.2f}%")
    print(f"Experience Match    : {experience_match:.2f}%")
    print(f"Education Match     : {education_match:.2f}%")
    print(f"Overall Score       : {overall_score:.2f}%")

    print("\nMatched Skills:")

    if comparison["matched"]:
        for skill in comparison["matched"]:
            print(f"  match {skill}")
    else:
        print("  None")

    print("\nMissing Skills:")

    if comparison["missing"]:
        for skill in comparison["missing"]:
            print(f"  missing {skill}")
    else:
        print("  None")
