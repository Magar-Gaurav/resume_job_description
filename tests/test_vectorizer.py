from resume_job_description.src.pdf_extractor import extract_text_from_pdf
from resume_job_description.src.preprocessing import preprocess_text
from resume_job_description.src.text_vectorizer import create_tfidf_vectors
from resume_job_description.src.similarity import calculate_similarity, similarity_to_percentage


# Resume
pdf_path = "data/raw/resume.pdf"

raw_resume = extract_text_from_pdf(pdf_path)
resume_text = preprocess_text(raw_resume)


# Sample Job Description
job_description = """
We are looking for a Junior Machine Learning Engineer.

The candidate should have strong knowledge of Python,
Pandas, NumPy, and Scikit-learn.

Experience with data preprocessing, exploratory data analysis,
feature engineering, classification, regression, and model evaluation
is required.

Knowledge of SQL, FastAPI, REST APIs, and PostgreSQL is preferred.

Experience with machine learning projects and deploying ML solutions
is an advantage.
"""

# Preprocess job description
clean_job_description = preprocess_text(job_description)


# Create TF-IDF vectors
resume_vector, job_vector, vectorizer = create_tfidf_vectors(
    resume_text,
    clean_job_description
)


print("\n===== TF-IDF INFORMATION =====")

print("Number of features:", len(vectorizer.get_feature_names_out()))

print("\n===== FIRST 30 FEATURES =====")

features = vectorizer.get_feature_names_out()

for feature in features[:30]:
    print(feature)


print("\n===== VECTOR SHAPES =====")

print("Resume vector shape:", resume_vector.shape)
print("Job vector shape: ", job_vector.shape)

# Calculate cosine similarity
similarity_score = calculate_similarity(
    resume_vector,
    job_vector
)

# Convert to percentage
match_percentage = similarity_to_percentage(
    similarity_score
)

print("\n===== SIMILARITY RESULT =====")

print("Cosine similarity:", similarity_score)
print("Match percentage:", f"{match_percentage}%")