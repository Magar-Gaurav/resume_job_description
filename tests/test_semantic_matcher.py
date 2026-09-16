from resume_job_description.src.semantic_matcher import (
    load_model,
    calculate_semantic_similarity,
    semantic_similarity_to_percentage,
)


resume_text = """
I am a Python developer with experience in
machine learning, data analysis, Pandas,
NumPy, Scikit-learn and predictive model
development.
"""


job_description = """
We are looking for a data scientist with
experience in Python, machine learning,
data analysis and building predictive models.
"""


print("Loading model...")

model = load_model()

print("\nCalculating semantic similarity...")

similarity_score = calculate_semantic_similarity(
    resume_text,
    job_description,
    model,
)

similarity_percentage = (
    semantic_similarity_to_percentage(
        similarity_score
    )
)

print("\n===== SEMANTIC MATCHING RESULT =====")

print(
    f"Similarity score: "
    f"{similarity_score:.4f}"
)

print(
    f"Match percentage: "
    f"{similarity_percentage:.2f}%"
)