from resume_job_description.src.education import (
    extract_education_level,
    calculate_education_match,
)

# Resume
resume_text = """
i hold a bachelor of technology degree in computer science
and have completed additional coursework toward a master
of science in data science
"""

# Job Description
job_text = """
the ideal candidate should have a master's degree in computer
science, data science, or a related field
"""

# Detect education levels independently
candidate_education, candidate_level = extract_education_level(resume_text)
required_education, required_level = extract_education_level(job_text)

# Compare resume and job education
education_match, detected_candidate, detected_required = calculate_education_match(
    resume_text,
    job_text,
)

# Display results
print("\n==== Detected Education Levels ====\n")
print(f"Candidate Education : {candidate_education} (level {candidate_level})")
print(f"Required Education  : {required_education} (level {required_level})")

print("\n==== Education Match ====\n")
print(f"Candidate Education : {detected_candidate}")
print(f"Required Education  : {detected_required}")
print(f"Education Match     : {education_match:.2f}%")
