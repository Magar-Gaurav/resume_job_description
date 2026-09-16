from resume_job_description.src.experience import (
    extract_required_experience,
    extract_candidate_experience,
    calculate_experience_match,
)

# Resume
resume_text = """
i am a software engineer with 3 years of experience
building web applications using python, django, and postgresql
"""

# Job Description
job_text = """
the ideal candidate should have at least 5 years of experience
in backend software development
"""

# Extract years of experience
candidate_years = extract_candidate_experience(resume_text)
required_years = extract_required_experience(job_text)

# Calculate experience match
experience_match = calculate_experience_match(
    candidate_years,
    required_years,
)

# Display results
print("\n==== Extracted Experience ====\n")
print(f"Candidate Experience : {candidate_years} years")
print(f"Required Experience  : {required_years} years")

print("\n==== Experience Match ====\n")
print(f"Experience Match : {experience_match:.2f}%")
