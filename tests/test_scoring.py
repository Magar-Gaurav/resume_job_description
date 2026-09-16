from resume_job_description.src.scoring import calculate_overall_score

text_similarity = 28.76
semantic_similarity = 65.42
skill_match = 73.68
experience_match = 100.0
education_match = 100.0

overall_score = calculate_overall_score(
    text_similarity,
    semantic_similarity,
    skill_match,
    experience_match,
    education_match,
)

print("\n==== Final Score ====\n")

print("Text Similarity :     ", f"{text_similarity:.2f}%")
print("Semantic Similarity : ", f"{semantic_similarity:.2f}%")
print("Skill Match :         ", f"{skill_match:.2f}%")
print("Experience Match :    ", f"{experience_match:.2f}%")
print("Education Match :     ", f"{education_match:.2f}%")
print("Overall Match Score : ", f"{overall_score:.2f}%")
