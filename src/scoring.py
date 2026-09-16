def calculate_overall_score(
    text_similarity: float,
    semantic_similarity: float,
    skill_match: float,
    experience_match: float,
    education_match: float
):
    """
    Calculate the final resume-job match score.

    Weights:
    Skill Match          = 45%
    Semantic Similarity  = 20%
    Text Similarity      = 10%
    Experience Match     = 15%
    Education Match      = 10%
    """

    score = (
        text_similarity * 0.10
        + semantic_similarity * 0.20
        + skill_match * 0.45
        + experience_match * 0.15
        + education_match * 0.10
    )

    return round(score, 2)


def get_match_category(score: float):
    """
    Convert the final score into a match category.
    """

    if score >= 75:
        return "Strong Match"

    elif score >= 50:
        return "Moderate Match"

    elif score >= 30:
        return "Weak Match"

    else:
        return "Low Match"
