def generate_skill_recommendations(missing_skills: list[str]) -> list[str]:
    """
    Build a list of recommendation messages for skills
    that are required by the job but missing from the resume.

    Args:
        missing_skills: Skills required by the job but not found
            in the resume.

    Returns:
        A list of human-readable recommendation strings. The first
        line is always a summary message.
    """

    if not missing_skills:
        return [
            "Your resume contains all detected required "
            "skills from this job description."
        ]

    recommendations = [
        "Consider improving the following skills "
        "to increase the match with this job:"
    ]

    for skill in missing_skills:
        recommendations.append(f"Learn or strengthen: {skill}")

    return recommendations


def generate_experience_recommendation(
    candidate_years: int,
    required_years: int,
) -> str | None:
    """
    Build a recommendation message about an experience gap.

    Args:
        candidate_years: Years of experience detected in the resume.
        required_years: Years of experience required by the job
            (0 means not specified).

    Returns:
        A recommendation string if the candidate falls short of the
        required experience, otherwise None.
    """

    if required_years > 0 and candidate_years < required_years:
        return (
            f"Required experience is {required_years} years, "
            f"while {candidate_years} years were detected "
            f"in the resume."
        )

    return None


def generate_education_recommendation(
    required_education: str | None,
    candidate_education: str | None,
    education_match: float,
) -> str | None:
    """
    Build a recommendation message about an education gap.

    Args:
        required_education: Highest education level detected in the
            job description.
        candidate_education: Highest education level detected in the
            resume.
        education_match: Education match percentage (0-100).

    Returns:
        A recommendation string if the candidate's education does
        not fully meet the requirement, otherwise None.
    """

    if (
        required_education
        and candidate_education
        and education_match < 100
    ):
        return (
            "The detected education level does not "
            "fully meet the required education level."
        )

    return None
