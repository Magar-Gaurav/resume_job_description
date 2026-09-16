EDUCATION_LEVELS = {
    "phd": 4,
    "doctorate": 4,

    "master's": 3,
    "masters": 3,
    "master": 3,
    "mca": 3,
    "m.tech": 3,
    "mtech": 3,
    "mba": 3,

    "bachelor's": 2,
    "bachelors": 2,
    "bachelor": 2,
    "bca": 2,
    "b.tech": 2,
    "btech": 2,
    "bsc": 2,
    "b.sc": 2,
    "bba": 2,
    "ba": 2,
    "b.com": 2,
    "bcom": 2,

    "diploma": 1,
}


def extract_education_level(text: str):
    """
    Detect the highest education level mentioned in the text.
    """

    text = text.lower()

    detected = None
    highest_level = 0

    for education, level in EDUCATION_LEVELS.items():

        if education in text:

            if level > highest_level:
                detected = education
                highest_level = level

    return detected, highest_level


def calculate_education_match(
    resume_text: str,
    job_text: str
):
    """
    Compare education level in resume and job description.
    """

    candidate_education, candidate_level = (
        extract_education_level(resume_text)
    )

    required_education, required_level = (
        extract_education_level(job_text)
    )

    # If education is not mentioned in the job,
    # do not penalize the candidate.
    if required_level == 0:

        return (
            100.0,
            candidate_education,
            required_education
        )

    # Candidate has the required or higher level.
    if candidate_level >= required_level:

        score = 100.0

    else:

        score = (
            candidate_level / required_level
        ) * 100

    return (
        round(score, 2),
        candidate_education,
        required_education
    )