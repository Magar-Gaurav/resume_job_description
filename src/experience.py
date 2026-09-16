import re


# Matches phrasing like:
#   "3 years of experience"
#   "3+ years experience"
#   "2-4 years of professional experience"     (hyphenated range)
#   "2 4 years of relevant experience"         (range where the
#       connector, e.g. an en dash or the word "to", was stripped
#       out earlier in preprocessing)
#   "5 years of hands-on industry experience"  (up to 3 filler
#       words - adjectives like "professional", "relevant",
#       "hands-on" - between "years" and "experience")
#
# Group 1 captures the first (i.e. lower-bound, for a range) number
# of years, which is used as the required/detected experience.
YEARS_EXPERIENCE_PATTERN = (
    r"(\d+)\+?"
    r"(?:\s*-\s*\d+\+?|\s+\d+\+?)?"
    r"\s*years?"
    r"(?:\s+(?:of\s+)?(?:[a-z\-]+\s+){0,3})?"
    r"experience"
)


def extract_required_experience(job_text: str):
    """
    Extract required years of experience from a job description.
    """

    text = job_text.lower()

    patterns = [
        YEARS_EXPERIENCE_PATTERN,
        r"minimum\s+of\s+(\d+)\s+years?",
        r"at\s+least\s+(\d+)\s+years?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return int(match.group(1))

    return 0


def extract_candidate_experience(resume_text: str):
    """
    Extract years of experience from a resume.
    """

    text = resume_text.lower()

    patterns = [
        YEARS_EXPERIENCE_PATTERN,
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return int(match.group(1))

    return 0


def calculate_experience_match(
    candidate_years: int,
    required_years: int
):
    """
    Calculate experience match percentage.
    """

    # If the job does not specify experience,
    # experience should not reduce the score.
    if required_years == 0:
        return 100.0

    if candidate_years >= required_years:
        return 100.0

    score = (
        candidate_years / required_years
    ) * 100

    return round(score, 2)
