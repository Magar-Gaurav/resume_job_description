import re

import nltk
from nltk.corpus import stopwords


def _ensure_stopwords_downloaded() -> None:
    """
    Make sure the NLTK 'stopwords' corpus is available locally,
    downloading it on first run if needed.
    """

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


_ensure_stopwords_downloaded()

STOP_WORDS = set(stopwords.words("english"))

# Non-ASCII dash variants (en dash, em dash, minus sign, etc.) that
# commonly appear in ranges like "2–4 years". These aren't in the
# allowed-character set below, so without normalizing them to a plain
# hyphen first, a range like "2–4" would be destroyed into "2 4"
# instead of staying readable as "2-4".
DASH_VARIANTS = ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212"]


def preprocess_text(text: str) -> str:
    """
    Clean and normalize resume or job description text.
    """

    # Convert text to lowercase
    text = text.lower()

    # Normalize non-ASCII dashes to a plain hyphen, so numeric ranges
    # (e.g. "2–4 years") survive as "2-4" instead of being destroyed
    for dash in DASH_VARIANTS:
        text = text.replace(dash, "-")

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove LinkedIn and GitHub URLs
    text = re.sub(
        r"\b(?:linkedin|github)\.com\S*",
        " ",
        text
    )

    # Remove phone numbers
    text = re.sub(
        r"\+?\d[\d\s().-]{7,}\d",
        " ",
        text
    )

    # Replace separators with spaces
    for separator in [",", ";", ":", "|", "\u2022", "\u00b7"]:
        text = text.replace(separator, " ")

    # Remove unwanted characters
    # Keep #, +, . and - for technical terms
    text = re.sub(
        r"[^a-z0-9#+.\-\s]",
        " ",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]

    return " ".join(tokens)
