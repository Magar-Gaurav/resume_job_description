from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


MODEL_NAME = "all-MiniLM-L6-v2"


def load_model():
    """
    Load the pre-trained Sentence Transformer model.
    """

    model = SentenceTransformer(MODEL_NAME)

    return model


def calculate_semantic_similarity(
    resume_text: str,
    job_description: str,
    model,
) -> float:
    """
    Calculate semantic similarity between
    resume and job description.

    Returns:
        Similarity score between 0 and 1.
    """

    embeddings = model.encode(
        [
            resume_text,
            job_description,
        ]
    )

    resume_embedding = embeddings[0]
    job_embedding = embeddings[1]

    similarity = cos_sim(
        resume_embedding,
        job_embedding,
    )

    return float(similarity[0][0])


def semantic_similarity_to_percentage(
    similarity_score: float,
) -> float:
    """
    Convert similarity score into percentage.
    """

    percentage = similarity_score * 100

    return max(
        0.0,
        min(percentage, 100.0),
    )