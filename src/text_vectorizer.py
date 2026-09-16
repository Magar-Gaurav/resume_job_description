from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectors(resume_text: str, job_description_text: str):
    """
    Convert resume and job description into TF-IDF vectors.
    """

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1
    )

    vectors = vectorizer.fit_transform(
        [resume_text, job_description_text]
    )

    resume_vector = vectors[0]
    job_vector = vectors[1]

    return resume_vector, job_vector, vectorizer