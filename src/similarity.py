from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(resume_vector, job_vector) -> float:
    """
        Calculate cosine similarity between a resume and job description using their TF-IDF vectors.

        Args:
            resume_vector (array-like): TF-IDF vector of the resume.
            job_vector (array-like): TF-IDF vector of the job description.
        
        Returns:
            float: Cosine similarity score between the resume and job description (0 and 1).
    
    """
    similarity = cosine_similarity(resume_vector,job_vector)

    return float(similarity[0][0])

def similarity_to_percentage(similarity_score: float) -> float:
    """
        Convert cosine similarity score to a percentage format.

        Args:
            similarity_score : Cosine similarity between 0 and 1    
        
        Returns:
            Similarity percentage between 0 and 100
    """
    return round(similarity_score * 100, 2)