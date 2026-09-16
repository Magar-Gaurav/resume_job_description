import re

# Technical skills that our system can recognize
SKILL_ALIASES = {
    "python": ["python"],
    "java": ["java"],
    "c": ["c programming", "c language"],
    "c++": ["c++"],
    "c#": ["c#", "c sharp"],
    "sql": ["sql"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "matplotlib": ["matplotlib"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "machine learning": ["machine learning", "machine-learning"],
    "deep learning": ["deep learning", "deep-learning"],
    "data science": ["data science", "data-science"],
    "statistics": ["statistics"],
    "probability": ["probability"],
    "pca": ["pca", "principal component analysis"],
    "eda": ["eda", "exploratory data analysis"],
    "data preprocessing": ["data preprocessing", "data pre-processing", "data preprocessing techniques"],
    "feature engineering": ["feature engineering"],
    "classification": ["classification"],
    "regression": ["regression"],
    "model evaluation": ["model evaluation", "model evaluation techniques"],
    "llm": ["llm", "llms", "large language model", "large language models"],
    "generative ai": ["generative ai", "generative artificial intelligence"],
    "prompt engineering": ["prompt engineering"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "vector databases": ["vector database", "vector databases"],
    "agentic ai": ["agentic ai"],
    "fastapi": ["fastapi", "fast api"],
    "rest apis": ["rest api", "rest apis", "restful api", "restful apis"],
    "mysql": ["mysql", "my sql"],
    "postgresql": ["postgresql", "postgres", "postgre sql"],
    "mongodb": ["mongodb", "mongo db"],
    "git": ["git"],
    "github": ["github", "git hub"],
    "jupyter notebook": ["jupyter notebook", "jupyter notebooks"],
    "vs code": ["vs code", "visual studio code"],
    "docker": ["docker"],
    "aws": ["aws", "amazon web services"],
    "kubernetes": ["kubernetes", "k8s"],
    "airflow": ["airflow", "apache airflow"],
    "html":["html","html5"],
    "css":["css","css3"],
    "javascript":["javascript","js"],
    "typescript":["typescript","ts"],
    "react":["react","reactjs","react.js"],
    "next.js":["next.js","nextjs","next js"],
    "tailwind css":["tailwind css","tailwindcss"]
}


def extract_skills(text : str) -> list[str]:
    """
        Extracts known technical skills from text.

        Args: 
            text: Resume or job-description text.
        Returns:
            Returns canonical skill names instead of different variations found in the text
    """
    text = text.lower()

    found_skills = []

    for canonical_skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            escaped_alias = re.escape(alias)

            # Special handling for C, C++ and C#
            if canonical_skill =="c":
                pattern = rf"(?<![\w+#]){escaped_alias}(?![\w+#])"
            else:
                pattern = rf"(?<!\w){escaped_alias}(?!\w)"

            if re.search(pattern, text):
                found_skills.append(canonical_skill)
                break

    return sorted(set(found_skills))

def compare_skills(resume_skills : list[str], job_skills : list[str]) -> dict:
    """
        Compare skills found in a resume with skills required for a job.

        Args:
            resume_skills: Skills detected in resume
            job_skills: Skills detected in the job description
        
        Returns:
            Dictionary containing matched, missing and additional skills
    """

    resume_set = set(resume_skills)
    job_set = set(job_skills)

    matched_skills = sorted(resume_set & job_set)
    missing_skills = sorted(job_set - resume_set)
    additional_skills = sorted(resume_set - job_set)

    return {
        "matched" : matched_skills,
        "missing" : missing_skills,
        "additional" : additional_skills
    }

def calculate_skill_match(resume_skills : list[str], job_skills : list[str]) -> float:
    """
        Calculate the percentage of required job skills that are present in the resume.

        Args:
            resume_skills: Skills detected in resume
            job_skills: Skills detected in the job description
        Returns:
            Skill match percentage between 0 and 100
    """
    if not job_skills:
        return 0.0

    matched_skills = set(resume_skills) & set(job_skills)
    score = len(matched_skills) / len(set(job_skills)) 

    return round(score * 100 , 2)