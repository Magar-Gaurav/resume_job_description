import tempfile

import streamlit as st

from resume_job_description.src.pdf_extractor import extract_text_from_pdf
from resume_job_description.src.preprocessing import preprocess_text

from resume_job_description.src.text_vectorizer import create_tfidf_vectors

from resume_job_description.src.similarity import (
    calculate_similarity,
    similarity_to_percentage,
)

from resume_job_description.src.semantic_matcher import (
    load_model,
    calculate_semantic_similarity,
    semantic_similarity_to_percentage,
)

from resume_job_description.src.skill_matcher import (
    extract_skills,
    compare_skills,
    calculate_skill_match,
)

from resume_job_description.src.experience import (
    extract_required_experience,
    extract_candidate_experience,
    calculate_experience_match,
)

from resume_job_description.src.education import (
    calculate_education_match,
)

from resume_job_description.src.scoring import (
    calculate_overall_score,
    get_match_category,
)

from resume_job_description.src.recommendations import (
    generate_skill_recommendations,
    generate_experience_recommendation,
    generate_education_recommendation,
)

from resume_job_description.src.ml_scoring import (
    is_model_available,
    predict_fit,
)


st.set_page_config(
    page_title="Resume Job Matcher",
    page_icon="R",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_semantic_model():
    """
    Load (and cache across reruns) the sentence-transformer model
    used for semantic similarity, so it's only downloaded/loaded once.
    """

    return load_model()


# Page title

st.title("Resume Job Matcher")

st.write(
    "Compare a resume with a job description using "
    "TF-IDF, cosine similarity, semantic similarity, "
    "skill matching, experience matching, and education matching."
)


# Input section

st.subheader("Resume and Job Description")

col1, col2 = st.columns(2)


with col1:

    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf"],
    )


with col2:

    job_description = st.text_area(
        "Paste Job Description",
        height=250,
        placeholder="Paste the job description here...",
    )


analyze_button = st.button(
    "Analyze Resume",
    type="primary",
)


# Analysis

if analyze_button:

    if uploaded_file is None:

        st.error(
            "Please upload a resume PDF."
        )

        st.stop()

    if not job_description.strip():

        st.error(
            "Please enter a job description."
        )

        st.stop()

    with st.spinner("Analyzing resume..."):

        # Save uploaded PDF

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            pdf_path = temp_file.name

        # Extract resume text

        raw_resume = extract_text_from_pdf(
            pdf_path
        )

        if not raw_resume.strip():

            st.error(
                "Could not extract text from the PDF."
            )

            st.stop()

        # Preprocess text

        resume_text = preprocess_text(
            raw_resume
        )

        clean_job_description = preprocess_text(
            job_description
        )

        # Create TF-IDF vectors

        resume_vector, job_vector, vectorizer = (
            create_tfidf_vectors(
                resume_text,
                clean_job_description
            )
        )

        # Calculate text similarity

        cosine_score = calculate_similarity(
            resume_vector,
            job_vector
        )

        text_similarity = similarity_to_percentage(
            cosine_score
        )

        # Calculate semantic similarity

        semantic_model = get_semantic_model()

        semantic_score = calculate_semantic_similarity(
            resume_text,
            clean_job_description,
            semantic_model,
        )

        semantic_similarity = semantic_similarity_to_percentage(
            semantic_score
        )

        # Extract skills

        resume_skills = extract_skills(
            resume_text
        )

        job_skills = extract_skills(
            clean_job_description
        )

        # Compare skills

        comparison = compare_skills(
            resume_skills,
            job_skills
        )

        skill_match = calculate_skill_match(
            resume_skills,
            job_skills
        )

        # Extract experience

        required_experience = (
            extract_required_experience(
                clean_job_description
            )
        )

        candidate_experience = (
            extract_candidate_experience(
                resume_text
            )
        )

        # Calculate experience match

        experience_match = (
            calculate_experience_match(
                candidate_experience,
                required_experience
            )
        )

        # Calculate education match

        (
            education_match,
            candidate_education,
            required_education
        ) = calculate_education_match(
            resume_text,
            clean_job_description
        )

        # Calculate overall score

        overall_score = calculate_overall_score(
            text_similarity,
            semantic_similarity,
            skill_match,
            experience_match,
            education_match
        )

        # Get match category

        category = get_match_category(
            overall_score
        )

    st.success(
        "Analysis completed successfully."
    )

    # Match results

    st.subheader("Match Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Overall Match",
            f"{overall_score:.2f}%"
        )

        st.write(
            f"Category: {category}"
        )

        st.progress(
            min(overall_score / 100, 1.0)
        )

    with col2:

        st.metric(
            "Text Similarity",
            f"{text_similarity:.2f}%"
        )

        st.progress(
            min(text_similarity / 100, 1.0)
        )

    with col3:

        st.metric(
            "Semantic Similarity",
            f"{semantic_similarity:.2f}%"
        )

        st.progress(
            min(semantic_similarity / 100, 1.0)
        )

    with col4:

        st.metric(
            "Skill Match",
            f"{skill_match:.2f}%"
        )

        st.progress(
            min(skill_match / 100, 1.0)
        )

    # Experience and education scores

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Experience Match",
            f"{experience_match:.2f}%"
        )

        st.progress(
            min(experience_match / 100, 1.0)
        )

    with col2:

        st.metric(
            "Education Match",
            f"{education_match:.2f}%"
        )

        st.progress(
            min(education_match / 100, 1.0)
        )

    # Score calculation

    st.divider()

    st.subheader("Score Calculation")

    st.write(
        "The overall score uses 45% skill matching, "
        "20% semantic similarity, 15% experience matching, "
        "10% text similarity, and 10% education matching."
    )

    st.code(
        "Overall Score = "
        "(Text Similarity x 0.10) + "
        "(Semantic Similarity x 0.20) + "
        "(Skill Match x 0.45) + "
        "(Experience Match x 0.15) + "
        "(Education Match x 0.10)"
    )

    st.write(
        f"({text_similarity:.2f} x 0.10) + "
        f"({semantic_similarity:.2f} x 0.20) + "
        f"({skill_match:.2f} x 0.45) + "
        f"({experience_match:.2f} x 0.15) + "
        f"({education_match:.2f} x 0.10) = "
        f"{overall_score:.2f}%"
    )

    # ML model prediction

    st.divider()

    st.subheader("ML Model Prediction")

    st.write(
        "In addition to the formula above, a Logistic Regression "
        "model trained on labeled resume-job pairs (no pretrained "
        "model involved) predicts a fit category from the same "
        "signals."
    )

    if is_model_available():

        ml_label, ml_probabilities = predict_fit(
            text_similarity,
            skill_match,
            experience_match,
            education_match,
        )

        st.metric(
            "Predicted Fit Category",
            ml_label,
        )

        for class_name, probability in sorted(
            ml_probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        ):

            st.write(
                f"{class_name}: {probability * 100:.2f}%"
            )

            st.progress(
                min(float(probability), 1.0)
            )

    else:

        st.info(
            "No trained model found. Run train_fit_model.py "
            "to train one on the resume-job-description-fit "
            "dataset, then reload this page."
        )

    # Skill analysis

    st.divider()

    st.subheader("Skill Analysis")

    col1, col2 = st.columns(2)

    # Matched skills

    with col1:

        st.write("Matched Skills")

        matched_skills = comparison.get(
            "matched",
            []
        )

        if matched_skills:

            for skill in matched_skills:

                st.write(
                    f"- {skill}"
                )

        else:

            st.write(
                "No matching skills found."
            )

    # Missing skills

    with col2:

        st.write("Missing Skills")

        missing_skills = comparison.get(
            "missing",
            []
        )

        if missing_skills:

            for skill in missing_skills:

                st.write(
                    f"- {skill}"
                )

        else:

            st.write(
                "No required skills are missing."
            )

    # Experience analysis

    st.divider()

    st.subheader("Experience Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write(
            "Required Experience"
        )

        if required_experience > 0:

            st.write(
                f"{required_experience} years"
            )

        else:

            st.write(
                "Not specified"
            )

    with col2:

        st.write(
            "Candidate Experience"
        )

        if candidate_experience > 0:

            st.write(
                f"{candidate_experience} years"
            )

        else:

            st.write(
                "Not detected"
            )

    with col3:

        st.write(
            "Experience Match"
        )

        st.write(
            f"{experience_match:.2f}%"
        )

    # Education analysis

    st.divider()

    st.subheader("Education Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write(
            "Required Education"
        )

        if required_education:

            st.write(
                required_education
            )

        else:

            st.write(
                "Not specified"
            )

    with col2:

        st.write(
            "Candidate Education"
        )

        if candidate_education:

            st.write(
                candidate_education
            )

        else:

            st.write(
                "Not detected"
            )

    with col3:

        st.write(
            "Education Match"
        )

        st.write(
            f"{education_match:.2f}%"
        )

    # Additional skills

    st.divider()

    st.subheader("Additional Resume Skills")

    additional_skills = comparison.get(
        "additional",
        []
    )

    if additional_skills:

        for skill in additional_skills:

            st.write(
                f"- {skill}"
            )

    else:

        st.write(
            "No additional skills detected."
        )

    # Recommendations

    st.divider()

    st.subheader("Recommendations")

    for line in generate_skill_recommendations(missing_skills):

        st.write(line)

    experience_recommendation = generate_experience_recommendation(
        candidate_experience,
        required_experience,
    )

    if experience_recommendation:

        st.write(experience_recommendation)

    education_recommendation = generate_education_recommendation(
        required_education,
        candidate_education,
        education_match,
    )

    if education_recommendation:

        st.write(education_recommendation)

    # Analysis details

    st.divider()

    st.subheader("Analysis Details")

    with st.expander(
        "View extracted resume text"
    ):

        st.text(
            raw_resume
        )

    with st.expander(
        "View processed resume text"
    ):

        st.text(
            resume_text
        )

    with st.expander(
        "View processed job description"
    ):

        st.text(
            clean_job_description
        )

    with st.expander(
        "View detected resume skills"
    ):

        st.write(
            resume_skills
        )

    with st.expander(
        "View detected job skills"
    ):

        st.write(
            job_skills
        )
