"""
Train a classifier that learns how much weight to give each of this
app's rule-based match signals, using real labeled resume/job pairs
instead of the hand-picked 45/20/15/10/10 formula in src/scoring.py.

IMPORTANT: this script uses NO pretrained / built-in model anywhere.
- No sentence-transformers, no downloaded weights of any kind.
- Every input signal is computed from scratch by this project's own
  code: TF-IDF is fit fresh per resume/job pair (not pretrained),
  and skill/experience/education matching are plain rule-based
  functions.
- Two candidate models are trained entirely on this dataset, from a
  random start, and the better one (by held-out macro F1) is kept:
  Logistic Regression (linear) and Random Forest (non-linear).

Data source:
    cnamuangtoun/resume-job-description-fit (Hugging Face)
    ~8,000 real resume / job-description pairs, each labeled
    "No Fit", "Potential Fit", or "Good Fit".

Usage:
    pip install datasets joblib
    python train_fit_model.py
"""

import os
import sys

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from resume_job_description.src.preprocessing import preprocess_text
from resume_job_description.src.text_vectorizer import create_tfidf_vectors
from resume_job_description.src.similarity import calculate_similarity, similarity_to_percentage
from resume_job_description.src.skill_matcher import extract_skills, calculate_skill_match
from resume_job_description.src.experience import (
    extract_required_experience,
    extract_candidate_experience,
    calculate_experience_match,
)
from resume_job_description.src.education import calculate_education_match


DATASET_NAME = "cnamuangtoun/resume-job-description-fit"
MODEL_OUTPUT_PATH = os.path.join("models", "fit_classifier.joblib")

# Keep this in the same order everywhere (training and inference)
FEATURE_NAMES = [
    "text_similarity",
    "skill_match",
    "experience_match",
    "education_match",
]


def find_column(columns, keywords):
    """
    Find the first column name containing any of the given keywords
    (case-insensitive). Used so this script keeps working even if
    the dataset's exact column names differ slightly from what's
    expected.
    """

    for column in columns:
        lowered = column.lower()

        if any(keyword in lowered for keyword in keywords):
            return column

    return None


def load_pairs():
    """
    Load the dataset and figure out which columns hold the resume
    text, job text, and label.
    """

    dataset = load_dataset(DATASET_NAME, split="train")

    columns = dataset.column_names

    resume_col = find_column(columns, ["resume"])
    job_col = find_column(columns, ["job", "description"])
    label_col = find_column(columns, ["label", "fit"])

    if not (resume_col and job_col and label_col):
        print("Could not automatically detect dataset columns.")
        print(f"Available columns: {columns}")
        print(
            "Open the dataset on Hugging Face, check the real "
            "column names, and edit load_pairs() to use them."
        )
        sys.exit(1)

    print(
        f"Using columns -> resume: '{resume_col}', "
        f"job: '{job_col}', label: '{label_col}'"
    )

    return dataset, resume_col, job_col, label_col


def extract_features(resume_text: str, job_text: str):
    """
    Compute this app's rule-based match signals for one resume/job
    pair. No pretrained model is used anywhere in this function.
    """

    clean_resume = preprocess_text(resume_text)
    clean_job = preprocess_text(job_text)

    # TF-IDF is fit fresh on this single pair (same as app.py does) -
    # this is NOT a pretrained model, just classic vector-space
    # similarity computed from the two documents themselves.
    resume_vector, job_vector, _ = create_tfidf_vectors(
        clean_resume,
        clean_job,
    )

    text_similarity = similarity_to_percentage(
        calculate_similarity(resume_vector, job_vector)
    )

    resume_skills = extract_skills(clean_resume)
    job_skills = extract_skills(clean_job)

    skill_match = calculate_skill_match(resume_skills, job_skills)

    required_experience = extract_required_experience(clean_job)
    candidate_experience = extract_candidate_experience(clean_resume)

    experience_match = calculate_experience_match(
        candidate_experience,
        required_experience,
    )

    education_match, _, _ = calculate_education_match(
        clean_resume,
        clean_job,
    )

    return [
        text_similarity,
        skill_match,
        experience_match,
        education_match,
    ]


def evaluate(name, model, X_test, y_test):
    """
    Print a held-out evaluation for one candidate model and return
    its macro F1, so candidates can be compared on equal footing
    (macro F1 accounts for class imbalance; plain accuracy does not).
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print(f"\n==== {name}: Held-out Test Results ====\n")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))
    print("Confusion matrix (labels order: "
          f"{list(model.classes_)}):")
    print(confusion_matrix(y_test, predictions, labels=model.classes_))

    return macro_f1


def main():
    dataset, resume_col, job_col, label_col = load_pairs()

    print(f"Extracting features for {len(dataset)} pairs...")

    features = []
    labels = []
    resume_groups = []

    for i, row in enumerate(dataset):
        resume_text = row[resume_col]
        job_text = row[job_col]
        label = row[label_col]

        features.append(extract_features(resume_text, job_text))
        labels.append(label)

        # Group by resume so the same resume can never end up split
        # across train and test. Some projects that used this
        # dataset's official split reported heavy resume overlap
        # between train and test, which inflates accuracy.
        resume_groups.append(hash(resume_text))

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(dataset)} done")

    features = np.array(features)
    labels = np.array(labels)
    resume_groups = np.array(resume_groups)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(features, labels, groups=resume_groups)
    )

    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    print(f"\nTrain size: {len(X_train)}  Test size: {len(X_test)}")

    # ------------------------------------------------------------
    # Candidate 1: Logistic Regression (linear, scaled features)
    # ------------------------------------------------------------

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )),
    ])

    logistic_model.fit(X_train, y_train)

    logistic_f1 = evaluate(
        "Logistic Regression",
        logistic_model,
        X_test,
        y_test,
    )

    logistic_classifier = logistic_model.named_steps["classifier"]

    print("\nLearned feature weights (on scaled features):")
    print(
        "(compare these to the hand-picked weights in "
        "src/scoring.py: skill 45%, semantic 20%, experience 15%, "
        "text 10%, education 10%. Larger magnitude here means the "
        "model leaned on that feature more.)"
    )

    for class_name, coefs in zip(
        logistic_classifier.classes_,
        logistic_classifier.coef_,
    ):
        print(f"\n  Class: {class_name}")

        for name, coef in zip(FEATURE_NAMES, coefs):
            print(f"    {name:20s}: {coef:+.4f}")

    # ------------------------------------------------------------
    # Candidate 2: Random Forest (non-linear, raw features - trees
    # don't need scaling)
    # ------------------------------------------------------------

    forest_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )

    forest_model.fit(X_train, y_train)

    forest_f1 = evaluate(
        "Random Forest",
        forest_model,
        X_test,
        y_test,
    )

    print("\nFeature importances:")

    for name, importance in zip(
        FEATURE_NAMES,
        forest_model.feature_importances_,
    ):
        print(f"    {name:20s}: {importance:.4f}")

    # ------------------------------------------------------------
    # Keep whichever candidate has the better held-out macro F1
    # ------------------------------------------------------------

    if forest_f1 >= logistic_f1:
        winner_name = "Random Forest"
        winner_model = forest_model
    else:
        winner_name = "Logistic Regression"
        winner_model = logistic_model

    print(
        f"\n==== Winner: {winner_name} "
        f"(macro F1 {max(forest_f1, logistic_f1):.4f}) ===="
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(winner_model, MODEL_OUTPUT_PATH)

    print(f"\nSaved {winner_name} to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()