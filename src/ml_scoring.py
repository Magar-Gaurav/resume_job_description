"""
Loads the classifier trained by train_fit_model.py and uses it to
predict a fit category from this app's own rule-based match signals.

This is separate from src/scoring.py's weighted formula: that
formula uses fixed weights chosen by hand, while this module uses a
model whose weights were learned from real labeled resume/job pairs
(see train_fit_model.py). No pretrained model is involved here -
just this project's own trained Logistic Regression.
"""

import os

import joblib


MODEL_PATH = os.path.join("models", "fit_classifier.joblib")

_model = None
_model_load_attempted = False


def is_model_available() -> bool:
    """
    Check whether a trained model file exists on disk.
    """

    return os.path.exists(MODEL_PATH)


def _get_model():
    """
    Load and cache the trained model from disk.
    """

    global _model, _model_load_attempted

    if _model is not None:
        return _model

    if _model_load_attempted:
        return None

    _model_load_attempted = True

    if not is_model_available():
        return None

    _model = joblib.load(MODEL_PATH)

    return _model


def predict_fit(
    text_similarity: float,
    skill_match: float,
    experience_match: float,
    education_match: float,
):
    """
    Predict a fit category using the trained model.

    Returns:
        A tuple of (predicted_label, probabilities_dict), or
        (None, None) if no trained model is available yet.
    """

    model = _get_model()

    if model is None:
        return None, None

    features = [[
        text_similarity,
        skill_match,
        experience_match,
        education_match,
    ]]

    predicted_label = model.predict(features)[0]

    probabilities = dict(
        zip(
            model.classes_,
            model.predict_proba(features)[0],
        )
    )

    return predicted_label, probabilities
