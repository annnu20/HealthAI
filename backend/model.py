"""
Model loading and prediction logic.

This wraps the exact prediction functions from the original notebook
(`predict_disease` / `predict_with_confidence`, cells 49-50) around the
artifacts saved by train_model.py. No prediction logic has been changed —
only adapted to load a persisted model instead of using in-notebook globals,
and to raise clear errors instead of relying on a Colab session.
"""

import os

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODELS_DIR, "best_disease_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "symptom_columns.pkl")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "Training.csv")


class ModelNotTrainedError(RuntimeError):
    """Raised when the saved model artifacts can't be found."""


def _artifacts_exist() -> bool:
    return (
        os.path.exists(MODEL_PATH)
        and os.path.exists(ENCODER_PATH)
        and os.path.exists(COLUMNS_PATH)
    )


class DiseasePredictor:
    """
    Loads the trained model, label encoder, and symptom column order once,
    and exposes the same prediction behavior as the original notebook.
    """

    def __init__(self):
        if not _artifacts_exist():
            raise ModelNotTrainedError(
                "Model artifacts not found in 'models/'. Run "
                "'python train_model.py' first to train and save the model."
            )
        self.model = joblib.load(MODEL_PATH)
        self.encoder = joblib.load(ENCODER_PATH)
        self.symptom_columns = joblib.load(COLUMNS_PATH)
        self._common_symptoms_cache = {}

    @property
    def symptoms(self):
        """All symptom names the model was trained on, in column order."""
        return list(self.symptom_columns)

    @property
    def diseases(self):
        """All disease labels the model can predict."""
        return list(self.encoder.classes_)

    def _vectorize(self, user_symptoms):
        input_data = np.zeros(len(self.symptom_columns))
        unrecognized = []

        columns_index = {col: i for i, col in enumerate(self.symptom_columns)}

        for symptom in user_symptoms:
            symptom = symptom.strip().lower().replace(" ", "_")
            if symptom in columns_index:
                input_data[columns_index[symptom]] = 1
            elif symptom:
                unrecognized.append(symptom)

        return input_data, unrecognized

    def _as_frame(self, input_data):
        # Wrap in a DataFrame with the original column names so sklearn
        # doesn't warn about missing feature names (purely cosmetic —
        # prediction behavior is identical to passing a raw array, which
        # is what the original notebook did).
        return pd.DataFrame([input_data], columns=self.symptom_columns)

    def predict_disease(self, user_symptoms):
        """Reproduces the notebook's predict_disease() (cell 49)."""
        input_data, _ = self._vectorize(user_symptoms)
        prediction = self.model.predict(self._as_frame(input_data))[0]
        disease = self.encoder.inverse_transform([prediction])[0]
        return disease

    def predict_with_confidence(self, user_symptoms):
        """
        Reproduces the notebook's predict_with_confidence() (cell 50).

        Returns:
            disease (str), confidence (float | None), unrecognized (list[str])
        """
        input_data, unrecognized = self._vectorize(user_symptoms)
        frame = self._as_frame(input_data)

        prediction = self.model.predict(frame)[0]
        disease = self.encoder.inverse_transform([prediction])[0]

        confidence = None
        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(frame)[0]
            confidence = max(probability) * 100

        return disease, confidence, unrecognized

    def top_predictions(self, user_symptoms, top_n=3):
        """
        Enhancement over the original notebook: returns the top-N most
        likely diseases with their probabilities, when the model supports
        predict_proba. Falls back to just the single prediction otherwise.
        """
        input_data, unrecognized = self._vectorize(user_symptoms)
        frame = self._as_frame(input_data)

        if not hasattr(self.model, "predict_proba"):
            disease = self.predict_disease(user_symptoms)
            return [(disease, None)], unrecognized

        probabilities = self.model.predict_proba(frame)[0]
        top_indices = np.argsort(probabilities)[::-1][:top_n]
        results = [
            (self.encoder.inverse_transform([i])[0], probabilities[i] * 100)
            for i in top_indices
        ]
        return results, unrecognized

    def common_symptoms_for(self, disease, min_frequency=0.5, exclude=None):
        """
        New helper (not in the original notebook): looks at the training
        data and returns the symptoms most commonly reported for `disease`
        (present in at least `min_frequency` of that disease's rows).

        This exists to address low prediction confidence for sparse input:
        a single symptom is often genuinely shared by several diseases, so
        the model's confidence for it is honestly low rather than wrong.
        Showing what other symptoms are typically reported for the
        predicted disease lets the user add relevant ones and get a
        sharper, more confident prediction — without the app inflating or
        faking the confidence number itself.

        `exclude` is an optional iterable of symptom names (e.g. the ones
        the user already entered) to leave out of the returned list.
        """
        disease_key = disease.strip() if isinstance(disease, str) else disease

        if disease_key not in self._common_symptoms_cache:
            self._common_symptoms_cache[disease_key] = self._compute_common_symptoms(
                disease_key, min_frequency
            )

        common = self._common_symptoms_cache[disease_key]
        if exclude:
            exclude_set = {s.strip().lower().replace(" ", "_") for s in exclude}
            common = [s for s in common if s not in exclude_set]
        return common

    def _compute_common_symptoms(self, disease_key, min_frequency):
        if not os.path.exists(TRAINING_DATA_PATH):
            return []

        df = pd.read_csv(TRAINING_DATA_PATH)
        if "Unnamed: 133" in df.columns:
            df = df.drop("Unnamed: 133", axis=1)

        disease_rows = df[df["prognosis"].str.strip() == disease_key]
        if disease_rows.empty:
            return []

        frequencies = disease_rows[self.symptom_columns].mean()
        common = frequencies[frequencies >= min_frequency].sort_values(ascending=False)
        return list(common.index)


_predictor_instance = None


def get_predictor():
    """Cached accessor so the model is only loaded from disk once."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = DiseasePredictor()
    return _predictor_instance
