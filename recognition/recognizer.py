"""
Character Recognizer Module
ML-based handwriting classifier (SVM baseline with probability calibration)
following Prompt Sections 17, 19, 20.
Classifies normalized air-drawn strokes into A-Z / a-z characters with confidence thresholds.
"""

import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from .feature_extractor import StrokeFeatureExtractor
from .dataset_manager import DatasetManager

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "character_svm.joblib")


class CharacterRecognizer:
    def __init__(self, model_path: str = None, min_confidence: float = 0.45):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.min_confidence = min_confidence
        self.extractor = StrokeFeatureExtractor(target_size=128, margin=14, num_resample_points=64)
        self.model = None
        self.classes_ = []
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.load_model()

    def is_trained(self) -> bool:
        return self.model is not None and len(self.classes_) > 1

    def load_model(self) -> bool:
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.model = data.get("model")
                self.classes_ = list(data.get("classes", []))
                print(f"[Recognizer] Loaded trained model from {self.model_path} ({len(self.classes_)} classes).")
                return True
            except Exception as e:
                print(f"[Recognizer] Error loading model: {e}")
        return False

    def save_model(self):
        if self.model is not None:
            joblib.dump({"model": self.model, "classes": self.classes_}, self.model_path)
            print(f"[Recognizer] Saved model to {self.model_path}")

    def train_from_dataset(self, dataset_mgr: DatasetManager) -> dict:
        """
        Trains the SVM classifier from collected dataset samples.
        """
        X, y = dataset_mgr.load_dataset()
        if len(y) == 0:
            return {"status": "error", "message": "No dataset samples found in data/dataset/"}

        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            return {
                "status": "error",
                "message": f"Need at least 2 distinct character classes to train (found {len(unique_labels)})."
            }

        # Train Support Vector Machine with RBF kernel and probability calibration
        if len(y) < 10:
            clf = KNeighborsClassifier(n_neighbors=min(3, len(y)))
        else:
            clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)

        clf.fit(X, y)
        self.model = clf
        self.classes_ = list(clf.classes_)
        self.save_model()

        acc = float(clf.score(X, y))
        return {
            "status": "success",
            "samples": len(y),
            "classes": len(unique_labels),
            "train_accuracy": acc
        }

    def predict_strokes(self, raw_strokes: list) -> tuple:
        """
        Full recognition pipeline (Prompt Section 20):
        Raw strokes -> Normalize -> Resample -> Feature extraction -> Classifier -> Prediction & Confidence.
        Returns:
            predicted_char: str or None
            confidence: float (0.0 to 1.0)
            status: str ('accepted', 'low_confidence', 'untrained', 'empty')
        """
        if not raw_strokes:
            return None, 0.0, "empty"

        if not self.is_trained():
            return None, 0.0, "untrained"

        features = self.extractor.extract_feature_vector(raw_strokes)
        features_2d = features.reshape(1, -1)

        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features_2d)[0]
                best_idx = int(np.argmax(probs))
                confidence = float(probs[best_idx])
                pred_label = self.classes_[best_idx]
            else:
                pred_label = self.model.predict(features_2d)[0]
                confidence = 1.0

            if confidence < self.min_confidence:
                return pred_label, confidence, "low_confidence"

            return pred_label, confidence, "accepted"

        except Exception as e:
            print(f"[Recognizer] Prediction error: {e}")
            return None, 0.0, "error"
