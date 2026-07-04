"""
Speech emotion classification.

Two modes:
  1. "pretrained" (default, easiest) — uses a HuggingFace Wav2Vec2 model
     fine-tuned for speech emotion recognition. No training required.
  2. "custom" — loads a locally trained scikit-learn model (see train_classifier.py)
     using MFCC features from audio_features.py. Use this if you want to train
     on your own dataset (e.g. RAVDESS) or experiment with your own labels.

Both return a simple emotion label string from a shared set:
    ["neutral", "calm", "happy", "sad", "angry", "fearful", "surprised", "disgust"]
(exact label sets vary slightly by backend — see EMOTION_MAP below)
"""

import numpy as np
import joblib
import os

PRETRAINED_MODEL_NAME = "superb/wav2vec2-base-superb-er"  # HuggingFace speech emotion model
CUSTOM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "emotion_svm.joblib")


class PretrainedEmotionClassifier:
    """Wraps a HuggingFace pipeline for speech emotion recognition."""

    def __init__(self, model_name: str = PRETRAINED_MODEL_NAME, device: str = "cpu"):
        from transformers import pipeline
        print(f"Loading pretrained emotion model: {model_name} ...")
        self.pipe = pipeline(
            "audio-classification",
            model=model_name,
            device=0 if device == "cuda" else -1,
        )

    def predict(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        audio: mono float32 numpy array in [-1, 1]
        Returns {"label": str, "score": float, "all_scores": list}
        """
        results = self.pipe({"array": audio, "sampling_rate": sample_rate})
        # results is a list of {"label":..., "score":...} sorted by score desc
        top = results[0]
        return {"label": top["label"], "score": top["score"], "all_scores": results}


class CustomEmotionClassifier:
    """Wraps a locally trained scikit-learn classifier using MFCC features."""

    LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "surprised", "disgust"]

    def __init__(self, model_path: str = CUSTOM_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                f"Run train_classifier.py first, or use the 'pretrained' classifier instead."
            )
        self.model = joblib.load(model_path)

    def predict(self, mfcc_mean: np.ndarray) -> dict:
        proba = self.model.predict_proba(mfcc_mean.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        label = self.model.classes_[idx]
        return {"label": label, "score": float(proba[idx]), "all_scores": dict(zip(self.model.classes_, proba))}


def get_classifier(mode: str = "pretrained", device: str = "cpu"):
    if mode == "pretrained":
        return PretrainedEmotionClassifier(device=device)
    elif mode == "custom":
        return CustomEmotionClassifier()
    else:
        raise ValueError(f"Unknown classifier mode: {mode}")
