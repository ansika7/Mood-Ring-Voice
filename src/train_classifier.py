"""
Optional: train your own emotion classifier on a labeled dataset (e.g. RAVDESS).

Expects a folder of .wav files where the emotion label can be parsed from the
filename (RAVDESS naming convention) or supplied via a CSV. This script uses the
RAVDESS filename convention by default:

    03-01-06-01-02-01-12.wav
             ^^ this third number is the emotion code:
    01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised

Download RAVDESS from: https://zenodo.org/record/1188976
Place the audio files (or a subset) in: data/ravdess/

Usage:
    python train_classifier.py --data_dir ../data/ravdess
"""

import argparse
import os
import glob
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_label_from_filename(filename: str) -> str:
    parts = os.path.basename(filename).split("-")
    code = parts[2]  # third field is emotion code in RAVDESS naming
    return RAVDESS_EMOTION_MAP.get(code, "unknown")


def extract_mfcc_mean(path: str, sample_rate: int = 16000) -> np.ndarray:
    audio, sr = librosa.load(path, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)


def build_dataset(data_dir: str):
    files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No .wav files found under {data_dir}")

    X, y = [], []
    print(f"Found {len(files)} audio files. Extracting features...")
    for i, f in enumerate(files):
        label = parse_label_from_filename(f)
        if label == "unknown":
            continue
        try:
            features = extract_mfcc_mean(f)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue
        X.append(features)
        y.append(label)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(files)}")

    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description="Train a custom speech emotion classifier.")
    parser.add_argument("--data_dir", required=True, help="Path to folder of labeled .wav files (RAVDESS format)")
    parser.add_argument("--output", default="../data/emotion_svm.joblib", help="Where to save the trained model")
    args = parser.parse_args()

    X, y = build_dataset(args.data_dir)
    print(f"Dataset built: {X.shape[0]} samples, {X.shape[1]} features each")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, C=10))
    print("Training SVM classifier...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n--- Evaluation ---")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(clf, args.output)
    print(f"\nModel saved to: {args.output}")


if __name__ == "__main__":
    main()
