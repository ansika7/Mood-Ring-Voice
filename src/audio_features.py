"""
Audio feature extraction for emotion analysis.

Extracts pitch, energy, tempo/rate proxy, and MFCCs from a short audio clip.
Used by the "train your own classifier" route (train_classifier.py + emotion_classifier.py's
lightweight mode). If you use the pretrained HuggingFace model instead, you don't need this
file's output directly, but pitch/energy are still used for visual intensity.
"""

import numpy as np
import librosa


def extract_features(audio: np.ndarray, sample_rate: int = 16000) -> dict:
    """
    Extract a feature dict from a mono float32 audio array.
    Returns both raw signal descriptors (for visualization) and
    an MFCC feature vector (for classification).
    """
    audio = audio.astype(np.float32)

    # Guard against completely silent/empty input
    if len(audio) == 0 or np.abs(audio).max() < 1e-4:
        return {
            "pitch": 0.0,
            "energy": 0.0,
            "zero_crossing_rate": 0.0,
            "mfcc_mean": np.zeros(13),
            "is_silent": True,
        }

    # --- Pitch (fundamental frequency) via pyin ---
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sample_rate
        )
        voiced_f0 = f0[~np.isnan(f0)]
        pitch = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    except Exception:
        pitch = 0.0

    # --- Energy (RMS loudness) ---
    rms = librosa.feature.rms(y=audio)[0]
    energy = float(np.mean(rms))

    # --- Zero crossing rate (proxy for "sharpness"/noisiness of speech) ---
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    zcr_mean = float(np.mean(zcr))

    # --- MFCCs (used as input features for the classifier) ---
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)

    return {
        "pitch": pitch,
        "energy": energy,
        "zero_crossing_rate": zcr_mean,
        "mfcc_mean": mfcc_mean,
        "is_silent": False,
    }


def normalize_for_visual(pitch: float, energy: float) -> tuple:
    """
    Map raw pitch/energy into 0-1 ranges roughly suited for visualization scaling.
    These bounds are rough heuristics for typical speaking voices — tune as needed.
    """
    pitch_norm = np.clip((pitch - 80) / (400 - 80), 0, 1)      # ~80-400 Hz speaking range
    energy_norm = np.clip(energy / 0.1, 0, 1)                   # rough loudness ceiling
    return float(pitch_norm), float(energy_norm)
