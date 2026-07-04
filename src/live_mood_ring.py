"""
Main entry point: "Mood Ring for Your Voice"

Captures short windows of microphone audio, extracts pitch/energy/MFCCs,
classifies the emotional tone, and drives a live pygame visualization that
reacts in real time.

Usage:
    python live_mood_ring.py                       # pretrained HF emotion model
    python live_mood_ring.py --classifier custom    # your own trained SVM model
    python live_mood_ring.py --window 2.5           # change analysis window length
"""

import argparse
import sys
import os
import threading
import time
import numpy as np
import sounddevice as sd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "visualizer"))

from audio_features import extract_features, normalize_for_visual
from emotion_classifier import get_classifier
from pygame_visualizer import MoodVisualizer

SAMPLE_RATE = 16000


class AudioBuffer:
    """Thread-safe rolling buffer that captures mic audio for windowed analysis."""

    def __init__(self, window_seconds: float, sample_rate: int = SAMPLE_RATE):
        self.window_samples = int(window_seconds * sample_rate)
        self.sample_rate = sample_rate
        self._buffer = np.zeros(self.window_samples, dtype=np.float32)
        self._lock = threading.Lock()

    def callback(self, indata, frames, time_info, status):
        with self._lock:
            new_data = indata[:, 0]
            n = len(new_data)
            if n >= self.window_samples:
                self._buffer = new_data[-self.window_samples:]
            else:
                self._buffer = np.concatenate([self._buffer[n:], new_data])

    def get_snapshot(self) -> np.ndarray:
        with self._lock:
            return self._buffer.copy()


def analysis_loop(buffer: AudioBuffer, classifier, classifier_mode: str,
                   visualizer: MoodVisualizer, stop_event: threading.Event,
                   analysis_interval: float = 1.0):
    """Runs in a background thread: periodically analyzes the audio buffer."""
    while not stop_event.is_set():
        audio = buffer.get_snapshot()
        features = extract_features(audio, buffer.sample_rate)

        if features["is_silent"]:
            visualizer.update_state("neutral", 0.0, 0.0)
            time.sleep(analysis_interval)
            continue

        pitch_norm, energy_norm = normalize_for_visual(features["pitch"], features["energy"])

        try:
            if classifier_mode == "pretrained":
                result = classifier.predict(audio, buffer.sample_rate)
                label = _map_pretrained_label(result["label"])
            else:
                result = classifier.predict(features["mfcc_mean"])
                label = result["label"]
        except Exception as e:
            print(f"Classification error: {e}")
            label = "neutral"

        visualizer.update_state(label, energy_norm, pitch_norm)
        print(f"[{label}]  pitch={features['pitch']:.1f}Hz  energy={features['energy']:.4f}")

        time.sleep(analysis_interval)


def _map_pretrained_label(raw_label: str) -> str:
    """
    Normalize labels from the HF 'superb/wav2vec2-base-superb-er' model
    (which uses: neu, hap, sad, ang) to our visualizer's label set.
    """
    mapping = {
        "neu": "neutral",
        "hap": "happy",
        "sad": "sad",
        "ang": "angry",
    }
    return mapping.get(raw_label.lower(), raw_label.lower())


def main():
    parser = argparse.ArgumentParser(description="Real-time voice mood visualizer.")
    parser.add_argument("--classifier", default="pretrained", choices=["pretrained", "custom"],
                         help="'pretrained' = HuggingFace model (no setup), "
                              "'custom' = your own trained model (run train_classifier.py first)")
    parser.add_argument("--window", type=float, default=2.0,
                         help="Analysis window length in seconds (default: 2.0)")
    parser.add_argument("--interval", type=float, default=1.0,
                         help="How often to re-analyze, in seconds (default: 1.0)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    print("Loading emotion classifier...")
    classifier = get_classifier(mode=args.classifier, device=args.device)

    buffer = AudioBuffer(window_seconds=args.window)
    visualizer = MoodVisualizer()
    stop_event = threading.Event()

    analysis_thread = threading.Thread(
        target=analysis_loop,
        args=(buffer, classifier, args.classifier, visualizer, stop_event, args.interval),
        daemon=True,
    )

    print("Starting microphone stream. Speak into your mic — close the window or Ctrl+C to stop.\n")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                         callback=buffer.callback):
        analysis_thread.start()
        try:
            running = True
            while running:
                running = visualizer.render_frame()
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            visualizer.close()
            print("\nStopped.")


if __name__ == "__main__":
    main()
