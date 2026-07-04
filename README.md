# 🎭 Mood Ring for Your Voice

A real-time voice emotion visualizer. Speak into your microphone and watch a
pulsing "mood blob" change color, speed, and size based on the emotional tone
of your voice — not just the words you say.

## Project Structure

```
mood-ring-voice/
├── requirements.txt
├── README.md
├── data/                       # (optional) place RAVDESS dataset here for custom training
├── src/
│   ├── audio_features.py       # pitch, energy, MFCC extraction (librosa)
│   ├── emotion_classifier.py   # pretrained HF model OR custom sklearn model
│   ├── train_classifier.py     # optional: train your own SVM on RAVDESS
│   └── live_mood_ring.py       # main entry point — mic capture + analysis + visuals
└── visualizer/
    └── pygame_visualizer.py    # the pulsing color-blob renderer
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No ffmpeg needed here — audio capture and feature extraction are handled by
`sounddevice` and `librosa` directly.

## 2. Run It (Easiest Path — Pretrained Model)

```bash
cd src
python live_mood_ring.py
```

This uses a pretrained HuggingFace model (`superb/wav2vec2-base-superb-er`) —
no training required. The first run will download the model (~90MB).

A window will open showing a pulsing circle:
- **Color** = detected emotion (blue = calm, yellow = happy, red = angry, purple = sad, etc.)
- **Pulse speed** = pitch (higher pitch = faster pulse)
- **Pulse size** = energy/loudness (louder = bigger swings)

Press `Ctrl+C` in the terminal or close the window to stop.

## 3. (Optional) Train Your Own Classifier

If you want more emotion categories or better accuracy on your own voice,
train a lightweight SVM on the RAVDESS dataset:

1. Download RAVDESS: https://zenodo.org/record/1188976
2. Extract the `.wav` files into `data/ravdess/`
3. Run:
   ```bash
   cd src
   python train_classifier.py --data_dir ../data/ravdess
   ```
4. Then run the live visualizer with your custom model:
   ```bash
   python live_mood_ring.py --classifier custom
   ```

## 4. Tuning

- `--window` — how many seconds of audio to analyze at once (default 2.0s). Shorter = more responsive but less accurate.
- `--interval` — how often to re-analyze (default 1.0s).
- `--device cuda` — use GPU for the pretrained model if available.

## 5. Extension Ideas

- **Voice mood journal** — log emotion + timestamp to a CSV throughout the day, then plot mood over time
- **Karaoke energy meter** — score how "into it" someone sounds and gamify it
- **Customer service tone monitor** — flag calls where frustration/anger is trending up
- **Companion character** — swap the blob for a simple animated face (SVG or sprite-based) that smiles/frowns
- **Multi-speaker mode** — combine with a speaker diarization step to track mood per person in a conversation
- **Web version** — port the visualizer to a browser with Web Audio API + Canvas so it's shareable via a link

## Troubleshooting

- **No sound detected / blob stays neutral** — check mic permissions and volume; try `python -m sounddevice` to confirm your input device is detected.
- **Pretrained model feels inaccurate** — it was trained on acted emotional speech (not natural conversation), so it works best with clearly expressive tone. Training a custom model on your own voice samples (Step 3) can help.
- **Laggy visualizer** — increase `--interval` so analysis runs less frequently, or use a smaller/faster classifier.
- **`librosa.pyin` errors on short/silent audio** — this is handled gracefully in `audio_features.py`, but very short windows (<1s) may give noisy pitch estimates.
