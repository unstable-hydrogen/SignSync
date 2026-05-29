"""
extract_and_train.py

1. Loads images from asl_source/asl_dataset/{letter}/  (a-z only)
2. Runs MediaPipe HandLandmarker to extract 21 landmarks per image
3. Z-score normalises per hand (same as normalize.ts)
4. Trains scikit-learn MLP: 63 inputs -> 26 letter outputs
5. Exports weights as JSON to public/asl_model/model_weights.json
   (no TF/ONNX needed — TypeScript does the forward pass directly)
"""

import os, sys, json
import numpy as np
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "asl_source" / "asl_dataset"
MODEL_OUT   = Path(__file__).parent.parent / "public" / "asl_model"
MODEL_OUT.mkdir(parents=True, exist_ok=True)

LETTERS   = list("abcdefghijklmnopqrstuvwxyz")
LABEL_MAP = {l: i for i, l in enumerate(LETTERS)}

# ── MediaPipe setup ────────────────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

LANDMARKER_MODEL = Path(__file__).parent / "hand_landmarker.task"
if not LANDMARKER_MODEL.exists():
    import urllib.request
    print("Downloading MediaPipe hand landmarker model (~24 MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
        LANDMARKER_MODEL
    )

hand_opts = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER_MODEL)),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.25,
    min_hand_presence_confidence=0.25,
)
landmarker = mp_vision.HandLandmarker.create_from_options(hand_opts)

# ── Helpers ────────────────────────────────────────────────────────────────────

def zscore(arr):
    std = arr.std()
    return (arr - arr.mean()) / std if std > 1e-6 else arr - arr.mean()

def extract(img_path):
    """Returns float32 (63,) or None."""
    import cv2
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None
    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res    = landmarker.detect(mp_img)
    if not res.hand_landmarks:
        return None
    lm = res.hand_landmarks[0]
    xs = zscore(np.array([p.x for p in lm], np.float32))
    ys = zscore(np.array([p.y for p in lm], np.float32))
    zs = zscore(np.array([p.z for p in lm], np.float32))
    return np.concatenate([xs, ys, zs])

# ── Extract landmarks ──────────────────────────────────────────────────────────

X, y = [], []
print(f"Dataset: {DATASET_DIR}\n")

for letter in LETTERS:
    folder = DATASET_DIR / letter
    if not folder.exists():
        print(f"  {letter}: folder missing — skipped")
        continue
    images  = sorted(folder.glob("*.jpeg")) + sorted(folder.glob("*.jpg"))
    ok = 0
    for p in images:
        feat = extract(p)
        if feat is not None:
            X.append(feat); y.append(LABEL_MAP[letter]); ok += 1
    print(f"  {letter.upper()}: {ok}/{len(images)}")

X = np.array(X, np.float32)
y = np.array(y, np.int32)
print(f"\nTotal: {len(X)} samples across {len(np.unique(y))} classes")

np.save(str(MODEL_OUT / "X.npy"), X)
np.save(str(MODEL_OUT / "y.npy"), y)

# ── Train ──────────────────────────────────────────────────────────────────────

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelBinarizer
from sklearn.metrics          import classification_report

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

print("\nTraining MLP (63->256->128->64->26) ...")
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-3,
    batch_size=32,
    learning_rate_init=1e-3,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=12,
    verbose=True,
    random_state=42,
)
mlp.fit(X_tr, y_tr)

y_pred = mlp.predict(X_val)
acc = (y_pred == y_val).mean()
print(f"\nVal accuracy: {acc*100:.1f}%")
print(classification_report(y_val, y_pred, target_names=[l.upper() for l in LETTERS],
                             zero_division=0))

# ── Export weights as JSON ─────────────────────────────────────────────────────
# Format: { coefs: [[...], ...], intercepts: [[...], ...], labels: [...] }
# TypeScript does: x = relu(x @ W + b) for each layer, softmax at end.

weights = {
    "coefs":      [w.tolist() for w in mlp.coefs_],
    "intercepts": [b.tolist() for b in mlp.intercepts_],
    "labels":     [l.upper() for l in LETTERS],
}
out_path = MODEL_OUT / "model_weights.json"
with open(out_path, "w") as f:
    json.dump(weights, f, separators=(",", ":"))

size_kb = out_path.stat().st_size / 1024
print(f"\nWeights saved → {out_path}  ({size_kb:.0f} KB)")
print("Done. Reload the SignSync app to use the trained model.")
