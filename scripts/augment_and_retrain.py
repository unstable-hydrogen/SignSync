"""
augment_and_retrain.py

Balances the dataset by augmenting under-represented letters to TARGET_COUNT,
then retrains the MLP and re-exports weights.

Augmentation for z-scored landmark vectors (63 dims):
  - Gaussian noise (simulates hand jitter / slight position change)
  - Small scale jitter before re-normalising (simulates distance variation)
"""
import json, numpy as np
from pathlib import Path
from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import classification_report

MODEL_OUT  = Path(__file__).parent.parent / "public" / "asl_model"
LETTERS    = list("abcdefghijklmnopqrstuvwxyz")
TARGET     = 70   # augment every letter up to this count
NOISE_STD  = 0.07 # std of Gaussian noise in z-score space
RNG        = np.random.default_rng(42)

X_orig = np.load(str(MODEL_OUT / "X.npy"))
y_orig = np.load(str(MODEL_OUT / "y.npy"))

# ── Augment ────────────────────────────────────────────────────────────────────
X_aug, y_aug = [X_orig], [y_orig]

counts = np.bincount(y_orig, minlength=26)
for cls in range(26):
    n = counts[cls]
    if n == 0 or n >= TARGET:
        continue
    need   = TARGET - n
    src    = X_orig[y_orig == cls]
    idx    = RNG.integers(0, len(src), size=need)
    sample = src[idx].copy()
    # Add Gaussian noise
    sample += RNG.normal(0, NOISE_STD, sample.shape).astype(np.float32)
    X_aug.append(sample)
    y_aug.append(np.full(need, cls, dtype=np.int32))
    print(f"  {LETTERS[cls].upper()}: {n} -> {n+need}")

X = np.concatenate(X_aug).astype(np.float32)
y = np.concatenate(y_aug).astype(np.int32)
print(f"\nAugmented dataset: {len(X)} samples")
print(f"Class distribution: min={np.bincount(y).min()} max={np.bincount(y).max()}")

# ── Train ──────────────────────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

print("\nTraining augmented MLP ...")
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    alpha=5e-4,
    batch_size=32,
    learning_rate_init=1e-3,
    max_iter=400,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=True,
    random_state=42,
)
mlp.fit(X_tr, y_tr)

y_pred = mlp.predict(X_val)
acc = (y_pred == y_val).mean()
print(f"\nVal accuracy: {acc*100:.1f}%")
print(classification_report(
    y_val, y_pred,
    target_names=[l.upper() for l in LETTERS],
    zero_division=0,
))

# ── Export ─────────────────────────────────────────────────────────────────────
weights = {
    "coefs":      [w.tolist() for w in mlp.coefs_],
    "intercepts": [b.tolist() for b in mlp.intercepts_],
    "labels":     [l.upper() for l in LETTERS],
}
out = MODEL_OUT / "model_weights.json"
with open(out, "w") as f:
    json.dump(weights, f, separators=(",", ":"))
print(f"Weights -> {out}  ({out.stat().st_size // 1024} KB)")
