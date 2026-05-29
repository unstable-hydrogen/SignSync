"""
retrain_with_csv.py

Merges user-collected CSV with the existing dataset, then retrains.

Usage:
  python scripts/retrain_with_csv.py path/to/asl_landmarks_*.csv
"""

import sys, json, numpy as np
from pathlib import Path
from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import classification_report

MODEL_OUT = Path(__file__).parent.parent / "public" / "asl_model"
LETTERS   = list("abcdefghijklmnopqrstuvwxyz")
LABEL_MAP = {l.upper(): i for i, l in enumerate(LETTERS)}

TARGET_PER_CLASS = 70
NOISE_STD        = 0.07
RNG              = np.random.default_rng(42)

# ── Load existing dataset ──────────────────────────────────────────────────────
X_base = np.load(str(MODEL_OUT / "X.npy"))   # already z-scored
y_base = np.load(str(MODEL_OUT / "y.npy"))
print(f"Existing dataset: {len(X_base)} samples")

# ── Load + z-score user CSV ────────────────────────────────────────────────────

def zscore(arr):
    std = arr.std()
    return (arr - arr.mean()) / std if std > 1e-6 else arr - arr.mean()

def zscore_row(row63):
    """Z-score the 63 raw landmark values (21x + 21y + 21z) independently."""
    xs = zscore(row63[0:21])
    ys = zscore(row63[21:42])
    zs = zscore(row63[42:63])
    return np.concatenate([xs, ys, zs])

csv_paths = sys.argv[1:]
if not csv_paths:
    print("Usage: python retrain_with_csv.py <path/to/*.csv>")
    sys.exit(1)

X_new, y_new = [], []
for csv_path in csv_paths:
    print(f"\nLoading {csv_path} ...")
    with open(csv_path) as f:
        lines = f.read().splitlines()

    # Skip header line
    data_lines = [l for l in lines[1:] if l.strip()]
    skipped = 0
    for line in data_lines:
        parts = line.split(",")
        label = parts[0].strip().upper()
        if label not in LABEL_MAP:
            skipped += 1
            continue
        # Collect the first 63 float values after the label
        try:
            vals = np.array([float(v) for v in parts[1:64]], dtype=np.float32)
        except ValueError:
            skipped += 1
            continue
        if len(vals) < 63:
            skipped += 1
            continue
        X_new.append(zscore_row(vals))
        y_new.append(LABEL_MAP[label])

    print(f"  Loaded {len(y_new)} rows  ({skipped} skipped)")

if not X_new:
    print("No valid rows found in CSV.")
    sys.exit(1)

X_new = np.array(X_new, dtype=np.float32)
y_new = np.array(y_new, dtype=np.int32)

# Show what was collected
new_counts = np.bincount(y_new, minlength=26)
print("\nUser samples per letter:")
for i, l in enumerate(LETTERS):
    if new_counts[i] > 0:
        print(f"  {l.upper()}: {new_counts[i]}")

# ── Merge ──────────────────────────────────────────────────────────────────────
# For letters the user collected, up-weight their samples (repeat 3x)
# so their hand shape dominates over the original dataset images.
X_user_boost = np.tile(X_new, (3, 1))
y_user_boost = np.tile(y_new, 3)

X_merged = np.concatenate([X_base, X_user_boost])
y_merged  = np.concatenate([y_base, y_user_boost])
print(f"\nMerged dataset: {len(X_merged)} samples (user samples weighted 3x)")

# ── Augment under-represented classes ─────────────────────────────────────────
counts = np.bincount(y_merged, minlength=26)
X_parts, y_parts = [X_merged], [y_merged]

for cls in range(26):
    n = counts[cls]
    if n == 0 or n >= TARGET_PER_CLASS:
        continue
    need   = TARGET_PER_CLASS - n
    src    = X_merged[y_merged == cls]
    idx    = RNG.integers(0, len(src), size=need)
    sample = src[idx].copy()
    sample += RNG.normal(0, NOISE_STD, sample.shape).astype(np.float32)
    X_parts.append(sample)
    y_parts.append(np.full(need, cls, dtype=np.int32))

X = np.concatenate(X_parts).astype(np.float32)
y = np.concatenate(y_parts).astype(np.int32)
print(f"After augmentation: {len(X)} samples  "
      f"(min={np.bincount(y).min()} max={np.bincount(y).max()} per class)")

# ── Train ──────────────────────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

print("\nTraining MLP (63->256->128->64->26) ...")
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
print(f"\nWeights -> {out}  ({out.stat().st_size // 1024} KB)")
print("Done. Reload the browser to use the updated model.")
