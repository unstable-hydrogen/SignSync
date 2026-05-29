"""Train MLP from pre-extracted landmarks and export weights as JSON."""
import json, numpy as np
from pathlib import Path
from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import classification_report

MODEL_OUT = Path(__file__).parent.parent / "public" / "asl_model"
LETTERS   = list("abcdefghijklmnopqrstuvwxyz")

X = np.load(str(MODEL_OUT / "X.npy"))
y = np.load(str(MODEL_OUT / "y.npy"))
print(f"Loaded {len(X)} samples, {len(np.unique(y))} classes")

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

print("Training MLP (63->256->128->64->26) ...")
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-3,
    batch_size=32,
    learning_rate_init=1e-3,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
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

weights = {
    "coefs":      [w.tolist() for w in mlp.coefs_],
    "intercepts": [b.tolist() for b in mlp.intercepts_],
    "labels":     [l.upper() for l in LETTERS],
}
out = MODEL_OUT / "model_weights.json"
with open(out, "w") as f:
    json.dump(weights, f, separators=(",", ":"))
print(f"Weights -> {out}  ({out.stat().st_size//1024} KB)")
