import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# -----------------------------
# Parameter
# -----------------------------
T = 25
TAU = 6.0
V_TH_HIDDEN = 0.22
V_TH_OUTPUT = 0.35
N_CLASSES = 10

TRAIN_SAMPLES = 12000
TEST_SAMPLES = 2000
EPOCHS = 12
LR = 0.06


# -----------------------------
# Daten laden
# -----------------------------
mnist = fetch_openml("mnist_784", version=1, parser="auto")
data = mnist.data.to_numpy() if hasattr(mnist.data, "to_numpy") else np.asarray(mnist.data)
labels = mnist.target.to_numpy() if hasattr(mnist.target, "to_numpy") else np.asarray(mnist.target)

images = data.reshape(-1, 28, 28).astype(np.float32) / 255.0
labels = labels.astype(int)


# -----------------------------
# Feature-Masken (Hidden Layer)
# -----------------------------
def _line_mask(axis: str, pos: int, width: int = 2):
    m = np.zeros((28, 28), dtype=np.float32)
    if axis == "v":
        m[:, max(0, pos - width):min(28, pos + width + 1)] = 1.0
    elif axis == "h":
        m[max(0, pos - width):min(28, pos + width + 1), :] = 1.0
    return m


def _diag_mask(direction: str, offset: int = 0, width: int = 1):
    y, x = np.indices((28, 28))
    if direction == "diag_main":
        d = np.abs(y - (x + offset))
    else:  # diag_anti
        d = np.abs(y - ((27 - x) + offset))
    return (d <= width).astype(np.float32)


def _curve_mask(center_x: float, center_y: float, radius: float, thickness: float, arc: str):
    y, x = np.indices((28, 28))
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    ring = np.abs(dist - radius) <= thickness

    if arc == "top":
        arc_region = y <= center_y
    elif arc == "bottom":
        arc_region = y >= center_y
    elif arc == "left":
        arc_region = x <= center_x
    else:  # right
        arc_region = x >= center_x

    return (ring & arc_region).astype(np.float32)


def build_feature_bank():
    features = []
    names = []

    # Vertikale Linien
    for pos in [5, 10, 14, 18, 23]:
        features.append(_line_mask("v", pos, width=1))
        names.append(f"vertikal_{pos}")

    # Horizontale Linien
    for pos in [5, 10, 14, 18, 23]:
        features.append(_line_mask("h", pos, width=1))
        names.append(f"horizontal_{pos}")

    # Diagonalen
    for off in [-5, 0, 5]:
        features.append(_diag_mask("diag_main", offset=off, width=1))
        names.append(f"diag_main_{off}")
        features.append(_diag_mask("diag_anti", offset=off, width=1))
        names.append(f"diag_anti_{off}")

    # Rundungen/Bögen
    curves = [
        (13.5, 13.5, 9.5, 1.3, "top", "bogen_top"),
        (13.5, 13.5, 9.5, 1.3, "bottom", "bogen_bottom"),
        (13.5, 13.5, 9.5, 1.3, "left", "bogen_left"),
        (13.5, 13.5, 9.5, 1.3, "right", "bogen_right"),
        (13.5, 13.5, 6.0, 1.2, "top", "bogen_klein_top"),
        (13.5, 13.5, 6.0, 1.2, "bottom", "bogen_klein_bottom"),
    ]
    for cx, cy, r, t, arc, name in curves:
        features.append(_curve_mask(cx, cy, r, t, arc))
        names.append(name)

    # Normierung
    features = np.array(features, dtype=np.float32)
    flat = features.reshape(features.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    flat = flat / norms
    return flat, names


feature_weights, feature_names = build_feature_bank()
N_HIDDEN = feature_weights.shape[0]


# -----------------------------
# Spike-Codierung und Simulation
# -----------------------------
def image_to_spike_train(image, T):
    flat = image.flatten()
    return (np.random.rand(T, flat.size) < (flat ** 1.8)).astype(np.float32)


def lif_counts(input_spikes, weights, v_th):
    """LIF für eine komplette Schicht; gibt Spike-Count pro Neuron zurück."""
    n_neurons = weights.shape[0]
    v = np.zeros(n_neurons, dtype=np.float32)
    counts = np.zeros(n_neurons, dtype=np.int32)

    for t in range(input_spikes.shape[0]):
        i_t = weights @ input_spikes[t]
        v += (-v + i_t) / TAU
        fired = v >= v_th
        counts[fired] += 1
        v[fired] = 0.0

    return counts


def forward_two_layer(image, w_in_hidden, w_hidden_out):
    in_spikes = image_to_spike_train(image, T)
    hidden_counts = lif_counts(in_spikes, w_in_hidden, V_TH_HIDDEN)

    # Hidden-Counts zu Spike-Train für Output umformen (Rate-basiert)
    hidden_rates = hidden_counts.astype(np.float32) / T
    hidden_spikes = (np.random.rand(T, hidden_rates.size) < hidden_rates).astype(np.float32)

    out_counts = lif_counts(hidden_spikes, w_hidden_out, V_TH_OUTPUT)
    return hidden_rates, out_counts


# -----------------------------
# Output-Layer trainieren
# -----------------------------
rng = np.random.default_rng(42)
w_hidden_out = rng.normal(loc=0.0, scale=0.08, size=(N_CLASSES, N_HIDDEN)).astype(np.float32)


def predict(image):
    hidden_rates, out_counts = forward_two_layer(image, feature_weights, w_hidden_out)
    if out_counts.max() == 0:
        logits = w_hidden_out @ hidden_rates
        pred = int(np.argmax(logits))
    else:
        pred = int(np.argmax(out_counts))
    return pred, hidden_rates, out_counts


for epoch in range(EPOCHS):
    order = np.random.permutation(TRAIN_SAMPLES)
    correct = 0

    for step, i in enumerate(order, 1):
        img = images[i]
        y = labels[i]

        pred, hidden_rates, out_counts = predict(img)
        if pred == y:
            correct += 1

        # Supervisierter Delta-Schritt auf Output-Layer
        target = np.zeros(N_CLASSES, dtype=np.float32)
        target[y] = 1.0
        probs = np.exp((w_hidden_out @ hidden_rates) * 3.0)
        probs /= probs.sum() + 1e-8

        grad = (target - probs)[:, None] * hidden_rates[None, :]
        w_hidden_out += LR * grad
        w_hidden_out = np.clip(w_hidden_out, -2.0, 2.0)

        if step % 2000 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Step {step}/{TRAIN_SAMPLES} | Running Acc: {correct / step:.3f}")

    print(f"Epoch {epoch + 1} fertig | Train-Acc: {correct / TRAIN_SAMPLES:.3f}")


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(start_idx, n_samples):
    end = min(start_idx + n_samples, len(images))
    correct = 0
    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)

    for i in range(start_idx, end):
        pred, _, _ = predict(images[i])
        y = labels[i]
        correct += int(pred == y)
        matrix[y, pred] += 1

    return correct / (end - start_idx), matrix


acc, conf = evaluate(60000, TEST_SAMPLES)
print(f"\nTest-Accuracy (2-Schicht-Modell): {acc:.3f}")


# -----------------------------
# Hidden-Features visualisieren
# -----------------------------
fig, axes = plt.subplots(4, 6, figsize=(12, 8))
for idx, ax in enumerate(axes.flat):
    if idx >= N_HIDDEN:
        ax.axis("off")
        continue
    ax.imshow(feature_weights[idx].reshape(28, 28), cmap="hot")
    ax.set_title(feature_names[idx], fontsize=8)
    ax.axis("off")
plt.suptitle("Hidden-Neuronen: Linien-/Diagonal-/Rundungs-Detektoren")
plt.tight_layout()
plt.show()


# -----------------------------
# Konfusionsmatrix visualisieren
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(conf, cmap="Blues")
ax.set_title("Konfusionsmatrix (2-Schicht-Modell)")
ax.set_xlabel("Vorhergesagt")
ax.set_ylabel("Wahr")
ax.set_xticks(range(N_CLASSES))
ax.set_yticks(range(N_CLASSES))
plt.colorbar(im)
plt.tight_layout()
plt.show()


# -----------------------------
# Einzelbild-Analyse
# -----------------------------
sample_idx = np.random.randint(60000, 70000)
img = images[sample_idx]
y = labels[sample_idx]
pred, hidden_rates, out_counts = predict(img)

print(f"\nBeispielindex: {sample_idx}")
print(f"True Label   : {y}")
print(f"Prediction   : {pred}")

# Top-Features anzeigen
best_hidden = np.argsort(hidden_rates)[-6:][::-1]
print("Aktivste Hidden-Features:")
for h in best_hidden:
    print(f"  - {feature_names[h]:<20} rate={hidden_rates[h]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img, cmap="gray")
axes[0].set_title(f"True={y} | Pred={pred}")
axes[0].axis("off")

axes[1].bar(range(N_CLASSES), out_counts / max(1, T))
axes[1].set_title("Output-Neuronen Aktivität")
axes[1].set_xlabel("Klasse")
axes[1].set_ylabel("Spike-Rate")
plt.tight_layout()
plt.show()
