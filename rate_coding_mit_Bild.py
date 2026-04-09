import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# -----------------------------
# Parameter
# -----------------------------

T = 100
tau = 6
V_th = 0.12
V_reset = 0.0

n_classes = 10

# Training-Parameter fuer das lineare Klassifikationsmodell
do_training = True
epochs = 5
learning_rate = 0.02
train_samples = 60000
test_samples = 4000


# -----------------------------
# MNIST laden
# -----------------------------

mnist = fetch_openml('mnist_784', version=1, parser='auto')

# Support both pandas and numpy-like outputs from fetch_openml.
data = mnist.data.to_numpy() if hasattr(mnist.data, "to_numpy") else np.asarray(mnist.data)
target = mnist.target.to_numpy() if hasattr(mnist.target, "to_numpy") else np.asarray(mnist.target)

images = data.reshape(-1, 28, 28) / 255.0               #Wahrscheinlichkeit: 0 (schwarz) bis 1 (weiß).
labels = target.astype(int)


def init_weights_from_prototypes(images, labels, n_classes, train_size=10000):
    """Build one weight vector per class from average training images."""
    x_train = images[:train_size].reshape(train_size, -1)
    y_train = labels[:train_size]

    class_weights = np.zeros((n_classes, x_train.shape[1]), dtype=float)

    for c in range(n_classes):
        class_samples = x_train[y_train == c]
        if len(class_samples) > 0:
            class_weights[c] = class_samples.mean(axis=0)

    # Normalize so classes are comparable in the dot-product current.
    norms = np.linalg.norm(class_weights, axis=1, keepdims=True) + 1e-8
    return class_weights / norms


weights = init_weights_from_prototypes(images, labels, n_classes)

def predict_with_weights(image, weights):
    x = image.flatten()
    scores = weights @ x
    return np.argmax(scores), scores


def evaluate_linear_classifier(images, labels, weights, start_idx, n_eval):
    end_idx = min(start_idx + n_eval, len(images))
    x_eval = images[start_idx:end_idx].reshape(-1, 28 * 28)
    y_eval = labels[start_idx:end_idx]
    scores = x_eval @ weights.T
    preds = np.argmax(scores, axis=1)
    return (preds == y_eval).mean()


def train_perceptron(images, labels, weights, train_samples, epochs, learning_rate):
    x_train = images[:train_samples].reshape(train_samples, -1)
    y_train = labels[:train_samples]

    for epoch in range(epochs):
        order = np.random.permutation(train_samples)
        mistakes = 0

        for i in order:
            x = x_train[i]
            y = y_train[i]

            scores = weights @ x
            pred = np.argmax(scores)

            if pred != y:
                mistakes += 1
                weights[y] += learning_rate * x
                weights[pred] -= learning_rate * x

        # Stabilisiert die Skalen pro Klasse fuer vergleichbare Scores.
        norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-8
        weights = weights / norms

        train_acc = evaluate_linear_classifier(images, labels, weights, 0, min(2000, train_samples))
        print(f"Epoch {epoch + 1}/{epochs} | Fehler: {mistakes} | Train-Acc (Teilmenge): {train_acc:.3f}")

    return weights


if do_training:
    acc_before = evaluate_linear_classifier(images, labels, weights, 60000, test_samples)
    print(f"Test-Accuracy vor Training: {acc_before:.3f}")

    weights = train_perceptron(
        images=images,
        labels=labels,
        weights=weights,
        train_samples=train_samples,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    acc_after = evaluate_linear_classifier(images, labels, weights, 60000, test_samples)
    print(f"Test-Accuracy nach Training: {acc_after:.3f}")


# -----------------------------
# EIN Bild auswählen
# -----------------------------

# Setze auf eine Zahl von 1 bis 70000, um gezielt ein Bild zu waehlen.
# Bei None wird bei jedem Lauf zufaellig ein Bild gewaehlt.
selected_number = None
if selected_number is None:
    selected_number = np.random.randint(1, 70001)

# Umrechnung auf 0-basierte Indizierung (MNIST: 0..69999).
idx = selected_number - 1
img = images[idx]
true_label = labels[idx]

pred, scores = predict_with_weights(img, weights)
print("Predicted (ohne LIF):", pred)
print("True:", true_label)
print("Scores:", np.round(scores, 3))

plt.imshow(img, cmap="gray")
plt.title(f"True label: {true_label}")
plt.axis("off")
plt.show()


# -----------------------------
# Bild → Spike-Train
# -----------------------------

def image_to_spike_train(image, T):
    flat = image.flatten()
    spikes = np.random.rand(T, len(flat)) < flat
    return spikes.astype(int)


# -----------------------------
# LIF Layer (10 Neuronen)
# -----------------------------

def lif_layer(input_spikes):

    spike_rates = []
    current_scores = []

    for c in range(n_classes):

        V = 0
        spikes = 0

        currents = input_spikes @ weights[c]
        current_scores.append(currents.mean())

        for t in range(input_spikes.shape[0]):

            # Class-specific current from pixel-wise similarity.
            I = currents[t] / 8.0

            V = V + (-V + I) / tau

            if V >= V_th:
                spikes += 1
                V = V_reset

        spike_rates.append(spikes / T)

    spike_rates = np.array(spike_rates)

    # If all neurons stay silent, use continuous current scores instead.
    if np.allclose(spike_rates, 0.0):
        current_scores = np.array(current_scores)
        spread = np.ptp(current_scores)
        if spread < 1e-12:
            return np.ones(n_classes) / n_classes
        return (current_scores - current_scores.min()) / spread

    return spike_rates


# -----------------------------
# Simulation
# -----------------------------

input_spikes = image_to_spike_train(img, T)


rates = lif_layer(input_spikes)

predicted_label = np.argmax(rates)

print("Predicted:", predicted_label)
print("True:", true_label)
print("Rates:", np.round(rates, 3))


# -----------------------------
# Plot: Spike-Rates pro Klasse
# -----------------------------

plt.figure()
plt.bar(range(10), rates)
plt.xlabel("Klasse")
plt.ylabel("Spike-Rate")
plt.title("Output neuron activity")
plt.show()