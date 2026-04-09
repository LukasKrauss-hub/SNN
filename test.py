import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# -----------------------------
# Allgemeine Parameter
# -----------------------------
T = 70
tau = 6
V_th = 0.14
V_reset = 0.0
n_classes = 10

# STDP Parameter
tau_plus  = 20.0
tau_minus = 20.0
A_plus    = 0.00001
A_minus   = 0.000015
w_max     = 1.0
w_min     = 0.0

epochs         = 3
train_samples  = 4000
test_samples   = 2000
val_start_idx  = 55000
val_samples    = 5000

# -----------------------------
# MNIST laden
# -----------------------------
mnist  = fetch_openml('mnist_784', version=1, parser='auto')
data   = mnist.data.to_numpy() if hasattr(mnist.data, "to_numpy") else np.asarray(mnist.data)
target = mnist.target.to_numpy() if hasattr(mnist.target, "to_numpy") else np.asarray(mnist.target)

images = data.reshape(-1, 28, 28) / 255.0
labels = target.astype(int)

# -----------------------------
# Gewichte initialisieren
# -----------------------------
def init_weights_from_prototypes(images, labels, n_classes, train_size=10000):
    x_train = images[:train_size].reshape(train_size, -1)
    y_train = labels[:train_size]

    class_weights = np.zeros((n_classes, x_train.shape[1]))
    for c in range(n_classes):
        mask = y_train == c
        if mask.any():
            class_weights[c] = x_train[mask].mean(axis=0)

    norms = np.linalg.norm(class_weights, axis=1, keepdims=True) + 1e-8
    return class_weights / norms

weights = init_weights_from_prototypes(images, labels, n_classes)

# -----------------------------
# Bild -> Spike-Train
# -----------------------------
def image_to_spike_train(image, T):
    flat = image.flatten()
    return (np.random.rand(T, len(flat)) < (flat ** 2)).astype(np.float32)

# -----------------------------
# LIF-Simulation
# -----------------------------
def lif_simulate(input_spikes, weights):
    spike_times = [[] for _ in range(n_classes)]
    V = np.zeros(n_classes)

    for t in range(T):
        I = (weights @ input_spikes[t]) / 8.0
        V += (-V + I) / tau

        fired = V >= V_th
        for c in np.where(fired)[0]:
            spike_times[c].append(t)
            V[c] = V_reset

    return spike_times

# -----------------------------
# Vorhersage über Spike-Aktivität
# -----------------------------
def predict_from_spikes(image, weights, T):
    input_spikes = image_to_spike_train(image, T)
    spike_times = lif_simulate(input_spikes, weights)
    spike_counts = np.array([len(st) for st in spike_times])

    if spike_counts.max() == 0:
        pred = np.argmax(weights @ image.flatten())
    else:
        pred = np.argmax(spike_counts)

    return pred, spike_times, spike_counts, input_spikes

# -----------------------------
# STDP-Update
# -----------------------------
def stdp_update(weights, input_spikes, spike_times, true_label, active_classes):
    pre_spike_times = [
        np.where(input_spikes[:, px] > 0)[0]
        for px in range(input_spikes.shape[1])
    ]

    for c in active_classes:
        if not spike_times[c]:
            continue

        if c == true_label:
            ltp_scale = 1.2
            ltd_scale = 0.8
        else:
            ltp_scale = 0.8
            ltd_scale = 1.2

        dW = np.zeros(weights.shape[1])

        for t_post in spike_times[c]:
            for px in range(weights.shape[1]):
                t_pres = pre_spike_times[px]
                if len(t_pres) == 0:
                    continue

                for t_pre in t_pres:
                    delta_t = t_post - t_pre

                    if delta_t > 0:
                        dW[px] += ltp_scale * A_plus * np.exp(-delta_t / tau_plus)
                    elif delta_t < 0:
                        dW[px] -= ltd_scale * A_minus * np.exp(delta_t / tau_minus)

        # Update begrenzen
        dW = np.clip(dW, -0.01, 0.01)
        weights[c] += dW

    np.clip(weights, w_min, w_max, out=weights)

    norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-8
    weights /= norms

    return weights

# -----------------------------
# Spike-basierte Evaluation
# -----------------------------
def evaluate_snn(images, labels, weights, start_idx, n_eval):
    end_idx = min(start_idx + n_eval, len(images))
    correct = 0

    for i in range(start_idx, end_idx):
        pred, _, _, _ = predict_from_spikes(images[i], weights, T)
        if pred == labels[i]:
            correct += 1

    return correct / (end_idx - start_idx)

# -----------------------------
# Vor Training
# -----------------------------
initial_test_acc = evaluate_snn(images, labels, weights, 60000, test_samples)
print(f"Test-Accuracy vor Training: {initial_test_acc:.3f}")

best_weights = weights.copy()
best_val_acc = evaluate_snn(images, labels, weights, val_start_idx, val_samples)

# -----------------------------
# Training
# -----------------------------
for epoch in range(epochs):
    order = np.random.permutation(train_samples)
    correct = 0

    for i_step, i in enumerate(order):
        img = images[i]
        true_lbl = labels[i]

        pred, spike_times, spike_counts, input_spikes = predict_from_spikes(img, weights, T)

        if pred == true_lbl:
            correct += 1
        else:
            active_classes = np.unique([pred, true_lbl])
            weights = stdp_update(weights, input_spikes, spike_times, true_lbl, active_classes)

        if (i_step + 1) % 500 == 0:
            running_acc = correct / (i_step + 1)
            print(f"  Epoch {epoch+1} | {i_step+1}/{train_samples} | Running Acc: {running_acc:.3f}")

    train_acc = evaluate_snn(images, labels, weights, 0, min(2000, train_samples))
    val_acc   = evaluate_snn(images, labels, weights, val_start_idx, val_samples)

    print(f"Epoch {epoch+1}/{epochs} abgeschlossen | Train-Acc: {train_acc:.3f} | Val-Acc: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = weights.copy()
        print("  Neues bestes Modell gespeichert.")
    else:
        weights = best_weights.copy()
        print("  Kein besseres Modell -> auf bestes Modell zurückgesetzt.")

# Bestes Modell verwenden
weights = best_weights.copy()

final_test_acc = evaluate_snn(images, labels, weights, 60000, test_samples)
print(f"\nTest-Accuracy nach STDP-Training: {final_test_acc:.3f}")

# -----------------------------
# Einzelbild klassifizieren
# -----------------------------
selected_number = None
if selected_number is None:
    selected_number = np.random.randint(1, 70001)

idx      = selected_number - 1
img      = images[idx]
true_lbl = labels[idx]

predicted_label, spike_times, spike_counts, input_spikes = predict_from_spikes(img, weights, T)

if spike_counts.max() == 0:
    rates = weights @ img.flatten()
    rates = (rates - rates.min()) / (np.ptp(rates) + 1e-12)
else:
    rates = spike_counts / T

print(f"\nBild #{selected_number}")
print(f"Predicted : {predicted_label}")
print(f"True      : {true_lbl}")
print(f"Spike-Rates: {np.round(rates, 3)}")

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(img, cmap="gray")
axes[0].set_title(f"True: {true_lbl} | Predicted: {predicted_label}")
axes[0].axis("off")

axes[1].bar(range(10), rates)
axes[1].set_xlabel("Klasse")
axes[1].set_ylabel("Spike-Rate")
axes[1].set_title("Output neuron activity (STDP)")

plt.tight_layout()
plt.show()