import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# -----------------------------
# Analyse-Tools ein/aus
# -----------------------------
ANALYSE_SPIKE_AKTIVITAET   = True   # Spike-Counts pro Klasse vor und nach Training
ANALYSE_GEWICHTE_VISUALISIERUNG = True   # Gewichte als 28x28 Bilder darstellen
ANALYSE_ACCURACY_PRO_KLASSE = True   # Accuracy getrennt pro Ziffer
ANALYSE_KONFUSIONSMATRIX    = True   # Verwechslungsmatrix aller Klassen
ANALYSE_UPDATE_RATE         = True   # Wie oft wird STDP ausgeloest?

# -----------------------------
# Allgemeine Parameter
# -----------------------------
T = 30
tau = 6
V_th = 0.15
V_reset = 0.0
n_classes = 10

# STDP Parameter
tau_plus = 20.0
tau_minus = 20.0
A_plus = 0.0001
A_minus = 0.00015
w_max = 1.0
w_min = 0.0

epochs = 10
train_samples = 8000
test_samples = 2000
val_start_idx = 50000
val_samples = 1000
ausgabe_intervall = 500

# -----------------------------
# MNIST laden
# -----------------------------
mnist = fetch_openml('mnist_784', version=1, parser='auto')
data = mnist.data.to_numpy() if hasattr(mnist.data, "to_numpy") else np.asarray(mnist.data)
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
            class_weights[c] = x_train[mask].mean(axis=0)  # Durchschnittsbild als Startgewicht

    norms = np.linalg.norm(class_weights, axis=1, keepdims=True) + 1e-8
    return class_weights / norms  # Normiert damit alle Klassen faire Startbedingungen haben

weights = init_weights_from_prototypes(images, labels, n_classes)

# -----------------------------
# Bild -> Spike-Train
# -----------------------------
def image_to_spike_train(image, T):
    flat = image.flatten()
    return (np.random.rand(T, len(flat)) < (flat ** 2)).astype(np.float32)  # quadratisch -> mehr Kontrast

# -----------------------------
# LIF-Simulation (Winner-Takes-All)
# -----------------------------
def lif_simulate_winner_takes_it_all(input_spikes, weights):
    spike_times = [[] for _ in range(n_classes)]
    V = np.zeros(n_classes)

    for t in range(T):
        I = (weights @ input_spikes[t]) / 8.0
        V += (-V + I) / tau

        fired = V >= V_th
        if fired.any():
            winner = np.argmax(V)  # nur das staerkste Neuron feuert wirklich
            for c in np.where(fired)[0]:
                if c != winner:
                    V[c] = V_reset
                    fired[c] = False
            spike_times[winner].append(t)
            V[winner] = V_reset

        # Fallback: wenn gar nichts gefeuert hat, nimm das Neuron mit dem höchsten V
    if all(len(st) == 0 for st in spike_times):
        fallback = int(np.argmax(V))
        spike_times[fallback].append(T - 1)

    return spike_times

# -----------------------------
# Vorhersage ueber Spike-Aktivitaet
# -----------------------------
def predict_from_spikes(image, weights, T):
    input_spikes = image_to_spike_train(image, T)
    spike_times = lif_simulate_winner_takes_it_all(input_spikes, weights)
    spike_counts = np.array([len(st) for st in spike_times])

    if spike_counts.max() == 0:
        pred = np.argmax(weights @ image.flatten())  # Fallback auf linearen Score
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
            ltp_scale = 1.2  # richtige Klasse: LTP verstaerkt, LTD gedaempft
            ltd_scale = 0.8
        else:
            ltp_scale = 0.8  # falsche Klasse: LTP gedaempft, LTD verstaerkt
            ltd_scale = 1.2

        dW = np.zeros(weights.shape[1])

        for t_post in spike_times[c]:
            for px in range(weights.shape[1]):
                t_pres = pre_spike_times[px]
                if len(t_pres) == 0:
                    continue

                for t_pre in t_pres:
                    delta_t = t_post - t_pre

                    if delta_t > 0:  # Prae vor Post -> LTP
                        dW[px] += ltp_scale * A_plus * np.exp(-delta_t / tau_plus)
                    elif delta_t < 0:  # Post vor Prae -> LTD
                        dW[px] -= ltd_scale * A_minus * np.exp(delta_t / tau_minus)

        dW = np.clip(dW, -0.01, 0.01)  # extremen Einzelupdate verhindern
        weights[c] += dW

    np.clip(weights, w_min, w_max, out=weights)

    norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-8
    weights /= norms  # Normierung damit kein Klasse durch schiere Gewichtsgroesse dominiert

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

# ==============================
# ANALYSE-FUNKTIONEN
# ==============================

def analyse_spike_aktivitaet(images, labels, weights, n_samples=200, label=""):
    """Zeigt pro Klasse wie oft das eigene Neuron feuert vs. die Konkurrenz."""
    print(f"\n--- Spike-Aktivitaet {label} ---")
    counts_per_class = [[] for _ in range(n_classes)]

    for i in range(n_samples):
        true_lbl = labels[i]
        _, _, spike_counts, _ = predict_from_spikes(images[i], weights, T)
        counts_per_class[true_lbl].append(spike_counts)

    print(f"{'Kl.':<5} {'eigene Spikes':<16} {'max Konkurrenz':<16} {'stumm %':<10} {'Marge'}")
    for c in range(n_classes):
        if not counts_per_class[c]:
            continue
        arr = np.array(counts_per_class[c])
        eigene  = arr[:, c].mean()
        andere  = np.delete(arr, c, axis=1).max(axis=1).mean()
        stumm   = (arr[:, c] == 0).mean() * 100
        marge   = eigene - andere
        flag    = " <- PROBLEM" if stumm > 30 or marge < 0 else ""
        print(f"{c:<5} {eigene:<16.2f} {andere:<16.2f} {stumm:<10.1f} {marge:+.2f}{flag}")


def analyse_gewichte_visualisierung(weights, titel="Gewichte"):
    """Stellt die Gewichte jeder Klasse als 28x28 Bild dar."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    vmax = weights.max()
    for c in range(n_classes):
        ax = axes[c // 5][c % 5]
        ax.imshow(weights[c].reshape(28, 28), cmap='hot', vmin=0, vmax=vmax)
        ax.set_title(f'Klasse {c}')
        ax.axis('off')
    plt.suptitle(titel)
    plt.tight_layout()
    plt.show()


def analyse_accuracy_pro_klasse(images, labels, weights, start_idx, n_eval, label=""):
    """Zeigt Accuracy getrennt pro Ziffer als Balkendiagramm im Terminal."""
    print(f"\n--- Accuracy pro Klasse {label} ---")
    correct = np.zeros(n_classes)
    total   = np.zeros(n_classes)
    end_idx = min(start_idx + n_eval, len(images))

    for i in range(start_idx, end_idx):
        true_lbl = labels[i]
        pred, _, _, _ = predict_from_spikes(images[i], weights, T)
        total[true_lbl] += 1
        if pred == true_lbl:
            correct[true_lbl] += 1

    accs = correct / (total + 1e-8)
    for c in range(n_classes):
        balken = '█' * int(accs[c] * 20)
        flag   = " <- SCHWACH" if accs[c] < 0.5 else ""
        print(f"Klasse {c}: {balken:<20} {accs[c]:.2f}{flag}")
    return accs


def analyse_konfusionsmatrix(images, labels, weights, start_idx, n_eval, label=""):
    """Berechnet und zeigt die Verwechslungsmatrix."""
    print(f"\n--- Konfusionsmatrix {label} ---")
    matrix  = np.zeros((n_classes, n_classes), dtype=int)
    end_idx = min(start_idx + n_eval, len(images))

    for i in range(start_idx, end_idx):
        pred, _, _, _ = predict_from_spikes(images[i], weights, T)
        matrix[labels[i]][pred] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xlabel('Vorhergesagt')
    ax.set_ylabel('Wahr')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            farbe = 'white' if matrix[i, j] > matrix.max() * 0.5 else 'black'
            ax.text(j, i, matrix[i, j], ha='center', va='center',
                    fontsize=8, color=farbe)
    plt.colorbar(im)
    plt.title(f'Konfusionsmatrix {label}')
    plt.tight_layout()
    plt.show()

    # groesste Verwechslungen ausgeben
    print("Groesste Verwechslungen (wahr -> pred: anzahl):")
    fehler = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and matrix[i, j] > 0:
                fehler.append((matrix[i, j], i, j))
    for anzahl, wahr, pred in sorted(fehler, reverse=True)[:5]:
        print(f"  {wahr} -> {pred}: {anzahl}x")


# -----------------------------
# Vor Training
# -----------------------------
print("=" * 50)
print("VOR TRAINING")
print("=" * 50)

initial_test_acc = evaluate_snn(images, labels, weights, 60000, test_samples)
print(f"Test-Accuracy vor Training: {initial_test_acc:.3f}")

best_weights = weights.copy()
best_val_acc = evaluate_snn(images, labels, weights, val_start_idx, val_samples)
print(f"Validation-Accuracy vor Training: {best_val_acc:.3f}")

if ANALYSE_SPIKE_AKTIVITAET:
    analyse_spike_aktivitaet(images, labels, weights, n_samples=200, label="(vor Training)")

if ANALYSE_GEWICHTE_VISUALISIERUNG:
    analyse_gewichte_visualisierung(weights, titel="Gewichte vor Training (Prototypen)")

if ANALYSE_ACCURACY_PRO_KLASSE:
    analyse_accuracy_pro_klasse(images, labels, weights, 60000, test_samples, label="(vor Training)")

if ANALYSE_KONFUSIONSMATRIX:
    analyse_konfusionsmatrix(images, labels, weights, 60000, min(500, test_samples), label="(vor Training)")

# -----------------------------
# Training
# -----------------------------
print("\n" + "=" * 50)
print("TRAINING")
print("=" * 50)

for epoch in range(epochs):
    order   = np.random.permutation(train_samples)
    correct = 0

    # Zaehler fuer Update-Rate Analyse
    updates        = 0
    no_spike_cases = 0

    for i_step, i in enumerate(order):
        img      = images[i]
        true_lbl = labels[i]

        pred, spike_times, spike_counts, input_spikes = predict_from_spikes(img, weights, T)

        if spike_counts.max() == 0:
            no_spike_cases += 1

        if pred == true_lbl:
            correct += 1
        else:
            active_classes = np.unique([pred, true_lbl])
            weights = stdp_update(weights, input_spikes, spike_times, true_lbl, active_classes)
            updates += 1

        if (i_step + 1) % ausgabe_intervall == 0:
            running_acc = correct / (i_step + 1)
            print(f"  Epoch {epoch+1} | {i_step+1}/{train_samples} | Running Acc: {running_acc:.3f}", end="")

            if ANALYSE_UPDATE_RATE:
                update_rate    = updates / (i_step + 1) * 100
                stumm_rate     = no_spike_cases / (i_step + 1) * 100
                update_flag    = " <- ZU WENIG" if update_rate < 5 else (" <- KONVERGIERT NICHT" if update_rate > 70 else "")
                stumm_flag     = " <- V_TH ZU HOCH?" if stumm_rate > 10 else ""
                print(f" | Updates: {update_rate:.1f}%{update_flag} | Stumm: {stumm_rate:.1f}%{stumm_flag}", end="")

            print()

    train_acc = evaluate_snn(images, labels, weights, 0, min(2000, train_samples))
    val_acc   = evaluate_snn(images, labels, weights, val_start_idx, val_samples)

    print(f"Epoch {epoch+1}/{epochs} abgeschlossen | Train-Acc: {train_acc:.3f} | Val-Acc: {val_acc:.3f}")

    previous_best_val_acc = best_val_acc

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = weights.copy()
        print(f"  Neues bestes Modell gespeichert ({previous_best_val_acc:.4f} -> {best_val_acc:.4f}).")
    else:
        weights = best_weights.copy()
        print(f"  Kein besseres Modell ({val_acc:.4f} <= Bestwert {previous_best_val_acc:.4f}) -> zurueckgesetzt.")

# Bestes Modell verwenden
weights = best_weights.copy()

# -----------------------------
# Nach Training
# -----------------------------
print("\n" + "=" * 50)
print("NACH TRAINING")
print("=" * 50)

final_test_acc = evaluate_snn(images, labels, weights, 60000, test_samples)
print(f"\nTest-Accuracy nach STDP-Training: {final_test_acc:.3f}")

if ANALYSE_SPIKE_AKTIVITAET:
    analyse_spike_aktivitaet(images, labels, weights, n_samples=200, label="(nach Training)")

if ANALYSE_GEWICHTE_VISUALISIERUNG:
    analyse_gewichte_visualisierung(weights, titel="Gewichte nach Training")

if ANALYSE_ACCURACY_PRO_KLASSE:
    analyse_accuracy_pro_klasse(images, labels, weights, 60000, test_samples, label="(nach Training)")

if ANALYSE_KONFUSIONSMATRIX:
    analyse_konfusionsmatrix(images, labels, weights, 60000, min(500, test_samples), label="(nach Training)")

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
# Plot Einzelbild
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