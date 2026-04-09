import numpy as np
from sklearn.datasets import fetch_openml

# -----------------------------
# Allgemeine Infos: Ein Neuron, also Ein Potential pro Klasse, das über die Zeit integriert wird. Es feuert, wenn es die Schwelle überschreitet. Die Gewichte werden durch STDP angepasst, abhängig von der zeitlichen Beziehung zwischen Pre- und Post-Spikes.
#
#
#
#
#       
#
#
#
#
#
#
#
# -----------------------------
T = 40
tau = 6
V_th = 0.12
V_reset = 0.0
n_classes = 10

# STDP Parameter (zeitbasiertes Lernen)
tau_plus  = 20.0   # ms - Zeitfenster LTP
tau_minus = 20.0   # ms - Zeitfenster LTD
A_plus    = 0.0001   # Lernrate LTP
A_minus   = 0.00015  # Lernrate LTD (etwas groesser -> Stabilisierung)
w_max     = 1.0
w_min     = 0.0

epochs         = 4
train_samples  = 3000
test_samples   = 2000   

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
# (Prototypen wie im Original)
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
    flat = image.flatten()          # macht aus 28x28 -> 784 Werte zwischen 0 und 1 
    return (np.random.rand(T, len(flat)) < (flat ** 2)).astype(np.float32) # je heller das Pixel, desto höher die Wahrscheinlichkeit für einen Spike in diesem Zeitschritt (quadratisch verstärkt) -> 0.5 -> 25% chance, 0.8 -> 64% chance, 0.2 -> 4% chance, 0.0 -> 0% chance, 1.0 -> 100% chance

# -----------------------------
# LIF-Simulation -> gibt Spike-Zeiten zurueck
# -----------------------------
def lif_simulate(input_spikes, weights):
    """
    Gibt fuer jede Klasse die Liste der Spike-Zeitpunkte zurueck.
    input_spikes: (T, n_pixels)
    weights:      (n_classes, n_pixels)
    """
    spike_times = [[] for _ in range(n_classes)]
    V = np.zeros(n_classes)

    for t in range(T):
        # Strom: Skalarprodukt jeder Klasse mit dem aktuellen Eingabevektor
        I = (weights @ input_spikes[t]) / 8.0 # also Heller Pixel -> hohe wahrscheinlichkeit für viele Spikes zb (1, 1, 1, 0, 1, 1, 1, 0) -> höherer wert bei klassen mit höheren gewichten
        V += (-V + I) / tau
        fired = V >= V_th       # Alle Neuronen, die die Schwelle überschreiten, feuern (alle Klassen werden gleichzeitig überprüft) fired = [True, False, False...]
        for c in np.where(fired)[0]:    # gibt die Indizes der True-Einträge zurück — also welche Klassen-Neuronen in diesem Zeitschritt gefeuert haben
            spike_times[c].append(t)    # Zeitpunkt merken → bei welchem t bei welcher klasse wurde gefeuert? brauchen wir für STDP
            V[c] = V_reset

    return spike_times


# -----------------------------
# STDP-Update (zeitbasiert)
# Reward-moduliert: korrektes Neuron erhaelt LTP-Bonus
# -----------------------------
# nur Top-Klassen lernen lassen


def stdp_update(weights, input_spikes, spike_times, true_label):
    """
    Fuer jeden Post-Spike eines Neurons werden alle Pre-Spikes bewertet:
      delta_t = t_post - t_pre
      delta_t > 0: pre kam zuerst -> LTP
      delta_t < 0: post kam zuerst -> LTD

    Reward-Modulation:
      Korrektes Neuron: voller LTP-Verstaerker
      Falsche Neuronen: LTD-Verstaerker (Konkurrenz)
    """
    # Vorberechnete Pre-Spike-Zeiten pro Pixel
    pre_spike_times = [np.where(input_spikes[:, px] > 0)[0]     # gibt die Zeitschritte zurück, in denen Pixel px einen Spike hatte (also wann das Pixel aktiv war) zb [0, 2, 7, ..., 99]
                       for px in range(input_spikes.shape[1])]  # input_spikes.shape[1] = 784

    for c in top_k:  # nur die Top-K Klassen updaten
        if not spike_times[c]:
            continue            # Wenn das Neuron in einer Klasse nie gefeuert hat, gibt es kein Post-Spike -> kein Update

        # Reward-Skalierung: richtige Klasse lernt, falsche werden gedaempft
        if c == true_label:
            ltp_scale =  1.2
            ltd_scale =  0.8
        else:
            ltp_scale =  0.8
            ltd_scale =  1.2

        dW = np.zeros(weights.shape[1])

        for t_post in spike_times[c]:
            for px in range(weights.shape[1]):
                t_pres = pre_spike_times[px]
                if len(t_pres) == 0:
                    continue

                for t_pre in t_pres:
                    delta_t = t_post - t_pre       # für jedes Post-Spike schauen, wann die Pre-Spikes waren -> delta_t berechnen -> entscheiden ob LTP oder LTD

                    if delta_t > 0:
                        # Pre vor Post -> LTP (Hebb'sches Lernen)
                        dW[px] += ltp_scale * A_plus * np.exp(-delta_t / tau_plus)      # STDP LTP Formel: A_plus * exp(-delta_t / tau_plus) -> je größer delta_t, desto kleiner der LTP Bonus
                    elif delta_t < 0:
                        # Post vor Pre -> LTD (Anti-Hebb)
                        dW[px] -= ltd_scale * A_minus * np.exp(delta_t / tau_minus)     # STDP LTD Formel: A_minus * exp(delta_t / tau_minus) -> je kleiner delta_t (also je früher das Post-Spike im Vergleich zum Pre-Spike), desto größer der LTD Bonus

        weights[c] += dW

    # Gewichte in biologisch plausiblen Grenzen halten
    np.clip(weights, w_min, w_max, out=weights)

    # Normalisierung fuer vergleichbare Scores
    norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-8
    weights /= norms

    return weights


# -----------------------------
# Klassifikation (unveraendert)
# -----------------------------
def predict(image, weights):
    x = image.flatten()
    scores = weights @ x
    return np.argmax(scores), scores


def evaluate(images, labels, weights, start_idx, n_eval): 
    end_idx = min(start_idx + n_eval, len(images))
    x_eval  = images[start_idx:end_idx].reshape(-1, 784) # macht aus (n_eval, 28, 28) -> (n_eval, 784)
    y_eval  = labels[start_idx:end_idx]   # gibt die wahren Labels für die Evaluationsbeispiele zurück
    preds   = np.argmax(x_eval @ weights.T, axis=1) # berechnet die Scores für alle Klassen und wählt die Klasse mit dem höchsten Score als Vorhersage aus
    return (preds == y_eval).mean()


# -----------------------------
# STDP-Training
# Hauptschleife
# -----------------------------
print(f"Test-Accuracy vor Training: "
      f"{evaluate(images, labels, weights, 60000, test_samples):.3f}")

for epoch in range(epochs):
    order = np.random.permutation(train_samples)
    correct = 0

    for i_step, i in enumerate(order):
        img_flat = images[i].flatten()
        true_lbl = labels[i]

        # Spike-Train erzeugen
        input_spikes = image_to_spike_train(images[i], T)

        # LIF simulieren -> Spike-Zeiten
        spike_times = lif_simulate(input_spikes, weights)

        # Vorhersage anhand Spike-Anzahl
        spike_counts = np.array([len(st) for st in spike_times])

        top_k = np.argsort(spike_counts)[-3:]

        if spike_counts.max() == 0:
            pred = np.argmax(weights @ img_flat)
        else:
            pred = np.argmax(spike_counts)

        if pred == true_lbl:
            correct += 1

        # STDP-Gewichtsupdate (zeitbasiert, kein Gradient)
        weights = stdp_update(weights, input_spikes, spike_times, true_lbl)

        # Fortschritt ausgeben
        if (i_step + 1) % 500 == 0:
            running_acc = correct / (i_step + 1)
            print(f"  Epoch {epoch+1} | {i_step+1}/{train_samples} "
                  f"| Running Acc: {running_acc:.3f}")

    train_acc = evaluate(images, labels, weights, 0, min(2000, train_samples))
    print(f"Epoch {epoch+1}/{epochs} abgeschlossen | "
          f"Train-Acc: {train_acc:.3f}")

print(f"\nTest-Accuracy nach STDP-Training: "
      f"{evaluate(images, labels, weights, 60000, test_samples):.3f}")


# -----------------------------
# Einzelbild klassifizieren
# -----------------------------
selected_number = None
if selected_number is None:
    selected_number = np.random.randint(1, 70001)

idx       = selected_number - 1
img       = images[idx]
true_lbl  = labels[idx]

input_spikes = image_to_spike_train(img, T)
spike_times  = lif_simulate(input_spikes, weights)
spike_counts = np.array([len(st) for st in spike_times])

if spike_counts.max() == 0:
    rates = weights @ img.flatten()
    rates = (rates - rates.min()) / (np.ptp(rates) + 1e-12)
else:
    rates = spike_counts / T

predicted_label = np.argmax(rates)

print(f"\nBild #{selected_number}")
print(f"Predicted : {predicted_label}")
print(f"True      : {true_lbl}")
print(f"Spike-Rates: {np.round(rates, 3)}")

# Plot
import matplotlib.pyplot as plt

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