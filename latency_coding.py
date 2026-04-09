import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(42)  # Fester Seed: gleiche Zufallszahlen bei jedem Lauf (reproduzierbare Ergebnisse).

# -----------------------------
# Parameter
# -----------------------------

T = 1000
tau = 12
V_th = 1.0
V_reset = 0.0
weight = 12.5
jitter_std = 8.0  # Standardabweichung des Spike-Zeit-Jitters in Zeitschritten.
noise_std = 0.03  # Standardabweichung des Membranrauschens pro Zeitschritt.
label_slope = 12.0  

n_samples = 100

# decision threshold (Zeitpunkt)
latency_threshold = T * 0.5     # wenn Spike vor t=500 → Klasse 1, sonst Klasse 0


# -----------------------------
# Latency coding
# -----------------------------

def latency_encode(x, T, jitter_std):  # jitter_std=0.0 erlaubt auch einen noisefreien Lauf.

    spikes = np.zeros(T)

    # großer Input -> früher Spike
    spike_time = int((1 - x) * (T - 1))  # Basis-Mapping: grosses x -> frueher Spike.
    # Timing jitter macht den Encoder realistischer.
    spike_time = int(np.clip(spike_time + np.random.normal(0, jitter_std), 0, T - 1))  # ungenauigkeit: 68 % der Spike-Zeiten liegen innerhalb von jitter_std Zeitschritten um die Basis-Spike-Zeit.

    spikes[spike_time] = 1

    return spikes


# -----------------------------
# LIF neuron
# -----------------------------

def lif_neuron(input_spikes, noise_std):  

    V = 0.0  # Membranpotential als float.
    output_spikes = []

    for spike in input_spikes:

        I = weight * spike

        V = V + (-V + I) / tau + np.random.normal(0, noise_std)  # Leak + Inputstrom + additives Membranrauschen.

 
        if V >= V_th:
            output_spikes.append(1)
            V = V_reset
        else:
            output_spikes.append(0)

    return np.array(output_spikes)


# -----------------------------
# Datensatz
# -----------------------------

inputs = np.random.rand(n_samples)  # 100 Werte zwischen 0 und 1

#Labels statt harter Schwelle bei x=0.5.

#Für x = 0.8 ist die Wahrscheinlichkeit für Klasse 1 ungefähr 97.4%.
p_class1 = 1 / (1 + np.exp(-label_slope * (inputs - 0.5)))  # nahe 0 -> wsh Klasse 0, nahe 1 -> wsh Klasse 1 mit steiler Übergang um 0.5 herum (je höher label_slope, desto schärfer der Übergang).
labels = (np.random.rand(n_samples) < p_class1).astype(int)  # Bernoulli-Sampling erzeugt realistisch verrauschte Labels.


# -----------------------------
# Simulation
# -----------------------------

predictions = []
latencies = []
spike_counts = []

for x in inputs:

    input_spikes = latency_encode(x, T, jitter_std=jitter_std)  # Encoder mit konfigurierbarem Timing-Jitter.

    output_spikes = lif_neuron(input_spikes, noise_std=noise_std)  # LIF mit konfigurierbarem Membranrauschen.

    spike_count = np.sum(output_spikes)
    spike_counts.append(spike_count)

    # Zeitpunkt des ersten Spikes
    spike_times = np.where(output_spikes == 1)[0]           # np.where gibt die Indizes zurück, an denen output_spikes == 1 ist (also die Zeitpunkte der Spikes)

    if len(spike_times) > 0:
        latency = spike_times[0]            # erster Spike bestimmt die Latenz
    else:
        latency = T  # kein Spike

    latencies.append(latency)

    # Klassifikation über Zeit
    if latency < latency_threshold:
        predictions.append(1)
    else:
        predictions.append(0)


latencies = np.array(latencies)
predictions = np.array(predictions)
spike_counts = np.array(spike_counts)


# -----------------------------
# Accuracy
# -----------------------------

accuracy = np.mean(predictions == labels)

print("Accuracy:", accuracy)
print("Average spikes:", np.mean(spike_counts))
print("Average latency:", np.mean(latencies))


# -----------------------------
# Plot 1: Input vs Latency
# -----------------------------

plt.figure()

plt.scatter(inputs, latencies)

plt.axvline(0.5, color="green", linestyle="--", label="true boundary")
plt.axhline(latency_threshold, color="red", linestyle="--", label="decision boundary")

plt.xlabel("Input value")
plt.ylabel("Latency (time step)")
plt.title("Latency Coding")

plt.legend()
plt.show()


# -----------------------------
# Plot 2: Accuracy vs Input
# -----------------------------

bins = np.linspace(0,1,20)

bin_centers = []
bin_accuracy = []

for i in range(len(bins)-1):

    low = bins[i]
    high = bins[i+1]

    mask = (inputs >= low) & (inputs < high)

    if np.sum(mask) > 0:

        acc = np.mean(predictions[mask] == labels[mask])

        bin_centers.append((low+high)/2)
        bin_accuracy.append(acc)


plt.figure()

plt.plot(bin_centers, bin_accuracy, marker="o")

plt.xlabel("Input value")
plt.ylabel("Accuracy")
plt.title("Latency Coding: Accuracy vs Input")

plt.ylim(0,1)

plt.show()


# -----------------------------
# Plot 3: Energy
# -----------------------------

plt.figure()

plt.scatter(inputs, spike_counts)

plt.xlabel("Input value")
plt.ylabel("Spike count")
plt.title("Energy usage (Latency Coding)")

plt.show()
