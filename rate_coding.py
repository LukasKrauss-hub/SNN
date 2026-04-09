import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameter
# -----------------------------

T = 1000
tau = 10
V_th = 1.0
V_reset = 0.0
weight = 3.2

n_samples = 100

# decision threshold als RATE
rate_threshold = 0.1


# -----------------------------
# Rate coding
# -----------------------------

def rate_encode(input_value, T):

    # viel schneller als Python-Schleife
    return (np.random.rand(T) < input_value).astype(int)


# -----------------------------
# LIF neuron
# -----------------------------

def lif_neuron(input_spikes):

    V = 0
    output_spikes = []

    for spike in input_spikes:

        I = weight * spike

        # Euler Diskretisierung
        V = V + (-V + I) / tau

        if V >= V_th:
            output_spikes.append(1)
            V = V_reset
        else:
            output_spikes.append(0)

    return np.array(output_spikes)


# -----------------------------
# Datensatz erzeugen
# -----------------------------

inputs = np.random.rand(n_samples)

labels = (inputs > 0.5).astype(int)     ##> 0.5 ==> Klasse 1, sonst Klasse 0


# -----------------------------
# Simulation
# -----------------------------

predictions = []
spike_counts = []

for x in inputs:

    input_spikes = rate_encode(x, T)

    output_spikes = lif_neuron(input_spikes)

    spike_count = np.sum(output_spikes)

    spike_counts.append(spike_count)

    spike_rate = spike_count / T

    if spike_rate > rate_threshold:
        predictions.append(1)
    else:
        predictions.append(0)


spike_counts = np.array(spike_counts)
predictions = np.array(predictions)


# -----------------------------
# Accuracy
# -----------------------------

accuracy = np.mean(predictions == labels)

print("Accuracy:", accuracy)


# -----------------------------
# Accuracy in Bins
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


# -----------------------------
# Plot 1: Input vs Spikes
# -----------------------------

plt.figure()

plt.scatter(inputs, spike_counts)

plt.axvline(0.5, color="green", linestyle="--", label="true boundary")

plt.axhline(rate_threshold*T, color="red", linestyle="--", label="decision boundary")

plt.xlabel("Input value")
plt.ylabel("Output spike count")
plt.title("Rate Coding + LIF neuron classification")

plt.legend()
plt.show()


# -----------------------------
# Plot 2: Accuracy vs Input
# -----------------------------

plt.figure()

plt.plot(bin_centers, bin_accuracy, marker="o")

plt.xlabel("Input value")
plt.ylabel("Accuracy")
plt.title("Classification accuracy vs input")

plt.ylim(0,1)

plt.show()


# -----------------------------
# Plot 3: Energy (Spikes)
# -----------------------------

plt.figure()

plt.scatter(inputs, spike_counts)

plt.xlabel("Input value")
plt.ylabel("Spike count")
plt.title("Energy usage (spikes)")

plt.show()


# -----------------------------
# Average Energy
# -----------------------------

print("Average spikes per sample:", np.mean(spike_counts))