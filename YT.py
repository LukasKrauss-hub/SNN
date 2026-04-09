import numpy as np
import matplotlib.pyplot as plt

T = 200                # Zeitsteps
tau = 10               # Zeitkonstante
V_th = 1.0             # Threshold
V_reset = 0.0

V = 0
membrane = []
spikes = []

for t in range(T):

    I = 0.2
    if 50 < t < 120:
        I = 1.3

    V = V + (-V + I) / tau

    if V >= V_th:
        spikes.append(t)
        V = V_reset

    membrane.append(V)

plt.plot(membrane)
plt.scatter(spikes, [1]*len(spikes), color="red")
plt.title("LIF Neuron")
plt.show()














