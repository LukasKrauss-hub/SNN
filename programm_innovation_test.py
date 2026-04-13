import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time

start_time = time.time()

# -----------------------------
# Analyse-Tools ein/aus
# -----------------------------
ANALYSE_SPIKE_AKTIVITAET        = True
ANALYSE_GEWICHTE_VISUALISIERUNG = True
ANALYSE_ACCURACY_PRO_KLASSE     = True
ANALYSE_KONFUSIONSMATRIX        = True
ANALYSE_UPDATE_RATE             = True

# ANALYSE_SPIKE_AKTIVITAET        = False
# ANALYSE_GEWICHTE_VISUALISIERUNG = False
# ANALYSE_ACCURACY_PRO_KLASSE     = False
# ANALYSE_KONFUSIONSMATRIX        = False
# ANALYSE_UPDATE_RATE             = False

# -----------------------------
# Allgemeine Parameter
# -----------------------------
T          = 20    # mehr Zeitschritte → stabilere Spike-Counts, weniger Rauschen
tau        = 6
V_th       = 0.15
V_reset    = 0.0
n_classes  = 10
N_RUNS     = 1    # mehrere stochastische Spike-Trains pro Bild → Rauschen mitteln

# STDP Parameter
tau_plus   = 20.0
tau_minus  = 20.0
A_plus = 7e-7
A_minus = 9e-7
w_max      = 1.0
w_min      = 0.0

# Lernrate-Decay: nach jeder Epoche wird A_plus/A_minus mit diesem Faktor multipliziert
LR_DECAY   = 0.9   # langsam weniger lernen → Feintuning in späten Epochen

# Adaptive Schwelle: nach jedem Spike steigt V_th kurz an → verhindert Dominanz
THETA_PLUS  = 0.02   # Schwellenerhöhung nach Spike
THETA_DECAY = 0.99   # Schwelle klingt pro Zeitschritt ab

# Homeostase: Klassen die zu selten feuern bekommen Bonus
TARGET_RATE        = 0.10   # Ziel-Spike-Rate (Spikes pro Zeitschritt)
HOMEOSTASIS_RATE   = 0.995  # gleitender Mittelwert der Aktivität
HOMEOSTASIS_SCALE  = 1.0    # wie stark der Homeostase-Effekt ist

epochs        = 5
train_samples = 2000
test_samples  = 1000
val_start_idx = 50000
val_samples   = 500
ausgabe_intervall = 1000

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

# Homeostase-Zustand: gleitender Durchschnitt der Spike-Rate pro Klasse
homeostasis_tracker = np.ones(n_classes) * TARGET_RATE

# -----------------------------
# Bild -> Spike-Train
# -----------------------------
def image_to_spike_train(image, T):
    flat = image.flatten()
    # quadratisch: erhöht Kontrast zwischen hellen und dunklen Pixeln
    return (np.random.rand(T, len(flat)) < (flat ** 2)).astype(np.float32)

# -----------------------------
# LIF-Simulation mit adaptiver Schwelle und Winner-Takes-All
# -----------------------------
def lif_simulate(input_spikes, weights, theta=None):
    """
    theta: adaptive Schwellenerhöhung pro Klasse (optional).
           Wenn None wird keine adaptive Schwelle verwendet.
    """
    spike_times = [[] for _ in range(n_classes)]
    V = np.zeros(n_classes)

    if theta is None:
        theta = np.zeros(n_classes)

    for t in range(T):
        I = (weights @ input_spikes[t]) / 8.0
        V += (-V + I) / tau

        # adaptive Schwelle klingt pro Zeitschritt ab
        theta *= THETA_DECAY

        # Winner-Takes-All mit adaptiver Schwelle
        effektive_schwelle = V_th + theta
        fired = V >= effektive_schwelle

        if fired.any():
            # unter den feuernden Neuronen gewinnt das mit höchstem V
            kandidaten = np.where(fired)[0]
            winner = kandidaten[np.argmax(V[kandidaten])]

            spike_times[winner].append(t)
            V[winner]     = V_reset
            theta[winner] += THETA_PLUS  # Schwelle nach Spike erhöhen

            # alle anderen feuernden Neuronen werden nur zurückgesetzt
            for c in kandidaten:
                if c != winner:
                    V[c] = V_reset

    # Fallback: wenn gar nichts gefeuert hat → Neuron mit höchstem V gewinnt
    # verhindert stumme Bilder komplett
    if all(len(st) == 0 for st in spike_times):
        fallback = int(np.argmax(V))
        spike_times[fallback].append(T - 1)

    return spike_times, theta

# -----------------------------
# Mehrere Runs mitteln → stabile Vorhersage
# -----------------------------
def predict_from_spikes(image, weights, T, n_runs=N_RUNS):
    """
    Führt n_runs stochastische Simulationen durch und
    summiert die Spike-Counts über alle Runs.
    Gibt den letzten input_spikes und spike_times für STDP zurück.
    """
    total_counts   = np.zeros(n_classes)
    last_spikes    = None
    last_input     = None
    theta          = np.zeros(n_classes)  # adaptive Schwelle über Runs hinweg

    for _ in range(n_runs):
        input_spikes = image_to_spike_train(image, T)
        spike_times, theta = lif_simulate(input_spikes, weights, theta.copy())
        counts = np.array([len(st) for st in spike_times])
        total_counts += counts
        last_spikes   = spike_times
        last_input    = input_spikes

    if total_counts.max() == 0:
        pred = np.argmax(weights @ image.flatten())
    else:
        pred = np.argmax(total_counts)

    return pred, last_spikes, total_counts, last_input

# -----------------------------
# STDP-Update mit Homeostase
# -----------------------------
def stdp_update(weights, input_spikes, spike_times, true_label,
                active_classes, homeostasis_tracker, a_plus, a_minus):
    """
    Homeostase-Modulation: Klassen die zu selten feuern bekommen
    einen LTP-Bonus, zu oft feuernde Klassen werden gedämpft.
    """
    pre_spike_times = [
        np.where(input_spikes[:, px] > 0)[0]
        for px in range(input_spikes.shape[1])
    ]

    for c in active_classes:
        post_spike_times = list(spike_times[c])

        if c == true_label and not post_spike_times:
            post_spike_times = [T - 1]

        if not post_spike_times:
            continue

        # Reward-Modulation: richtig vs. falsch
        if c == true_label:
            ltp_scale = 1.2
            ltd_scale = 0.8
        else:
            ltp_scale = 0.8
            ltd_scale = 1.2

        # Homeostase-Modulation: zu selten feuernde Klassen bekommen Bonus
        # homeostasis_tracker[c] < TARGET_RATE → Neuron feuert zu selten → LTP stärken
        h_ratio = TARGET_RATE / (homeostasis_tracker[c] + 1e-6)
        h_scale = np.clip(h_ratio ** HOMEOSTASIS_SCALE, 0.3, 3.0)

        dW = np.zeros(weights.shape[1])

        for t_post in post_spike_times:
            for px in range(weights.shape[1]):
                t_pres = pre_spike_times[px]
                if len(t_pres) == 0:
                    continue

                for t_pre in t_pres:
                    delta_t = t_post - t_pre

                    if delta_t > 0:
                        dW[px] += h_scale * ltp_scale * a_plus * np.exp(-delta_t / tau_plus)
                    elif delta_t < 0:
                        dW[px] -= ltd_scale * a_minus * np.exp(delta_t / tau_minus)
                        # Note: für LTD keine h_scale → Homeostase wirkt nur auf LTP

        dW = np.clip(dW, -0.005, 0.005)  # konservativer Clip für stabiles Lernen
        weights[c] += dW

    np.clip(weights, w_min, w_max, out=weights)
    norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-8
    weights /= norms

    return weights

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_snn(images, labels, weights, start_idx, n_eval, n_runs=N_RUNS):
    end_idx = min(start_idx + n_eval, len(images))
    correct = 0
    for i in range(start_idx, end_idx):
        pred, _, _, _ = predict_from_spikes(images[i], weights, T, n_runs)
        if pred == labels[i]:
            correct += 1
    return correct / (end_idx - start_idx)

# ==============================
# ANALYSE-FUNKTIONEN
# ==============================

def analyse_spike_aktivitaet(images, labels, weights, n_samples=300, label=""):
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
        arr    = np.array(counts_per_class[c])
        eigene = arr[:, c].mean()
        andere = np.delete(arr, c, axis=1).max(axis=1).mean()
        stumm  = (arr[:, c] == 0).mean() * 100
        marge  = eigene - andere
        flag   = " <- PROBLEM" if stumm > 20 or marge < 0 else ""
        print(f"{c:<5} {eigene:<16.2f} {andere:<16.2f} {stumm:<10.1f} {marge:+.2f}{flag}")

def analyse_gewichte_visualisierung(weights, titel="Gewichte"):
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
        flag   = " <- SCHWACH" if accs[c] < 0.6 else ""
        print(f"Klasse {c}: {balken:<20} {accs[c]:.2f}{flag}")
    return accs

def analyse_konfusionsmatrix(images, labels, weights, start_idx, n_eval, label=""):
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
            ax.text(j, i, matrix[i, j], ha='center', va='center', fontsize=8, color=farbe)
    plt.colorbar(im)
    plt.title(f'Konfusionsmatrix {label}')
    plt.tight_layout()
    plt.show()
    print("Groesste Verwechslungen:")
    fehler = [(matrix[i, j], i, j) for i in range(n_classes)
              for j in range(n_classes) if i != j and matrix[i, j] > 0]
    for anzahl, wahr, pred in sorted(fehler, reverse=True)[:5]:
        print(f"  {wahr} -> {pred}: {anzahl}x")

# -----------------------------
# Vor Training
# -----------------------------
print("=" * 50)
print("VOR TRAINING")
print("=" * 50)

initial_test_acc = evaluate_snn(images, labels, weights, 60000, test_samples)
print(f"Test-Accuracy vor Training:       {initial_test_acc:.3f}")

best_weights  = weights.copy()
best_val_acc  = evaluate_snn(images, labels, weights, val_start_idx, val_samples)
best_homeostasis_tracker = homeostasis_tracker.copy()
print(f"Validation-Accuracy vor Training: {best_val_acc:.3f}")

if ANALYSE_SPIKE_AKTIVITAET:
    analyse_spike_aktivitaet(images, labels, weights, label="(vor Training)")
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

# aktuelle Lernraten — werden pro Epoche mit LR_DECAY multipliziert
a_plus_eff  = A_plus
a_minus_eff = A_minus

val_history = []  # Verlauf der Val-Accuracy für Diagnose

for epoch in range(epochs):
    order   = np.random.permutation(train_samples)
    correct = 0
    updates        = 0
    no_spike_cases = 0

    for i_step, i in enumerate(order):
        img      = images[i]
        true_lbl = labels[i]

        pred, spike_times, spike_counts, input_spikes = predict_from_spikes(
            img, weights, T, n_runs=N_RUNS
        )

        # Homeostase-Tracker aktualisieren: gleitender Mittelwert der Spike-Rate
        spike_rates = spike_counts / (T * N_RUNS)
        homeostasis_tracker = (HOMEOSTASIS_RATE * homeostasis_tracker
                               + (1 - HOMEOSTASIS_RATE) * spike_rates)

        if spike_counts.max() == 0:
            no_spike_cases += 1

        if pred == true_lbl:
            correct += 1
        else:
            active_classes = np.unique([pred, true_lbl])
            weights = stdp_update(
                weights, input_spikes, spike_times, true_lbl,
                active_classes, homeostasis_tracker, a_plus_eff, a_minus_eff
            )
            updates += 1

        if (i_step + 1) % ausgabe_intervall == 0:
            running_acc = correct / (i_step + 1)
            print(f"  Epoch {epoch+1} | {i_step+1}/{train_samples}"
                  f" | Running Acc: {running_acc:.3f}", end="")
            if ANALYSE_UPDATE_RATE:
                update_rate = updates / (i_step + 1) * 100
                stumm_rate  = no_spike_cases / (i_step + 1) * 100
                u_flag = " <- ZU WENIG"        if update_rate < 5  else (
                         " <- KONVERGIERT NICHT" if update_rate > 70 else "")
                s_flag = " <- V_TH ZU HOCH?"   if stumm_rate  > 5  else ""
                print(f" | Updates: {update_rate:.1f}%{u_flag}"
                      f" | Stumm: {stumm_rate:.1f}%{s_flag}", end="")
            print()

    train_acc = evaluate_snn(images, labels, weights, 0, min(2000, train_samples))
    val_acc   = evaluate_snn(images, labels, weights, val_start_idx, val_samples)
    val_history.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.3f}"
          f" | Val: {val_acc:.3f} | LR: {a_plus_eff:.2e}")

    previous_best = best_val_acc

    if val_acc > best_val_acc:
        best_val_acc  = val_acc
        best_weights  = weights.copy()
        best_homeostasis_tracker = homeostasis_tracker.copy()
        print(f"  Neues bestes Modell ({previous_best:.4f} -> {best_val_acc:.4f}).")
        # nach Verbesserung: Lernrate leicht reduzieren
        a_plus_eff  *= LR_DECAY
        a_minus_eff *= LR_DECAY
    else:
        weights = best_weights.copy()
        homeostasis_tracker = best_homeostasis_tracker.copy()
        a_plus_eff  *= LR_DECAY
        a_minus_eff *= LR_DECAY
        print(f"  Keine Verbesserung ({val_acc:.4f} <= {previous_best:.4f})."
              f" Rollback auf bestes Modell, Lernrate reduziert auf {a_plus_eff:.2e}.")

    # Frühes Abbrechen wenn Lernrate zu klein geworden ist
    if a_plus_eff < 1e-7:
        print("  Lernrate zu klein — Training beendet.")
        break

# Bestes Modell laden
weights = best_weights.copy()

# Val-Verlauf ausgeben
print(f"\nVal-Accuracy Verlauf: {[f'{v:.3f}' for v in val_history]}")
print(f"Bestes Val-Ergebnis:  {best_val_acc:.3f}")

# -----------------------------
# Nach Training
# -----------------------------
print("\n" + "=" * 50)
print("NACH TRAINING")
print("=" * 50)

final_test_acc = evaluate_snn(images, labels, weights, 60000, test_samples)
print(f"\nTest-Accuracy nach Training: {final_test_acc:.3f}")

if ANALYSE_SPIKE_AKTIVITAET:
    analyse_spike_aktivitaet(images, labels, weights, label="(nach Training)")
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

predicted_label, spike_times, spike_counts, input_spikes = predict_from_spikes(
    img, weights, T, n_runs=N_RUNS
)

if spike_counts.max() == 0:
    rates = weights @ img.flatten()
    rates = (rates - rates.min()) / (np.ptp(rates) + 1e-12)
else:
    rates = spike_counts / (T * N_RUNS)

print(f"\nBild #{selected_number}")
print(f"Predicted : {predicted_label}")
print(f"True      : {true_lbl}")
print(f"Spike-Rates: {np.round(rates, 3)}")

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

elapsed_seconds = time.time() - start_time
hours, remainder = divmod(int(elapsed_seconds), 3600)
minutes, seconds = divmod(remainder, 60)
print(f"\nGesamtlaufzeit: {hours:02d}:{minutes:02d}:{seconds:02d} ({elapsed_seconds:.1f} Sekunden)")