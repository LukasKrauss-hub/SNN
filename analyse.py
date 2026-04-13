import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# -----------------------------
# Analyse-Tools ein/aus
# -----------------------------
ANALYSE_SPIKE_AKTIVITAET = True
ANALYSE_GEWICHTE_VISUALISIERUNG = True
ANALYSE_ACCURACY_PRO_KLASSE = True
ANALYSE_KONFUSIONSMATRIX = True
ANALYSE_UPDATE_RATE = True
ANALYSE_EINZELBILD_DETAIL = True

# -----------------------------
# Allgemeine Parameter
# -----------------------------
SEED = 42
rng = np.random.default_rng(SEED)

T = 30
tau = 6.0
n_input = 28 * 28
n_hidden = 64
n_classes = 10

V_th_hidden = 0.20
V_th_output = 0.17
V_reset = 0.0

# STDP Input -> Hidden
stdp_tau_pre = 20.0
stdp_tau_post = 20.0
A_plus_h = 0.0020
A_minus_h = 0.0012

# Supervisiertes Lernen Hidden -> Output (spike-nahe Hybrid-Regel)
lr_out = 0.010

w_max = 1.0
w_min = 0.0

epochs = 8
train_samples = 8000
test_samples = 2000
val_start_idx = 50000
val_samples = 1000
ausgabe_intervall = 400

# -----------------------------
# MNIST laden
# -----------------------------
mnist = fetch_openml("mnist_784", version=1, parser="auto")
data = mnist.data.to_numpy() if hasattr(mnist.data, "to_numpy") else np.asarray(mnist.data)
target = mnist.target.to_numpy() if hasattr(mnist.target, "to_numpy") else np.asarray(mnist.target)

images = data.reshape(-1, 28, 28).astype(np.float32) / 255.0
labels = target.astype(int)


# -----------------------------
# Gewichte initialisieren
# -----------------------------
def init_input_hidden_weights(images, n_hidden, train_size=12000):
    """Initialisiert Hidden-Gewichte aus zufaelligen Trainingsbildern (prototype-like, aber nicht klassengebunden)."""
    x_train = images[:train_size].reshape(train_size, -1)
    idx = rng.choice(train_size, size=n_hidden, replace=False)
    w = x_train[idx].copy()
    w += rng.normal(0.0, 0.03, size=w.shape)
    w = np.clip(w, 0.0, None)
    norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
    return (w / norms).astype(np.float32)


def init_hidden_output_weights(n_classes, n_hidden):
    """Startet Output nahe neutral, damit Hidden-Features erst entstehen koennen."""
    w = rng.uniform(0.02, 0.08, size=(n_classes, n_hidden)).astype(np.float32)
    norms = np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
    return w / norms


W_in_hidden = init_input_hidden_weights(images, n_hidden)
W_hidden_out = init_hidden_output_weights(n_classes, n_hidden)


# -----------------------------
# Bild -> Spike-Train
# -----------------------------
def image_to_spike_train(image, T):
    flat = image.flatten()
    # beibehaltener Kontrast-Boost
    return (rng.random((T, flat.size)) < (flat ** 2)).astype(np.float32)


# -----------------------------
# 2-Schicht LIF-Simulation
# -----------------------------
def lif_simulate_two_layer(input_spikes, w_in_hidden, w_hidden_out):
    hidden_spike_times = [[] for _ in range(n_hidden)]
    output_spike_times = [[] for _ in range(n_classes)]

    hidden_spike_train = np.zeros((T, n_hidden), dtype=np.float32)
    output_spike_train = np.zeros((T, n_classes), dtype=np.float32)

    V_hidden = np.zeros(n_hidden, dtype=np.float32)
    V_output = np.zeros(n_classes, dtype=np.float32)

    for t in range(T):
        # Input -> Hidden (LIF)
        I_hidden = (w_in_hidden @ input_spikes[t]) / 8.0
        V_hidden += (-V_hidden + I_hidden) / tau

        fired_hidden = V_hidden >= V_th_hidden
        if fired_hidden.any():
            # WTA im Hidden-Layer fuer Spezialisierung
            winner_h = int(np.argmax(V_hidden))
            fired_hidden[:] = False
            fired_hidden[winner_h] = True
            hidden_spike_train[t, winner_h] = 1.0
            hidden_spike_times[winner_h].append(t)
            V_hidden[winner_h] = V_reset

        # Hidden -> Output (LIF)
        I_output = (w_hidden_out @ hidden_spike_train[t]) / 2.0
        V_output += (-V_output + I_output) / tau

        fired_output = V_output >= V_th_output
        if fired_output.any():
            # WTA in Klassenschicht wie im alten Modell
            winner_o = int(np.argmax(V_output))
            fired_output[:] = False
            fired_output[winner_o] = True
            output_spike_train[t, winner_o] = 1.0
            output_spike_times[winner_o].append(t)
            V_output[winner_o] = V_reset

    # Fallback wie bisher: wenn keine Output-Spikes, nehme hoechstes Membranpotential
    if output_spike_train.sum() == 0:
        fallback = int(np.argmax(V_output))
        output_spike_train[T - 1, fallback] = 1.0
        output_spike_times[fallback].append(T - 1)

    hidden_counts = hidden_spike_train.sum(axis=0).astype(np.int32)
    output_counts = output_spike_train.sum(axis=0).astype(np.int32)

    return {
        "hidden_spike_times": hidden_spike_times,
        "output_spike_times": output_spike_times,
        "hidden_spike_train": hidden_spike_train,
        "output_spike_train": output_spike_train,
        "hidden_counts": hidden_counts,
        "output_counts": output_counts,
        "V_hidden_end": V_hidden,
        "V_output_end": V_output,
    }


# -----------------------------
# Vorhersage
# -----------------------------
def predict_from_spikes(image, w_in_hidden, w_hidden_out):
    input_spikes = image_to_spike_train(image, T)
    sim = lif_simulate_two_layer(input_spikes, w_in_hidden, w_hidden_out)

    output_counts = sim["output_counts"]
    if output_counts.max() == 0:
        pred = int(np.argmax(sim["V_output_end"]))
    else:
        pred = int(np.argmax(output_counts))

    return pred, sim, input_spikes


# -----------------------------
# STDP: Input -> Hidden (unsupervised)
# -----------------------------
def stdp_update_hidden(w_in_hidden, input_spikes, hidden_spike_train):
    pre_trace = np.zeros(n_input, dtype=np.float32)
    post_trace = np.zeros(n_hidden, dtype=np.float32)

    decay_pre = np.exp(-1.0 / stdp_tau_pre)
    decay_post = np.exp(-1.0 / stdp_tau_post)

    dW = np.zeros_like(w_in_hidden)

    for t in range(T):
        pre = input_spikes[t]
        post = hidden_spike_train[t]

        pre_trace = pre_trace * decay_pre + pre
        post_trace = post_trace * decay_post + post

        # LTP: pre kurz vor post
        dW += A_plus_h * np.outer(post, pre_trace)
        # LTD: post kurz vor spaeterem pre
        dW -= A_minus_h * np.outer(post_trace, pre)

    dW = np.clip(dW, -0.02, 0.02)
    w_in_hidden += dW
    np.clip(w_in_hidden, w_min, w_max, out=w_in_hidden)

    norms = np.linalg.norm(w_in_hidden, axis=1, keepdims=True) + 1e-8
    w_in_hidden /= norms

    updates = int(np.count_nonzero(np.abs(dW) > 1e-12))
    return w_in_hidden, updates


# -----------------------------
# Hidden -> Output Lernen (supervised, spike-nahe Hybrid)
# -----------------------------
def supervised_update_output(w_hidden_out, hidden_counts, pred, true_lbl):
    if pred == true_lbl:
        return w_hidden_out, 0

    x = hidden_counts.astype(np.float32)
    x = x / (x.sum() + 1e-8)

    # Richtige Klasse verstaerken, falsche Klasse schwaechen
    w_hidden_out[true_lbl] += lr_out * x
    w_hidden_out[pred] -= lr_out * x

    np.clip(w_hidden_out, w_min, w_max, out=w_hidden_out)
    norms = np.linalg.norm(w_hidden_out, axis=1, keepdims=True) + 1e-8
    w_hidden_out /= norms

    updates = int(np.count_nonzero(x > 0))
    return w_hidden_out, updates


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_snn(images, labels, w_in_hidden, w_hidden_out, start_idx, n_eval):
    end_idx = min(start_idx + n_eval, len(images))
    correct = 0
    for i in range(start_idx, end_idx):
        pred, _, _ = predict_from_spikes(images[i], w_in_hidden, w_hidden_out)
        if pred == labels[i]:
            correct += 1
    return correct / (end_idx - start_idx)


# ==============================
# ANALYSE-FUNKTIONEN
# ==============================
def analyse_spike_aktivitaet(images, labels, w_in_hidden, w_hidden_out, n_samples=200, label=""):
    print(f"\n--- Spike-Aktivitaet {label} ---")

    hidden_activity = np.zeros((n_classes, n_hidden), dtype=np.float32)
    output_activity = np.zeros((n_classes, n_classes), dtype=np.float32)
    per_class_count = np.zeros(n_classes, dtype=np.float32)

    for i in range(n_samples):
        y = labels[i]
        _, sim, _ = predict_from_spikes(images[i], w_in_hidden, w_hidden_out)
        hidden_activity[y] += sim["hidden_counts"]
        output_activity[y] += sim["output_counts"]
        per_class_count[y] += 1

    for c in range(n_classes):
        if per_class_count[c] == 0:
            continue
        hidden_mean = hidden_activity[c] / per_class_count[c]
        output_mean = output_activity[c] / per_class_count[c]
        stumm_hidden = (hidden_mean == 0).mean() * 100
        eigene_out = output_mean[c]
        andere_out = np.delete(output_mean, c).max()
        marge = eigene_out - andere_out
        flag = " <- PROBLEM" if stumm_hidden > 40 or marge < 0 else ""

        top_hidden = np.argsort(hidden_mean)[-3:][::-1]
        print(
            f"Klasse {c}: Output(eigen={eigene_out:.2f}, max_konk={andere_out:.2f}, Marge={marge:+.2f}) | "
            f"stumm Hidden={stumm_hidden:.1f}% | Top-Features={top_hidden.tolist()}{flag}"
        )


def analyse_gewichte_visualisierung(w_in_hidden, w_hidden_out, titel="Gewichte"):
    # Input -> Hidden als 28x28 Muster (erste 25 Neuronen)
    n_show = min(25, n_hidden)
    grid = int(np.ceil(np.sqrt(n_show)))

    fig, axes = plt.subplots(grid, grid, figsize=(10, 10))
    vmax = w_in_hidden.max()
    for h, ax in enumerate(axes.flat):
        if h >= n_show:
            ax.axis("off")
            continue
        ax.imshow(w_in_hidden[h].reshape(28, 28), cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(f"H{h}", fontsize=8)
        ax.axis("off")
    plt.suptitle(f"{titel} | Input->Hidden")
    plt.tight_layout()
    plt.show()

    # Hidden -> Output als Heatmap
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(w_hidden_out, aspect="auto", cmap="viridis")
    ax.set_title(f"{titel} | Hidden->Output")
    ax.set_xlabel("Hidden-Neuron")
    ax.set_ylabel("Output-Klasse")
    ax.set_yticks(range(n_classes))
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


def analyse_accuracy_pro_klasse(images, labels, w_in_hidden, w_hidden_out, start_idx, n_eval, label=""):
    print(f"\n--- Accuracy pro Klasse {label} ---")
    correct = np.zeros(n_classes)
    total = np.zeros(n_classes)
    end_idx = min(start_idx + n_eval, len(images))

    for i in range(start_idx, end_idx):
        y = labels[i]
        pred, _, _ = predict_from_spikes(images[i], w_in_hidden, w_hidden_out)
        total[y] += 1
        if pred == y:
            correct[y] += 1

    accs = correct / (total + 1e-8)
    for c in range(n_classes):
        balken = "█" * int(accs[c] * 20)
        flag = " <- SCHWACH" if accs[c] < 0.5 else ""
        print(f"Klasse {c}: {balken:<20} {accs[c]:.2f}{flag}")
    return accs


def analyse_konfusionsmatrix(images, labels, w_in_hidden, w_hidden_out, start_idx, n_eval, label=""):
    print(f"\n--- Konfusionsmatrix {label} ---")
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    end_idx = min(start_idx + n_eval, len(images))

    for i in range(start_idx, end_idx):
        pred, _, _ = predict_from_spikes(images[i], w_in_hidden, w_hidden_out)
        matrix[labels[i], pred] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Wahr")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            farbe = "white" if matrix[i, j] > matrix.max() * 0.5 else "black"
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8, color=farbe)

    plt.colorbar(im)
    plt.title(f"Konfusionsmatrix {label}")
    plt.tight_layout()
    plt.show()

    print("Groesste Verwechslungen (wahr -> pred: anzahl):")
    fehler = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and matrix[i, j] > 0:
                fehler.append((matrix[i, j], i, j))
    for anzahl, wahr, pred in sorted(fehler, reverse=True)[:5]:
        print(f"  {wahr} -> {pred}: {anzahl}x")


def analyse_einzelbild_detail(image, true_lbl, w_in_hidden, w_hidden_out, idx=None):
    pred, sim, _ = predict_from_spikes(image, w_in_hidden, w_hidden_out)
    hidden_counts = sim["hidden_counts"]
    output_counts = sim["output_counts"]

    top_hidden = np.argsort(hidden_counts)[-5:][::-1]
    print("\n--- Einzelbildanalyse ---")
    if idx is not None:
        print(f"Bildindex: {idx}")
    print(f"True Label: {true_lbl}")
    print(f"Predicted : {pred}")
    print(f"Top Hidden-Neuronen: {top_hidden.tolist()}")
    print(f"Top Hidden-Spikes: {hidden_counts[top_hidden].tolist()}")
    print(f"Output-Spike-Counts: {output_counts.tolist()}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"True: {true_lbl} | Pred: {pred}")
    axes[0].axis("off")

    axes[1].bar(range(n_classes), output_counts)
    axes[1].set_title("Output Spike-Counts")
    axes[1].set_xlabel("Klasse")

    axes[2].bar(range(len(top_hidden)), hidden_counts[top_hidden])
    axes[2].set_xticks(range(len(top_hidden)))
    axes[2].set_xticklabels([f"H{h}" for h in top_hidden])
    axes[2].set_title("Staerkste Hidden-Features")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Vor Training
# -----------------------------
print("=" * 50)
print("VOR TRAINING (2-Schicht-SNN)")
print("=" * 50)

initial_test_acc = evaluate_snn(images, labels, W_in_hidden, W_hidden_out, 60000, test_samples)
print(f"Test-Accuracy vor Training: {initial_test_acc:.3f}")

best_W_in_hidden = W_in_hidden.copy()
best_W_hidden_out = W_hidden_out.copy()
best_val_acc = evaluate_snn(images, labels, W_in_hidden, W_hidden_out, val_start_idx, val_samples)
print(f"Validation-Accuracy vor Training: {best_val_acc:.3f}")

if ANALYSE_SPIKE_AKTIVITAET:
    analyse_spike_aktivitaet(images, labels, W_in_hidden, W_hidden_out, n_samples=200, label="(vor Training)")

if ANALYSE_GEWICHTE_VISUALISIERUNG:
    analyse_gewichte_visualisierung(W_in_hidden, W_hidden_out, titel="Gewichte vor Training")

if ANALYSE_ACCURACY_PRO_KLASSE:
    analyse_accuracy_pro_klasse(images, labels, W_in_hidden, W_hidden_out, 60000, test_samples, label="(vor Training)")

if ANALYSE_KONFUSIONSMATRIX:
    analyse_konfusionsmatrix(images, labels, W_in_hidden, W_hidden_out, 60000, min(500, test_samples), label="(vor Training)")


# -----------------------------
# Training
# -----------------------------
print("\n" + "=" * 50)
print("TRAINING")
print("=" * 50)

for epoch in range(epochs):
    order = rng.permutation(train_samples)
    correct = 0

    updates_hidden = 0
    updates_out = 0
    no_output_spike_cases = 0

    for i_step, i in enumerate(order):
        img = images[i]
        true_lbl = labels[i]

        pred, sim, input_spikes = predict_from_spikes(img, W_in_hidden, W_hidden_out)
        hidden_counts = sim["hidden_counts"]
        output_counts = sim["output_counts"]

        if output_counts.max() == 0:
            no_output_spike_cases += 1

        if pred == true_lbl:
            correct += 1

        # Hidden-STDP immer aktiv (unsupervised Feature-Lernen)
        W_in_hidden, upd_h = stdp_update_hidden(W_in_hidden, input_spikes, sim["hidden_spike_train"])
        updates_hidden += upd_h

        # Output-Lernen nur bei Fehler
        W_hidden_out, upd_o = supervised_update_output(W_hidden_out, hidden_counts, pred, true_lbl)
        updates_out += upd_o

        if (i_step + 1) % ausgabe_intervall == 0:
            running_acc = correct / (i_step + 1)
            print(f"  Epoch {epoch + 1} | {i_step + 1}/{train_samples} | Running Acc: {running_acc:.3f}", end="")

            if ANALYSE_UPDATE_RATE:
                upd_h_rate = updates_hidden / ((i_step + 1) * n_hidden * n_input) * 100
                upd_o_rate = updates_out / ((i_step + 1) * n_hidden) * 100
                stumm_rate = no_output_spike_cases / (i_step + 1) * 100
                stumm_flag = " <- V_TH OUTPUT ZU HOCH?" if stumm_rate > 10 else ""
                print(
                    f" | Hidden-STDP Aktivitaet: {upd_h_rate:.3f}%"
                    f" | Output-Update: {upd_o_rate:.2f}%"
                    f" | Kein Output-Spike: {stumm_rate:.1f}%{stumm_flag}",
                    end="",
                )

            print()

    train_acc = evaluate_snn(images, labels, W_in_hidden, W_hidden_out, 0, min(2000, train_samples))
    val_acc = evaluate_snn(images, labels, W_in_hidden, W_hidden_out, val_start_idx, val_samples)

    print(f"Epoch {epoch + 1}/{epochs} abgeschlossen | Train-Acc: {train_acc:.3f} | Val-Acc: {val_acc:.3f}")

    prev_best = best_val_acc
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_W_in_hidden = W_in_hidden.copy()
        best_W_hidden_out = W_hidden_out.copy()
        print(f"  Neues bestes Modell gespeichert ({prev_best:.4f} -> {best_val_acc:.4f}).")
    else:
        W_in_hidden = best_W_in_hidden.copy()
        W_hidden_out = best_W_hidden_out.copy()
        print(f"  Kein besseres Modell ({val_acc:.4f} <= Bestwert {prev_best:.4f}) -> zurueckgesetzt.")


# Bestes Modell verwenden
W_in_hidden = best_W_in_hidden.copy()
W_hidden_out = best_W_hidden_out.copy()

# -----------------------------
# Nach Training
# -----------------------------
print("\n" + "=" * 50)
print("NACH TRAINING")
print("=" * 50)

final_test_acc = evaluate_snn(images, labels, W_in_hidden, W_hidden_out, 60000, test_samples)
print(f"\nTest-Accuracy nach Training: {final_test_acc:.3f}")

if ANALYSE_SPIKE_AKTIVITAET:
    analyse_spike_aktivitaet(images, labels, W_in_hidden, W_hidden_out, n_samples=200, label="(nach Training)")

if ANALYSE_GEWICHTE_VISUALISIERUNG:
    analyse_gewichte_visualisierung(W_in_hidden, W_hidden_out, titel="Gewichte nach Training")

if ANALYSE_ACCURACY_PRO_KLASSE:
    analyse_accuracy_pro_klasse(images, labels, W_in_hidden, W_hidden_out, 60000, test_samples, label="(nach Training)")

if ANALYSE_KONFUSIONSMATRIX:
    analyse_konfusionsmatrix(images, labels, W_in_hidden, W_hidden_out, 60000, min(500, test_samples), label="(nach Training)")


# -----------------------------
# Einzelbild klassifizieren
# -----------------------------
selected_number = None
if selected_number is None:
    selected_number = int(rng.integers(1, 70001))

idx = selected_number - 1
img = images[idx]
true_lbl = labels[idx]

predicted_label, sim, _ = predict_from_spikes(img, W_in_hidden, W_hidden_out)
out_counts = sim["output_counts"]
rates = out_counts / T

print(f"\nBild #{selected_number}")
print(f"Predicted  : {predicted_label}")
print(f"True       : {true_lbl}")
print(f"Spike-Rates: {np.round(rates, 3)}")

if ANALYSE_EINZELBILD_DETAIL:
    analyse_einzelbild_detail(img, true_lbl, W_in_hidden, W_hidden_out, idx=idx)
