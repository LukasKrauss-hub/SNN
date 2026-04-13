import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

start_time = time.time()


def show_all_plots_at_end():
    backend = plt.get_backend().lower()
    if plt.get_fignums() and "agg" not in backend:
        print("\nZeige alle erzeugten Plots am Ende...")
        plt.show()


# ==============================
# Analyse-Tools ein/aus
# ==============================
ANALYSE_GEWICHTE_VISUALISIERUNG = True
ANALYSE_ACCURACY_PRO_KLASSE = True
ANALYSE_KONFUSIONSMATRIX = True
ANALYSE_LOSS_KURVE = True

# ==============================
# Architektur / Training
# ==============================
n_input = 784
n_hidden = 400
n_classes = 10

epochs = 15
batch_size = 250
train_samples = 10000
val_start_idx = 50000
val_samples = 500
test_start_idx = 60000
test_samples = 2000
ausgabe_intervall = 10

learning_rate = 1e-3
alpha = 1e-4
random_seed = 42


def iterate_minibatches(features, targets, batch_size, rng):
    order = rng.permutation(len(features))
    for start in range(0, len(features), batch_size):
        batch_idx = order[start:start + batch_size]
        yield features[batch_idx], targets[batch_idx]


def evaluate_model(model, features, targets):
    predictions = model.predict(features)
    return np.mean(predictions == targets), predictions


def analyse_gewichte_hidden(model, titel="Hidden-Gewichte"):
    weights_input_hidden = model.coefs_[0]
    n_show = min(40, weights_input_hidden.shape[1])
    fig, axes = plt.subplots(4, 10, figsize=(14, 6))
    vmax = np.max(np.abs(weights_input_hidden))

    for hidden_index in range(40):
        ax = axes[hidden_index // 10][hidden_index % 10]
        if hidden_index < n_show:
            ax.imshow(
                weights_input_hidden[:, hidden_index].reshape(28, 28),
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
            )
        ax.axis("off")

    plt.suptitle(f"{titel} (erste {n_show} von {n_hidden})")
    plt.tight_layout()


def analyse_gewichte_output(model, titel="Output-Gewichte"):
    weights_hidden_output = model.coefs_[1]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for class_index in range(n_classes):
        ax = axes[class_index // 5][class_index % 5]
        ax.bar(range(n_hidden), weights_hidden_output[:, class_index], width=1.0, color="steelblue")
        ax.set_title(f"Klasse {class_index}")
        ax.axis("off")

    plt.suptitle(titel)
    plt.tight_layout()


def analyse_accuracy_pro_klasse(true_labels, predicted_labels, label=""):
    print(f"\n--- Accuracy pro Klasse {label} ---")
    for class_index in range(n_classes):
        class_mask = true_labels == class_index
        class_accuracy = np.mean(predicted_labels[class_mask] == true_labels[class_mask])
        bar = "█" * int(class_accuracy * 20)
        flag = " <- SCHWACH" if class_accuracy < 0.7 else ""
        print(f"Klasse {class_index}: {bar:<20} {class_accuracy:.2f}{flag}")


def analyse_konfusionsmatrix(true_labels, predicted_labels, label=""):
    print(f"\n--- Konfusionsmatrix {label} ---")
    matrix = confusion_matrix(true_labels, predicted_labels, labels=np.arange(n_classes))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Wahr")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))

    for row_index in range(n_classes):
        for col_index in range(n_classes):
            color = "white" if matrix[row_index, col_index] > matrix.max() * 0.5 else "black"
            ax.text(col_index, row_index, matrix[row_index, col_index], ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im)
    plt.title(f"Konfusionsmatrix {label}")
    plt.tight_layout()

    print("Groesste Verwechslungen:")
    confusions = [
        (matrix[row_index, col_index], row_index, col_index)
        for row_index in range(n_classes)
        for col_index in range(n_classes)
        if row_index != col_index and matrix[row_index, col_index] > 0
    ]
    for count, true_class, predicted_class in sorted(confusions, reverse=True)[:5]:
        print(f"  {true_class} -> {predicted_class}: {count}x")


def plot_loss_curve(loss_history, val_history):
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(loss_history, color="firebrick", marker="o", label="Train-Loss")
    ax1.set_xlabel("Epoche")
    ax1.set_ylabel("Loss")
    ax1.set_title("Trainingsverlauf")

    ax2 = ax1.twinx()
    ax2.plot(val_history, color="darkgreen", marker="s", label="Val-Accuracy")
    ax2.set_ylabel("Validation-Accuracy")

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="center right")
    plt.tight_layout()


# ==============================
# MNIST laden
# ==============================
mnist = fetch_openml("mnist_784", version=1, parser="auto")
data = mnist.data.to_numpy() if hasattr(mnist.data, "to_numpy") else np.asarray(mnist.data)
target = mnist.target.to_numpy() if hasattr(mnist.target, "to_numpy") else np.asarray(mnist.target)

features = data.astype(np.float32) / 255.0
images = features.reshape(-1, 28, 28)
labels = target.astype(int)

X_train = features[:train_samples]
y_train = labels[:train_samples]
X_val = features[val_start_idx:val_start_idx + val_samples]
y_val = labels[val_start_idx:val_start_idx + val_samples]
X_test = features[test_start_idx:test_start_idx + test_samples]
y_test = labels[test_start_idx:test_start_idx + test_samples]

print("=" * 55)
print("VOR TRAINING")
print("=" * 55)
print(f"Train-Samples: {len(X_train)} | Val-Samples: {len(X_val)} | Test-Samples: {len(X_test)}")
print("Zwei-Schicht-Netz: 784 -> 400 -> 10")
print("Ueberwachtes Training statt reinem STDP, damit Accuracy im Bereich 0.9+ erreichbar wird.")

model = MLPClassifier(
    hidden_layer_sizes=(n_hidden,),
    activation="relu",
    solver="adam",
    alpha=alpha,
    batch_size=batch_size,
    learning_rate_init=learning_rate,
    max_iter=1,
    shuffle=False,
    random_state=random_seed,
)

classes = np.arange(n_classes)
rng = np.random.default_rng(random_seed)
best_model_state = None
best_val_acc = -np.inf
loss_history = []
val_history = []

print("\n" + "=" * 55)
print(f"TRAINING ({epochs} Epochen)")
print("=" * 55)

for epoch_index in range(epochs):
    batch_losses = []
    batch_counter = 0

    for batch_features, batch_labels in iterate_minibatches(X_train, y_train, batch_size, rng):
        if best_model_state is None and batch_counter == 0 and epoch_index == 0:
            model.partial_fit(batch_features, batch_labels, classes=classes)
        else:
            model.partial_fit(batch_features, batch_labels)

        batch_losses.append(model.loss_)
        batch_counter += 1

        if batch_counter % ausgabe_intervall == 0:
            print(
                f"  Epoch {epoch_index + 1} | Batch {batch_counter}"
                f" | Loss: {model.loss_:.4f}"
            )

    train_acc, _ = evaluate_model(model, X_train, y_train)
    val_acc, _ = evaluate_model(model, X_val, y_val)
    epoch_loss = float(np.mean(batch_losses))

    loss_history.append(epoch_loss)
    val_history.append(val_acc)

    print(
        f"Epoch {epoch_index + 1}/{epochs}"
        f" | Train: {train_acc:.3f}"
        f" | Val: {val_acc:.3f}"
        f" | Loss: {epoch_loss:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {
            "coefs": [coef.copy() for coef in model.coefs_],
            "intercepts": [intercept.copy() for intercept in model.intercepts_],
            "n_layers": model.n_layers_,
            "n_outputs": model.n_outputs_,
            "out_activation": model.out_activation_,
            "classes": model.classes_.copy(),
            "loss": model.loss_,
        }
        print(f"  Neues bestes Modell gespeichert ({best_val_acc:.3f}).")

if best_model_state is not None:
    model.coefs_ = [coef.copy() for coef in best_model_state["coefs"]]
    model.intercepts_ = [intercept.copy() for intercept in best_model_state["intercepts"]]
    model.n_layers_ = best_model_state["n_layers"]
    model.n_outputs_ = best_model_state["n_outputs"]
    model.out_activation_ = best_model_state["out_activation"]
    model.classes_ = best_model_state["classes"].copy()
    model.loss_ = best_model_state["loss"]

print(f"\nVal-Verlauf: {[f'{value:.3f}' for value in val_history]}")
print(f"Bestes Val:  {best_val_acc:.3f}")

print("\n" + "=" * 55)
print("NACH TRAINING")
print("=" * 55)

test_acc, test_predictions = evaluate_model(model, X_test, y_test)
print(f"\nTest-Accuracy: {test_acc:.3f}")

if ANALYSE_GEWICHTE_VISUALISIERUNG:
    analyse_gewichte_hidden(model, titel="Hidden-Gewichte nach Training")
    analyse_gewichte_output(model, titel="Output-Gewichte nach Training")
if ANALYSE_ACCURACY_PRO_KLASSE:
    analyse_accuracy_pro_klasse(y_test, test_predictions, label="(nach Training)")
if ANALYSE_KONFUSIONSMATRIX:
    analyse_konfusionsmatrix(y_test, test_predictions, label="(nach Training)")
if ANALYSE_LOSS_KURVE:
    plot_loss_curve(loss_history, val_history)

selected_number = np.random.randint(1, len(images) + 1)
sample_index = selected_number - 1
sample_image = images[sample_index]
sample_true_label = labels[sample_index]
sample_probabilities = model.predict_proba(features[sample_index:sample_index + 1])[0]
sample_prediction = int(np.argmax(sample_probabilities))

print(f"\nBild #{selected_number} | True: {sample_true_label} | Pred: {sample_prediction}")
print(f"Klassenwahrscheinlichkeiten: {np.round(sample_probabilities, 3)}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(sample_image, cmap="gray")
axes[0].set_title(f"True: {sample_true_label} | Predicted: {sample_prediction}")
axes[0].axis("off")
axes[1].bar(range(n_classes), sample_probabilities)
axes[1].set_xlabel("Klasse")
axes[1].set_ylabel("Wahrscheinlichkeit")
axes[1].set_title("Output-Aktivitaet (2-Layer Netz)")
plt.tight_layout()

show_all_plots_at_end()

elapsed = time.time() - start_time
hours, remainder = divmod(int(elapsed), 3600)
minutes, seconds = divmod(remainder, 60)
print(f"\nGesamtlaufzeit: {hours:02d}:{minutes:02d}:{seconds:02d}")
