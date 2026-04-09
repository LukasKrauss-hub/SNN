import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# -----------------------------
# Allgemeine Parameter
# -----------------------------
T = 30
tau = 6
V_th = 0.15
V_reset = 0.0
n_classes = 10

# STDP Parameter
tau_plus = 20.0  # Zeitkonstante fuer potenzierende Updates, wenn Prae-Spike vor Post-Spike kommt.
tau_minus = 20.0  # Zeitkonstante fuer depressive Updates, wenn Post-Spike vor Prae-Spike kommt.
A_plus = 0.000001  # Grundstaerke der Long-Term Potentiation im STDP-Lernschritt.
A_minus = 0.0000015  # Grundstaerke der Long-Term Depression im STDP-Lernschritt.
w_max = 1.0  # Obergrenze fuer Gewichte, damit sie nicht unkontrolliert wachsen.
w_min = 0.0  # Untergrenze fuer Gewichte, damit keine negativen Werte entstehen.

epochs = 4  # Anzahl der vollstaendigen Trainingsdurchlaeufe ueber die Trainingsbeispiele.
train_samples = 1000  # Anzahl der MNIST-Bilder, die fuer das Training verwendet werden.
test_samples = 5000  # Anzahl der Testbilder aus dem Testbereich des Datensatzes.
val_start_idx = 50000  # Startindex des Validierungsbereichs innerhalb des Gesamtdatensatzes.
val_samples = 1000  # Anzahl der Beispiele, die fuer die Validierung verwendet werden.
ausgabe_intervall = 500  # Anzahl der Trainingsschritte, nach denen ein Zwischenstand ausgegeben wird.

# -----------------------------
# MNIST laden
# -----------------------------
mnist = fetch_openml('mnist_784', version=1, parser='auto')  # Laedt den klassischen MNIST-Datensatz mit 784 Pixeln pro Bild.
data = mnist.data.to_numpy() if hasattr(mnist.data, "to_numpy") else np.asarray(mnist.data)  # Wandelt die Bilddaten robust in ein NumPy-Array um.
target = mnist.target.to_numpy() if hasattr(mnist.target, "to_numpy") else np.asarray(mnist.target)  # Wandelt auch die Zielklassen robust in ein NumPy-Array um.

images = data.reshape(-1, 28, 28) / 255.0  # Formt die flachen 784er-Vektoren zu 28x28-Bildern um und normalisiert Pixel auf den Bereich 0 bis 1.
labels = target.astype(int)  # Konvertiert die Klassenlabels von Strings in Ganzzahlen.

# -----------------------------
# Gewichte initialisieren
# -----------------------------
def init_weights_from_prototypes(images, labels, n_classes, train_size=10000):  # Erzeugt Startgewichte aus mittleren Klassen-Prototypen.
    x_train = images[:train_size].reshape(train_size, -1)  # Nimmt die ersten Trainingsbilder und klappt jedes Bild in einen 784er-Vektor auf.
    y_train = labels[:train_size]  # Nimmt die zugehoerigen Labels fuer genau diese Trainingsbilder.

    class_weights = np.zeros((n_classes, x_train.shape[1]))  # Legt fuer jede Klasse einen Gewichtungsvektor ueber alle Pixel an.
    for c in range(n_classes):  # Durchlaeuft nacheinander alle Klassen von 0 bis 9.
        mask = y_train == c  # Erstellt eine boolesche Maske, die anzeigt, welche Trainingsbeispiele zur aktuellen Klasse gehoeren.
        if mask.any():  # Prueft, ob es im betrachteten Ausschnitt ueberhaupt Beispiele dieser Klasse gibt.
            class_weights[c] = x_train[mask].mean(axis=0)  # Setzt den Gewichtsvektor auf das mittlere Bild dieser Klasse. also durchschnittl Helligkeit ist startgewicht

    norms = np.linalg.norm(class_weights, axis=1, keepdims=True) + 1e-8  # Berechnet die Betrags-Norm jedes Klassenvektors und vermeidet Division durch null.
    return class_weights / norms  # Gibt normierte Startgewichte zurueck, damit alle Klassen vergleichbare Skalen haben. Helle Klassen mit vielen Hellen Pixeln zb 8 würden sonst immer gegen zb 1 gewinnen, da sie mehr Strom liefern wuerden. Normalisierung sorgt fuer faire Startbedingungen. Jede Klasse bekommt glecihe "Gesamtpunktzahl"

weights = init_weights_from_prototypes(images, labels, n_classes)  # Initialisiert die Gewichte einmal aus den berechneten Klassen-Prototypen.

# -----------------------------
# Bild -> Spike-Train
# -----------------------------
def image_to_spike_train(image, T):  # Wandelt ein Graustufenbild in eine zeitliche Folge binaerer Eingangsspikes um.
    flat = image.flatten()  # Klappt das 28x28-Bild zu einem eindimensionalen Pixelvektor auf.
    return (np.random.rand(T, len(flat)) < (flat **2)).astype(np.float32)  # Erzeugt fuer jeden Zeitschritt und jedes Pixel stochastische Spikes mit intensitaetsabhaengiger Wahrscheinlichkeit.

# -----------------------------
# LIF-Simulation
# -----------------------------
def lif_simulate(input_spikes, weights):  # Simuliert die Antwort der Output-Neuronen auf die gegebene Spike-Folge.
    spike_times = [[] for _ in range(n_classes)]  # Legt fuer jede Klasse eine Liste an, in der die Ausgabespike-Zeitpunkte gesammelt werden.
    V = np.zeros(n_classes)  # Initialisiert das Membranpotenzial aller Output-Neuronen mit null.

    for t in range(T):  # Durchlaeuft nacheinander alle diskreten Zeitschritte der Simulation.
        I = (weights @ input_spikes[t]) / 8.0  # Berechnet den Eingangsstrom als gewichtete Summe der aktiven Eingangsspikes.
        V += (-V + I) / tau  # Aktualisiert das Membranpotenzial nach der LIF-Dynamik mit Leck und Eingangsstrom.

        fired = V >= V_th  # Bestimmt, welche Neuronen im aktuellen Zeitschritt den Schwellwert erreicht haben. Fired ist auch Vrektor mit True/False für jedes Neuron.
        for c in np.where(fired)[0]:  # Durchlaeuft nur die Indizes der Neuronen, die wirklich gefeuert haben. c ist der Index der Klasse, die in diesem Zeitschritt feuert.
            spike_times[c].append(t)  # Speichert den aktuellen Zeitpunkt als Spike dieser Klasse.
            V[c] = V_reset  # Setzt das Potenzial des feuernden Neurons direkt nach dem Spike zurueck.

    return spike_times  # Gibt fuer jede Klasse die Liste der beobachteten Spike-Zeiten zurueck.

def lif_simulate_winner_takes_it_all(input_spikes, weights):
    spike_times = [[] for _ in range(n_classes)]
    V = np.zeros(n_classes)

    for t in range(T):
        I = (weights @ input_spikes[t]) / 8.0
        V += (-V + I) / tau

        fired = V >= V_th
        if fired.any():
            winner = np.argmax(V)
            # alle anderen Neuronen werden zurückgesetzt
            for c in np.where(fired)[0]:
                if c != winner:
                    V[c] = V_reset
                    fired[c] = False
            # nur winner feuert wirklich
            spike_times[winner].append(t)
            V[winner] = V_reset

    return spike_times
# -----------------------------
# Vorhersage über Spike-Aktivität
# -----------------------------
def predict_from_spikes(image, weights, T):  # Fuehrt fuer ein Bild die Spike-Kodierung, Simulation und Klassenvorhersage aus.
    input_spikes = image_to_spike_train(image, T)  # Kodiert das Bild in eine stochastische zeitliche Spike-Matrix.
#    spike_times = lif_simulate(input_spikes, weights)  # gibt mir Output Spikes zurück, also wann welches Output-Neuron gefeuert hat
    spike_times = lif_simulate_winner_takes_it_all(input_spikes, weights)  # gibt mir Output Spikes zurück, also wann welches Output-Neuron gefeuert hat
    spike_counts = np.array([len(st) for st in spike_times])  # Zaehlt, wie viele Spikes jedes Output-Neuron insgesamt erzeugt hat.

    if spike_counts.max() == 0:  # Prueft den Spezialfall, dass kein einziges Output-Neuron gespiket hat.
        pred = np.argmax(weights @ image.flatten())  # Faellt in diesem Fall auf eine direkte lineare Aehnlichkeitsentscheidung zurueck. Also Prototyp ohne lernen
    else:  # Behandelt den Normalfall, dass wenigstens ein Spike erzeugt wurde.
        pred = np.argmax(spike_counts)  # Waehlt die Klasse mit der hoechsten Spike-Anzahl als Vorhersage. bei gleichsstand wird erste ausgewählt.

    return pred, spike_times, spike_counts, input_spikes  # Gibt Vorhersage und Zwischenresultate fuer Analyse und Lernen zurueck.

# -----------------------------
# STDP-Update
# -----------------------------
def stdp_update(weights, input_spikes, spike_times, true_label, active_classes):  # Passt Gewichte mit einer vereinfachten STDP-Regel an.
    pre_spike_times = [  # Sammelt fuer jedes Pixel alle Zeitpunkte, an denen dort ein Pre-Spike auftrat.
        np.where(input_spikes[:, px] > 0)[0]  # Liest fuer ein Pixel alle Zeitschritte aus, in denen der Eingang aktiv war. np.where([False, True, False, True]) -> gibt [1, 3] zurück, also die Zeitschritte, in denen das Pixel gespikelt hat. Das wird fuer alle Pixel gemacht, um spaeter die zeitlichen Abstaende zu den Post-Spikes zu berechnen.
        for px in range(input_spikes.shape[1])  # Wiederholt das fuer alle Eingabedimensionen beziehungsweise Pixel. [1] bedeutet ich nimm aus input spikes pixel also in range 784
    ]

    for c in active_classes:  # Aktualisiert nur relevante Klassen, typischerweise die wahre und die vorhergesagte Klasse.
        if not spike_times[c]:  # Ueberspringt Klassen, die gar keinen Output-Spike produziert haben.
            continue  # Geht direkt zur naechsten aktiven Klasse weiter.

        if c == true_label:  # Prueft, ob die betrachtete Klasse die korrekte Zielklasse ist.
            ltp_scale = 1.2  # Verstaerkt fuer die Zielklasse die Potenzierung etwas staerker.
            ltd_scale = 0.8  # Schwaecht fuer die Zielklasse die Depression etwas ab.
        else:  # Behandelt die faelschlich aktive Konkurrenzklasse.
            ltp_scale = 0.8  # Verkleinert dort die Potenzierung, damit falsche Klassen weniger profitieren.
            ltd_scale = 1.2  # Verstaerkt dort die Depression, damit falsche Klassen eher bestraft werden.

        dW = np.zeros(weights.shape[1])  # Legt einen Delta-Vektor fuer alle Gewichte der aktuellen Klasse an. also 784 0er

        for t_post in spike_times[c]:  # Durchlaeuft alle Post-Spike-Zeitpunkte des aktuellen Output-Neurons.
            for px in range(weights.shape[1]):  # Durchlaeuft alle Eingabepixel beziehungsweise Synapsen.
                t_pres = pre_spike_times[px]  # Holt die Pre-Spike-Zeitpunkte des aktuellen Pixels. Ich hol einfach nur den Eintrag an der Stelle
                if len(t_pres) == 0:  # Prueft, ob dieses Pixel im gesamten Zeitfenster jemals aktiv war.
                    continue  # Ueberspringt ungenutzte Synapsen ohne weiteren Rechenaufwand.

                for t_pre in t_pres:  # Vergleicht jeden Prae-Spike dieses Pixels mit dem aktuellen Post-Spike.
                    delta_t = t_post - t_pre  # Berechnet den zeitlichen Abstand zwischen Post- und Prae-Spike.

                    if delta_t > 0:  # Fall: Prae-Spike kam vor dem Post-Spike.
                        dW[px] += ltp_scale * A_plus * np.exp(-delta_t / tau_plus)  # Erhoeht das Gewicht nach einer exponentiell abklingenden LTP-Regel.
                    elif delta_t < 0:  # Fall: Prae-Spike kam nach dem Post-Spike.
                        dW[px] -= ltd_scale * A_minus * np.exp(delta_t / tau_minus)  # Verringert das Gewicht nach einer exponentiell abklingenden LTD-Regel.

        dW = np.clip(dW, -0.01, 0.01)  # Begrenzt das Gewichtsupdate pro Synapse, um extreme Aenderungen zu vermeiden.
        weights[c] += dW  # Addiert das berechnete Update auf den Gewichtsvektor der aktuellen Klasse.

    np.clip(weights, w_min, w_max, out=weights)  # Schneidet alle Gewichte global auf den erlaubten Bereich zwischen Minimum und Maximum zu.

    norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-8  # Berechnet erneut die Norm jedes Klassenvektors fuer die spaetere Normalisierung. +e-8 verhindert Division durch Null, falls ein Vektor komplett auf null gesetzt wurde.
    weights /= norms  # Normiert alle Gewichtsvektoren, damit ihre Groessenordnung stabil bleibt.

    return weights  # Gibt die aktualisierten und normalisierten Gewichte zurueck.

# -----------------------------
# Spike-basierte Evaluation
# -----------------------------
def evaluate_snn(images, labels, weights, start_idx, n_eval):  # Misst die Klassifikationsgenauigkeit auf einem zusammenhaengenden Datensatzabschnitt.
    end_idx = min(start_idx + n_eval, len(images))  # Bestimmt das Ende des Auswertebereichs, ohne ueber die Datensatzgrenze hinauszugehen.
    correct = 0  # Zaehlt, wie viele Vorhersagen im betrachteten Bereich korrekt sind.

    for i in range(start_idx, end_idx):  # Durchlaeuft alle Beispiele des gewuenschten Evaluationsbereichs.
        pred, _, _, _ = predict_from_spikes(images[i], weights, T)  # Sagt fuer das aktuelle Bild die Klasse voraus und ignoriert unnoetige Zusatzwerte.
        if pred == labels[i]:  # Prueft, ob die Vorhersage dem echten Label entspricht.
            correct += 1  # Erhoeht den Zaehler fuer korrekte Klassifikationen.

    return correct / (end_idx - start_idx)  # Gibt die Accuracy als Anteil korrekter Vorhersagen zurueck.

# -----------------------------
# Vor Training
# -----------------------------
initial_test_acc = evaluate_snn(images, labels, weights, 60000, test_samples)  # Bewertet das untrainierte Modell auf dem Testbereich.
print(f"Test-Accuracy vor Training: {initial_test_acc:.3f}")  # Gibt die Testgenauigkeit vor jedem Lernschritt auf der Konsole aus.

best_weights = weights.copy()  # Prototypengewichte
best_val_acc = evaluate_snn(images, labels, weights, val_start_idx, val_samples)  # Misst die Validierungsgenauigkeit des Startmodells.

# -----------------------------
# Training
# -----------------------------
for epoch in range(epochs):  # Startet die äußere Trainingsschleife über mehrere Epochen.
    order = np.random.permutation(train_samples)  # Erzeugt fuer diese Epoche eine zufaellige Reihenfolge der Trainingsbeispiele.
    correct = 0  # Zaehlt korrekte Vorhersagen innerhalb der aktuellen Epoche.

    for i_step, i in enumerate(order):  # Durchlaeuft alle Trainingsindizes zusammen mit dem aktuellen Schrittzaehler. i step einfach mien zähler
        img = images[i]  # Holt das aktuelle Trainingsbild aus dem Datensatz.
        true_lbl = labels[i]  # Holt das dazugehoerige korrekte Klassenlabel.

        pred, spike_times, spike_counts, input_spikes = predict_from_spikes(img, weights, T)  # Berechnet Vorhersage und alle fuer STDP noetigen Zwischenwerte.

        if pred == true_lbl:  # Prueft, ob das Modell dieses Trainingsbeispiel bereits korrekt klassifiziert hat.
            correct += 1  # Erhoeht in diesem Fall nur den Zaehler fuer die laufende Trainingsgenauigkeit.
        else:  # Behandelt den Fall einer Fehlklassifikation.
            active_classes = np.unique([pred, true_lbl])  # Bestimmt die Klassen, deren Gewichte angepasst werden sollen. unique entfernt doppelte einträge
            weights = stdp_update(weights, input_spikes, spike_times, true_lbl, active_classes)  # Fuehrt das STDP-Lernupdate fuer diese Klassen aus.

        if (i_step + 1) % ausgabe_intervall == 0:  # Prueft, ob ein Zwischenstand nach jeweils ... Trainingsschritten ausgegeben werden soll.
            running_acc = correct / (i_step + 1)  # Berechnet die bisherige Trefferquote innerhalb der aktuellen Epoche.
            print(f"  Epoch {epoch+1} | {i_step+1}/{train_samples} | Running Acc: {running_acc:.3f}")  # Druckt den Trainingsfortschritt mit laufender Accuracy.

    train_acc = evaluate_snn(images, labels, weights, 0, min(2000, train_samples))  # Bewertet das Modell am Ende der Epoche auf einem Trainingsausschnitt. für val max 2000
    val_acc = evaluate_snn(images, labels, weights, val_start_idx, val_samples)  # Bewertet das Modell am Ende der Epoche auf dem Validierungsbereich.

    print(f"Epoch {epoch+1}/{epochs} abgeschlossen | Train-Acc: {train_acc:.3f} | Val-Acc: {val_acc:.3f}")  # Gibt die zusammengefassten Metriken dieser Epoche aus.

    if val_acc > best_val_acc:  # Prueft, ob das aktuelle Modell besser validiert als alle bisherigen Modelle.
        best_val_acc = val_acc  # Aktualisiert den bisher besten Validierungswert.
        best_weights = weights.copy()  # Speichert die aktuellen Gewichte als neues bestes Modell.
        print("  Neues bestes Modell gespeichert.")  # Meldet, dass ein neuer Bestwert gefunden wurde.
    else:  # Behandelt den Fall, dass sich das Modell auf der Validierung nicht verbessert hat.
        weights = best_weights.copy()  # Setzt die Gewichte auf das bisher beste Modell zurueck.
        print("  Kein besseres Modell -> auf bestes Modell zurueckgesetzt.")  # Meldet das Zuruecksetzen auf den besten bekannten Stand.

# Bestes Modell verwenden
weights = best_weights.copy()  # Stellt sicher, dass fuer den Abschlusstest wirklich das beste Validierungsmodell genutzt wird.

final_test_acc = evaluate_snn(images, labels, weights, 60000, test_samples)  # Misst die finale Testgenauigkeit nach dem STDP-Training.
print(f"\nTest-Accuracy nach STDP-Training: {final_test_acc:.3f}")  # Gibt die finale Testgenauigkeit formatiert aus.

# -----------------------------
# Einzelbild klassifizieren
# -----------------------------
selected_number = None  # Erlaubt optional die manuelle Auswahl einer Bildnummer; None bedeutet zufaellige Auswahl.
if selected_number is None:  # Prueft, ob keine feste Bildnummer vorgegeben wurde.
    selected_number = np.random.randint(1, 70001)  # Waehlt dann zufaellig eine Bildnummer zwischen 1 und 70000.

idx = selected_number - 1  # Wandelt die 1-basierte Bildnummer in einen 0-basierten Array-Index um.
img = images[idx]  # Holt das ausgewaehlte Bild aus dem Datensatz.
true_lbl = labels[idx]  # Holt das korrekte Label dieses ausgewaehlten Bildes.

predicted_label, spike_times, spike_counts, input_spikes = predict_from_spikes(img, weights, T)  # Klassifiziert das ausgewaehlte Bild mit dem trainierten Modell.

if spike_counts.max() == 0:  # Prueft erneut den Sonderfall ohne Output-Spikes.
    rates = weights @ img.flatten()  # Berechnet dann stattdessen direkte Aktivierungswerte ueber die Gewichte.
    rates = (rates - rates.min()) / (np.ptp(rates) + 1e-12)  # Normalisiert diese Aktivierungen auf einen gut vergleichbaren Bereich.
else:  # Behandelt den Normalfall mit vorhandenen Spikes.
    rates = spike_counts / T  # Berechnet die Spike-Rate jeder Klasse als Spikes pro Zeitschritt.

print(f"\nBild #{selected_number}")  # Gibt aus, welches Bild aus dem Datensatz betrachtet wird.
print(f"Predicted : {predicted_label}")  # Gibt die vom Modell vorhergesagte Klasse aus.
print(f"True      : {true_lbl}")  # Gibt das tatsaechliche Klassenlabel aus.
print(f"Spike-Rates: {np.round(rates, 3)}")  # Gibt die Aktivitaet aller Output-Klassen gerundet aus.

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Erstellt eine Abbildung mit zwei Teilplots nebeneinander.

axes[0].imshow(img, cmap="gray")  # Zeigt links das ausgewaehlte MNIST-Bild in Graustufen an.
axes[0].set_title(f"True: {true_lbl} | Predicted: {predicted_label}")  # Setzt einen Titel mit echter und vorhergesagter Klasse.
axes[0].axis("off")  # Blendet fuer das Bild die Achsenbeschriftungen aus.

axes[1].bar(range(10), rates)  # Zeichnet rechts ein Balkendiagramm der Aktivitaet fuer alle zehn Klassen.
axes[1].set_xlabel("Klasse")  # Beschriftet die x-Achse mit den Klassenindizes.
axes[1].set_ylabel("Spike-Rate")  # Beschriftet die y-Achse mit der gemessenen Aktivitaet.
axes[1].set_title("Output neuron activity (STDP)")  # Setzt einen Titel fuer das Aktivitaetsdiagramm.

plt.tight_layout()  # Optimiert automatisch die Abstaende, damit sich Beschriftungen nicht ueberlappen.
plt.show()  # Zeigt die fertig aufgebaute Grafik auf dem Bildschirm an.







# ============================================================
# DATENFORMEN & BEISPIELE (Quick Reference)
# ============================================================

# images: (70000, 28, 28)
# Beispiel:
# images[0] =
# [[0.0, 0.1, ..., 0.0],
#  ...
#  [0.0, 0.0, ..., 0.2]]
# → 70000 Bilder, jedes 28x28

# labels: (70000,)
# Beispiel:
# [5, 0, 4, 1, 9, ...]
# → Klassenlabels (0–9)

# ------------------------------------------------------------

# weights: (10, 784)
# Beispiel:
# weights[0] = [0.02, 0.1, 0.0, ..., 0.3]
# weights[1] = [0.5, 0.01, 0.2, ..., 0.0]
# → 10 Klassen, je ein 784er Vektor (Prototyp)

# class_weights: (10, 784)
# → wie weights, aber vor Normierung

# norms: (10, 1)
# Beispiel:
# [[5.3],
#  [4.8],
#  ...
#  [6.1]]
# → Länge jedes Klassenvektors

# ------------------------------------------------------------

# image.flatten(): (784,)
# Beispiel:
# [0.0, 0.1, 0.2, ..., 0.0]
# → Bild als Vektor

# input_spikes: (T, 784)
# Beispiel (T=3):
# [[0,1,0,...],
#  [1,0,0,...],
#  [0,1,1,...]]
# → Zeit x Pixel (0/1 Spikes)

# input_spikes[t]: (784,)
# → Spike-Vektor zu Zeitpunkt t

# ------------------------------------------------------------

# V: (10,)
# Beispiel:
# [0.0, 0.12, 0.05, ..., 0.2]
# → Membranpotenzial pro Klasse

# I: (10,)
# Beispiel:
# [0.03, 0.15, 0.07, ..., 0.01]
# → Inputstrom pro Klasse

# fired: (10,)
# Beispiel:
# [False, True, False, ..., False]
# → welche Neuronen feuern

# ------------------------------------------------------------

# spike_times: Liste mit 10 Listen
# Beispiel:
# [
#   [2, 5],     # Klasse 0 feuert bei t=2,5
#   [1],        # Klasse 1 feuert bei t=1
#   [],
#   ...
# ]
# → Output-Spikezeiten pro Klasse

# spike_counts: (10,)
# Beispiel:
# [2, 1, 0, ..., 3]
# → Anzahl Spikes pro Klasse

# ------------------------------------------------------------

# pre_spike_times: Liste mit 784 Arrays
# Beispiel:
# [
#   [1,3],      # Pixel 0 hat bei t=1,3 gespikt
#   [0],        # Pixel 1 hat bei t=0 gespikt
#   [2,4,5],    # Pixel 2 hat bei t=2,4,5 gespikt
#   [],
#   ...
# ]
# → Input-Spikezeiten pro Pixel

# ------------------------------------------------------------

# dW: (784,)
# Beispiel:
# [0.001, -0.0005, 0.0, ..., 0.002]
# → Gewichtsänderung für eine Klasse

# active_classes: (2,)
# Beispiel:
# [3, 7]
# → vorhergesagte + echte Klasse

# ------------------------------------------------------------

# scores = weights @ image.flatten(): (10,)
# Beispiel:
# [1.2, 0.5, 3.7, ..., 0.9]
# → Ähnlichkeit zu jeder Klasse

# pred: int
# Beispiel:
# 2
# → vorhergesagte Klasse

# ------------------------------------------------------------

# rates: (10,)
# Beispiel:
# [0.1, 0.0, 0.3, ..., 0.05]
# → Spike-Rate pro Klasse

# ============================================================