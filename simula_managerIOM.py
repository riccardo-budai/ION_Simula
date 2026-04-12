
import math
import numpy as np
import matplotlib.pyplot as plt

def generate_sfap(t, arrival_time):
    """
    Genera un singolo potenziale d'azione di una fibra (Single Fiber Action Potential - SFAP).

    Questa funzione modella un SFAP come la somma di due curve gaussiane:
    una per la fase positiva (depolarizzazione) e una per la fase negativa
    (ripolarizzazione).

    Args:
        t (float): Il punto temporale corrente in cui calcolare il potenziale (in ms).
        arrival_time (float): Il tempo di arrivo del potenziale al punto di registrazione (in ms).

    Returns:
        float: L'ampiezza del potenziale (in µV) al tempo t.
    """
    amplitude = 1.0  # Ampiezza relativa della fase positiva
    width1 = 0.1  # Larghezza della fase positiva (controlla la durata)
    width2 = 0.2  # Larghezza della fase negativa

    relative_time = t - arrival_time

    # Calcola la fase positiva (più rapida e di ampiezza maggiore)
    positive_phase = -1.0 * amplitude * math.exp(-((relative_time / width1) ** 2))

    # Calcola la fase negativa (più lenta, ritardata e di ampiezza minore)
    negative_phase = 0.5 * amplitude * math.exp(-(((relative_time - 0.15) / width2) ** 2))

    # Il potenziale totale è la somma delle due fasi
    return positive_phase + negative_phase


# --- Esempio di utilizzo ---
if __name__ == "__main__":
    # Parametri per l'esempio
    tempo_di_arrivo = 5.0  # L'impulso arriva a 5 ms

    print("Esempio di calcolo del potenziale SFAP a diversi istanti di tempo:")
    print("-" * 60)

    # Calcoliamo il potenziale in alcuni punti temporali attorno al tempo di arrivo
    for tempo in [4.8, 5.0, 5.2, 5.5]:
        potenziale = generate_sfap(tempo, tempo_di_arrivo)
        print(f"Al tempo t = {tempo:.2f} ms, l'ampiezza del potenziale è {potenziale:.4f} µV")

    # Esempio per generare una piccola forma d'onda
    print("\nGenerazione di una serie di punti per un grafico:")
    print("-" * 60)
    waveform_points = []
    for i in range(400, 601):  # genera punti da t=4.00 a t=6.00
        t = i / 100.0
        potential = generate_sfap(t, tempo_di_arrivo)
        waveform_points.append((t, potential))
        if i % 20 == 0:  # Stampa un punto ogni tanto per brevità
            print(f"t={t:.2f} ms, V={potential:.4f} µV")

    # 2. Converti la lista in un array NumPy per poterla manipolare facilmente
    waveform_points_np = np.array(waveform_points)

    print(f"Dimensione dell'array di punti generato: {np.shape(waveform_points_np)}")

    # 3. Esegui il plot usando lo slicing di NumPy per separare le colonne x (tempo) e y (potenziale)
    plt.figure(figsize=(10, 6))  # Crea una figura di dimensioni adeguate

    # La prima colonna (indice 0) è l'asse x, la seconda (indice 1) è l'asse y
    plt.plot(waveform_points_np[:, 0], waveform_points_np[:, 1], label=f'SFAP (arrivo a {tempo_di_arrivo} ms)',
             color='b')

    # Aggiungi etichette e titolo per chiarezza
    plt.title("Forma d'Onda di un Singolo Potenziale d'Azione (SFAP)")
    plt.xlabel("Tempo (ms)")
    plt.ylabel("Ampiezza Relativa (µV)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.5)  # Linea dello zero
    plt.axvline(tempo_di_arrivo, color='r', linestyle=':',
                label=f'Tempo di arrivo = {tempo_di_arrivo} ms')  # Linea del tempo di arrivo
    plt.legend()
    plt.gca().invert_yaxis()

    # 4. Mostra il grafico
    plt.show()
