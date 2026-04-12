import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def create_dummy_csv_files():
    """ Crea due file CSV finti per testare lo script """
    print("Creazione file di test...")
    # File 1: Tempi "ideali"
    triggers = [1, 2, 3] * 10  # 30 eventi totali
    times_1 = np.linspace(1.0, 30.0, 30)  # Un evento ogni secondo

    df1 = pd.DataFrame({'Timestamp': times_1, 'Marker': triggers})
    df1.to_csv('test_file_A.csv', index=False)

    # File 2: Tempi con ritardo (offset) e jitter (rumore)
    # Aggiungiamo 50ms di ritardo fisso + rumore casuale (jitter)
    offset = 0.050
    jitter_noise = np.random.normal(0, 0.005, 30)  # 5ms di deviazione standard
    times_2 = times_1 + offset + jitter_noise

    df2 = pd.DataFrame({'Time_sec': times_2, 'Trigger_Val': triggers})
    df2.to_csv('test_file_B.csv', index=False)
    print("File 'test_file_A.csv' e 'test_file_B.csv' creati.")


def analyze_jitter(file_path_a, file_path_b, col_map_a, col_map_b, trigger_values=[1.0, 2.0, 3.0]):
    """
    Carica due file e calcola il jitter per specifici trigger.

    col_map: dizionario {'time': 'nome_colonna_tempo', 'trig': 'nome_colonna_trigger'}
    """
    try:
        # 1. Caricamento
        df_a = pd.read_csv(file_path_a)
        df_b = pd.read_csv(file_path_b)

        print(f"\n--- Analisi Jitter: {file_path_a} vs {file_path_b} ---")

        results = []

        # 2. Iterazione per ogni tipo di trigger
        for trig_val in trigger_values:
            # Estrazione tempi per il trigger specifico
            times_a = df_a[df_a[col_map_a['trig']] == trig_val][col_map_a['time']].values
            times_b = df_b[df_b[col_map_b['trig']] == trig_val][col_map_b['time']].values

            # Controllo numero eventi
            len_a = len(times_a)
            len_b = len(times_b)

            if len_a == 0 and len_b == 0:
                print(f"Trigger {trig_val}: Nessun evento trovato.")
                continue

            # Allineamento alla lunghezza minima (se un file ha perso dei trigger)
            min_len = min(len_a, len_b)
            if len_a != len_b:
                print(
                    f"ATTENZIONE Trigger {trig_val}: Discrepanza numero eventi (File A: {len_a}, File B: {len_b}). Analizzo i primi {min_len}.")

            t_a = times_a[:min_len]
            t_b = times_b[:min_len]

            # 3. Calcolo Delta (Differenze)
            # Delta = Tempo B - Tempo A
            deltas = t_a - t_b

            # Statistiche
            mean_latency = np.mean(deltas)
            jitter_std = np.std(deltas)
            max_diff = np.max(np.abs(deltas))

            results.append({
                'Trigger': trig_val,
                'Count': min_len,
                'Mean Latency (s)': round(mean_latency, 6),
                'Jitter (StdDev) (s)': round(jitter_std, 6),
                'Max Abs Diff (s)': round(max_diff, 6),
                'Deltas': deltas
            })

        # 4. Creazione Tabella Risultati
        if results:
            res_df = pd.DataFrame(results).drop(columns=['Deltas'])
            print("\nRISULTATI ANALISI:")
            print(res_df.to_string(index=False))

            # 5. Visualizzazione Grafica
            plot_jitter(results)
        else:
            print("Nessun risultato da mostrare.")

    except Exception as e:
        print(f"Errore durante l'analisi: {e}")


def plot_jitter(results):
    """ Plotta istogrammi e scatter plot dei ritardi """
    fig, axes = plt.subplots(len(results), 2, figsize=(12, 4 * len(results)))
    if len(results) == 1: axes = np.array([axes])  # Gestione caso singolo trigger

    fig.suptitle('Analisi Sincronizzazione Trigger (File Event - File Record)')

    for i, res in enumerate(results):
        deltas = res['Deltas']
        trig = res['Trigger']

        # Istogramma (Distribuzione del Jitter)
        ax_hist = axes[i][0]
        ax_hist.hist(deltas * 1000, bins=20, color='skyblue', edgecolor='black')  # Convertito in ms
        ax_hist.set_title(f'Istogramma Ritardi Trigger {trig}')
        ax_hist.set_xlabel('Ritardo (ms)')
        ax_hist.set_ylabel('Conteggio')

        # Scatter (Stabilità nel tempo)
        ax_scat = axes[i][1]
        ax_scat.plot(deltas * 1000, 'o-', markersize=4, color='salmon')
        ax_scat.set_title(f'Stabilità Temporale Trigger {trig}')
        ax_scat.set_xlabel('Numero Evento')
        ax_scat.set_ylabel('Ritardo (ms)')
        ax_scat.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# --- ESECUZIONE ---
if __name__ == "__main__":
    # 1. Genera file finti (commenta questa riga se hai già i tuoi file)
    # create_dummy_csv_files()

    # 2. Configurazione Nomi Colonne (ADATTARE AI TUOI FILE)
    # Esempio: Se nel file A la colonna tempo si chiama 'Time' e il trigger 'Marker'
    map_a = {'time': 'Timestamp', 'trig': 'Event'}

    # Esempio: Se nel file B le colonne hanno nomi diversi
    map_b = {'time': 'timestamp_s', 'trig': 'code_event'}

    # 3. Lancia analisi: file a = Event, Timestamp
    # fiel b = timestamp_s, code_event
    file_a = 'Muse_data/Stimuli_MUSE-S.csv'
    file_b = 'Muse_data/EEG_MUSE-S_output.csv'

    if os.path.exists(file_a) and os.path.exists(file_b):
        analyze_jitter(file_a, file_b, map_a, map_b, trigger_values=[1, 2, 3])
    else:
        print("File non trovati.")