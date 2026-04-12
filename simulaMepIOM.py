import numpy as np
import matplotlib.pyplot as plt
import random

# ==============================================================================
# 1. PARAMETRI GENERALI E MODELLAZIONE CONCETTUALE
# ==============================================================================

# Parametri di Simulazione
SIM_DURATION_MS = 50.0      # Durata totale della simulazione
DT_MS = 0.05                # Passo temporale (alta risoluzione)
TIME = np.arange(0, SIM_DURATION_MS, DT_MS)

# 1.1 Modello di Stimolazione Corticale (Volley D/I)
# La tES con 3-4 impulsi evoca una salva D seguita da onde I (I1, I2, I3)[cite: 70].
# I tempi sono approssimativi, con un intervallo inter-onda di ~1.5 ms[cite: 253].
CORTICAL_SPIKE_TIMES = [
    2.0,  # 5.0 Onda D (attivazione assonale diretta) [cite: 72]
    3.5,  # 6.5 Onda I1 (sinapsi dendritica prossimale) [cite: 78]
    5.5,  # 8.0 Onda I2
    7.0,  # 9.5 Onda I3 (input sinaptici con ritardi maggiori) [cite: 79]
]

# 1.2 Modello di Conduzione (CST)
CONDUCTION_DELAY_MS = 19.0  # (5 ms per upper limb) (19 ms per lower limb)
                            # 15.0 Ritardo medio per raggiungere il midollo spinale
NUM_CST_FIBERS = 10000  # Popolazione CST (circa 1 milione in totale [cite: 112])
DISPERSION_STD_MS = 0.5  # Deviazione standard per la dispersione temporale [cite: 113]

# 1.3 Modello di Integrazione Spinale (Alfa Motoneurone)
MOTONEURON_THRESHOLD_MV = 20.0  # Soglia di attivazione (simbolica)
EPSP_PEAK_MS = 1.0  # Tempo al picco dell'EPSP (cinetica del glutammato) [cite: 132]
EPSP_DECAY_TAU_MS = 3.0  # Costante di decadimento dell'EPSP [cite: 139]
EPSP_AMPLITUDE_MV = 7.0  # Ampiezza di un singolo EPSP sottom soglia [cite: 134]


# ==============================================================================
# 2. FUNZIONI DI SIMULAZIONE
# ==============================================================================

def generate_epsp(time_array, time_peak, tau_decay, amplitude, arrival_time):
    """
    Genera un EPSP (Potenziale Postsinaptico Eccitatorio) modellato da una funzione alfa
    (o doppia esponenziale semplificata) [cite: 132].
    """
    t_shifted = time_array - arrival_time
    epsp = np.zeros_like(t_shifted)

    # Funzione alfa: t/tau * exp(-t/tau)
    # Usiamo una forma semplificata basata sulla costante di tempo
    valid_t = t_shifted > 0
    epsp[valid_t] = amplitude * (t_shifted[valid_t] / time_peak) * np.exp(-t_shifted[valid_t] / tau_decay)

    # Normalizza l'altezza
    epsp_max = np.max(epsp)
    if epsp_max > 0:
        epsp *= (amplitude / epsp_max)

    return epsp


def simulate_cst_conduction(cortical_times, num_fibers, delay_ms, dispersion_std_ms):
    """
    Simula la propagazione lungo il CST, applicando ritardo e dispersione temporale
    a ogni salva (D/I)[cite: 113].
    """
    arrival_times = {}

    # Distribuzione dei ritardi (simula la distribuzione del diametro delle fibre) [cite: 112]
    dispersion_population = np.random.normal(0, dispersion_std_ms, num_fibers)

    for volley_index, cortical_time in enumerate(cortical_times):
        # Ritardo medio di conduzione (CST) + Tempo di spike corticale
        mean_arrival = cortical_time + delay_ms

        # Simula l'arrivo al midollo spinale per ogni fibra (dispersione temporale)
        # La dispersione è la stessa per tutte le salve per simulare la coerenza delle fibre
        # ma i tempi di arrivo sono diversi per ogni salva.
        final_arrivals = mean_arrival + dispersion_population

        # Traccia solo l'arrivo della fibra più sincrona (per la precisione del CMAP)
        # e usiamo solo un sottoinsieme casuale di fibre per semplicità.
        # Raggruppa i tempi di arrivo per l'integrazione
        arrival_times[f'V{volley_index}'] = final_arrivals[::(num_fibers // 100)]  # Semplifica a 100 fibre

    return arrival_times


def simulate_spinal_integration(cst_arrival_times):
    """
    Simula la sommazione temporale degli EPSP e l'attivazione del Motoneurone Alfa[cite: 137].
    """
    # Motoneurone singolo (per la sommazione)
    motoneuron_potential = np.zeros_like(TIME)

    # Traccia i tempi di attivazione del motoneurone
    motoneuron_spike_times = []

    # Somma gli EPSP generati dalle fibre CST
    for volley_key, arrival_times_ms in cst_arrival_times.items():
        # Ogni fibra CST che arriva genera un EPSP sottom soglia nel motoneurone[cite: 134].

        # Per semplicità, consideriamo un solo EPSP risultante per ogni salva (V0..V3)
        # con un'ampiezza scalata per simulare la sommazione spaziale della popolazione

        # Somma di tutti i micro-EPSP (qui è una sommazione *temporale* di salve)

        # Latenza sinaptica (circa 1ms per la sinapsi monosinaptica)
        synaptic_delay = 1.0

        # L'onda D (V0) è più resistente all'anestesia, ma qui non modelliamo l'anestesia esplicitamente.
        # Applichiamo un EPSP per l'arrivo medio della salva:
        mean_arrival = np.mean(arrival_times_ms)

        # Latenza sinaptica inclusa
        epsp_arrival_time = mean_arrival + synaptic_delay

        # Genera il Potenziale Postsinaptico Eccitatorio (EPSP)
        single_epsp = generate_epsp(
            TIME,
            EPSP_PEAK_MS,
            EPSP_DECAY_TAU_MS,
            EPSP_AMPLITUDE_MV,
            epsp_arrival_time
        )
        motoneuron_potential += single_epsp

        # Controllo della soglia (attivazione "tutto o nulla") [cite: 140]
        if not motoneuron_spike_times:
            # Controllo solo se non si è già attivato (per simulare 1 spike per salva)
            if np.max(motoneuron_potential) >= MOTONEURON_THRESHOLD_MV:
                # Registra il tempo in cui si verifica l'attivazione
                spike_time_index = np.argmax(motoneuron_potential)
                motoneuron_spike_times.append(TIME[spike_time_index])
                # Azzeriamo il potenziale dopo lo spike (refrattarietà)
                motoneuron_potential[spike_time_index:] = 0.0

    return motoneuron_potential, motoneuron_spike_times


def generate_cmap_waveform(motoneuron_spike_times):
    """
    Simula la sintesi del CMAP (Potenziale d'Azione Muscolare Composto)[cite: 173].
    """
    cmap = np.zeros_like(TIME)

    # 1. Conduzione periferica (motoneurone alfa -> muscolo) [cite: 110]
    PERIPHERAL_DELAY = 5.0  # Ritardo fisso (periferia + NMJ)

    for spike_time in motoneuron_spike_times:
        # Il potenziale del motoneurone genera un MFAP nella fibra muscolare [cite: 152]

        mfap_arrival_time = spike_time + PERIPHERAL_DELAY

        # Modello semplificato di Potenziale d'Azione dell'Unità Motoria (MUAP) [cite: 164]
        # Usiamo una forma biphasic/triphasic semplificata
        def generate_muap(t, arrival_t):
            t_rel = t - arrival_t
            # Modello tri-fasico semplificato: Onda positiva-negativa-positiva
            muap = 20 * np.exp(-((t_rel - 1.5) ** 2) / 0.5)  # Picco positivo
            muap -= 40 * np.exp(-((t_rel - 3.0) ** 2) / 0.8)  # Picco negativo (più ampio)
            muap += 10 * np.exp(-((t_rel - 5.0) ** 2) / 1.0)  # Post-potenziale
            return muap

        muap_wave = generate_muap(TIME, mfap_arrival_time)

        # Sommazione spaziale dei MUAP per formare il CMAP [cite: 173]
        cmap += muap_wave * 0.5  # Scala per simulare la somma di una popolazione di UM attivate

    return cmap


# ==============================================================================
# 3. ESECUZIONE E VISUALIZZAZIONE
# ==============================================================================

def run_simulation():
    """Esegue l'intera pipeline di simulazione."""

    # Fase 1: Conduzione CST (Generazione Volley D/I e Dispersione)
    cst_arrival_times = simulate_cst_conduction(
        CORTICAL_SPIKE_TIMES,
        NUM_CST_FIBERS,
        CONDUCTION_DELAY_MS,
        DISPERSION_STD_MS
    )

    # Fase 2: Integrazione Spinale (Sommazione EPSP)
    motoneuron_potential, motoneuron_spike_times = simulate_spinal_integration(
        cst_arrival_times
    )

    # Fase 3: Sintesi CMAP
    cmap_waveform = generate_cmap_waveform(motoneuron_spike_times)

    # Report
    print("-" * 50)
    print(f"SALVA CORTICALE (D/I) INVIATA A: {CORTICAL_SPIKE_TIMES} ms")
    print(f"RITARDO MEDIO CST: {CONDUCTION_DELAY_MS} ms")
    print(f"DISPERSIONE TEMPORALE CST: {DISPERSION_STD_MS} ms")
    print("-" * 50)

    if motoneuron_spike_times:
        print(f"✅ MOTONEURONE ALFA ATTIVATO A: {motoneuron_spike_times[0]:.2f} ms")
        print(f"Lat. Insorgenza CMAP (appross.): {motoneuron_spike_times[0] + 5.0:.2f} ms")

    else:
        print("❌ MOTONEURONE ALFA NON ATTIVATO (SOTTOM SOGLIA).")

    print("-" * 50)

    return motoneuron_potential, cmap_waveform, motoneuron_spike_times


# --- Plotting ---
if __name__ == "__main__":
    potential, cmap, spikes = run_simulation()

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Simulazione Concettuale della Via Motoria (tES -> CMAP)", fontsize=14)

    # Plot 1: Integrazione Spinale (Motoneurone)
    axs[0].plot(TIME, potential, label='Somma EPSP', color='blue')
    axs[0].axhline(MOTONEURON_THRESHOLD_MV, color='red', linestyle='--', label='Soglia Attivazione')

    for spike_time in spikes:
        axs[0].axvline(spike_time, color='green', linestyle=':', label='Spike MN')

    axs[0].set_title("Integrazione Sinaptica (Alfa Motoneurone Spinale)")
    axs[0].set_ylabel("Potenziale (mV)")
    axs[0].set_xlim(min(CORTICAL_SPIKE_TIMES) - 2, SIM_DURATION_MS)
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle=':', alpha=0.6)

    # Plot 2: CMAP (Risposta Muscolare)
    axs[1].plot(TIME, cmap, label='CMAP Simulato', color='black', linewidth=2)
    axs[1].set_title("Potenziale d'Azione Muscolare Composto (CMAP) Registrato")
    axs[1].set_xlabel("Tempo dalla Stimolazione (ms)")
    axs[1].set_ylabel("Ampiezza (µV) - Simbolica")
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle=':', alpha=0.6)


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()