import numpy as np
import matplotlib.pyplot as plt


# --- 1. FUNZIONE TEMPORALE (Impulso di Corrente I(t)) ---

def current_pulse(t, peak_time_s, duration_tau_s):
    """
    Modella l'evoluzione temporale della corrente I(t) usando la differenza di due esponenziali.
    Questa forma d'onda raggiunge il picco a circa 3ms e decade.

    Args:
        t (np.array): Vettore del tempo in secondi.
        peak_time_s (float): Tempo in cui la funzione raggiunge il picco (circa).
        duration_tau_s (float): Costante di tempo per la durata/decadimento.

    Returns:
        np.array: L'ampiezza relativa della corrente nel tempo (tra 0 e 1).
    """
    # Usiamo una forma comune (Difference of Exponentials)
    t_minus_delay = t - 0.001  # Aggiunge un piccolo ritardo iniziale
    t_minus_delay[t_minus_delay < 0] = 0

    # Due costanti di tempo per salita veloce e decadimento lento
    tau_rise = peak_time_s / 2
    tau_decay = duration_tau_s

    I_t = (np.exp(-t_minus_delay / tau_decay) - np.exp(-t_minus_delay / tau_rise))

    # Normalizza per assicurare che il picco massimo sia 1
    max_val = np.max(I_t)
    return I_t / max_val if max_val > 0 else I_t


# --- 2. PARAMETRI FISICI ---
SIGMA_E = 0.3
DIPOLE_LENGTH_M = 100e-6
CURRENT_MAX_MICROA = 1000e-6  # Corrente massima aumentata a 1000 uA per segnale visibile

# --- PARAMETRI TEMPORALI ---
DT = 0.0001  # Passo temporale (0.1 ms)
T_TOTAL = 0.012  # Tempo totale (25 ms)
TIME_VECTOR = np.linspace(0, T_TOTAL, int(T_TOTAL / DT))
PEAK_TIME = 0.002  # 3 ms per il picco
DECAY_TIME = 0.003  # 5 ms per il decadimento (tau decay)

# --- PARAMETRI SPAZIALI ---
DISTANCE_X_M = 0.05  # 4 cm dal centro del dipolo (Registrazione superficiale/lontana)

# Il dipolo è sull'asse X (Antero-Posteriore), elettrodo a Z fisso (superficie)
DIPOLE_POSITIONS = {
    'Source': np.array([DIPOLE_LENGTH_M / 2, 0.0, 0.0]),
    'Sink': np.array([-DIPOLE_LENGTH_M / 2, 0.0, 0.0])
}
# La registrazione avviene ad una profondità Z_fixed (es. superficie del tessuto)
Z_fixed = 0.05  # 1 mm al di sopra del piano del dipolo


# --- 3. FUNZIONE DI CALCOLO (Invariata) ---

def calculate_field_potential_static(electrode_pos, current, sigma_e, dipole_pos_dict):
    """
    Calcola il potenziale di campo (FP) statico per una data posizione
    e una data corrente.
    """
    K = 1.0 / (4 * np.pi * sigma_e)
    FP_total = 0.0
    r_source = np.linalg.norm(electrode_pos - dipole_pos_dict['Source'])
    r_sink = np.linalg.norm(electrode_pos - dipole_pos_dict['Sink'])

    if r_source > 0:
        FP_total += K * (current / r_source)
    if r_sink > 0:
        FP_total += K * (-current / r_sink)
    return FP_total


# --- 4. SIMULAZIONE DELLA DINAMICA TEMPORALE ---

# 1. Calcola il potenziale di campo statico (il fattore spaziale costante)
# L'elettrodo è fisso a X=4cm, Y=0, Z=1mm
STATIC_ELECTRODE_POS = np.array([DISTANCE_X_M, 0.0, Z_fixed])

FP_static_V = calculate_field_potential_static(
    STATIC_ELECTRODE_POS,
    CURRENT_MAX_MICROA,  # Usiamo la corrente massima per trovare l'ampiezza massima
    SIGMA_E,
    DIPOLE_POSITIONS
)

# 2. Calcola l'impulso di corrente nel tempo I(t)
I_t_pulse = current_pulse(TIME_VECTOR, PEAK_TIME, DECAY_TIME)

# 3. Combina Spazio e Tempo: FP(t) = FP_static * I(t)
FP_temporal_V = FP_static_V * I_t_pulse

# 4. Conversione per la visualizzazione
FP_temporal_uV = FP_temporal_V * 1e6  # Converti in microVolt (uV)
TIME_VECTOR_MS = TIME_VECTOR * 1e3  # Converti in millisecondi (ms)

# --- 5. VISUALIZZAZIONE DEI RISULTATI ---
plt.figure(figsize=(10, 6))
plt.plot(TIME_VECTOR_MS, FP_temporal_uV,
         linestyle='-', linewidth=2,
         label=f'FP Temporale a X={DISTANCE_X_M * 100:.1f} cm, Z={Z_fixed * 1000:.1f} mm')

plt.axhline(0, color='gray', linestyle='--')
plt.axvline(PEAK_TIME * 1000, color='r', linestyle=':', label=f'Picco (approx. {PEAK_TIME * 1000:.1f} ms)')

plt.title('Simulazione del Potenziale Post-Sinaptico di Campo (fPSP)')
plt.xlabel('Tempo (ms)')
plt.ylabel('Potenziale di Campo (uV)')
plt.grid(True)
plt.legend()
plt.show()