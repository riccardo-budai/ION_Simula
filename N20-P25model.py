import mne
import numpy as np
import pywt
# from scipy.signal import ricker  # Per creare la forma d'onda del dipolo (tipo wavelet)
import matplotlib.pyplot as plt


# --- 1. PARAMETRI FISICI CRITICI (Secondo i tuoi requisiti) ---
# Conduttività del modello sferico a 4 strati (Brain, CSF, Skull, Scalp)
# La conduttività del cranio (0.004 S/m) divisa per la conduttività del cervello/scalpo (0.33 S/m)
# dà un rapporto di circa 82.5 (tradizionale BEM).
# Per ottenere un BSCR più vicino a 30-50 (come richiesto), aumentiamo il valore del cranio.
# BSCR = sigma_brain / sigma_skull. Vogliamo 33. -> sigma_skull = 0.33 / 33 = 0.01

SIGMAS_OPTIMAL = (0.33, 1.0, 0.01, 0.33)  # [Brain, CSF, Skull, Scalp] in S/m
# Controllo BSCR: 0.33 / 0.01 = 33.0. -> OTTIMALE! (Rientra nel range 30-50)

# --- 2. PARAMETRI TEMPORALI E SPAZIALI DEL DIPOLO N20-P25 ---

SFREQ = 1000  # Frequenza di campionamento (1 kHz)
T_MIN = -0.050  # Tempo iniziale (-50 ms)
T_MAX = 0.050  # Tempo finale (+50 ms)
T_VECTOR = np.arange(T_MIN, T_MAX, 1 / SFREQ)

# Latencies (in secondi)
LATENCY_N20 = 0.020
LATENCY_P25 = 0.025

# Posizioni (coordinate MRI / Head in metri)
# Assumiamo una posizione generica di S1 per un dipolo tangenziale.
# Coordinata X: Laterale, Y: Antero-Posteriore, Z: Superiore-Inferiore
POS_N20 = np.array([-0.040, -0.010, 0.070])  # 4cm lateralmente, 1cm dietro, 7cm in alto
POS_P25 = np.array([-0.040, -0.010, 0.070])  # Per semplicità, la posizione è la stessa, cambia l'orientamento/polarità

# Orientamenti (vettori unitari)
# N20: Tangenziale (es. rivolto in avanti/indietro)
ORI_N20 = np.array([0.0, 0.8, 0.2])  # Orientato prevalentemente in Y (AP)
ORI_N20 /= np.linalg.norm(ORI_N20)  # Normalizza
# P25: Radiale (es. rivolto verso la superficie)
ORI_P25 = np.array([0.2, 0.0, 0.8])  # Orientato prevalentemente in Z (Radiale)
ORI_P25 /= np.linalg.norm(ORI_P25)  # Normalizza

'''
def make_dipole_moment(time_vector, latency_s, peak_width_s, amplitude):
    """ Crea il momento dipolare Q(t) usando una Ricker wavelet (simil-Gaussiana bipolare). """
    # La ricker (Mexican Hat wavelet) ha una forma eccellente per un evento SEP
    # Calcola l'indice del picco
    peak_idx = int((latency_s - T_MIN) * SFREQ)

    # Crea la forma d'onda
    # Usiamo una ricker che è già centrata. Spostiamo la finestra.

    # Crea una Ricker centrata su zero e poi la sposta
    # 5.0 è un fattore di scala per la larghezza del picco
    ricker_wave = ricker(len(time_vector), a=peak_width_s * SFREQ)

    # Sposta il picco all'indice corretto
    shift_samples = peak_idx - int(len(time_vector) / 2)
    # Roll sposta l'array ciclicamente
    q_t = np.roll(ricker_wave, shift_samples)

    # Modula l'ampiezza
    # Normalizza l'ampiezza massima a 1 e moltiplica per l'ampiezza desiderata (in Am^2)
    q_t /= np.max(np.abs(q_t))
    q_t *= amplitude * 1e-9  # Usiamo 1 nAm^2 come momento tipico per l'EEG

    # Applica un filtro per eliminare il segnale prima del T_MIN (causalità)
    q_t[time_vector < latency_s - 0.015] = 0  # Taglia il segnale prima dell'insorgenza

    return q_t
'''

import numpy as np
import pywt  # Importa la libreria PyWavelets


# ... (il resto degli import e dei parametri iniziali) ...

def make_dipole_moment_pywt(time_vector, latency_s, peak_width_s, amplitude, sfreq):
    """
    Crea il momento dipolare Q(t) usando la Wavelet di Morlet (o Ricker/mexh)
    di PyWavelets.

    Args:
        time_vector (np.array): Vettore del tempo.
        latency_s (float): Latenza del picco (es. 0.020 s).
        peak_width_s (float): Larghezza approssimativa del picco (simile a 'a' in pywt).
        amplitude (float): Ampiezza desiderata (in nAm^2).
        sfreq (int): Frequenza di campionamento (Hz).

    Returns:
        np.array: Il momento dipolare Q(t).
    """

    # --- 1. Definizione della Wavelet e della Scala ---
    # Frequenza in cicli/s. Usiamo l'inverso della larghezza del picco.
    # La scala in PyWavelets è inversamente proporzionale alla frequenza.
    # Se il picco è largo 3ms, la frequenza è ~333 Hz.
    # La scala deve essere calibrata in base alla forma d'onda desiderata.

    # La scala 'a' per la Ricker (mexh) è la larghezza temporale.
    # Manteniamo la logica di 'peak_width_s' che definisce la larghezza.
    scale = peak_width_s * sfreq

    # 2. Genera la Wavelet
    # La Wavelet Ricker/Mexican Hat ('mexh') è la più simile a scipy.signal.ricker
    # 'wavelet' è la forma d'onda. 'x' è il vettore del tempo.
    wavelet_form, x = pywt.cwt(np.array([1]), scales=[scale], wavelet='mexh')

    # La forma della wavelet è nel primo (e unico) elemento dell'output CWT
    # Ma pywt.cwt è pensato per l'analisi, non per la sintesi della forma.

    # *** ALTERNATIVA PIÙ SEMPLICE: pywt.ContinuousWavelet ***
    # Usiamo il metodo corretto per ottenere la forma della wavelet:

    # Ottieni i coefficienti della wavelet per la data scala (la forma del segnale)
    # L'output è solo la forma d'onda.
    wavelet_form, x = pywt.integrate_wavelet('mexh', precision=10)

    # 3. Interpolazione e Normalizzazione

    # Il vettore 'x' ha solo pochi punti. Dobbiamo interpolare e allungare.
    # Interpoliamo la forma della wavelet per adattarla al nostro vettore temporale.
    from scipy.interpolate import interp1d

    # Calcola il numero di campioni desiderato
    n_samples = len(time_vector)

    # La forma d'onda 'wavelet_form' è centrata attorno a 0 su 'x'.
    # Interpoliamo per ottenere la forma della wavelet con il numero di punti corretto
    f_interp = interp1d(x, wavelet_form, kind='cubic', bounds_error=False, fill_value=0)

    # Determiniamo la finestra temporale corretta per l'interpolazione
    # (basata sulla larghezza del picco)
    time_span = x[-1] - x[0]
    scaled_x = np.linspace(-time_span * peak_width_s, time_span * peak_width_s, n_samples)

    q_t_centered = f_interp(scaled_x)

    # 4. Spostamento Temporale e Modulazione dell'Ampiezza

    # Spostiamo il segnale per centrarlo sulla latenza corretta
    # Calcola l'offset temporale desiderato
    time_offset = latency_s - (time_vector[0] + time_vector[-1]) / 2

    # Sposta l'onda con l'interpolazione (o un semplice roll se l'interpolazione è su una griglia fissa)
    # Per semplicità, usiamo l'indice roll:
    peak_idx = np.argmax(np.abs(q_t_centered))
    latency_idx = int((latency_s - time_vector[0]) * sfreq)
    shift_samples = latency_idx - peak_idx

    q_t_shifted = np.roll(q_t_centered, shift_samples)

    # Normalizza e modula l'ampiezza
    max_val = np.max(np.abs(q_t_shifted))
    if max_val > 0:
        q_t_shifted /= max_val

    q_t_final = q_t_shifted * amplitude * 1e-9  # Moltiplica per l'ampiezza desiderata

    # Applica un filtro per eliminare il segnale prima del T_MIN (causalità)
    q_t_final[time_vector < latency_s - 0.015] = 0

    return q_t_final


# Esempio di Utilizzo:
# q_n20 = make_dipole_moment_pywt(T_VECTOR, LATENCY_N20, WIDTH_N20, AMPLITUDE_N20, SFREQ)
# q_p25 = make_dipole_moment_pywt(T_VECTOR, LATENCY_P25, WIDTH_P25, AMPLITUDE_P25, SFREQ)

# --- 3. CREAZIONE DEL MODELLO DEL VOLUME CONDUTTORE (Sphere Model) ---

# Usiamo un Info object di base (necessario per definire i sensori)
info = mne.create_info(ch_names=['Cz', 'C3', 'C4', 'Fz'], sfreq=SFREQ, ch_types='eeg')
# Aggiungiamo posizioni standard 10/20 (necessarie per MNE)
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# Creazione del modello sferico a 4 strati con il BSCR ottimale 33:1
sphere_model = mne.make_sphere_model(
    head_radius=0.09, # 9 cm di raggio
    info=info,
    relative_radii=(0.9, 0.92, 0.97, 1.0),
    sigmas=SIGMAS_OPTIMAL
)

print(f"Modello Sferico creato con BSCR: {SIGMAS_OPTIMAL[0] / SIGMAS_OPTIMAL[2]:.1f}")


# --- 4. GENERAZIONE DEL MOMENTO DIPOLARE TEMPORALE Q(t) ---

# Parametri del pulse (ampiezza in nAm^2, larghezza in s)
AMPLITUDE_N20 = -1.5 # Negativo (per la polarità N20)
WIDTH_N20 = 0.003    # 3 ms di larghezza del picco

AMPLITUDE_P25 = 1.0 # Positivo (per la polarità P25)
WIDTH_P25 = 0.004   # 4 ms di larghezza del picco

# Q(t) del dipolo N20 (tangenziale)
# q_n20 = make_dipole_moment(T_VECTOR, LATENCY_N20, WIDTH_N20, AMPLITUDE_N20)
q_n20 = make_dipole_moment_pywt(T_VECTOR, LATENCY_N20, WIDTH_N20, AMPLITUDE_N20, SFREQ)
# Q(t) del dipolo P25 (radiale)
# q_p25 = make_dipole_moment(T_VECTOR, LATENCY_P25, WIDTH_P25, AMPLITUDE_P25)
q_p25 = make_dipole_moment_pywt(T_VECTOR, LATENCY_P25, WIDTH_P25, AMPLITUDE_P25, SFREQ)


# --- 5. CREAZIONE DELLA SORGENTE DINAMICA (Dipole MNE Object) ---

# 1. Calcola il MOMENTO DIPOLARE TEMPORALE totale (Vettore 3D nel tempo)
combined_moment = q_n20[:, np.newaxis] * ORI_N20[np.newaxis, :] + \
                  q_p25[:, np.newaxis] * ORI_P25[np.newaxis, :]

# 2. Ampiezza (la norma del momento in ogni istante temporale)
amplitude_t = np.linalg.norm(combined_moment, axis=1)

# 3. Orientamento (Vettore unitario del momento in ogni istante temporale)
# Inizializza l'orientamento con una forma corretta
orientation_t = np.zeros_like(combined_moment)

# Imposta una soglia per definire "zero" per evitare problemi di floating point
ZERO_TOLERANCE = 1e-15

# Trova gli indici dove l'ampiezza NON è zero
non_zero_mask = amplitude_t >= ZERO_TOLERANCE

# Normalizza solo i momenti che non sono zero
orientation_t[non_zero_mask] = combined_moment[non_zero_mask] / amplitude_t[non_zero_mask, np.newaxis]

# Per i punti in cui l'ampiezza è zero (zero_mask = ~non_zero_mask):
# MNE vuole un vettore di lunghezza 1. Diamo un orientamento di default (es. l'orientamento N20)
# Poiché l'ampiezza è zero, il forward model risulterà comunque zero per questi istanti,
# ma MNE è soddisfatto dalla lunghezza 1 del vettore 'ori'.
orientation_t[~non_zero_mask] = ORI_N20 # 4. gof (Goodness of Fit)
gof_array = np.ones(len(T_VECTOR))

# --- 5. CREAZIONE DELLA SORGENTE DINAMICA (Dipole MNE Object) ---
# 1. Ricalcolo del vettore temporale per garantire che sia corretto
T_VECTOR = np.arange(T_MIN, T_MAX, 1/SFREQ)
N_TIME = len(T_VECTOR)

# --- 5. CREAZIONE DELLA SORGENTE DINAMICA (Dipole MNE Object) ---

# Per un dipolo dinamico con posizione fissa, MNE preferisce che `pos` sia replicato
# per ogni punto temporale per garantire che le dimensioni del tempo vengano mantenute.
POS_DYN = np.tile(POS_N20, (N_TIME, 3)) # Dimensione (N_TIME, 3)

dipole = mne.Dipole(
    times=T_VECTOR,
    pos=POS_DYN,                      # Usa la posizione replicata nel tempo
    ori=orientation_t,                # Orientamento (N_TIME, 3)
    amplitude=amplitude_t,            # Ampiezza (N_TIME,)
    gof=gof_array
)

# --- 6. CALCOLO DEL MODELLO FORWARD E SIMULAZIONE ---

# Calcola la leadfield (forward solution) per il dipolo dinamico
fwd, stc = mne.make_forward_dipole(dipole, sphere_model, info)

# Proietta la sorgente (stc) nello spazio dei sensori (evoked potential)
evoked = mne.simulation.simulate_evoked(fwd, stc, info, nave=np.inf)

# Converti in microVolt (uV)
evoked.data *= 1e6

# --- 7. PLOT DEI RISULTATI ---

# Plot del potenziale evocato simulato
fig, ax = plt.subplots(figsize=(8, 5))
evoked.plot(titles='Simulazione N20-P25 (BSCR 33:1)', axes=ax, time_unit='ms', spatial_colors=True)
ax.set_title(f'SEP S1 Simulato (BSCR: 33:1)')
ax.axvline(LATENCY_N20 * 1000, color='r', linestyle='--', label='N20 (20 ms)')
ax.axvline(LATENCY_P25 * 1000, color='g', linestyle='--', label='P25 (25 ms)')
ax.set_xlabel('Tempo (ms)')
ax.set_ylabel('Ampiezza (µV)')
ax.legend()
plt.tight_layout()
plt.show()

# Stampa la leadfield (il fattore spaziale che proietta il segnale sui sensori)
leadfield = fwd['sol']['data']
print(f"\nLeadfield (Sensori x Dipoli) shape: {leadfield.shape}")
