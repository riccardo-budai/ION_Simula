"""
generazione della risposta muscolare CMAP alla stimolazione del nervo motore in diverse sedi:
    nervo periferico (polso, gomito, ascella, erb)
    corteccia motoria (mep dallo scalpo)
"""

import numpy as np
import matplotlib.pyplot as plt

'''
def single_muap_waveform(t, amplitude, t_peak, duration_sigma):
    """
    Modella una forma d'onda MUAP bi-fasica (negativa-positiva) utilizzando Gaussiane/derivata
    per semplicità. MUAP classico è spesso tri-fasico (p-n-p o n-p-n).
    Usiamo un approccio tri-fasico (P1-N1-P2) approssimato con differenze di Gaussiane.
    """

    t_shifted = t - t_peak

    # Parametri per la tri-fasicità tipica (P-N-P)
    sigma_fase = duration_sigma

    # Fase 1 (Positiva) - Veloce e piccola
    P1 = (amplitude / 4) * np.exp(-(t_shifted ** 2) / (1 * (sigma_fase / 2) ** 2))

    # Fase 2 (Negativa) - Veloce e dominante
    N1 = amplitude * np.exp(-((t_shifted - sigma_fase / 2) ** 2) / (2 * (sigma_fase) ** 2))

    # Fase 3 (Positiva) - Lenta
    P2 = (amplitude / 2) * np.exp(-((t_shifted - sigma_fase) ** 2) / (2 * (sigma_fase * 2) ** 2))

    return P1 - N1 + P2
'''

'''
def calculate_cmap(time_vector,
                   n_mu_units,
                   mean_vc_ms,
                   distance_m,
                   cmap_delay_s=0.001,
                   vc_std_ms=5.0,
                   muap_duration_s=0.005):
    """
    Calcola il CMAP come somma di N MUAP, tenendo conto della dispersione
    della velocità di conduzione (VC).

    :param time_vector: Vettore tempo (secondi).
    :param n_mu_units: Numero di unità motorie da sommare.
    :param mean_vc_ms: Velocità di conduzione media (m/s).
    :param distance_m: Distanza nervo-elettrodo (metri).
    :param vc_std_ms: Deviazione standard della VC per la dispersione.
    :param muap_duration_s: Durata di base del singolo MUAP.
    :return: Array numpy del segnale CMAP.
    """

    time_points = len(time_vector)
    cmap_sum = np.zeros_like(time_vector)

    # 1. Distribuzione dei parametri per ogni unità motore

    # a. Velocità di Conduzione (VC): Variazione gaussiana attorno alla media
    # Usiamo una distribuzione normale (in m/s)
    mu_vcs = np.random.normal(loc=mean_vc_ms, scale=vc_std_ms, size=n_mu_units)
    # Assicurati che le VC siano positive e ragionevoli (es. > 1 m/s)
    mu_vcs[mu_vcs < 1.0] = 1.0

    # b. Amplitudine dei MUAP: Variazione casuale per simulare dimensioni diverse delle MU
    mu_amplitudes = np.random.uniform(low=0.1, high=1.0, size=n_mu_units)

    # 2. Somma dei MUAP con i loro specifici ritardi
    for i in range(n_mu_units):
        # Ritardo di propagazione: Distanza / Velocità
        propagation_delay = distance_m / mu_vcs[i]

        # Ritardo totale (includendo il ritardo sinaptico/elettrodico costante)
        total_delay = propagation_delay + cmap_delay_s

        # Genera il MUAP
        muap = single_muap_waveform(
            time_vector,
            amplitude=mu_amplitudes[i],
            t_peak=total_delay,
            duration_sigma=muap_duration_s
        )
        # Somma algebrica
        cmap_sum += muap
    return cmap_sum
'''


def single_muap_waveform(t, amplitude, t_peak, duration_sigma):
    """
    Modella una forma d'onda MUAP tri-fasica (P1-N1-P2) con ampiezze e larghezze regolate.
    """

    t_shifted = t - t_peak

    # Parametri per la tri-fasicità tipica (P-N-P)
    sigma_fase = duration_sigma

    # Rimuoviamo il fattore '1' non necessario e ci concentriamo sui coefficienti.

    # Fase 1 (Positiva) - Veloce e piccola
    # Riduciamo l'ampiezza (es. 1/6) e usiamo una base per lo smorzamento.
    # [MODIFICA QUI]: Ampiezza ridotta (amplitude / 6) e smorzamento Gaussiano standard (fattore 2)
    P1 = (amplitude / 2) * np.exp(-(t_shifted ** 2) / (2 * (sigma_fase / 8) ** 2))  # Usa /4 per essere stretta
    # P1 = 0.0

    # Fase 2 (Negativa) - Veloce e dominante (picco negativo)
    # Spostiamo leggermente il picco (t_shifted - sigma_fase / 4) e usiamo sigma_fase/2 come larghezza
    N1 = amplitude * np.exp(-((t_shifted - sigma_fase / 4) ** 2) / (2 * (sigma_fase / 2) ** 2))

    # Fase 3 (Positiva) - Lenta (coda positiva)
    # L'ampiezza è metà della N1, la larghezza è maggiore e il picco è spostato più lontano.
    P2 = (amplitude / 2) * np.exp(-((t_shifted - sigma_fase) ** 2) / (2 * (sigma_fase * 1.5) ** 2))

    # La forma MUAP è la combinazione algebrica:
    return -P1 - N1 + P2


def single_mep_waveform(t, amplitude, t_peak, duration_sigma):
    # ... (Il tuo codice MUAP tri-fasico P1-N1-P2)
    t_shifted = t - t_peak
    sigma_fase = duration_sigma

    # Fase 1 (Positiva) - (Corretta come modificato prima)
    P1 = (amplitude / 6) * np.exp(-(t_shifted ** 2) / (2 * (sigma_fase / 4) ** 2))

    # Fase 2 (Negativa) - (Corretta come modificato prima)
    N1 = amplitude * np.exp(-((t_shifted - sigma_fase / 4) ** 2) / (2 * (sigma_fase / 2) ** 2))

    # Fase 3 (Positiva) - (Corretta come modificato prima)
    P2 = (amplitude / 2) * np.exp(-((t_shifted - sigma_fase) ** 2) / (2 * (sigma_fase * 1.5) ** 2))

    return P1 - N1 + P2

def calculate_cmap(time_vector,
                   n_mu_units,
                   mean_vc_ms,
                   distance_m,
                   cmap_delay_s=0.001,
                   vc_std_ms=3.0,
                   muap_duration_min_s=0.002,  # <--- Durata minima (2 ms)
                   muap_duration_max_s=0.015,  # <--- Durata massima (15 ms)
                   muap_duration_base_s=0.005):  # Parametro originale per la forma d'onda base

    time_points = len(time_vector)
    cmap_sum = np.zeros_like(time_vector)

    # ... (Il resto dei parametri VC e Amplitudine) ...
    mu_vcs = np.random.normal(loc=mean_vc_ms, scale=vc_std_ms, size=n_mu_units)
    mu_vcs[mu_vcs < 1.0] = 1.0
    mu_amplitudes = np.random.uniform(low=0.1, high=1.0, size=n_mu_units) / 10

    # [NUOVA LOGICA]: Variazione della durata (sigma)
    MEAN_MUAP_DURATION_S = 0.0085  # 8.5 ms
    STD_MUAP_DURATION_S = 0.00325
    MIN_DURATION_S = muap_duration_min_s
    MAX_DURATION_S = muap_duration_max_s
    # Generiamo un array di durate MUAP nel range desiderato (distribuzione uniforme o normale)
    mu_durations_s_uni = np.random.uniform(low=muap_duration_min_s,
                                       high=muap_duration_max_s,
                                       size=n_mu_units)
    mu_durations_s = np.random.normal(loc=MEAN_MUAP_DURATION_S,
                                          scale=STD_MUAP_DURATION_S,
                                          size=n_mu_units)
    mu_durations_s = np.clip(mu_durations_s,
                             a_min=MIN_DURATION_S,
                             a_max=MAX_DURATION_S)

    # Se il tuo MUAP è definito da un parametro sigma, devi convertire la durata
    # (T_duration) nel parametro sigma (sigma_duration).
    # Assumiamo che sigma_duration sia proporzionale alla durata totale desiderata (es. 1/4 della durata).
    mu_sigmas_s = mu_durations_s / 4.0  # Adattamento al tuo modello single_muap_waveform

    # 2. Somma dei MUAP
    for i in range(n_mu_units):
        propagation_delay = distance_m / mu_vcs[i]
        total_delay = propagation_delay + cmap_delay_s

        # Genera il MUAP, ORA PASSANDO LA DURATA/SIGMA SPECIFICA
        muap = single_muap_waveform(
            time_vector,
            amplitude=mu_amplitudes[i],
            t_peak=total_delay,
            duration_sigma=mu_sigmas_s[i]  # <--- PASSAGGIO DELLA DURATA VARIABILE
        )

        cmap_sum += muap
        plt.plot(TIME_VECTOR * 1e3, muap)
        plt.xlabel("Tempo (ms)")
        plt.ylabel("Ampiezza (mV)")
    plt.gca().invert_yaxis()
    plt.show()

    return cmap_sum


def calculate_mep(time_vector,
                  n_mu_units,
                  mep_latency_ms=24.0,  # Latenza totale target (es. 24 ms a riposo)
                  tccm_spread_ms=1.5,  # Dispersione temporale aggiunta dal tratto centrale
                  muap_duration_min_s=0.002,
                  muap_duration_max_s=0.015):
    """
    Simula il Potenziale Motorio Evocato (MEP) per l'FDI come somma di MUAP.

    A differenza del CMAP, il MEP non dipende solo dalla VC periferica. La dispersione
    temporale (Jitter) è la chiave per la sua morfologia.

    :param mep_latency_ms: La latenza totale clinica target (es. 24 ms).
    :param tccm_spread_ms: Jitter/dispersione aggiunto dalla stimolazione centrale.
    """

    time_points = len(time_vector)
    mep_sum = np.zeros_like(time_vector)

    # 1. PARAMETRI MUAP (Variazione di Ampiezza e Durata)
    n_mu_units = int(n_mu_units * 0.5)  # Un MEP sub-massimale potrebbe attivare meno unità del CMAP
    mu_amplitudes = np.random.uniform(low=0.1, high=1.0, size=n_mu_units)

    # Durata MUAP (Basata sulla distribuzione normale)
    MEAN_MUAP_DURATION_S = 0.0085
    STD_MUAP_DURATION_S = 0.00325
    mu_durations_s_raw = np.random.normal(loc=MEAN_MUAP_DURATION_S,
                                          scale=STD_MUAP_DURATION_S,
                                          size=n_mu_units)
    mu_durations_s = np.clip(mu_durations_s_raw,
                             a_min=muap_duration_min_s,
                             a_max=muap_duration_max_s)
    mu_sigmas_s = mu_durations_s / 3.0

    # 2. CALCOLO DEI RITARDI (Jitter è la chiave)

    # Latenza base per TUTTE le unità motorie (MEP Latenza Totale)
    base_delay_s = mep_latency_ms / 1000.0

    # Aggiunta di Jitter/Dispersione dal percorso centrale (TCCM Jitter)
    # Questa dispersione modella l'attivazione non perfettamente sincrona della corteccia
    # e del tratto corticospinale (i-waves, desincronizzazione spinale).
    # Usiamo una distribuzione uniforme o normale attorno allo 0.

    # TCCM Spread determina quanto il segnale 'si disperde' all'arrivo alla spina
    central_jitter_s = np.random.uniform(low=0.0,
                                         high=tccm_spread_ms / 1000.0,
                                         size=n_mu_units)

    # 3. SOMMA DEI MUAP
    for i in range(n_mu_units):
        # Ritardo totale = Latenza di base + Jitter individuale
        total_delay = base_delay_s + central_jitter_s[i]

        muap = single_mep_waveform(
            time_vector,
            amplitude=mu_amplitudes[i],
            t_peak=total_delay,
            duration_sigma=mu_sigmas_s[i]
        )

        mep_sum += muap

    # Normalizzazione per ampiezza (optional)
    return mep_sum / np.max(np.abs(mep_sum)) * 500  # Normalizza ad es. a 500 uV picco a picco

# Nel tuo simulatore o worker, dove si calcola il CMAP/M-wave
# (Assumendo che tu voglia modellare l'M-wave registrato localmente)

FS = 15000
MAX_TIME_S = 0.030 # 50 ms
TIME_VECTOR = np.linspace(0, MAX_TIME_S, int(FS * MAX_TIME_S))

# Parametri specifici della via efferente
DISTANCE = 0.60 # 15 cm dal sito di registrazione/stimolazione
VC_MOTOR = 60.0 # Velocità di conduzione tipica motoria (m/s)
N_MU = 1000     # Numero di unità motorie attivate

# Calcolo del CMAP simulato
simulated_cmap = calculate_cmap(
    TIME_VECTOR,
    n_mu_units=N_MU,
    mean_vc_ms=VC_MOTOR,
    distance_m=DISTANCE,
    cmap_delay_s=0.0023, # Ritardo sinaptico/di giunzione (2.3 ms)
    vc_std_ms=10.0
)

# Ora puoi tracciare 'simulated_cmap' in funzione di 'TIME_VECTOR'
plt.plot(TIME_VECTOR * 1e3, simulated_cmap)
plt.xlabel("Tempo (ms)")
plt.ylabel("Ampiezza (mV)")
plt.gca().invert_yaxis()
plt.show()

FS = 2000
MAX_TIME_S = 0.080 # 80 ms per vedere la risposta completa
TIME_VECTOR = np.linspace(0, MAX_TIME_S, int(FS * MAX_TIME_S))

mep_fdi = calculate_mep(
    TIME_VECTOR,
    n_mu_units=1500,
    mep_latency_ms=24.0, # Latenza target
    tccm_spread_ms=5.0   # Dispersione totale di 2 ms
)
plt.plot(TIME_VECTOR * 1e3, mep_fdi)
plt.xlabel("Tempo (ms)")
plt.ylabel("Ampiezza (mV)")
plt.gca().invert_yaxis()
plt.show()