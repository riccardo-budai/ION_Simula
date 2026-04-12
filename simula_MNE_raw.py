


import mne
from mne.datasets import sample
from mne.viz import plot_alignment, set_3d_view

import numpy as np

# --- 1. FUNZIONE TEMPORALE (Impulso di Corrente I(t)) ---

def current_pulse(t, peak_time_s, duration_tau_s):
    """
    Modella l'evoluzione temporale della corrente I(t) usando la differenza di due esponenziali.
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


# -------------------------------------------------------------------------
# SETUP DATI MNE
# -------------------------------------------------------------------------

data_path = sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
trans_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw-trans.fif'
subjects_dir = data_path / 'subjects'
subject = 'sample'

# Carica le informazioni sui canali EEG
info = mne.io.read_info(raw_fname)
info = mne.pick_info(info, mne.pick_types(info, eeg=True, meg=False, stim=False, eog=False))
trans = mne.read_trans(trans_fname)

# Questo mantiene l'applicazione aperta e il plot visibile
# mne.viz.backends

# 2. DEFINISCI IL TUO SPAZIO SORGENTE
src = mne.setup_source_space(subject, subjects_dir=subjects_dir, spacing='oct6', n_jobs=-1)


# 3. CALCOLA LA FORWARD SOLUTION (OPERATORE STATICO)
fwd = mne.make_forward_solution(
    info,
    trans=trans_fname,
    src=src,
    bem=subjects_dir / subject / 'bem' / f'{subject}-5120-5120-5120-bem-sol.fif',
    eeg=True, meg=False, verbose=False
)

# plot degli elettrodi
fig = plot_alignment(
    info,
    trans=trans,
    subject=subject,
    subjects_dir=subjects_dir,
    # Parametri di visualizzazione:
    dig=False,  # Non mostrare i punti di digitalizzazione
    eeg=['original', 'projected'],  # Mostra sia le posizioni originali che proiettate
    meg=[],  # Non mostrare i sensori MEG
    fwd=fwd,
    coord_frame="head",
    surfaces=["head", "pial"]  # Mostra la superficie del capo (scalpo) e la corteccia
)
# --- 3. IMPOSTA L'ANGOLO DI VISUALIZZAZIONE E MANTIENI APERTA LA FINESTRA ---
# Imposta un angolo di visione tipico per vedere lo scalpo dall'alto/laterale
set_3d_view(figure=fig, azimuth=135, elevation=80, distance=0.6)

# CRUCIALMENTE: Riduciamo la FWD per includere solo i vertici attivi
# MNE consiglia di usare la funzione di riduzione per coerenza:
# Non è necessario se usiamo la Forward Solution nel suo formato completo.
# Continuiamo a usare FWD completo, ma assicuriamoci di selezionare il vertice attivo da FWD.



# 4. CREA L'ATTIVITÀ TEMPORALE (I(t)) E LO SOURCEESTIMATE (STC)
# -------------------------------------------------------------------------

sfreq = info['sfreq']
duration_s = 0.200
n_samples = int(sfreq * duration_s)
times = np.arange(n_samples) / sfreq

# --- Parametri del tuo impulso I(t) ---
PEAK_TIME = 0.003
DECAY_TIME = 0.005
I_t_pulse = current_pulse(times, PEAK_TIME, DECAY_TIME)

# 1. Definisci l'attività (DATI) solo per la sorgente attiva
# Vogliamo 1 riga (per il vertice attivo) e n_samples colonne.
# Usiamo reshape per garantire la forma (1, N_samples)
source_activity_data_1_row = I_t_pulse.reshape(1, -1)
# La dimensione è: (1, n_samples)

# 1. Trova l'indice del vertice attivo nel Forward Solution
# Il vertice che scegliamo deve essere presente negli indici che MNE ha ritenuto validi.
# Usiamo il primo vertice valido nell'emisfero sinistro:
active_vert_idx = fwd['src'][0]['vertno'][0]

# 2. Definisci gli indici dei vertici ATTIVI per STC
# stc.data sarà 1 riga, quindi dobbiamo specificare che solo un vertice è attivo.
vertices_left = np.array([active_vert_idx], dtype=np.int64) # Usa l'indice del vertice trovato
vertices_right = np.array([], dtype=np.int64)
vertices_stc = [vertices_left, vertices_right]

# 3. Definisci l'attività (DATI) solo per la sorgente attiva (1 riga)
source_activity_data_1_row = I_t_pulse.reshape(1, -1)

# 4. CREA L'OGGETTO SOURCEESTIMATE
stc = mne.SourceEstimate(
    data=source_activity_data_1_row,
    # Passa i vertici ATTIVI
    vertices=vertices_stc,
    tmin=times[0],
    tstep=1/sfreq,
    subject=subject
)

# 5. SIMULA I DATI EEG GREZZI
# Applichiamo la funzione di riduzione al FWD per farlo combaciare con l'STC
fwd_sim = mne.forward.apply_forward(fwd, stc, info=info)

#
raw_sim = mne.simulation.simulate_raw(info, stc, forward=fwd, n_jobs=-1)

# 6. VISUALIZZA IL RISULTATO (L'FP ai sensori)
# -------------------------------------------------------------------------
raw_sim.plot_psd()
raw_sim.plot(duration=duration_s, butterfly=True, block=True)



