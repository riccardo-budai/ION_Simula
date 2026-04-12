import mne
import numpy as np
import os
import h5py
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
from mne.minimum_norm import write_inverse_operator, read_inverse_operator

# ======================================================================================================================
# 1. SETUP E GENERAZIONE MODELLO PER CARATTERISTICHE DEL NOISE DI UNA REGISTRAZIONE SIMMETRICA 0036
# ======================================================================================================================
path_eeg = '/media/ric23/Extreme SSD/Multimodal_dataset_hdf5'
file_path = os.path.join(path_eeg, 'sub-0142.h5')
print('path to eeg file =', file_path)

with h5py.File(file_path, 'r') as f:
    eeg = f['eeg_data'][:, 0:5000]

print(eeg.shape, eeg.shape[1]/500, ' seconds')
print(f"avg amp = {np.mean(eeg, axis=0)}")

sampling_rate = 500
all_ch_names_in_file = ['FC3', 'FC4', 'CP3', 'CP4', 'FT7', 'FT8', 'TP7', 'TP8']
ch_map = {name: idx for idx, name in enumerate(all_ch_names_in_file)}
print("Mappa indici canali:", ch_map)

info = mne.create_info(all_ch_names_in_file, sampling_rate, 'eeg')
raw = mne.io.RawArray(eeg, info, verbose=False)
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)
raw.set_montage(montage)
raw.filter(l_freq=1.0, h_freq=20, verbose=False)
raw.plot(block=True)
'''
print("1. Configurazione e simulazione dati...")
# Canali specifici per il montaggio 4+4
selected_channels = ['FC3', 'CP3', 'FT7', 'TP7',  # Emisfero Sinistro
                     'FC4', 'CP4', 'FT8', 'TP8']  # Emisfero Destro
# Creazione info e montaggio
info = mne.create_info(ch_names=selected_channels, sfreq=256, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)
# Generiamo dati con una struttura di segnale (sinusoidi) per vedere qualcosa di sensato
# invece di rumore bianco puro.
n_samples = 256 * 4  # 2 secondi
times = np.arange(n_samples) / info['sfreq']
rng = np.random.RandomState(42)
# Simuliamo 2 sorgenti latenti distinte (es. una frontale e una parietale)
source_signal_1 = np.sin(2 * np.pi * 4 * times) * 1e-5  # 4 Hz
source_signal_2 = np.cos(2 * np.pi * 20 * times) * 1e-5  # 20 Hz
# Proiettiamo questi segnali sugli 8 canali in modo misto
# (In un caso reale, questo lo fa la fisica del cervello)
data = np.zeros((8, n_samples))
data[[0, 1, 4, 5], :] += source_signal_1 # Attività più frontale
data[[2, 3, 6, 7], :] += source_signal_2 # Attività più temporale
data += rng.randn(8, n_samples) * 1e-7 # Aggiunta rumore sensore
'''

# raw = mne.io.RawArray(data, info)
raw.set_eeg_reference('average', projection=True)

# =========================================================
# 2. MODELLO DIRETTO (FORWARD) E ROI
# =========================================================
print("2. Calcolo Forward Model e selezione ROI...")

# Scarica fsaverage
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = fs_dir.parent

# Source Space (oct6 è standard, circa 8196 dipoli)
src = mne.setup_source_space('fsaverage', spacing='oct6', subjects_dir=subjects_dir, add_dist=False)

# BEM Model
model = mne.make_bem_model(subject='fsaverage', ico=4, conductivity=(0.3, 0.006, 0.3), subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

# Forward Solution
fwd = mne.make_forward_solution(raw.info, trans='fsaverage', src=src, bem=bem, eeg=True, mindist=5.0)

# Caricamento Labels (Atlante)
labels = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=subjects_dir)

# Nomi delle ROI target (corrispondenti agli 8 elettrodi)
'''
FC3        | caudalmiddlefrontal-lh (62.6mm)          | superiorfrontal-lh (74.5mm)             
CP3        | postcentral-lh (57.9mm)                  | precentral-lh (58.9mm)                  
FT7        | parsopercularis-lh (42.1mm)              | rostralmiddlefrontal-lh (47.7mm)        
TP7        | supramarginal-lh (35.6mm)                | postcentral-lh (38.9mm)                 
FC4        | caudalmiddlefrontal-rh (62.8mm)          | superiorfrontal-rh (72.4mm)             
CP4        | precentral-rh (56.2mm)                   | postcentral-rh (57.3mm)                 
FT8        | parsopercularis-rh (41.1mm)              | parstriangularis-rh (43.1mm)            
TP8        | supramarginal-rh (28.5mm)                | postcentral-rh (36.8mm)                 
'''
target_labels_names = [
    # FC3
    'caudalmiddlefrontal-lh',   # (62.6mm)
    'superiorfrontal-lh',       # (74.5mm)
    # CP3
    'postcentral-lh',           # (57.9mm)
    'precentral-lh',            # (58.9mm)
    # FT7
    'parsopercularis-lh',       # (42.1mm)
    'rostralmiddlefrontal-lh',  # (47.7mm)
    # TP7
    'supramarginal-lh',         # (35.6mm)
    #'postcentral-lh',           # (38.9mm)
    # FC4
    'caudalmiddlefrontal-rh',   # (62.8mm)
    'superiorfrontal-rh',       # (72.4mm)
    # CP4
    'precentral-rh',            # (56.2mm)
    'postcentral-rh',           # (57.3mm)
    # FT8
    'parsopercularis-rh',       # (41.1mm)
    'rostralmiddlefrontal-rh',      # (43.1mm)
    # TP8
    'supramarginal-rh'         # (28.5mm)
    #'postcentral-rh'            # (36.8mm)
]

# Estraiamo gli oggetti Label nell'ordine esatto della lista nomi
target_labels = [next(l for l in labels if l.name == name) for name in target_labels_names]


# =========================================================
# 3. SOLUZIONE INVERSA E UNMIXING
# =========================================================
print("3. Calcolo Inversa e Unmixing sulle ROI...")

noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)
inverse_operator = make_inverse_operator(raw.info,
                                         fwd,
                                         noise_cov,
                                         fixed=False,
                                         loose=0.2,
                                         depth=0.8)

mne.write_forward_solution('modello_8ch-fwd.fif', fwd, overwrite=True)
# Salva su file .fif (standard MNE)
write_inverse_operator('modello_8ch-inv.fif', inverse_operator, overwrite=True)

########################################################################################################################
#   usa bem_8sources_sLORETA per calcolare le timeseries sorgente e plottare sulle ROI l'ampiezza media con elettrodi
########################################################################################################################
