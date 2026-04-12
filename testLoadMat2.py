import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import mne


# Variabili specifiche per l'estrazione delle localizzazioni
MAT_LOCS_KEY = 'locs'
MAT_DATA_KEY = 'data'
SFREQ_FISSO = 1000.0  # Assumiamo ancora 1000.0 Hz per coerenza, da verificare

# Tipo di canale per MNE
MNE_CHANNEL_TYPE = 'ecog'


def load_ecog_locs_and_create_names(brain_filepath: str):
    """
    Carica le localizzazioni degli elettrodi dal file 'brain' .mat,
    crea i nomi dei canali come segnaposto e restituisce le coordinate.
    """
    print(f"Caricamento delle localizzazioni dal file: {brain_filepath}")

    try:
        mat_contents = sio.loadmat(brain_filepath)
    except Exception as e:
        print(f"ERRORE durante il caricamento del file 'brain' .mat: {e}")
        return None, None

    if MAT_LOCS_KEY not in mat_contents:
        print(f"ERRORE: La chiave '{MAT_LOCS_KEY}' non è stata trovata nel file.")
        return None, None

    # Estrazione della matrice delle localizzazioni (N_elettrodi x 3)
    locs_matrix = mat_contents[MAT_LOCS_KEY]

    # Determinazione del numero di elettrodi
    num_electrodes = locs_matrix.shape[0]

    # Creazione dei nomi dei canali come segnaposto (es. ECoG 01, ECoG 02, ...)
    # Iniziamo da 1 (range(1, N+1)) e usiamo 3 cifre di riempimento se N > 99
    ch_names = [f'ECoG {i:03d}' for i in range(1, num_electrodes + 1)]

    print(f"\nLocalizzazioni estratte per {num_electrodes} elettrodi.")

    return locs_matrix, ch_names


def load_ecog_data_with_locs(data_filepath: str, brain_filepath: str, sfreq: float):
    """
    Carica i dati ECoG e le localizzazioni, poi crea l'oggetto MNE RawArray.
    """

    # --- 1. CARICAMENTO DELLE LOCALIZZAZIONI E NOMI CANALI ---
    locs_matrix, ch_names = load_ecog_locs_and_create_names(brain_filepath)
    if locs_matrix is None:
        return None

    num_channels = len(ch_names)

    # --- 2. CARICAMENTO DEI DATI ECoG (dal file di dati precedente) ---
    try:
        mat_contents = sio.loadmat(data_filepath)
        data_matrix = mat_contents[MAT_DATA_KEY]
    except Exception as e:
        print(f"ERRORE durante il caricamento dei dati da {data_filepath}: {e}")
        return None

    # --- 3. FORMATTAZIONE DEI DATI (Assumendo forma (Tempo x Canali) e trasponiamo) ---
    if data_matrix.shape[0] > data_matrix.shape[1] and data_matrix.shape[1] == num_channels:
        print(f"Trasposizione dei dati da {data_matrix.shape} in (Canali x Tempo)...")
        data_matrix = data_matrix.T * 1e-8  # mne needs amplitudes in Volt 1e-6
    elif data_matrix.shape[0] != num_channels:
        print(
            f"ERRORE GRAVE: Numero di canali nei dati ({data_matrix.shape[0]}) non corrisponde a 'locs' ({num_channels}).")
        return None

    # --- 4. PREPARAZIONE PER MNE (RawArray e Montaggio) ---

    # Creazione dell'oggetto Info di MNE
    ch_types = [MNE_CHANNEL_TYPE] * num_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Creazione del RawArray
    raw = mne.io.RawArray(data_matrix, info)

    # Creazione dell'oggetto Montaggio (Montage) da coordinate 3D (in mm o cm - MNE usa metri)
    # Assumiamo che le tue coordinate siano in millimetri (tipico per Brainstorm/FieldTrip)
    # Dobbiamo convertirle in metri (unità standard MNE)
    locs_in_meters = locs_matrix / 1000.0

    # Mappa dei canali: {Nome_Canale: [x, y, z]}
    dig_ch_pos = dict(zip(ch_names, locs_in_meters))

    # Creazione del Montaggio
    montage = mne.channels.make_dig_montage(ch_pos=dig_ch_pos, coord_frame='mri')  # 'mri' è comune per ECoG

    # Applicazione del Montaggio ai dati Raw
    raw.set_montage(montage)

    print("Oggetto MNE RawArray con Montaggio 3D creato con successo. ✅")
    return raw


# --- ESEMPIO DI UTILIZZO ---

# 1. Definisci i percorsi dei file
file_ecog_data = 'ecogDataSets/speech_basic/speech_basic/data/jc_nouns.mat'
file_ecog_brain = 'ecogDataSets/speech_basic/speech_basic/brains/jc_brain.mat'

# 2. Chiama la funzione principale
raw_with_montage = load_ecog_data_with_locs(
    data_filepath=file_ecog_data,
    brain_filepath=file_ecog_brain,
    sfreq=SFREQ_FISSO  # Ricorda di verificare SFREQ_FISSO!
)

if raw_with_montage:
    print(f"\nTipo di oggetto finale: {type(raw_with_montage)}")
    # Ora puoi plottare la localizzazione 3D degli elettrodi della grid subdurale
    raw_with_montage.plot_sensors(kind='3d', title='ECoG Electrode Locations')
    raw_with_montage.plot()
    plt.show()