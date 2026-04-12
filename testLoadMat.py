import scipy.io as sio
import numpy as np
import mne

# --- VARIABILI GLOBALI SPECIFICHE PER IL TUO FILE .MAT ---
MAT_DATA_KEY = 'data'

# "data" (time x number of channels): These are the data. sampled at 1000Hz
#     • scale factor: 1 amplifier unit = .0298 microvolts
#     • built-in band pass 0.15 to 200 Hz
# “stimsites” (1xN): “N” channels which were stimulated as part of a stimulation pair.
# “ecssites” (Mx2): Channel pairs where ECS produced interruption of naming during clinical mapping.
# The individualized MRI anatomy files are in the folder brains/, with files named – “##_brain.mat”, with variables:
# ◦ "brain": This is a structure representing the tessellated brain surface. It can easily be plotted with the CTMR
# package (see “Automated electrocorticographic electrode localization on individually rendered brain surfaces”
# by D Hermes, et al in Journal of Neuroscience Methods, 2009)
# ◦ "locs" (number of channels x 3): Electrode locations, for plotting on the rendered brain.


# 1. Frequenza di campionamento (sfreq)
SFREQ_FISSO = 1000.0

# 2. Lista dei nomi dei canali
# Se non hai i nomi esatti, puoi usare dei segnaposto come "ECoG 01", "ECoG 02", etc.
# L'importante è che la LUNGHEZZA di questa lista corrisponda al numero di canali in 'data'.
# Esempio per 64 canali:
# NOMI_CANALI_FISSI = [f'ECoG {i:02d}' for i in range(1, 65)]

# Se non conosci il numero di canali, dovremo ricavarlo dalla shape di 'data' DOPO il caricamento.
MAT_LOCS_KEY = 'locs'
MAT_DATA_KEY = 'data'

# Tipo di canale per MNE
MNE_CHANNEL_TYPE = 'ecog'

def load_brain_info(cnt_filepath: str):
    try:
        mat_contents = sio.loadmat(cnt_filepath)
        print(f"File '{cnt_filepath}' caricato con successo!")
        print(f"{mat_contents}")
    except Exception as e:
        print(f"ERRORE durante il caricamento del file .mat: {e}")
        return None



def load_ecog_data_mne_from_mat(cnt_filepath: str, fixed_sfreq: float, fixed_ch_names: list):
    """
    Carica i dati ECoG da un file .mat usando la chiave 'data'
    e parametri fissi per sfreq e nomi dei canali.
    """
    print(f"Caricamento da file ECoG .mat: {cnt_filepath}")

    # --- 1. CARICAMENTO DEL FILE .MAT ---
    # ... (Il codice di caricamento è lo stesso di prima, ma usa MAT_DATA_KEY) ...
    try:
        mat_contents = sio.loadmat(cnt_filepath)
        print(f"File '{cnt_filepath}' caricato con successo!")
    except Exception as e:
        print(f"ERRORE durante il caricamento del file .mat: {e}")
        return None

    # --- 2. ESTRAZIONE DEI PARAMETRI CHIAVE ---
    try:
        # Estrai la matrice dei dati
        data_matrix = mat_contents[MAT_DATA_KEY]
        sfreq = fixed_sfreq
        ch_names = fixed_ch_names
        num_expected_channels = len(ch_names)

        if data_matrix.shape[0] > data_matrix.shape[1] and data_matrix.shape[1] == num_expected_channels:
            # Se (Campioni x Canali) e il numero di canali è corretto, trasponi.
            print(f"La matrice dati ha forma {data_matrix.shape} (Campioni x Canali). Eseguo la trasposizione...")
            data_matrix = data_matrix.T  # Ora dovrebbe essere (Canali x Campioni)

        elif data_matrix.shape[0] == num_expected_channels and data_matrix.shape[1] > data_matrix.shape[0]:
            # Se è già (Canali x Campioni) e il numero di canali è corretto, non fare nulla.
            print(f"La matrice dati ha forma {data_matrix.shape} (Canali x Campioni). OK.")

        else:
            # Se la forma non corrisponde al numero di canali attesi su NESSUN asse
            print(
                f"ERRORE GRAVE: La forma dei dati ({data_matrix.shape}) non corrisponde al numero di canali atteso ({num_expected_channels}) su nessun asse.")
            print(
                "Ciò suggerisce che la variabile 'data' potrebbe contenere dati complessi o la lista NOMI_CANALI_FISSI è errata.")
            return None

        num_channels = data_matrix.shape[0]
        num_samples = data_matrix.shape[1]

        print(
            f"\nParametri utilizzati dopo la formattazione: {num_channels} canali, {num_samples} campioni, {sfreq} Hz")

    except KeyError as e:
        print(f"\nERRORE: La chiave {e} non è disponibile nel file .mat.")
        return None

    # --- 3. PULIZIA E FORMATTAZIONE DEI DATI ---

    # Verifica e Adattamento della Forma (Shape) dei Dati
    # MNE si aspetta (canali x tempo). Assumiamo che 'data' sia (campioni x canali)
    if data_matrix.shape[0] > data_matrix.shape[1] and data_matrix.shape[1] == len(ch_names):
        print(f"AVVISO: La forma della matrice dati è {data_matrix.shape}. Trasposizione in (canali x tempo)...")
        data_matrix = data_matrix.T
    elif data_matrix.shape[0] != len(ch_names):
        print(
            f"ERRORE GRAVE: Dopo l'estrazione, la dimensione dei canali ({data_matrix.shape[0]}) NON corrisponde al numero di nomi dei canali forniti ({len(ch_names)}).")
        # Tentativo di trasposizione per vedere se risolve
        data_matrix_T = data_matrix.T
        if data_matrix_T.shape[0] == len(ch_names):
            print("Trasposizione forzata effettuata.")
            data_matrix = data_matrix_T
        else:
            print("Impossibile far corrispondere i dati ai nomi dei canali. Verifica i valori NOMI_CANALI_FISSI.")
            return None

    num_channels = data_matrix.shape[0]
    num_samples = data_matrix.shape[1]

    print(f"\nParametri utilizzati: {num_channels} canali, {num_samples} campioni, {sfreq} Hz")

    # --- 4. PREPARAZIONE PER MNE (Creazione di info e raw) ---

    ch_types = [MNE_CHANNEL_TYPE] * num_channels

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )

    # MNE lavora in Volt (V). Se i tuoi dati sono in microVolt (uV), moltiplica per 1e-6
    # Esempio: data_matrix = data_matrix * 1e-6

    raw = mne.io.RawArray(data_matrix, info)

    print("Oggetto MNE RawArray creato con successo.")

    return {
        'raw': raw,
        'ch_names': ch_names,
        'template_len': num_samples
    }


# --- ESEMPIO DI UTILIZZO E ISTRUZIONI DI IMPLEMENTAZIONE ---
# 1. Definizione dei parametri fissi (ADATTALI!)
FS = 1000.0  # <--- IL VALORE REALE DEL TUO DATASET
# Se la tua matrice 'data' ha 96 colonne (canali)
CANALI = [f'ECoG {i:02d}' for i in range(1, 97)]  # Esempio per 96 canali

# 2. Percorso del file
file_ecog = 'ecogDataSets/speech_basic/speech_basic/data/jc_nouns.mat'
file_brain = 'ecogDataSets/speech_basic/speech_basic/brains/jc_brain.mat'

# chiamta funzione load brain
load_brain_info(file_brain)

# 3. Chiamata alla funzione
# mne_data = load_ecog_data_mne_from_mat(file_ecog, fixed_sfreq=FS, fixed_ch_names=CANALI)
# if mne_data:
#    print("Pronto per l'analisi con MNE:", mne_data['raw'])