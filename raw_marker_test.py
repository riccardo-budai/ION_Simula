

import mne
import numpy as np
import matplotlib.pyplot as plt
import os  # Necessario per verificare il percorso del file

# --- PARAMETRI FISICI GLOBALI (Estratti dal tuo codice) ---
EEG_CHANNELS = ['FP2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'FP1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5']
N_EEG_CHANNELS = len(EEG_CHANNELS)
AUX_CHANNELS = ['EOG', 'ECG']
ALL_PLOT_CHANNELS = EEG_CHANNELS + AUX_CHANNELS
N_ALL_CHANNELS = len(ALL_PLOT_CHANNELS)

FS_EEG = 512
TIME_DURATION_S = 10.0
TIME_POINTS_EEG = int(FS_EEG * TIME_DURATION_S)
TIME_VECTOR_EEG = np.linspace(0, TIME_DURATION_S, TIME_POINTS_EEG, endpoint=False)


def load_eeg_data_mne_cnt(cnt_filepath: str, target_time_vector: np.ndarray, target_fs: float):
    """
    Carica un file .cnt usando MNE, lo ricampiona alla target_fs, estrae dati/eventi e visualizza.
    """
    if not os.path.exists(cnt_filepath):
        print(f"ERRORE: File non trovato al percorso specificato: {cnt_filepath}")
        return _generate_dummy_data(target_time_vector)  # Fallback

    print(f"Caricamento da file CNT (Neuroscan) con MNE: {cnt_filepath}")

    try:
        # 1. Caricamento del file CNT
        raw = mne.io.read_raw_cnt(cnt_filepath, preload=True, verbose=False)
        original_sfreq = raw.info['sfreq']

        # 2. Estrazione degli eventi PRIMA del resampling (MNE li scala internamente)
        events, event_id_map = mne.events_from_annotations(raw, verbose=False)
        print(f"EVENTS = {events} {len(event_id_map)}")

        raw.plot()
        plt.show()

        mne.viz.plot_events(events=events)

        '''
        # 3. Ricampionamento (Resampling) se necessario
        if original_sfreq != target_fs:
            print(f"Ricampionamento da {original_sfreq} Hz a {target_fs} Hz...")
            raw.resample(target_fs, npad="auto")

        # 4. Estrazione finale degli eventi ricalibrati dopo il resampling
        # Ricalcoliamo gli eventi nel nuovo spazio di campionamento
        events, event_id_map = mne.events_from_annotations(raw, verbose=False)
        '''
        # 5. Gestione della lunghezza temporale (Cropping/Padding)
        n_target_samples = len(target_time_vector)
        n_raw_samples = raw.n_times

        all_data, times = raw.get_data(return_times=True)
        all_data = all_data * -1e6  # Conversione in microVolt

        if n_raw_samples >= n_target_samples:
            # Cropping di dati ed eventi
            all_data = all_data[:, :n_target_samples]
            # if events.size > 0:
            #    events = events[events[:, 0] < n_target_samples]
        else:
            # Padding (gli eventi non sono influenzati dal padding)
            padding_len = n_target_samples - n_raw_samples
            all_data = np.pad(all_data, ((0, 0), (0, padding_len)), 'constant')

        # 6. Mappatura dei Canali (Logica invariata per l'output)
        raw_ch_names = raw.ch_names
        data_map = {name.upper(): all_data[i] for i, name in enumerate(raw_ch_names)}

        eeg_list = []
        for ch_name in EEG_CHANNELS:
            key = ch_name.upper()
            eeg_list.append(data_map[key] if key in data_map else np.zeros(n_target_samples))
        eeg_data_2d = np.array(eeg_list)

        eog_data = data_map.get('EOG', data_map.get('VEOG', np.zeros(n_target_samples)))
        ecg_data = data_map.get('ECG', data_map.get('EKG', np.zeros(n_target_samples)))

        # 7. Visualizzazione MNE (OPZIONALE PER DEBUG)
        if events.size > 0:
            print("Visualizzazione eventi MNE in corso (chiudi la finestra per continuare).")
            # Convertiamo l'array events in Annotazioni per la visualizzazione grafica MNE
            mne.io.set_annotations_from_events(raw, events, event_id=event_id_map, verbose=False)

            # Eseguiamo il plot finale (la raw ha ora la dimensione corretta)
            raw.plot(
                duration=TIME_DURATION_S,
                n_channels=8,
                scalings={'eeg': 50e-6, 'eog': 50e-6},
                show=True,
                block=True  # Mantiene la finestra aperta finché non la chiudi
            )
            plt.show()

        return {
            'eeg_data_2d': eeg_data_2d,
            'eog_v': eog_data,
            'ecg': ecg_data,
            'template_len': n_target_samples,
            'events': events,
            'event_id_map': event_id_map
        }

    except Exception as e:
        print(f"ERRORE durante il caricamento/generazione dati MNE: {e}")
        return _generate_dummy_data(target_time_vector)


def _generate_dummy_data(target_time_vector):
    """ Funzione helper per generare dati dummy in caso di errore """
    len_dummy = len(target_time_vector)
    return {
        'eeg_data_2d': np.random.normal(0, 1, (len(EEG_CHANNELS), len_dummy)),
        'eog_v': np.random.normal(0, 1, len_dummy),
        'ecg': np.random.normal(0, 1, len_dummy),
        'template_len': len_dummy,
        'events': np.array([]),
        'event_id_map': {}
    }


# --- ESEMPIO DI ESECUZIONE PER DEBUG ---

# Nota: Assicurati che il file 'eegDataSets/ALBAEGIT.CNT' esista!
worker_params = {
    'FS': FS_EEG,
    'time_vector': TIME_VECTOR_EEG,
    'eeg_channels': EEG_CHANNELS,
    'aux_channels': AUX_CHANNELS,
    'cnt_filepath': 'eegDataSets/ALBAEGIT.CNT',
    'timer_interval_ms': 100
}

# La funzione restituisce il dizionario 'loaded'
loaded = load_eeg_data_mne_cnt(
    worker_params['cnt_filepath'],
    worker_params['time_vector'],
    FS_EEG
)

# Esempio di utilizzo dei dati restituiti
print("\n--- Risultati Caricamento ---")
print(f"Eventi trovati: {len(loaded['events'])}")
print(f"Mappatura Eventi: {loaded['event_id_map']}")
print(f"Dimensione Dati EEG: {loaded['eeg_data_2d'].shape}")