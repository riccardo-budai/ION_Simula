
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

# --- CONFIGURAZIONE ---
SIGNAL_LENGTH = 1024
CHANNELS = 16
LATENT_DIM = 100
SAMPLING_RATE = 128
MODEL_PATH = "saved_models/generator_eeg.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Generate with device = {device}")


# --- 1. CLASSE GENERATORE ---
class Generator(nn.Module):
    def __init__(self, latent_dim, signal_length, channels):
        super(Generator, self).__init__()

        self.signal_length = signal_length
        self.channels = channels
        self.lstm_input_size = 64
        self.linear = nn.Linear(latent_dim, signal_length * self.lstm_input_size)
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=128, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.final_mapping = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(0.2, inplace=True),
                                           nn.Linear(128, channels))

    def forward(self, z):
        out = self.linear(z)
        out = out.view(out.shape[0], self.signal_length, self.lstm_input_size)
        lstm_out, _ = self.lstm(out)
        signal = self.final_mapping(lstm_out)
        signal = signal.transpose(1, 2)
        return signal


def load_generator():
    model = Generator(LATENT_DIM, SIGNAL_LENGTH, CHANNELS).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except:
        print("Modello non trovato, assicurati di averlo addestrato.")
        exit()
    return model


# --- 2. IL MODULO DI INIEZIONE ANOMALIE ---
class AnomalyInjector:
    def __init__(self, sampling_rate):
        self.fs = sampling_rate

    def modify_amplitude(self, signal_data, channels_idx, factor, start_idx=0, end_idx=None):
        modified_signal = signal_data.copy()
        if end_idx is None:
            end_idx = modified_signal.shape[1]
        for ch in channels_idx:
            modified_signal[ch, start_idx:end_idx] *= factor
        return modified_signal

    def inject_frequency(self, signal_data, channels_idx, freq_type, amplitude=0.5):
        modified_signal = signal_data.copy()
        length = modified_signal.shape[1]
        t = np.linspace(0, length / self.fs, length)

        if freq_type == 'delta':
            freq = 2.0
        elif freq_type == 'theta':
            freq = 5.5
        elif freq_type == 'alpha':
            freq = 10.5
        elif freq_type == 'beta':
            freq = 20.0
        else:
            freq = 10.0

        wave = amplitude * np.sin(2 * np.pi * freq * t)
        for ch in channels_idx:
            modified_signal[ch, :] += wave
        return modified_signal

    def add_spike_and_wave(self, signal_data, channels_idx, position_idx, amplitude=2.0, width=5):
        """
        Aggiunge un complesso Punta-Onda (Spike-and-Wave).
        Correction: Aggiunto int() per gestire width decimali (es. 1.5).
        """
        modified_signal = signal_data.copy()
        length = modified_signal.shape[1]
        x = np.arange(length)

        # --- 1. SPIKE ---
        # width può essere float (es. 1.5) per spike stretti
        gauss = amplitude * np.exp(-0.5 * ((x - position_idx) / width) ** 2)
        spike_shape = -np.diff(gauss, append=0) * (width / 1.5)
        spike_max_val = np.max(np.abs(spike_shape))

        # --- 2. ONDA LENTA (DELTA) ---
        delta_freq = 2.0
        wave_points = int((1.0 / delta_freq) * self.fs)
        t_wave = np.linspace(0, 1 * np.pi, wave_points)
        slow_wave = np.sin(t_wave) * (spike_max_val * 2)

        # --- 3. FIX ERRORE SLICE ---
        # width * 3 potrebbe dare un float (es. 4.5).
        # position_idx è int. La somma diventa float.
        # Dobbiamo forzare la conversione a int() per usarlo come indice di array.
        wave_start_idx = int(position_idx + (width * 3))

        full_wave_shape = np.zeros(length)
        end_idx = wave_start_idx + wave_points

        if wave_start_idx < length:
            points_to_copy = min(length, end_idx) - wave_start_idx
            # Ora wave_start_idx è un intero, quindi lo slicing funziona
            full_wave_shape[wave_start_idx: wave_start_idx + points_to_copy] = slow_wave[:points_to_copy]

        total_artifact = spike_shape + full_wave_shape

        for ch in channels_idx:
            modified_signal[ch, :] += total_artifact

        return modified_signal

    def add_spike(self, signal_data, channels_idx, position_idx, amplitude=3.0, width=1.5):
        """
        Aggiunge uno 'spike' (picco improvviso).
        Per stringerlo, abbassa il valore 'width'.
        Nota: width non dovrebbe scendere sotto 1.0 per evitare artefatti digitali.

        Default width originale: 5
        Nuovo width (3x più stretto): 1.5
        """
        modified_signal = signal_data.copy()
        length = modified_signal.shape[1]
        x = np.arange(length)

        # Creiamo la forma dello spike
        # Se width è piccolo (es. 1.5), la curva scende a zero molto velocemente
        spike_shape = amplitude * np.exp(-0.5 * ((x - position_idx) / width) ** 2)

        # La derivata (np.diff) rende lo spike bifasico (su e giù)
        # Quando stringi width, la pendenza aumenta, quindi l'ampiezza potrebbe variare.
        spike_shape = np.diff(spike_shape, append=0)

        # Compensazione opzionale: quando stringi molto lo spike con np.diff,
        # l'ampiezza visiva tende a ridursi leggermente o diventare troppo "secca".
        # Moltiplichiamo per un fattore correttivo se necessario, o aumentiamo l'amplitude in ingresso.
        spike_shape = spike_shape * 1.5

        for ch in channels_idx:
            modified_signal[ch, :] += spike_shape

        return modified_signal


# --- 3. ESECUZIONE E VISUALIZZAZIONE ---

if __name__ == "__main__":
    # 1. Genera segnale Base
    gen = load_generator()
    noise = torch.randn(1, LATENT_DIM).to(device)
    with torch.no_grad():
        base_signal = gen(noise).cpu().numpy().squeeze()  # (8, 1024)

    print(f"base signal shape = {base_signal.shape}")
    injector = AnomalyInjector(SAMPLING_RATE)

    # --- APPLICAZIONE ANOMALIE ---

    # A: Attenuazione Ch0
    signal_amp = injector.modify_amplitude(base_signal, channels_idx=[0], factor=0.5)

    # B: Beta Ch1
    signal_beta = injector.inject_frequency(base_signal, channels_idx=[1], freq_type='beta', amplitude=0.04)

    # C: Spike Ch2 (Cumulativo corretto)
    # Nota: passo il risultato del primo add_spike come input del secondo per non raddoppiare il segnale base
    temp_sig = injector.add_spike(base_signal, channels_idx=[2], position_idx=250, amplitude=5.0, width=1.5)
    signal_spike = injector.add_spike(temp_sig, channels_idx=[2], position_idx=350, amplitude=6.0, width=2.0)

    # D: Spike and Wave Ch3 (Cumulativo corretto)
    temp_sw = injector.add_spike_and_wave(base_signal, channels_idx=[3], position_idx=150, amplitude=4.0, width=1.5)
    signal_sw = injector.add_spike_and_wave(temp_sw, channels_idx=[3], position_idx=350, amplitude=4.0, width=1.5)

    # --- CREAZIONE LISTA SEGNALI PER PLOT ---
    # CORREZIONE QUI: Selezioniamo specificamente IL CANALE che ci interessa plottare
    # così la lista contiene solo array 1D (shape 1024,)

    final_signals_to_plot = [
        signal_amp[0, :],  # Prendo solo la riga 0
        signal_beta[1, :],  # Prendo solo la riga 1
        signal_spike[2, :],  # Prendo solo la riga 2
        signal_sw[3, :],  # Prendo solo la riga 3
        base_signal[4, :],  # Già 1D
        base_signal[5, :],
        base_signal[6, :],
        base_signal[7, :],
        base_signal[8, :],
        base_signal[9, :],
        base_signal[10, :],
        base_signal[11, :],
        base_signal[12, :],
        base_signal[13, :],
        base_signal[14, :],
        base_signal[15, :]
    ]

    titles = [
        "Ch0: Attenuazione",
        "Ch1: Iniezione Beta",
        "Ch2: Doppi Spikes",
        "Ch3: Spike and Wave multipli",
        "Ch4: Normale", "Ch5: Normale", "Ch6: Normale", "Ch7: Normale"
    ]

    # --- PLOT COMPARATIVO ---
    time_axis = np.linspace(0, SIGNAL_LENGTH / SAMPLING_RATE, SIGNAL_LENGTH)

    plt.figure(figsize=(12, 12))  # Aumentato altezza

    for i in range(CHANNELS):
        ax = plt.subplot(CHANNELS, 1, i + 1)

        # 1. Disegna il segnale NORMALE (base)
        plt.plot(time_axis, base_signal[i], color='gray', linestyle='--', alpha=0.5, label='Base')

        # 2. Disegna il segnale MODIFICATO
        # Ora current_trace è già un vettore (1024,)
        current_trace = final_signals_to_plot[i]

        # Disegno solo se è diverso dal base (per i primi 4 canali) oppure disegno sempre
        color = 'red' if i < 4 else 'green'
        label = 'Anomalia' if i < 4 else 'Normale'

        plt.plot(time_axis, current_trace, color=color, label=label)

        # Gestione sicura dei titoli
        if i < len(titles):
            plt.title(titles[i], fontsize=10)

        # Legenda solo nel primo grafico per non affollare
        if i == 0:
            plt.legend(loc='upper right', fontsize='small')

        plt.grid(True, alpha=0.3)
        plt.yticks([])  # Rimuovo asse Y per pulizia

        # Asse X solo in fondo
        if i == CHANNELS - 1:
            plt.xlabel("Tempo (s)")
        else:
            plt.xticks([])

    plt.tight_layout()
    plt.show()