
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURAZIONE ---
SIGNAL_LENGTH = 1024
CHANNELS = 8
LATENT_DIM = 100
# Assicurati che il percorso sia corretto
MODEL_PATH = "saved_models/generator_eeg.pth"

SAMPLING_RATE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. DEFINIZIONE DELLA CLASSE ---
# (Deve essere identica a quella usata durante il training)
class Generator(nn.Module):
    def __init__(self, latent_dim, signal_length, channels):
        super(Generator, self).__init__()
        self.signal_length = signal_length
        self.channels = channels
        self.lstm_input_size = 64

        self.linear = nn.Linear(latent_dim, signal_length * self.lstm_input_size)

        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)

        self.final_mapping = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, channels)
        )

    def forward(self, z):
        out = self.linear(z)
        out = out.view(out.shape[0], self.signal_length, self.lstm_input_size)
        lstm_out, _ = self.lstm(out)
        signal = self.final_mapping(lstm_out)
        signal = signal.transpose(1, 2)
        return signal


# --- 2. CARICAMENTO DEL MODELLO ---
def load_generator():
    print(f"Caricamento modello da {MODEL_PATH}...")

    # Inizializziamo il modello vuoto
    model = Generator(LATENT_DIM, SIGNAL_LENGTH, CHANNELS).to(device)

    # Carichiamo i pesi salvati
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"ERRORE: Non trovo il file '{MODEL_PATH}'. Verifica il percorso.")
        exit()

    # Mettiamo il modello in modalità "valutazione" (Disabilita dropout, ecc.)
    model.eval()
    return model


# --- 3. GENERAZIONE E VISUALIZZAZIONE ---

if __name__ == "__main__":
    generator = load_generator()

    # Quanti esempi (segnali) diversi vuoi generare?
    num_signals = 8

    # Creiamo rumore casuale
    noise = torch.randn(num_signals, LATENT_DIM).to(device)

    # Generiamo i dati
    with torch.no_grad():
        # Output shape: (num_signals, CHANNELS, SIGNAL_LENGTH)
        synthetic_signals = generator(noise).cpu().numpy()

    duration = SIGNAL_LENGTH / SAMPLING_RATE
    time_axis = np.linspace(0, duration, SIGNAL_LENGTH)

    print(f"Generazione grafici con durata: {duration} secondi (Freq: {SAMPLING_RATE} Hz)")

    # Valore per separare le linee (aumentalo se si sovrappongono troppo)
    offset_val = 6

    plt.figure(figsize=(12, 8))

    for i in range(num_signals):
        ax = plt.subplot(num_signals, 1, i + 1)
        current_data = synthetic_signals[i]

        for ch in range(CHANNELS):
            trace = current_data[ch, :]
            vertical_shift = ch * offset_val

            # QUI LA MODIFICA: Uso time_axis invece dell'indice automatico
            plt.plot(time_axis, trace + vertical_shift, label=f'Ch {ch + 1}')

            plt.text(- (duration * 0.02), vertical_shift, f"Ch{ch + 1}",
                     va='center', fontweight='bold', fontsize=9)

        plt.title(f"Segnale Sintetico #{i + 1}")
        plt.ylabel("Ampiezza (uV)")
        plt.grid(True, alpha=0.3)
        plt.yticks([])  # Nascondo i numeri Y perché l'offset li rende falsi

        # Etichetta X solo sull'ultimo grafico in basso
        if i == num_signals - 1:
            plt.xlabel("Tempo (secondi)")
            # Imposto i limiti dell'asse X esatti
            plt.xlim(0, duration)
        else:
            # Nascondo i numeri dell'asse X per i grafici superiori per pulizia
            plt.xticks([])
            plt.xlim(0, duration)

    plt.tight_layout()
    plt.show()

    print("Generazione completata.")

'''
# --- 3. GENERAZIONE CONCATENATA ---
if __name__ == "__main__":
    generator = load_generator()

    # Numero di segmenti da concatenare
    num_segments = 4

    # Generiamo il rumore per 4 segmenti
    noise = torch.randn(num_segments, LATENT_DIM).to(device)

    with torch.no_grad():
        # Output shape originale: (4, 4, 512) -> (Batch, Canali, Lunghezza)
        raw_signals = generator(noise).cpu().numpy()

    # --- CONCATENAZIONE ---
    # raw_signals è una lista di array 4x512.
    # Vogliamo unirli lungo l'asse del tempo (axis=1) per ottenere un array 4x2048
    # Sintassi: prendiamo ogni elemento del batch e lo incolliamo orizzontalmente
    long_signal = np.concatenate([raw_signals[i] for i in range(num_segments)], axis=1)

    print(f"Shape finale concatenata: {long_signal.shape}")
    # Dovrebbe stampare (4, 2048) se SIGNAL_LENGTH=512 e num_segments=4

    # --- CALCOLO ASSE TEMPORALE LUNGO ---
    total_points = long_signal.shape[1]
    total_duration = total_points / SAMPLING_RATE

    # Crea l'asse temporale da 0 a total_duration
    time_axis = np.linspace(0, total_duration, total_points)

    print(f"Durata totale segnale: {total_duration:.2f} secondi")

    # --- VISUALIZZAZIONE UNICA ---
    offset_val = 6  # Spaziatura verticale tra i canali

    plt.figure(figsize=(15, 6))  # Figura più larga per vedere meglio i dettagli

    # Iteriamo sui 4 canali
    for ch in range(CHANNELS):
        # Estraiamo i dati lunghi del canale corrente
        trace = long_signal[ch, :]

        # Calcoliamo lo spostamento verticale
        vertical_shift = ch * offset_val

        # Plottiamo
        plt.plot(time_axis, trace + vertical_shift, label=f'Ch {ch + 1}')

        # Etichetta canale a sinistra
        plt.text(- (total_duration * 0.01), vertical_shift, f"Ch{ch + 1}",
                 va='center', fontweight='bold', fontsize=10)

    # --- DECORAZIONI ---
    # Aggiungiamo linee verticali per mostrare dove sono stati incollati i segmenti
    segment_duration = SIGNAL_LENGTH / SAMPLING_RATE
    for i in range(1, num_segments):
        boundary_time = i * segment_duration
        plt.axvline(x=boundary_time, color='red', linestyle='--', alpha=0.3, linewidth=1)
        if ch == 0:  # Scriviamo solo una volta in alto
            plt.text(boundary_time, (CHANNELS * offset_val), "Join",
                     color='red', fontsize=8, ha='center')

    plt.title(f"Segnale EEG Sintetico Concatenato ({num_segments} segmenti uniti)")
    plt.xlabel("Tempo (secondi)")
    plt.ylabel("Ampiezza (con Offset)")
    plt.yticks([])  # Nascondiamo i numeri dell'asse Y
    plt.grid(True, alpha=0.3)
    plt.xlim(0, total_duration)

    plt.tight_layout()
    plt.show()

    print("Generazione e concatenazione completata.")
'''