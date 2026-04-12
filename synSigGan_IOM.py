import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE ---
# Parametri del segnale
SIGNAL_LENGTH = 1024    # Lunghezza del segnale
CHANNELS = 16            # Numero di canali (es. 4 elettrodi)
LATENT_DIM = 100        # Input rumore
BATCH_SIZE = 64
LR = 0.0002
EPOCHS = 200

# Se hai una GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo in uso: {device}")


# --- 1. IL GENERATORE ---
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


# --- 2. IL DISCRIMINATORE ---
class Discriminator(nn.Module):
    def __init__(self, signal_length, channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(256 * (signal_length // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, signal):
        return self.model(signal)


# --- DATASET LOADER ---
class EEGDataset(Dataset):
    def __init__(self, csv_file, signal_length, channels_to_use):
        # Caricamento dati grezzi
        raw_data = pd.read_csv(csv_file).values

        # Selezione canali
        self.data = raw_data[:, :channels_to_use]
        print(f"Dati caricati. Shape: {self.data.shape}")

        self.signal_length = signal_length

        # Normalizzazione
        mean = np.mean(self.data)
        std = np.std(self.data)
        self.data = (self.data - mean) / (std + 1e-6)

        # Calcolo numero segmenti
        self.num_samples = len(self.data) // signal_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.signal_length
        end = start + self.signal_length
        segment = self.data[start:end]
        segment_tensor = torch.from_numpy(segment).float()
        # Trasponi per PyTorch: (Canali, Lunghezza)
        segment_tensor = segment_tensor.transpose(0, 1)
        return segment_tensor


# --- CONFIGURAZIONE DATALOADER ---
# Assicurati che il percorso del file sia corretto
dataset = EEGDataset(
    csv_file="eegDataSets/TUGI01_Q1.csv",
    signal_length=SIGNAL_LENGTH,
    channels_to_use=CHANNELS
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# --- 3. INIZIALIZZAZIONE MODELLI ---
generator = Generator(LATENT_DIM, SIGNAL_LENGTH, CHANNELS).to(device)
discriminator = Discriminator(SIGNAL_LENGTH, CHANNELS).to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))


# --- 4. TRAINING LOOP ---
def train_gan(epochs=10):
    print("Inizio addestramento...")

    # Liste per salvare l'andamento delle loss per i grafici futuri
    loss_d_history = []
    loss_g_history = []

    for epoch in range(epochs):
        for i, real_data in enumerate(dataloader):
            real_data = real_data.to(device)
            current_batch_size = real_data.size(0)

            # --- A. Addestramento Discriminatore ---
            optimizer_D.zero_grad()

            # Etichette
            label_real = torch.ones(current_batch_size, 1).to(device)
            label_fake = torch.zeros(current_batch_size, 1).to(device)

            # Loss su dati veri
            output_real = discriminator(real_data)
            loss_real = criterion(output_real, label_real)

            # Loss su dati falsi
            noise = torch.randn(current_batch_size, LATENT_DIM).to(device)
            fake_data = generator(noise)
            output_fake = discriminator(fake_data.detach())
            loss_fake = criterion(output_fake, label_fake)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # --- B. Addestramento Generatore ---
            optimizer_G.zero_grad()
            output_fake_for_G = discriminator(fake_data)
            loss_G = criterion(output_fake_for_G, label_real)  # Inganniamo D
            loss_G.backward()
            optimizer_G.step()

        # Stampa e salvataggio storico
        print(f"[Epoca {epoch + 1}/{epochs}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        loss_d_history.append(loss_D.item())
        loss_g_history.append(loss_G.item())

    print("Addestramento completato.")
    return generator, loss_d_history, loss_g_history


# --- ESECUZIONE ---
# Nota: aumento le epoche a 50 per vedere risultati migliori
trained_gen, d_losses, g_losses = train_gan(epochs=100)

# Salvataggio
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")
torch.save(trained_gen.state_dict(), "saved_models/generator_eeg.pth")
torch.save(discriminator.state_dict(), "saved_models/discriminator_eeg.pth")
print("Modelli salvati.")


# --- 5. VISUALIZZAZIONE CON OFFSET ---
def plot_generated_signals(generator, device, latent_dim, channels, offset=3):
    """
    Genera un segnale e lo plotta separando i canali verticalmente.
    Args:
        offset (float): Quanto spazio verticale mettere tra un canale e l'altro.
                        Se i segnali si sovrappongono ancora, aumenta questo numero.
    """
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(1, latent_dim).to(device)
        # Output shape: (1, Channels, Length) -> rimuoviamo il batch con squeeze
        synthetic_signal = generator(noise).cpu().numpy().squeeze()

    plt.figure(figsize=(12, 6))

    # Ciclo sui canali per applicare l'offset
    # synthetic_signal shape ora è (Channels, Length) -> es. (4, 512)
    for i in range(channels):
        # Selezioniamo il canale i-esimo
        channel_data = synthetic_signal[i, :]

        # Aggiungiamo l'offset: i * offset
        # Canale 0 -> +0
        # Canale 1 -> +3
        # Canale 2 -> +6 ...
        shifted_data = channel_data + (i * offset)

        plt.plot(shifted_data, label=f"Canale {i + 1}")

    plt.title("Segnale EEG/EMG Sintetico (Multi-canale)")
    plt.xlabel("Campioni Temporali")
    plt.ylabel("Ampiezza (con Offset)")

    # Mettiamo la legenda a destra
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Eseguiamo la funzione di plot
# Puoi cambiare 'offset=5' se i segnali sono troppo vicini o troppo lontani
plot_generated_signals(trained_gen, device, LATENT_DIM, CHANNELS, offset=5)