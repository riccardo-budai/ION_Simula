import time
import numpy as np
from PySide6.QtCore import QThread, Signal


class LocalGeneratorWorker(QThread):
    """
    Generatore di segnali simulati per il Self-Training.
    Non richiede rete.
    """
    data_signal = Signal(object)  # Invia il dato (lista o float) alla GUI

    def __init__(self, module_type, config=None):
        super().__init__()
        self.module_type = module_type
        self.config = config if config else {}
        self.running = True

    def run(self):
        # Configurazioni di base
        srate = 500  # Hz standard
        chunk_size = 10  # Campioni per pacchetto
        t = 0

        # Parametri simulazione
        noise_level = self.config.get("noise", 0.1)
        anomaly = self.config.get("anomaly", "Normal")

        while self.running:
            data = []

            # --- LOGICA EEG ---
            if self.module_type == "EEG":
                # Base sinusoide (Alpha wave 10Hz)
                base = np.sin(np.linspace(t, t + chunk_size, chunk_size) * 0.1)

                # Se Anomalia "Seizure", aggiungi picchi e frequenza alta
                if anomaly == "Seizure" and (t % 500 < 200):
                    base += np.sin(np.linspace(t, t + chunk_size, chunk_size) * 0.5) * 5.0

                noise = np.random.normal(0, noise_level, chunk_size)
                data = (base + noise).tolist()

            # --- LOGICA SEP (Somatosensory) ---
            elif self.module_type == "SEP":
                data = np.random.normal(0, 0.05, chunk_size).tolist()  # Rumore di fondo
                # Genera un picco ripetitivo (simula lo stimolo)
                cycle = t % 500  # Ogni 500 campioni (1 secondo)
                if 20 < cycle < 40:  # Latenza N20
                    amp = 0.5 if anomaly == "Low Amplitude" else 5.0
                    # Crea un picco gaussiano manuale
                    peak = [amp] * chunk_size
                    data = peak

            # --- LOGICA MEP (Motor) ---
            elif self.module_type == "MEP":
                data = np.random.normal(0, 0.02, chunk_size).tolist()
                # Trigger casuale o a comando (qui simulato ciclico)
                if t % 1000 < 20:
                    data = [10.0] * chunk_size  # Grande risposta motoria

            # --- LOGICA VITALS ---
            elif self.module_type == "VITALS":
                # Invia [HR, SpO2]
                hr = 70 + np.random.normal(0, 2)
                spo2 = 98 + np.random.normal(0, 0.5)

                if anomaly == "Bradycardia": hr = 45

                # Vitals mandano solo 1 campione per ciclo (non chunk)
                data = [hr, spo2]
                # Rallenta il loop per i vitali (1Hz)
                time.sleep(0.9)

                # Emissione dati
            self.data_signal.emit(data)

            # Avanzamento tempo e sleep per rispettare il sampling rate
            t += chunk_size
            if self.module_type != "VITALS":
                time.sleep(1.0 / (srate / chunk_size))

    def stop(self):
        self.running = False
        self.wait()