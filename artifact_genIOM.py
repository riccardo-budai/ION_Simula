import numpy as np
import scipy.signal
import scipy.ndimage
import random
from typing import Any, List, Dict


class ArtifactGenerator:
    """
    Gestisce la generazione e l'iniezione di artefatti e anomalie cliniche.
    Include la logica per selezionare quali canali colpire (Targeting).
    """

    def __init__(self, fs: float):
        self.fs = fs

        # Definizioni dei gruppi muscolari per il targeting rapido
        # Puoi espandere questa lista con i nomi reali usati nel tuo DB
        self.MUSCLE_GROUPS = {
            "PROX": ["DELTOID", "BICEPS", "TRICEPS", "QUAD", "HAMSTRING", "GLUTEUS", "TRAPEZIUS"],
            "DISTAL": ["APB", "ADM", "TIBIALIS", "GASTRO", "AH", "TA", "FDI"],
            "UPPER": ["DELTOID", "BICEPS", "TRICEPS", "APB", "ADM", "FDI", "TRAPEZIUS"],
            "LOWER": ["QUAD", "HAMSTRING", "GLUTEUS", "TIBIALIS", "GASTRO", "AH", "TA"],
            "LEFT": ["L ", "Left"],         # Keyword per lato sinistro
            "RIGHT": ["R ", "Right"]        # Keyword per lato destro
        }

    def _is_channel_targeted(self, channel_name: str, target_setting: Any) -> bool:
        """
        Verifica se il canale corrente deve ricevere l'anomalia.
        """
        # 1. Caso "ALL" (Default)
        if target_setting == "ALL" or target_setting is None:
            return True

        # 2. Caso Lista specifica (es. ["C3", "C4"])
        if isinstance(target_setting, list):
            return channel_name in target_setting

        # 3. Caso Gruppi definiti (PROX, DISTAL) o keywords
        if isinstance(target_setting, str):
            # Controllo diretto
            if target_setting == channel_name:
                return True
            # Controllo gruppi
            if target_setting in self.MUSCLE_GROUPS:
                keywords = self.MUSCLE_GROUPS[target_setting]
                return any(k.upper() in channel_name.upper() for k in keywords)

        return False

    def apply_anomaly(self, data_matrix: np.ndarray, time_vector: np.ndarray, params: Dict,
                      channel_names: List[str] = None) -> np.ndarray:
        """
        Applica l'anomalia sui canali selezionati della matrice dati.
        Gestisce automaticamente input 1D (singolo canale) e 2D (multicanale).
        """
        if not params.get('active', False):
            return data_matrix

        # --- 1. GESTIONE DIMENSIONI (FIX PER L'ERRORE) ---
        input_was_1d = False
        if data_matrix.ndim == 1:
            input_was_1d = True
            # Trasformiamo il 1D (samples,) in 2D (1, samples) per uniformare la logica
            data_matrix = data_matrix[np.newaxis, :]

        # Se i dati sono 2D ma channel_names non è coerente, fix rapido
        n_channels = data_matrix.shape[0]

        # --- 2. GESTIONE NOMI CANALI ---
        if channel_names is None:
            target = "ALL"
            # Generiamo nomi fittizi se non passati
            fake_names = [f"Ch_{i}" for i in range(n_channels)]
        else:
            target = params.get('target_signal', 'ALL')
            fake_names = channel_names

            # Sicurezza: se ho 1 canale dati ma mi passi 10 nomi (o viceversa), evito crash
            if len(fake_names) != n_channels:
                # Se è 1D, prendiamo solo il primo nome o usiamo un placeholder
                if input_was_1d and len(fake_names) >= 1:
                    fake_names = [fake_names[0]]
                else:
                    # Fallback brutale per evitare index error
                    fake_names = [f"Ch_{i}" for i in range(n_channels)]

        # Copia i dati per sicurezza
        modified_matrix = data_matrix.copy()

        # Determina il tipo di anomalia
        atype = params.get('type', '').upper()

        # --- 3. CICLO DI ELABORAZIONE ---
        for i in range(n_channels):
            ch_name = fake_names[i]

            # Controllo Target
            if channel_names is not None and not self._is_channel_targeted(ch_name, target):
                continue  # Salta questo canale

            # Estrai la riga singola (Ora sicuro perché la matrice è sicuramente 2D)
            single_signal = modified_matrix[i, :]

            # Applicazione anomalia _ singola ---
            if atype == 'EP_MODIFICATION':
                modified_matrix[i, :] = self._modify_ep(single_signal, params)

            elif atype == 'BOVIE':
                modified_matrix[i, :] = self._inject_bovie(single_signal, time_vector, params)

            elif atype == 'DRILL':
                modified_matrix[i, :] = self._inject_drill(single_signal, time_vector, params)

            elif atype == 'EPILEPSY':
                modified_matrix[i, :] = self._inject_epilepsy(single_signal, time_vector, params)

            elif atype == 'EMG_DISCHARGE':
                modified_matrix[i, :] = self._inject_emg_discharge(single_signal, params)

            elif atype == 'SPECTRAL_CHANGE':
                modified_matrix[i, :] = self._modify_spectrum(single_signal, params)

        # --- 4. RITORNO AL FORMATO ORIGINALE ---
        if input_was_1d:
            # Se era entrato come 1D, deve uscire come 1D (n_samples,)
            return modified_matrix[0]
        else:
            return modified_matrix

    # -------------------------------------------------------------------------
    # IMPLEMENTAZIONE ALGORITMI DSP
    # -------------------------------------------------------------------------

    def _modify_ep(self, signal, params):
        """ Modifica Ampiezza e Latenza (shift) """
        out_sig = signal.copy()

        # 1. Ampiezza
        amp_factor = params.get('amplitude_factor', 1.0)
        out_sig = out_sig * amp_factor

        # 2. Latenza
        latency_shift_ms = params.get('latency_shift_ms', 0.0)
        if latency_shift_ms != 0:
            shift_samples = int((latency_shift_ms / 1000.0) * self.fs)
            start_shift = int((3.0 / 1000.0) * self.fs)
            if shift_samples > 0:
                # Ritardo: pad a sinistra con zeri, taglia a destra
                out_sig = np.pad(out_sig, (shift_samples, start_shift), mode='edge')[:len(signal)]
            elif shift_samples < 0:
                # Anticipo
                shift_samples = abs(shift_samples)
                out_sig = np.pad(out_sig, (0, shift_samples), mode='edge')[shift_samples:]

        return out_sig

    def _inject_bovie(self, signal, time_vector, params):
        amplitude = params.get('amplitude', 50.0)
        burst_prob = params.get('probability', 0.05)

        if np.random.random() < burst_prob:
            noise = np.random.normal(0, 1, len(signal))
            envelope = scipy.signal.windows.tukey(len(signal), alpha=0.5)
            bovie_sig = noise * amplitude * envelope
            limit = 1000
            bovie_sig = np.clip(bovie_sig, -limit, limit)
            return signal + bovie_sig
        return signal

    def _inject_drill(self, signal, time_vector, params):
        freq = params.get('frequency', 200.0)
        amp = params.get('amplitude', 50.0)
        drill_noise = np.sin(2 * np.pi * freq * time_vector) * amp
        drill_noise += np.sin(2 * np.pi * (freq * 2) * time_vector) * (amp * 0.3)
        return signal + drill_noise

    def _inject_epilepsy(self, signal, time_vector, params):
        freq = params.get('frequency', 3.0)
        amp = params.get('amplitude', 300.0)

        period = 1.0 / freq
        phase = np.fmod(time_vector, period)
        spike_width = 0.05

        spikes = np.exp(-((phase - (period / 2)) ** 2) / (2 * (spike_width / 4) ** 2))
        wave = np.sin(2 * np.pi * freq * time_vector)

        epilepsy_sig = (spikes * amp) + (wave * (amp * 0.3))
        return signal + epilepsy_sig

    def _inject_emg_discharge(self, signal, params):
        pattern = params.get('pattern', 'NEUROTONIC')
        amp = params.get('amplitude', 100.0)
        prob = params.get('density', 0.1)

        if pattern == 'NEUROTONIC':
            noise = np.random.normal(0, 1, len(signal))
            b, a = scipy.signal.butter(2, 100, 'high', fs=self.fs)
            high_freq_noise = scipy.signal.lfilter(b, a, noise)
            mask = (np.random.random(len(signal)) < prob).astype(float)
            mask = scipy.ndimage.binary_dilation(mask, iterations=int(self.fs * 0.05)).astype(float)
            return signal + (high_freq_noise * amp * mask)

        elif pattern == 'MUAP':
            # Semplice implementazione MUAP
            num_spikes = int(len(signal) * prob * 0.01)
            spike_locs = np.random.randint(0, len(signal), num_spikes)
            muap_sig = np.zeros_like(signal)
            muap_template = np.array([-0.5, 1.0, -0.5]) * amp
            for loc in spike_locs:
                if loc < len(signal) - 3:
                    muap_sig[loc:loc + 3] += muap_template
            return signal + muap_sig

        return signal

    def _modify_spectrum(self, signal, params):
        mode = params.get('mode', 'NONE')
        out_sig = signal.copy()
        if mode == 'SLOWING':
            b, a = scipy.signal.butter(2, 12, 'low', fs=self.fs)
            out_sig = scipy.signal.lfilter(b, a, out_sig)
        return out_sig
