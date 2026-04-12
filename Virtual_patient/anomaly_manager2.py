import json
import numpy as np
import random
import os
from typing import Dict, Any, List

# Assicurati che l'import corrisponda al nome del file creato sopra
from artifact_genIOM import ArtifactGenerator


class AnomalyInjector:
    """
    Gestisce il caricamento e l'iniezione dinamica di un'anomalia.
    Delega la manipolazione matematica a ArtifactGenerator.
    """

    def __init__(self, json_anomaly_id: str):
        """
        Inizializza l'iniettore caricando la configurazione JSON.
        """
        self.config: Dict[str, Any] = {}
        self.is_active = False
        self.start_time = 0.0

        # Gestione ArtifactGenerator (Lazy loading)
        self.artifact_gen = None

        # Costruisce il percorso completo del file JSON
        full_path = json_anomaly_id

        try:
            with open(full_path, 'r') as f:
                self.config = json.load(f)
            print(f"✅ Anomalia JSON '{json_anomaly_id}' caricata con successo.")
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            print(f"❌ ERRORE CRITICO: Impossibile caricare JSON: {full_path}. {e}")
            self.config = {}

        # --- ESTRAZIONE PARAMETRI ---
        self.anom_config = self.config.get('anom_config', {})
        timing_control = self.config.get('timing_control', {})

        self.trigger_type = timing_control.get('trigger_type', 'Timer')
        self.appearance_pct = timing_control.get('appearance_pct', 100)
        self.time_model = timing_control.get('time_model', 'single')
        self.trigger_time_s = timing_control.get('trigger_time_s', 5.0)
        self.repeat_interval_s = timing_control.get('repeat_interval_s', 10.0)
        self.duration_s = self.anom_config.get('duration_s', 5.0)  # La durata è una proprietà fisica dell'anomalia

        # --- VARIABILI DI STATO ---
        self.is_armed = False
        self.arming_time = 0.0
        self.has_triggered_random = False
        self.has_completed_single_run = False
        self.next_trigger_time = self.trigger_time_s

        print(f"Injector init: time_model={self.time_model}, trigger_type={self.trigger_type}")

    def arm_trigger(self, arm_time: float):
        """ 'Arma' il sistema di trigger impostando il tempo assoluto. """
        if self.is_armed:
            return
        self.is_armed = True
        self.arming_time = arm_time
        self.next_trigger_time = self.arming_time + self.trigger_time_s
        print(f"🔩 Anomaly ARMED at {arm_time:.1f}s. First check at {self.next_trigger_time:.2f}s")

    def reset(self):
        """ Resetta l'iniettore. """
        self.is_active = False
        self.is_armed = False
        self.arming_time = 0.0
        self.has_completed_single_run = False
        self.next_trigger_time = self.trigger_time_s
        print("🔄 Anomaly Injector RESET.")

    def _check_trigger(self, current_time: float) -> bool:
        """ Logica temporale per decidere se attivare l'anomalia. """
        if not self.is_armed:
            return False
        if self.is_active:
            return False
        if self.time_model == 'single' and self.has_completed_single_run:
            return False

        should_trigger = False

        # Logica Timer
        if self.trigger_type == 'Timer':
            if current_time >= self.next_trigger_time:
                should_trigger = True

        # Logica Random
        elif self.trigger_type == 'Randomized':
            if current_time >= self.next_trigger_time:
                # Se è 'single' e non ha ancora triggerato
                if self.time_model == 'single' and not self.has_triggered_random:
                    should_trigger = random.randint(1, 100) <= self.appearance_pct
                    self.has_triggered_random = True  # Segna come tentato
                # Se è ripetitiva
                elif self.time_model == 'ripetitiva':
                    should_trigger = random.randint(1, 100) <= self.appearance_pct

        # Aggiornamento tempi per modelli ripetitivi
        if should_trigger:
            if self.time_model == 'ripetitiva':
                self.next_trigger_time = current_time + self.repeat_interval_s
        elif self.time_model == 'ripetitiva' and current_time >= self.next_trigger_time:
            # Se non ha triggerato (es. random fallito), aggiorna comunque il timer
            self.next_trigger_time = current_time + self.repeat_interval_s

        return should_trigger

    def apply_anomaly(self, current_time: float, output_data: np.ndarray, time_points: np.ndarray,
                      channel_labels: List[str] = None) -> np.ndarray:
        """
        Gestisce il ciclo di vita dell'anomalia e delega la modifica dei dati.
        Args:
            current_time: tempo simulazione attuale
            output_data: matrice dati (n_canali, n_samples)
            time_points: vettore tempo
            channel_labels: lista dei nomi dei canali ["C3", "C4"...]
        """
        # --- 1. Inizializzazione Lazy del Generator ---
        if self.artifact_gen is None and len(time_points) > 1:
            dt = time_points[1] - time_points[0]
            if dt > 0:
                fs = 1.0 / dt
                self.artifact_gen = ArtifactGenerator(fs)
            else:
                pass  # Aspettiamo dati migliori

        # --- 2. Gestione Stato Attivo/Inattivo ---
        # Verifica fine anomalia
        if self.is_active:
            if current_time >= self.start_time + self.duration_s:
                self.is_active = False
                print(f"🛑 Anomalia terminata a {current_time:.2f} s.")
                if self.time_model == 'single':
                    self.has_completed_single_run = True
                return output_data  # Ritorna pulito appena finisce

        # Verifica inizio anomalia
        elif self._check_trigger(current_time):
            self.is_active = True
            self.start_time = current_time
            print(f"🔥 Anomalia innescata a {current_time:.2f} s.")

        # --- 3. Iniezione Dati ---
        if self.is_active:
            return self._inject_data(output_data, time_points, channel_labels)

        return output_data

    def _inject_data(self, data: np.ndarray, time_points: np.ndarray, channel_labels: List[str]) -> np.ndarray:
        """
        Collega l'Injector all'ArtifactGenerator.
        """
        if self.artifact_gen is None:
            return data

            # Prepariamo i parametri per il generatore
        gen_params = self.anom_config.copy()

        # FORZIAMO 'active': True perché l'Injector ha già deciso che è tempo di agire
        gen_params['active'] = True

        # --- COMPATIBILITÀ RETROATTIVA ---
        if 'type' not in gen_params:
            if 'amplitude_factor' in gen_params or 'latency_shift_ms' in gen_params:
                gen_params['type'] = 'EP_MODIFICATION'

        # Chiamata al core DSP passando anche i nomi dei canali
        try:
            return self.artifact_gen.apply_anomaly(data, time_points, gen_params, channel_names=channel_labels)
        except Exception as e:
            print(f"Errore in ArtifactGenerator: {e}")
            return data

    # Metodi accessori per interfaccia utente
    def get_expected_action(self) -> str:
        return self.anom_config.get('expected_action', 'IGNORE')

    def get_points_value(self) -> int:
        return self.config.get('metadata_ui', {}).get('points_value', 5)

    def get_difficulty_level(self) -> int:
        return self.config.get('metadata_ui', {}).get('difficulty_level', 1)