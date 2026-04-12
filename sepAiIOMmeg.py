"""
Simulazione completa della risposta da stimolazione afferente della via sensitiva
sensory CAP dal nervo mediano stimolato al polso, con iniezione di dati target MEG.
"""
import json
import os
import logging
import sys
from datetime import datetime
import numpy as np
import random
import pyqtgraph as pg
from PySide6.QtUiTools import QUiLoader
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline, interp1d
from PySide6 import QtCore
from PySide6.QtCore import Qt, Signal, Slot, QFile
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget
import webbrowser as wb
import mne
from mne.datasets import hf_sef
from sepReportWin import ReportSepWindow

# --- PARAMETRI FISICI GLOBALI (Generazione di base, non usati per FP) ---
SIGMA_E = 0.3
DIPOLE_LENGTH_M = 100e-6
CURRENT_MAX_MICROA = 1000e-6
DIPOLE_POSITIONS = {
    'Source': np.array([DIPOLE_LENGTH_M / 2, 0.0, 0.0]),
    'Sink': np.array([-DIPOLE_LENGTH_M / 2, 0.0, 0.0])
}
# Canali MEG da pre-caricare per la selezione dinamica del SEP (S1 e CV7)
ALL_CORTICAL_MEG_CHANNELS = ['MEG0331', 'MEG0631', 'MEG1241', 'MEG0421',    # pre F3 F4
                             'MEG0431', 'MEG0721', 'MEG1141', 'MEG1141',    # central C3 C4
                             'MEG0741', 'MEG0731', 'MEG1821', 'MEG2211'     # central S1
                             ]


CV7_TARGET_CH_NAME = 'MEG0631'
S1_INITIAL_CH_NAME = 'MEG0721'  # Corrisponde a C3^ iniziale

CV7_VC_M_S = 65.0           # Velocità di Conduzione della via dorsale
CV7_DISTANCE_M = 0.6        # Distanza dal sito di stimolazione (es. polso -> CV7)
CV7_LATENCY_S = CV7_DISTANCE_M / CV7_VC_M_S # Latenza: ~0.012 s (13 ms)
CV7_DISTANCE_X_M = 0.04     # 4 cm dalla colonna vertebrale (per la registrazione superficiale)
CV7_Z_FIXED = 0.05          # Distanza verticale dell'elettrodo (superficie)



# --- FUNZIONI DI UTILITÀ GLOBALI ---
def calculate_field_potential_static(electrode_pos, current, sigma_e, dipole_pos_dict):
    """
    Calcola il potenziale di campo (FP) statico per una data posizione
    e una data corrente.
    """
    K = 1.0 / (4 * np.pi * sigma_e)
    FP_total = 0.0
    r_source = np.linalg.norm(electrode_pos - dipole_pos_dict['Source'])
    r_sink = np.linalg.norm(electrode_pos - dipole_pos_dict['Sink'])
    if r_source > 0:
        FP_total += K * (current / r_source)
    if r_sink > 0:
        FP_total += K * (-current / r_sink)
    return FP_total

def current_pulse(t, peak_time_s, duration_tau_s):
    """ Modella l'evoluzione temporale della corrente I(t). """
    t_minus_delay = t - 0.001
    t_minus_delay[t_minus_delay < 0] = 0
    tau_rise = peak_time_s / 2
    tau_decay = duration_tau_s
    I_t = (np.exp(-t_minus_delay / tau_decay) - np.exp(-t_minus_delay / tau_rise))
    max_val = np.max(I_t)
    return I_t / max_val if max_val > 0 else I_t

def single_fiber_ap(t, amplitude, duration_tau, delay):
    """ Calcola un singolo Potenziale d'Azione (SFAP) (usato per i CAP periferici). """
    t_shifted = t - delay
    A_pos = amplitude * 1.5
    sigma_pos = duration_tau * 0.5
    A_neg = amplitude * 0.5
    sigma_neg = duration_tau * 1.0
    phase_pos = A_pos * np.exp(-(t_shifted ** 2) / (2 * sigma_pos ** 2))
    phase_neg = A_neg * np.exp(-(t_shifted ** 2) / (2 * sigma_neg ** 2))
    return phase_pos - phase_neg

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def apply_spline_smoothing(data, time, smoothing_factor=5):
    n_points = len(data)
    sparse_indices = np.arange(0, n_points, smoothing_factor)
    if sparse_indices[-1] != n_points - 1:
        sparse_indices = np.append(sparse_indices, n_points - 1)
    cs = CubicSpline(time[sparse_indices], data[sparse_indices])
    return cs(time)

def apply_simple_smooth(data, window_len=15):
    if window_len % 2 == 0:
        window_len += 1
    w = np.ones(window_len, 'd')
    smoothed_data = np.convolve(data, w / w.sum(), mode='same')
    return smoothed_data

def prepare_target_sep_data(time_vector, sfreq):
    """
    Carica e interpola i segnali SEP target da MNE per tutti i canali utili.
    """
    print("\n--- Preparazione Dati SEP Target ---")
    try:
        fname_evoked = os.path.join(mne.datasets.hf_sef.data_path(), "MEG/subject_a/sef2_right-ave.fif")
        evoked = mne.Evoked(fname_evoked)
    except Exception as e:
        print(f"ERRORE MNE: Impossibile caricare i dati campione. {e}")
        return {}

    meg_channels_map = {}
    ch_names_to_load = ALL_CORTICAL_MEG_CHANNELS

    # Pre-carica tutti i canali target e li interpola
    for ch_name in ch_names_to_load:
        picks_ch = mne.pick_channels(evoked.ch_names, include=[ch_name])

        if picks_ch.size > 0:
            evoked_ch = evoked.copy().pick(picks_ch)
            # Estrazione del singolo canale [0, :] per ottenere l'array 1D
            ch_data = evoked_ch.data[0, :] * 1e12  # Scaling in unità arbitrarie per PyQtGraph

            # Interpolazione
            time_original = evoked_ch.times
            interp_func = interp1d(time_original, ch_data, kind='cubic', bounds_error=False, fill_value=0)
            ch_data_resampled = interp_func(time_vector)

            meg_channels_map[ch_name] = ch_data_resampled
        else:
            print(f"ATTENZIONE: Canale {ch_name} non trovato nel set dati. Usando zeri.")
            meg_channels_map[ch_name] = np.zeros_like(time_vector)

    # Assegnazione iniziale (S1 e CV7)
    s1_initial_data = meg_channels_map[S1_INITIAL_CH_NAME]
    cv7_initial_data = meg_channels_map[CV7_TARGET_CH_NAME]

    print(
        f"Segnali Target S1 ({S1_INITIAL_CH_NAME}) campionato a {len(s1_initial_data)} punti.")

    return {
        'S1': s1_initial_data,
        'CV7': cv7_initial_data,
        'MEG_CHANNELS': meg_channels_map
    }


# --- CLASSE WORKER PER IL THREAD SEPARATO ----------------------------------------------------------------------------
class CAPWorker(QtCore.QObject):
    data_ready = Signal(dict, int, float, dict)
    waterfall_trace_ready = Signal(dict)

    def __init__(self, simulator_params):
        super().__init__()

        for key, value in simulator_params.items():
            setattr(self, key, value)

        self.anomaly_config = simulator_params.get('anomaly_config')
        if self.anomaly_config:
            print(f"Worker: Anomalia configurata con trigger: {self.anomaly_config.get('trigger_type')}")
            # todo Imposta le variabili di iniezione dell'anomalia in base al JSON
            # Esempio:
            # self.trigger_type = self.anomaly_config.get('trigger_type')
            # self.anom_data = self.anomaly_config.get('anom_config', {})
        else:
            print("Worker: Nessuna anomalia attiva.")

        self.update_counter = 0
        self.cap_data_buffers = {dist: [] for dist in self.DISTANCES_CM}
        self.fp_data_buffer = []
        self.s1_data_buffer = []
        self.noise_level = 0.0
        self.FIFO_BLOCK_SIZE = 50

        self.current_filter_index = 0
        self.FS = simulator_params['FS']

        # PARAMETRI RUMORE
        self.noise_level = 1000  # CAP noise
        self.noise_level_CV7 = 0.5 # SEP/FP noise
        self.noise_level_SEP = 1

        # Carica i dati target all'inizializzazione del worker
        self.target_data = prepare_target_sep_data(self.time, self.FS)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._run_simulation_step)

        # todo load json anomaly file


    @Slot()
    def _init_timer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._run_simulation_step)

    @Slot(int)
    def start_simulation(self, interval_ms):
        self.timer.start(interval_ms)

    @Slot()
    def stop_simulation(self):
        self.timer.stop()

    @Slot(int)
    def update_filter_state(self, new_index):
        self.current_filter_index = new_index

    @Slot(str)
    def update_channel_target(self, new_ch_name):
        """ Aggiorna il segnale S1 target dal pool di canali pre-caricati. """
        if new_ch_name in self.target_data['MEG_CHANNELS']:
            # Sostituisce il segnale S1 corrente con il nuovo segnale
            self.target_data['S1'] = self.target_data['MEG_CHANNELS'][new_ch_name]
            self.s1_data_buffer = []
            print(f"Worker: Segnale target S1 aggiornato a {new_ch_name}.")
        else:
            print(f"Worker: Canale {new_ch_name} non trovato per l'aggiornamento.")

    @Slot(int, int)
    def update_averaging_params(self, new_n_avg, new_fifo_block_size):
        self.N_AVERAGES = new_n_avg
        self.FIFO_BLOCK_SIZE = new_fifo_block_size

    def _calculate_cap_at_distance(self, distance_m, noise_amplitude=0):
        """ Logica di calcolo del CAP per una distanza specifica. """
        cap_sum = np.zeros_like(self.time)

        for i in range(self.N_FIBERS):
            t_delay = (distance_m / self.fiber_vcs[i]) + self.MIN_DISPLAY_DELAY
            sfap_i = single_fiber_ap(self.time, self.fiber_amplitudes[i],
                                     self.fiber_taus[i], t_delay)
            cap_sum += sfap_i

        if noise_amplitude > 0:
            noise = np.random.normal(0, noise_amplitude, self.time_points)
            cap_sum += noise
        return cap_sum

    @staticmethod
    def _calculate_cervical_fp(time_vector, fs, noise_amplitude, min_display_delay):
        """
        Calcola il potenziale di campo cervicale (CV7).
        Ora riceve min_display_delay come argomento perché è una costante del worker.
        """
        # Variabili globali del dipolo (devono essere accessibili qui, altrimenti bisogna passarle)
        # Assumiamo che le costanti (SIGMA_E, CURRENT_MAX_MICROA, DIPOLE_POSITIONS, ecc.)
        # siano state definite all'inizio del file e siano accessibili.

        # 1. Calcola il potenziale di campo statico (il fattore spaziale costante)
        static_electrode_pos = np.array([CV7_DISTANCE_X_M, 0.0, CV7_Z_FIXED])
        # Se la funzione 'calculate_field_potential_static' è globale, chiamala direttamente
        DIPOLE_POSITIONS = {
            'Source': np.array([DIPOLE_LENGTH_M / 2, 0.0, 0.0]),
            'Sink': np.array([-DIPOLE_LENGTH_M / 2, 0.0, 0.0])
        }
        FP_static_V = calculate_field_potential_static(
            static_electrode_pos,
            CURRENT_MAX_MICROA,
            SIGMA_E,
            DIPOLE_POSITIONS
        )
        # 2. Calcola l'impulso di corrente nel tempo I(t), tenendo conto del ritardo
        CV7_TOTAL_DELAY = CV7_LATENCY_S + min_display_delay  # Usa l'argomento passato
        PEAK_TIME = 0.003
        DECAY_TIME = 0.004
        I_t_pulse = current_pulse(time_vector - CV7_TOTAL_DELAY, PEAK_TIME, DECAY_TIME)

        # 3. Combina Spazio e Tempo: FP(t)
        FP_temporal_V = FP_static_V * I_t_pulse

        # 4. Conversione per la visualizzazione e aggiunta del rumore
        FP_temporal_uV = FP_temporal_V * 1e4
        if noise_amplitude > 0:
            noise = np.random.normal(0, noise_amplitude / 2, len(time_vector))
            FP_temporal_uV += noise
        return FP_temporal_uV

    def _run_simulation_step(self):
        self.update_counter += 1
        current_avg_count = 0
        averaged_data = {}

        # --- 1. CAP PERIFERICO (GENERAZIONE + AVERAGING) ---
        for dist_cm, dist_m in zip(self.DISTANCES_CM, self.DISTANCES_M):
            new_cap_data = self._calculate_cap_at_distance(dist_m, noise_amplitude=self.noise_level)
            buffer = self.cap_data_buffers[dist_cm]
            buffer.append(new_cap_data)

            if len(buffer) > self.N_AVERAGES:
                for _ in range(self.FIFO_BLOCK_SIZE):
                    if buffer: buffer.pop(0)

            current_avg_count = len(buffer)
            if current_avg_count > 0:
                averaged_data[dist_cm] = np.mean(buffer, axis=0)
            else:
                averaged_data[dist_cm] = np.zeros_like(self.time)

            # APPLICAZIONE FILTRO (Logica invariata)
            if self.current_filter_index == 1:
                window_size = 3
                averaged_data[dist_cm] = apply_simple_smooth(averaged_data[dist_cm], window_len=window_size)
            elif self.current_filter_index == 2:
                averaged_data[dist_cm] = apply_spline_smoothing(averaged_data[dist_cm], self.time, smoothing_factor=5)
            elif self.current_filter_index == 3:
                averaged_data[dist_cm] = apply_bandpass_filter(averaged_data[dist_cm], 5, 3000, self.FS)
            elif self.current_filter_index == 4:
                averaged_data[dist_cm] = apply_bandpass_filter(averaged_data[dist_cm], 10, 1500, self.FS)

        # --- 2. FP CERVICALE (CV7) - INIEZIONE DATI TARGET ---
        averaged_fp_data = np.zeros_like(self.time)
        if 'CV7' in self.target_data:
            # noise_fp = np.random.normal(0, self.noise_level_CV7, self.time_points)
            # new_fp_data = self.target_data['CV7'] + noise_fp
            # Calcola il segnale FP a CV7 con rumore
            new_fp_data = self._calculate_cervical_fp(
                self.time,
                self.FS,
                self.noise_level_CV7,
                self.MIN_DISPLAY_DELAY  # Passiamo MIN_DISPLAY_DELAY, che è una variabile d'istanza
            )

            self.fp_data_buffer.append(new_fp_data)
            if len(self.fp_data_buffer) > self.N_AVERAGES:
                for _ in range(self.FIFO_BLOCK_SIZE):
                    if self.fp_data_buffer: self.fp_data_buffer.pop(0)

            averaged_fp_data = apply_simple_smooth(np.mean(self.fp_data_buffer, axis=0), window_len=5)

        # --- 3. FP CORTICALE (S1) - INIEZIONE DATI TARGET DINAMICI ---
        averaged_s1_data = np.zeros_like(self.time)
        if 'S1' in self.target_data:
            noise_s1 = np.random.normal(0, self.noise_level_SEP, self.time_points)
            new_s1_data = self.target_data['S1'] + noise_s1
            #
            self.s1_data_buffer.append(new_s1_data)
            if len(self.s1_data_buffer) > self.N_AVERAGES:
                for _ in range(self.FIFO_BLOCK_SIZE):
                    if self.s1_data_buffer: self.s1_data_buffer.pop(0)

            averaged_s1_data = apply_simple_smooth(np.mean(self.s1_data_buffer, axis=0), window_len=11)

        cervical_fp_data = {
            'CV7': averaged_fp_data,
            'S1': averaged_s1_data
        }

        # Emette il segnale completo al thread principale
        self.data_ready.emit(averaged_data, current_avg_count, self.noise_level, cervical_fp_data)

        # --- LOGICA WATERFALL (Invio dati combinati) ---
        if current_avg_count == self.N_AVERAGES:
            waterfall_data = averaged_data.copy()
            waterfall_data.update(cervical_fp_data)
            self.waterfall_trace_ready.emit(waterfall_data)
            # print(f"Waterfall: Emissione traccia a {self.N_AVERAGES} medie.")

    @Slot(int)
    def update_fiber_structure(self, new_n_fibers):
        self.N_FIBERS = new_n_fibers
        self.fiber_amplitudes = np.random.uniform(5, 15, self.N_FIBERS)
        self.fiber_taus = np.full(self.N_FIBERS, 0.0015)
        self.fiber_vcs = np.linspace(self.MIN_VC_MS, self.MAX_VC_MS, self.N_FIBERS)
        for dist in self.DISTANCES_CM:
            self.cap_data_buffers[dist] = []


# --- CLASSE SIMULATORE SEP from MEG recording #########################################################################
class SepSimulatorAI(QWidget):
    simulation_closed = Signal(str)
    def __init__(self, json_anomaly_path=None, learning_manager=None, current_anomaly=None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setWindowFlag(Qt.WindowType.Window)

        self.json_anomaly_path = json_anomaly_path
        self.anomaly_config = None
        self.learning_manager = learning_manager
        self.current_anomaly = current_anomaly

        self.load_json_anomaly()

        # --- CARICAMENTO UI ---
        # self.ui = uic.loadUi('res/sepAiIOMForm.ui', self)
        ui_file_path = "res/sepAiIOMForm.ui"
        ui_file = QFile(ui_file_path)

        if not ui_file:
            print(f"Errore: Impossibile aprire il file {ui_file_path}")
            sys.exit(-1)

        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        ui_file.close()

        self.setFixedSize(1000, 480)
        self.move(50, 415)
        percorso_logo = 'res/logos/logoSEP3.png'
        pixmap = QPixmap(percorso_logo)
        scaled_pixmap = pixmap.scaled(self.ui.labelLogo.size(),
                                      Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        self.ui.labelLogo.setPixmap(scaled_pixmap)
        # watrefall report window
        self.report_window = ReportSepWindow(parent=self)
        self.waterfall_log_timestamps = []

        # --- INIZIALIZZAZIONE VARIABILI ---
        self.MEG_EEG_MAP = {
            'F3': ALL_CORTICAL_MEG_CHANNELS[0], 'F4': ALL_CORTICAL_MEG_CHANNELS[1],
            'F3L': ALL_CORTICAL_MEG_CHANNELS[3], 'F4L': ALL_CORTICAL_MEG_CHANNELS[4],
            'C3': ALL_CORTICAL_MEG_CHANNELS[5], 'C4': ALL_CORTICAL_MEG_CHANNELS[6],
            'C3^': ALL_CORTICAL_MEG_CHANNELS[7], 'C4^': ALL_CORTICAL_MEG_CHANNELS[8],
        }

        self.interStim = 250
        self.CURRENT_MEG_CH = self.MEG_EEG_MAP['C3^']
        VC_REF_DIST_CM = 20.0 - 7.0  # Differenza tra Wrist (7.0) ed Elbow (20.0)
        self.VC_REF_DIST_M = VC_REF_DIST_CM / 100.0

        self.TRACE_MAP = [7.0, 20.0, 40.0, 55.0, 'CV7', 'S1']
        self.DISTANCES_CM = [7.0, 20.0, 40.0, 55.0]
        self.DISTANCES_M = [d / 100.0 for d in self.DISTANCES_CM]
        self.MIN_DISPLAY_DELAY = 0.005
        self.time_points = 500
        self.max_time = 0.075 + self.MIN_DISPLAY_DELAY
        self.time = np.linspace(0, self.max_time, self.time_points)
        self.FS = (self.time_points - 1) / self.max_time

        self.N_AVERAGES = 100
        self.FIFO_BLOCK_SIZE = 50
        self.MIN_VC_MS = 20.0
        self.MAX_VC_MS = 65.0
        self.N_FIBERS = 5000

        self.plot_visibility = {dist: True for dist in self.DISTANCES_CM}

        self.TRACE_KEY = self.TRACE_MAP[0]
        self.waterfall_offset = 0.0
        self.waterfall_plots = []
        self.max_waterfall_traces = 20
        self.main_averages_blocks = 0

        self.cv7_curve = None
        self.s1_curve = None
        self.cursor_1 = None
        self.cursor_2 = None

        # --- PREPARAZIONE WORKER ---
        self.worker_params = {
            'DISTANCES_CM': self.DISTANCES_CM, 'DISTANCES_M': self.DISTANCES_M,
            'MIN_DISPLAY_DELAY': self.MIN_DISPLAY_DELAY, 'N_FIBERS': self.N_FIBERS,
            'FS': self.FS, 'time': self.time, 'N_AVERAGES': self.N_AVERAGES,
            'FIFO_BLOCK_SIZE': self.FIFO_BLOCK_SIZE, 'MIN_VC_MS': self.MIN_VC_MS,
            'MAX_VC_MS': self.MAX_VC_MS,
            # Le fibre sono rigenerate nel worker
            'fiber_amplitudes': np.random.uniform(5, 15, self.N_FIBERS),
            'fiber_taus': np.full(self.N_FIBERS, 0.0015),
            'fiber_vcs': np.linspace(self.MIN_VC_MS, self.MAX_VC_MS, self.N_FIBERS),
            'anomaly_config': self.anomaly_config,
            'time_points': self.time_points
        }

        self.thread = QtCore.QThread()
        self.worker = CAPWorker(self.worker_params)
        self.worker.moveToThread(self.thread)

        self.worker.data_ready.connect(self.update_gui_plot)
        self.worker.waterfall_trace_ready.connect(self.add_waterfall_trace)

        self.thread.start()
        QtCore.QMetaObject.invokeMethod(self.worker, "_init_timer", QtCore.Qt.ConnectionType.QueuedConnection)

        self.anomaly_active = False
        self.ui.pushBtAnomaly.setText('ION-Sim Anomaly -> inactive')
        self.ui.comboAnomaly.setEnabled(False)

        # --- SETUP GUI ---
        self._setup_gui()
        self.setup_choice_buttons()

        self.response_start_time = None

    def _setup_gui(self):
        """ Configura l'interfaccia utente con PyQtGraph e le curve multiple. """

        # --- CONNESSIONI ---
        self.ui.pushBtExit.clicked.connect(self.closeWin)
        self.ui.pushBtStart.clicked.connect(self.startStop)
        self.ui.pushBtHelp.clicked.connect(self.handle_helpSep)
        self.ui.pushBtClear.clicked.connect(self.clear_waterfall_traces)
        self.ui.pushBtTimes.clicked.connect(self.show_report_window)
        self.ui.pushBtAnomaly.clicked.connect(self.handle_anomaly)
        self.ui.pushBtRetray.clicked.connect(self.retray_answer)

        self.ui.checkWrist.stateChanged.connect(lambda state: self.handle_rec_site_toggle(state, 7.0))
        self.ui.checkElbow.stateChanged.connect(lambda state: self.handle_rec_site_toggle(state, 20.0))
        self.ui.checkArmpit.stateChanged.connect(lambda state: self.handle_rec_site_toggle(state, 40.0))
        self.ui.checkErb.stateChanged.connect(lambda state: self.handle_rec_site_toggle(state, 55.0))

        # --- COMBO BOX SELEZIONE TRACCE (WATERFALL) ---
        self.ui.comboTrace.addItem('wrist');
        self.ui.comboTrace.addItem('elbow')
        self.ui.comboTrace.addItem('armsit');
        self.ui.comboTrace.addItem('Erb')
        self.ui.comboTrace.addItem('CV7');
        self.ui.comboTrace.addItem('C3/4')
        self.ui.comboTrace.currentIndexChanged.connect(self.handle_traceWaterfall)

        # --- COMBO BOX SELEZIONE CANALE (S1 TARGET) ---
        self.ui.comboChn.addItem('F3')
        self.ui.comboChn.addItem('F4')
        self.ui.comboChn.addItem('F3L')
        self.ui.comboChn.addItem('F4L')
        self.ui.comboChn.addItem('C3')
        self.ui.comboChn.addItem('C4')
        self.ui.comboChn.addItem('C3^')
        self.ui.comboChn.addItem('C4^')
        self.ui.comboChn.currentIndexChanged.connect(self.handle_chnSep)
        self.ui.comboChn.setCurrentIndex(0)

        # --- CONFIGURAZIONE PLOT CAP ---
        self.ui.plt_cap.setBackground(background='#000000')
        self.capPLT = self.ui.plt_cap.addPlot()
        self.capPLT.setLabel('left', r"Ampiezza (uV)");
        self.capPLT.setLabel('bottom', "Tempo (ms)")
        self.capPLT.showGrid(x=True, y=True);
        self.capPLT.enableAutoRange(axis='x', enable=True)

        self.cap_curves = {}
        colors = [(255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 150, 0)]
        for i, dist_cm in enumerate(self.DISTANCES_CM):
            curve = self.capPLT.plot(self.time, np.zeros_like(self.time),
                                     pen=pg.mkPen(color=colors[i], width=1), name=f"CAP a {dist_cm:.1f} cm")
            self.cap_curves[dist_cm] = curve
        self.capPLT.addLegend()

        # --- CONFIGURAZIONE PLOT SEP (CV7/S1) ---
        self.ui.plt_sep.setBackground(background='#000000')
        self.sepPLT = self.ui.plt_sep.addPlot()
        self.sepPLT.showGrid(x=True, y=True)
        self.cv7_curve = self.sepPLT.plot(self.time * 1000, np.zeros_like(self.time),
                                          pen=pg.mkPen(color='c', width=1), name="Field Potential CV7")
        self.s1_curve = self.sepPLT.plot(self.time * 1000, np.zeros_like(self.time),
                                         pen=pg.mkPen('w', width=1, style=QtCore.Qt.PenStyle.DotLine),
                                         name="Cortical FP (S1)")

        # --- CONFIGURAZIONE PLOT WATERFALL ---
        self.ui.plt_fall.setBackground(background='#000000')
        self.plt_fall = self.ui.plt_fall.addPlot()
        self.plt_fall.setLabel('bottom', "Tempo (ms)")
        self.plt_fall.getAxis('left').hide()
        self.plt_fall.showGrid(x=True, y=False);
        self.plt_fall.enableAutoRange(axis='y', enable=False)
        self.plt_fall.setYRange(0, self.max_waterfall_traces * 50)

        # --- CURSORI E FILTRI ---
        self.ui.spinBoxIsi.valueChanged.connect(self.handle_stimIsi)
        self.ui.spinBoxAvg.valueChanged.connect(self.handle_avg)
        self.ui.spinBoxOvl.valueChanged.connect(self.handle_fifo_block)
        self.ui.spinBoxFibers.valueChanged.connect(self.handle_num_fibers)

        self.cursor_1 = pg.InfiniteLine(movable=True, pen=pg.mkPen('r', width=2, style=QtCore.Qt.PenStyle.DashLine),
                                        label='T1: {value:.2f} ms', labelOpts={'position': 0.8, 'color': 'r'});
        self.cursor_1.setValue(self.MIN_DISPLAY_DELAY * 1000 + 1);
        self.capPLT.addItem(self.cursor_1)
        self.cursor_2 = pg.InfiniteLine(movable=True, pen=pg.mkPen('b', width=2, style=QtCore.Qt.PenStyle.DashLine),
                                        label='T2: {value:.2f} ms', labelOpts={'position': 0.2, 'color': 'b'});
        self.cursor_2.setValue(self.MIN_DISPLAY_DELAY * 1000 + 5);
        self.capPLT.addItem(self.cursor_2)

        self.ui.comboFilter.addItem('Raw signal');
        self.ui.comboFilter.addItem('Smooth')
        self.ui.comboFilter.addItem('Spline');
        self.ui.comboFilter.addItem('5-3000 Hz')
        self.ui.comboFilter.addItem('10-1500 Hz')
        self.ui.comboFilter.currentIndexChanged.connect(self.handle_filter);
        self.ui.comboFilter.setCurrentIndex(0)

        self.cursor_1.sigPositionChanged.connect(self.update_vc_measurement)
        self.cursor_2.sigPositionChanged.connect(self.update_vc_measurement)
        self.update_vc_measurement()

    def setup_choice_buttons(self):
        """Prepara i pulsanti delle alternative leggendoli dal file JSON."""
        # TODO numero di pulsanti max = 4 compresa la risposta corretta
        print(f"ANOMALY SET = {self.anomaly_config}")
        combo = self.ui.comboAnswers
        confirm_button = self.ui.pushBtConfirm
        if not self.anomaly_config or 'learning_assessment' not in self.anomaly_config:
            combo.hide()
            confirm_button.hide()
            return
        combo.show()
        confirm_button.show()
        combo.setEnabled(True)
        confirm_button.setEnabled(True)

        assessment_data = self.anomaly_config['learning_assessment']
        correct_answer = assessment_data.get('correct_answer')
        distractors = assessment_data.get('distractors', [])
        if not correct_answer:
            self.logger.error("Dati di apprendimento incompleti nel JSON: 'correct_answer' mancante.")
            combo.hide()
            confirm_button.hide()
            return
        # Mettiamo tutte le scelte in una lista e mescoliamole
        combo.clear()
        choices = [correct_answer] + distractors
        random.shuffle(choices)
        combo.addItem("...")
        combo.addItems(choices)
        try:
            confirm_button.clicked.disconnect()
        except TypeError:
            pass  # Nessuna connessione da rimuovere
        confirm_button.clicked.connect(self._handle_choice_confirmation)

    def _handle_choice_confirmation(self):
        """
        Questa funzione viene chiamata quando l'utente clicca su "Conferma Risposta".
        Valuta la scelta fatta nella ComboBox.
        """
        combo = self.ui.comboAnswers
        confirm_button = self.ui.pushBtConfirm

        # Recupera il testo dell'opzione attualmente selezionata
        chosen_action = combo.currentText()

        # 1. Controlla che l'utente abbia selezionato una risposta valida (non il placeholder)
        if combo.currentIndex() == 0:  # L'indice 0 è "Seleziona la tua risposta..."
            self.logger.warning("Nessuna risposta selezionata. Per favore, scegli un'opzione.")
            # Qui potresti mostrare un messaggio all'utente
            return

        if not self.learning_manager or not self.anomaly_config:
            return

        # Calcolo del tempo di risposta dal momento in cui l'anomalia è stata attivata
        tempo_risposta = None
        if self.response_start_time:
            tempo_risposta = (datetime.now() - self.response_start_time).total_seconds()
            self.logger.info(f"Tempo di risposta calcolato: {tempo_risposta:.2f} secondi.")

        # 2. Disabilita i controlli per evitare risposte multiple
        combo.setEnabled(False)
        confirm_button.setEnabled(False)
        # 3. Recupera la risposta corretta dal JSON caricato
        correct_action = self.anomaly_config['learning_assessment']['correct_answer']
        # 4. Il resto della logica di confronto e registrazione è IDENTICO a prima!
        if chosen_action == correct_action:
            is_correct = True
            punti = 100
            self.ui.pushBtRetray.setEnabled(False)
            self.logger.info(f"Risposta CORRETTA! Scelta: '{chosen_action}'")
            self.ui.textAnswers.append(f"{chosen_action} : CORRECT = points 100")
        else:
            is_correct = False
            punti = -20
            self.ui.pushBtRetray.setEnabled(True)
            self.logger.warning(f"Risposta SBAGLIATA. Scelta: '{chosen_action}', Corretta: '{correct_action}'")
            self.ui.textAnswers.append(f"{chosen_action} : ERROR = points -20")
        #
        self.learning_manager.record_decision(
            azione_presa=chosen_action,
            esito_corretto=is_correct,
            punti=punti,
            tempo_risposta_sec = tempo_risposta
        )

    def retray_answer(self):
        self.setup_choice_buttons()

    def load_json_anomaly(self):
        # json file
        if self.json_anomaly_path:
            try:
                with open(self.json_anomaly_path, 'r') as f:
                    self.anomaly_config = json.load(f)
                # print(f"{self.anomaly_config}")
                self.logger.info(f"EEG Simulator: Caricata configurazione anomalia da {self.json_anomaly_path}.")
            except json.JSONDecodeError as e:
                self.logger.error(f"Errore nella decodifica JSON per l'anomalia: {e}")
                self.anomaly_config = None  # Reimposta a None se c'è un errore
        #
        metadata = self.anomaly_config.get('metadata_db_mapping', {})
        timing = self.anomaly_config.get('timing_control', {})
        injection = self.anomaly_config.get('injection_parameters', {})
        #
        print("\n--- Parametri Anomalia Caricati ---")
        print(f"Codice Anomalia (label_id): {metadata.get('db_label_id', 'N/A')}")
        print(f"Schema Temporale: {timing.get('schema_temporale', 'N/A')}")
        print(f"Modalità Temporale: {timing.get('time_modal', 'N/A')}")
        print(f"Trigger Type (tipo): {metadata.get('db_trigger_type', 'N/A')}")
        print(f"Percentuale Apparizione: {injection.get('appearance_pct', 'N/A')}%")
        #
        scenario_config = self.anomaly_config.get('scenario_specific_config', {})
        print("\n--- Parametri Specifici dello Scenario ---")
        if scenario_config:
            # Iteriamo su tutte le coppie chiave-valore nel dizionario specifico
            for key, value in scenario_config.items():
                # Usiamo 'str()' per assicurarci che anche liste o numeri siano stampati correttamente
                print(f"  > {key}: {value}")
        else:
            print("Nessun parametro specifico per lo scenario trovato.")

    def handle_anomaly(self):
        if not self.anomaly_active:
            # **ATTIVAZIONE** della modalità Anomalia
            self.ui.pushBtAnomaly.setText('ION-Sim Anomaly: ACTIVE')
            # self.setGeometry(50, 415, 1100, 480)  # La dimensione maggiore
            self.setFixedSize(1150, 480)
            self.ui.comboAnomaly.setEnabled(True)
            self.anomaly_active = True
            self.response_start_time = datetime.now()
            self.ui.pushBtAnomaly.setStyleSheet("background-color: darkred;")
            self.logger.info(f"Tempo di risposta INIZIATO: {self.response_start_time}")
        else:
            # **DISATTIVAZIONE** della modalità Anomalia (se era già attiva)
            self.ui.pushBtAnomaly.setText('ION-Sim Anomaly -> inactive')
            self.setFixedSize(1000, 480)  # La dimensione minore
            self.ui.comboAnomaly.setEnabled(False)
            self.anomaly_active = False
            self.ui.pushBtAnomaly.setStyleSheet("background-color: darkgreen;")
    '''
    def _handle_user_decision(self, is_correct: bool):
        """
        Questa funzione viene chiamata quando l'utente prende una decisione.
        """
        if not self.learning_manager:
            print("LearningManager non è disponibile, impossibile registrare la decisione.")
            return
        if is_correct:
            azione = "Anomalia Identificata Correttamente"

            punti = 100
        else:
            azione = "Decisione Errata"
            punti = -20

        print(f"L'utente ha preso una decisione: {azione}")

        # Usiamo il manager per registrare tutto!
        success = self.learning_manager.record_decision(
            azione_presa=azione,
            esito_corretto=is_correct,
            punti=punti
        )
        if success:
            # Qui puoi mostrare un feedback all'utente (es. un messaggio "Corretto!" o "Sbagliato!")
            print("Decisione registrata con successo nel database.")
        else:
            print("Fallimento nella registrazione della decisione.")

        # Potresti voler fermare o resettare la simulazione qui
    '''
    '''
    def _handle_user_choice(self, chosen_action: str):
        """
        Questa funzione ora confronta la scelta dell'utente con la risposta
        corretta letta direttamente dal JSON.
        """
        if not self.learning_manager or not self.anomaly_config:
            return

        # Disabilitiamo i pulsanti
        # ...

        # Recuperiamo la risposta corretta dal JSON caricato
        correct_action = self.anomaly_config['learning_assessment']['correct_answer']

        # Il confronto rimane identico!
        if chosen_action == correct_action:
            is_correct = True
            punti = 100
        else:
            is_correct = False
            punti = -20

        self.learning_manager.record_decision(
            azione_presa=chosen_action,
            esito_corretto=is_correct,
            punti=punti
        )
    '''

    def show_report_window(self):
        """ Mostra la finestra dei report/log. """
        self.report_window.show()
        self.report_window.raise_()

    def handle_helpSep(self):
        # todo eventually passed argument as contextual file pdf to open
        pdf_path = 'help_docs/docs_simula/SEP_tutorial_Gemini.pdf'
        wb.open_new(pdf_path)

    def handle_chnSep(self, new_index):
        """ Gestisce la selezione del canale 10-20 (C3, C4, F3, F4) e invia il canale MEG equivalente al Worker. """
        eeg_name = self.ui.comboChn.currentText()
        meg_name = self.MEG_EEG_MAP.get(eeg_name, S1_INITIAL_CH_NAME)  # Fallback al canale iniziale
        self.ui.labelChannel.setText(f"channel = {meg_name}")
        self.CURRENT_MEG_CH = meg_name

        QtCore.QMetaObject.invokeMethod(self.worker, "update_channel_target",
                                        QtCore.Qt.ConnectionType.QueuedConnection,
                                        QtCore.Q_ARG(str, self.CURRENT_MEG_CH))

        self.clear_waterfall_traces()
        if self.ui.pushBtStart.text() == 'Stop':
            self.startStop();
            self.startStop()

    def handle_traceWaterfall(self, new_index):
        self.TRACE_KEY = self.TRACE_MAP[new_index]
        self.clear_waterfall_traces()
        print(f"Waterfall trace selezionata: {self.TRACE_KEY}")

    @Slot(dict)
    def add_waterfall_trace(self, waterfall_data):
        """ Slot per ricevere i dati mediati (a N_AVERAGES) e aggiungerli al grafico waterfall. """
        self.main_averages_blocks += 1
        timems = self.time * 1000

        if len(self.waterfall_plots) >= self.max_waterfall_traces:
            # rimuove il primo item aggiunto
            oldest_plot = self.waterfall_plots.pop(0)
            self.plt_fall.removeItem(oldest_plot)
            # remove oldest entry
            self.report_window.remove_oldest_waterfall_entry()

        if self.TRACE_KEY in waterfall_data:
            if self.TRACE_KEY == 'S1':
                data = waterfall_data[self.TRACE_KEY] * -1e3    # negativo in alto
            else:
                data = waterfall_data[self.TRACE_KEY]
            self.waterfall_offset_step = int(abs(float(np.max(data)) - float(np.min(data))) / 5)
            offset_data = data + self.waterfall_offset
            color = (200, 200, 255, 180)
            new_curve = self.plt_fall.plot(timems, offset_data, pen=pg.mkPen(color=color, width=1))
            self.waterfall_plots.append(new_curve)

            # aggiorna lo step dell'offeset con la nuova ampiezza del segnale picco/picco
            self.waterfall_offset += self.waterfall_offset_step
            # print(f"offset e step = {np.max(data)} {np.min(data)} {self.waterfall_offset} + {self.waterfall_offset_step}")

            # Registrazione del timestamp quando la traccia viene aggiunta
            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.report_window.add_waterfall_entry(current_time_str)  # <--- AGGIUNTA LOG WATERFALL
            self.report_window.add_full_log_entry(current_time_str)

            self.plt_fall.enableAutoRange(axis='x', enable=False)
            self.plt_fall.enableAutoRange(axis='y', enable=True)

        self.ui.lcdWaterFall.display(self.main_averages_blocks)

    def clear_waterfall_traces(self):
        """ Rimuove tutte le tracce dal plot waterfall e resetta il contatore dell'offset. """
        for curve in self.waterfall_plots:
            self.plt_fall.removeItem(curve)
        self.waterfall_plots = []
        self.waterfall_offset = 0.0
        self.plt_fall.enableAutoRange(axis='y', enable=True)

    def handle_filter(self, new_index):
        QtCore.QMetaObject.invokeMethod(self.worker, "update_filter_state",
                                        QtCore.Qt.ConnectionType.QueuedConnection, QtCore.Q_ARG(int, new_index))

    def update_vc_measurement(self):
        """ Calcola la differenza di tempo (delta T) tra i due cursori e l'equivalente VC. """
        if self.cursor_1 is None or self.cursor_2 is None: return
        t1_ms = self.cursor_1.value();
        t2_ms = self.cursor_2.value()
        delta_t_ms = abs(t2_ms - t1_ms);
        delta_t_s = delta_t_ms / 1000.0
        vc_ms = self.VC_REF_DIST_M / delta_t_s if delta_t_s != 0 else 0.0
        vc_text = f"VC misurata: {vc_ms:.2f} m/s"
        self.ui.labelLatency.setText(vc_text)

    def handle_rec_site_toggle(self, state, distance_cm):
        is_visible = (state == Qt.CheckState.Checked.value)
        self.plot_visibility[distance_cm] = is_visible
        if distance_cm in self.cap_curves:
            self.cap_curves[distance_cm].setVisible(is_visible)

    def handle_avg(self, new_value):
        self.N_AVERAGES = new_value
        QtCore.QMetaObject.invokeMethod(self.worker, "update_averaging_params",
                                        QtCore.Qt.ConnectionType.QueuedConnection,
                                        QtCore.Q_ARG(int, self.N_AVERAGES), QtCore.Q_ARG(int, self.FIFO_BLOCK_SIZE))

    def handle_fifo_block(self, new_value):
        self.FIFO_BLOCK_SIZE = new_value
        QtCore.QMetaObject.invokeMethod(self.worker, "update_averaging_params",
                                        QtCore.Qt.ConnectionType.QueuedConnection,
                                        QtCore.Q_ARG(int, self.N_AVERAGES), QtCore.Q_ARG(int, self.FIFO_BLOCK_SIZE))

    def handle_num_fibers(self, new_value):
        self.N_FIBERS = new_value * 1000;
        self.ui.label_fiber.setText(f"fibers = {self.N_FIBERS}")
        QtCore.QMetaObject.invokeMethod(self.worker, "update_fiber_structure",
                                        QtCore.Qt.ConnectionType.QueuedConnection, QtCore.Q_ARG(int, self.N_FIBERS))
        if self.ui.pushBtStart.text() == 'Stop': self.startStop(); self.startStop()

    def handle_stimIsi(self, new_value):
        new_value = round((1 / new_value) * 1000)
        self.ui.label_isi.setText(f"ISI = {new_value} ms");
        self.interStim = new_value
        if self.ui.pushBtStart.text() == 'Stop': self.startStop(); self.startStop()

    def closeEvent(self, event):
        # Emette il segnale con la chiave corretta prima della chiusura
        self.simulation_closed.emit('SEP')  # o 'SEPAS'
        self.closeWin()
        # Accetta l'evento di chiusura
        event.accept()

    def closeWin(self):
        QtCore.QMetaObject.invokeMethod(self.worker, "stop_simulation", QtCore.Qt.ConnectionType.QueuedConnection)
        self.worker.setParent(None)
        # self.worker.thread = None
        self.thread.quit()
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.close()

    def startStop(self):
        self.capPLT.enableAutoRange(axis='y', enable=True);
        self.capPLT.enableAutoRange(axis='x', enable=True)
        if self.ui.pushBtStart.text() == 'Start':
            self.ui.pushBtStart.setText('Stop')
            QtCore.QMetaObject.invokeMethod(self.worker, "start_simulation",
                                            QtCore.Qt.ConnectionType.QueuedConnection,
                                            QtCore.Q_ARG(int, self.interStim))
        else:
            self.ui.pushBtStart.setText('Start')
            QtCore.QMetaObject.invokeMethod(self.worker, "stop_simulation", QtCore.Qt.ConnectionType.QueuedConnection)

    @Slot(dict, int, float, dict)
    def update_gui_plot(self, averaged_data, current_avg_count, noise_level, cervical_fp_data):
        # current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        #self.report_window.add_full_log_entry(current_time_str)
        #
        timems = self.time * 1000
        for dist_cm, data in averaged_data.items():
            if dist_cm in self.cap_curves:
                self.cap_curves[dist_cm].setData(timems, data)
                self.cap_curves[dist_cm].setVisible(self.plot_visibility.get(dist_cm, True))

        self.sepPLT.invertY(True)
        if 'CV7' in cervical_fp_data:
            self.cv7_curve.setData(timems, (cervical_fp_data['CV7'] * -10) +1.5)    # 2.0 di offset

        if 'S1' in cervical_fp_data:
            self.s1_curve.setData(timems, cervical_fp_data['S1'] * 10)

        self.capPLT.setTitle(
            rf"CAP Averaging: {current_avg_count} / {self.N_AVERAGES} Campioni | Rumore Max: {noise_level:.1f} $\mu$V")