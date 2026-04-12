"""
Simulazione completa della risposta da stimolazione afferente della via sensitiva
sensory CAP dal nervo mediano stimolato al polso, con iniezione di dati target MEG.
File aggiornato con gestione AnomalyInjector e LearningManager.
CORRETTO: Ordine inizializzazione GUI per evitare crash all'avvio.
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

from scipy.interpolate import interp1d
from PySide6 import QtCore
from PySide6.QtCore import Qt, Signal, Slot, QFile
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QMessageBox, QVBoxLayout
import webbrowser as wb
import mne
# from mne.datasets import hf_sef
from sepReportWin import ReportSepWindow
from anomaly_manager2 import AnomalyInjector
from utils_funcIOM import apply_simple_smooth, apply_spline_smoothing, apply_bandpass_filter

# --- PARAMETRI FISICI GLOBALI -------------------------------------------------
SIGMA_E = 0.3
DIPOLE_LENGTH_M = 100e-6
CURRENT_MAX_MICROA = 1000e-6
DIPOLE_POSITIONS = {
    'Source': np.array([DIPOLE_LENGTH_M / 2, 0.0, 0.0]),
    'Sink': np.array([-DIPOLE_LENGTH_M / 2, 0.0, 0.0])
}
# Canali MEG da pre-caricare per la selezione dinamica del SEP (S1 e CV7)
ALL_CORTICAL_MEG_CHANNELS = ['MEG0331', 'MEG0631', 'MEG1241', 'MEG0421',  # pre F3 F4
                             'MEG0431', 'MEG0721', 'MEG1141', 'MEG1141',  # central C3 C4
                             'MEG0741', 'MEG0731', 'MEG1821', 'MEG2211'  # central S1
                             ]

CV7_TARGET_CH_NAME = 'MEG0631'
S1_INITIAL_CH_NAME = 'MEG0721'  # Corrisponde a C3^ iniziale

CV7_VC_M_S = 65.0  # Velocità di Conduzione della via dorsale
CV7_DISTANCE_M = 0.6  # Distanza dal sito di stimolazione (es. polso -> CV7)
CV7_LATENCY_S = CV7_DISTANCE_M / CV7_VC_M_S  # Latenza: ~0.012 s (13 ms)
CV7_DISTANCE_X_M = 0.04  # 4 cm dalla colonna vertebrale (per la registrazione superficiale)
CV7_Z_FIXED = 0.05  # Distanza verticale dell'elettrodo (superficie)


# --- FUNZIONI DI UTILITÀ GLOBALI ---
def calculate_field_potential_static(electrode_pos, current, sigma_e, dipole_pos_dict):
    """
    Calcola il potenziale di campo (FP) statico per una data posizione e corrente.
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
    """ Calcola un singolo Potenziale d'Azione (SFAP). """
    t_shifted = t - delay
    A_pos = amplitude * 1.5
    sigma_pos = duration_tau * 0.5
    A_neg = amplitude * 0.5
    sigma_neg = duration_tau * 1.0
    phase_pos = A_pos * np.exp(-(t_shifted ** 2) / (2 * sigma_pos ** 2))
    phase_neg = A_neg * np.exp(-(t_shifted ** 2) / (2 * sigma_neg ** 2))
    return phase_pos - phase_neg


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
            ch_data = evoked_ch.data[0, :] * 1e12  # Scaling
            time_original = evoked_ch.times
            interp_func = interp1d(time_original, ch_data, kind='cubic', bounds_error=False, fill_value=0)
            ch_data_resampled = interp_func(time_vector)
            meg_channels_map[ch_name] = ch_data_resampled
        else:
            print(f"ATTENZIONE: Canale {ch_name} non trovato nel set dati. Usando zeri.")
            meg_channels_map[ch_name] = np.zeros_like(time_vector)

    s1_initial_data = meg_channels_map[S1_INITIAL_CH_NAME]
    cv7_initial_data = meg_channels_map[CV7_TARGET_CH_NAME]

    print(f"Segnali Target S1 ({S1_INITIAL_CH_NAME}) caricati.")

    return {
        'S1': s1_initial_data,
        'CV7': cv7_initial_data,
        'MEG_CHANNELS': meg_channels_map
    }


# --- CLASSE WORKER PER IL THREAD SEPARATO -----------------------------------------------------------------------------
class CAPWorker(QtCore.QObject):
    # data_ready = Signal(dict, int, float, dict)
    #waterfall_trace_ready = Signal(dict)
    data_ready = Signal(object, int, float, object)
    waterfall_trace_ready = Signal(object)
    anomaly_triggered = Signal()

    def __init__(self, simulator_params):
        super().__init__()

        self.prev_anomaly_active = False

        for key, value in simulator_params.items():
            setattr(self, key, value)

        # --- GESTIONE ANOMALIA (Iniezione Fisica) ---
        self.anomaly_config = simulator_params.get('anomaly_config')
        self.json_anomaly_path = simulator_params.get('json_anomaly_path')

        self.injector = None
        if self.json_anomaly_path:
            print(f"Worker: Inizializzazione AnomalyInjector con {self.json_anomaly_path}")
            try:
                self.injector = AnomalyInjector(self.json_anomaly_path)
                # Resetta e arma subito (o gestisci l'armamento via segnale se preferisci trigger manuali)
                self.injector.reset()
                self.injector.arm_trigger(arm_time=0.0)
            except Exception as e:
                print(f"Worker Error: Fallita inizializzazione AnomalyInjector: {e}")
                self.injector = None
        else:
            print("Worker: Nessuna anomalia attiva (modalità normale).")

        self.simulation_time = 0.0  # Tempo trascorso in secondi
        self.interval_s = 0.250  # Valore di default, aggiornato in start_simulation

        self.update_counter = 0
        self.cap_data_buffers = {dist: [] for dist in self.DISTANCES_CM}
        self.fp_data_buffer = []
        self.s1_data_buffer = []
        self.FIFO_BLOCK_SIZE = 50

        self.current_filter_index = 0
        self.FS = simulator_params['FS']

        # PARAMETRI RUMORE BASE
        self.noise_level = 100  # CAP noise
        self.noise_level_CV7 = 0.5  # SEP noise
        self.noise_level_SEP = 1.0

        # Carica i dati target
        self.target_data = prepare_target_sep_data(self.time, self.FS)

    @Slot()
    def _init_timer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._run_simulation_step)

    @Slot(int)
    def start_simulation(self, interval_ms):
        self.interval_s = interval_ms / 1000.0
        self.simulation_time = 0.0
        if self.injector:
            self.injector.reset()
            self.injector.arm_trigger(0.0)
        self.timer.start(interval_ms)

    @Slot()
    def stop_simulation(self):
        self.timer.stop()

    @Slot(int)
    def update_filter_state(self, new_index):
        self.current_filter_index = new_index

    @Slot(str)
    def update_channel_target(self, new_ch_name):
        if new_ch_name in self.target_data['MEG_CHANNELS']:
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
        static_electrode_pos = np.array([CV7_DISTANCE_X_M, 0.0, CV7_Z_FIXED])
        DIPOLE_POSITIONS = {
            'Source': np.array([DIPOLE_LENGTH_M / 2, 0.0, 0.0]),
            'Sink': np.array([-DIPOLE_LENGTH_M / 2, 0.0, 0.0])
        }
        FP_static_V = calculate_field_potential_static(
            static_electrode_pos, CURRENT_MAX_MICROA, SIGMA_E, DIPOLE_POSITIONS
        )
        CV7_TOTAL_DELAY = CV7_LATENCY_S + min_display_delay
        PEAK_TIME = 0.003
        DECAY_TIME = 0.004
        I_t_pulse = current_pulse(time_vector - CV7_TOTAL_DELAY, PEAK_TIME, DECAY_TIME)
        FP_temporal_V = FP_static_V * I_t_pulse
        FP_temporal_uV = FP_temporal_V * 1e4
        if noise_amplitude > 0:
            noise = np.random.normal(0, noise_amplitude / 2, len(time_vector))
            FP_temporal_uV += noise
        return FP_temporal_uV

    def _run_simulation_step(self):
        self.update_counter += 1
        self.simulation_time += self.interval_s

        current_avg_count = 0
        averaged_data = {}

        is_active_now = False
        if self.injector:
            is_active_now = self.injector.is_active

        # Se l'anomalia è appena partita (era False, ora è True)
        if is_active_now and not self.prev_anomaly_active:
            self.anomaly_triggered.emit()
            print("Worker: Segnale anomaly_triggered EMESSO.")

        self.prev_anomaly_active = is_active_now

        # --- 1. CAP PERIFERICO ---
        for dist_cm, dist_m in zip(self.DISTANCES_CM, self.DISTANCES_M):
            raw_cap_data = self._calculate_cap_at_distance(dist_m, noise_amplitude=self.noise_level)

            # [INIEZIONE ANOMALIA]
            if self.injector:
                raw_cap_data = self.injector.apply_anomaly(self.simulation_time, raw_cap_data, self.time)

            buffer = self.cap_data_buffers[dist_cm]
            buffer.append(raw_cap_data)

            if len(buffer) > self.N_AVERAGES:
                for _ in range(self.FIFO_BLOCK_SIZE):
                    if buffer: buffer.pop(0)

            current_avg_count = len(buffer)
            if current_avg_count > 0:
                averaged_data[dist_cm] = np.mean(buffer, axis=0)
            else:
                averaged_data[dist_cm] = np.zeros_like(self.time)

            # FILTRI
            if self.current_filter_index == 1:
                averaged_data[dist_cm] = apply_simple_smooth(averaged_data[dist_cm], window_len=3)
            elif self.current_filter_index == 2:
                averaged_data[dist_cm] = apply_spline_smoothing(averaged_data[dist_cm], self.time, smoothing_factor=5)
            elif self.current_filter_index == 3:
                averaged_data[dist_cm] = apply_bandpass_filter(averaged_data[dist_cm], 5, 3000, self.FS)
            elif self.current_filter_index == 4:
                averaged_data[dist_cm] = apply_bandpass_filter(averaged_data[dist_cm], 10, 1500, self.FS)

        # --- 2. FP CERVICALE (CV7) ---
        averaged_fp_data = np.zeros_like(self.time)
        if 'CV7' in self.target_data:
            new_fp_data = self._calculate_cervical_fp(
                self.time, self.FS, self.noise_level_CV7, self.MIN_DISPLAY_DELAY
            )

            # [INIEZIONE ANOMALIA]
            if self.injector:
                new_fp_data = self.injector.apply_anomaly(self.simulation_time, new_fp_data, self.time)

            self.fp_data_buffer.append(new_fp_data)
            if len(self.fp_data_buffer) > self.N_AVERAGES:
                for _ in range(self.FIFO_BLOCK_SIZE):
                    if self.fp_data_buffer: self.fp_data_buffer.pop(0)

            averaged_fp_data = apply_simple_smooth(np.mean(self.fp_data_buffer, axis=0), window_len=5)

        # --- 3. FP CORTICALE (S1) ---
        averaged_s1_data = np.zeros_like(self.time)
        if 'S1' in self.target_data:
            noise_s1 = np.random.normal(0, self.noise_level_SEP, self.time_points)
            new_s1_data = self.target_data['S1'] + noise_s1

            # [INIEZIONE ANOMALIA]
            if self.injector:
                new_s1_data = self.injector.apply_anomaly(self.simulation_time, new_s1_data, self.time)

            self.s1_data_buffer.append(new_s1_data)
            if len(self.s1_data_buffer) > self.N_AVERAGES:
                for _ in range(self.FIFO_BLOCK_SIZE):
                    if self.s1_data_buffer: self.s1_data_buffer.pop(0)

            averaged_s1_data = apply_simple_smooth(np.mean(self.s1_data_buffer, axis=0), window_len=11)

        cervical_fp_data = {
            'CV7': averaged_fp_data,
            'S1': averaged_s1_data
        }

        self.data_ready.emit(averaged_data, current_avg_count, self.noise_level, cervical_fp_data)

        if current_avg_count == self.N_AVERAGES:
            waterfall_data = averaged_data.copy()
            waterfall_data.update(cervical_fp_data)
            self.waterfall_trace_ready.emit(waterfall_data)

    @Slot(int)
    def update_fiber_structure(self, new_n_fibers):
        self.N_FIBERS = new_n_fibers
        self.fiber_amplitudes = np.random.uniform(5, 15, self.N_FIBERS)
        self.fiber_taus = np.full(self.N_FIBERS, 0.0015)
        self.fiber_vcs = np.linspace(self.MIN_VC_MS, self.MAX_VC_MS, self.N_FIBERS)
        for dist in self.DISTANCES_CM:
            self.cap_data_buffers[dist] = []


# --- CLASSE SIMULATORE SEP from MEG recording #########################################################################
class SepSimulator(QWidget):
    simulation_closed = Signal(str)

    def __init__(self, json_anomaly_path=None, learning_manager=None, current_anomaly=None, ai_agent=None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setWindowFlag(Qt.WindowType.Window)

        self.json_anomaly_path = json_anomaly_path
        self.anomaly_config = None

        self.learning_manager = learning_manager
        self.current_anomaly = current_anomaly
        self.ai_agent = ai_agent

        if self.ai_agent is None:
            self.logger.warning("AI Agent non passato al simulatore. Le funzioni adattive saranno disabilitate.")

        self.load_json_anomaly()

        # --- CARICAMENTO UI ---
        # self.ui = uic.loadUi('res/sepIOMForm.ui', self)
        ui_file_path = "res/sepIOMForm.ui"
        ui_file = QFile(ui_file_path)

        if not ui_file:
            print(f"Errore: Impossibile aprire il file {ui_file_path}")
            sys.exit(-1)
        loader = QUiLoader()

        # Registriamo le classi di pyqtgraph in modo che il loader le crei correttamente
        loader.registerCustomWidget(pg.GraphicsLayoutWidget)
        loader.registerCustomWidget(pg.PlotWidget)
        self.ui = loader.load(ui_file)
        ui_file.close()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Rimuove margini vuoti ai bordi
        layout.addWidget(self.ui)

        self.setFixedSize(1000, 480)
        self.move(50, 415)
        percorso_logo = 'res/logos/logoSEP2.png'
        pixmap = QPixmap(percorso_logo)
        scaled_pixmap = pixmap.scaled(self.ui.labelLogo.size(),
                                      Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        self.ui.labelLogo.setPixmap(scaled_pixmap)

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
        VC_REF_DIST_CM = 20.0 - 7.0
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
        self.N_FIBERS = 100

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

        # --- PREPARAZIONE WORKER CON ANOMALY PARAMS ---
        self.worker_params = {
            'DISTANCES_CM': self.DISTANCES_CM, 'DISTANCES_M': self.DISTANCES_M,
            'MIN_DISPLAY_DELAY': self.MIN_DISPLAY_DELAY, 'N_FIBERS': self.N_FIBERS,
            'FS': self.FS, 'time': self.time, 'N_AVERAGES': self.N_AVERAGES,
            'FIFO_BLOCK_SIZE': self.FIFO_BLOCK_SIZE, 'MIN_VC_MS': self.MIN_VC_MS,
            'MAX_VC_MS': self.MAX_VC_MS,
            'fiber_amplitudes': np.random.uniform(5, 15, self.N_FIBERS),
            'fiber_taus': np.full(self.N_FIBERS, 0.0015),
            'fiber_vcs': np.linspace(self.MIN_VC_MS, self.MAX_VC_MS, self.N_FIBERS),

            # Passiamo esplicitamente il percorso JSON e la config per l'iniezione fisica
            'anomaly_config': self.anomaly_config,
            'json_anomaly_path': self.json_anomaly_path,
            'time_points': self.time_points
        }

        self.thread = QtCore.QThread()
        self.worker = CAPWorker(self.worker_params)
        self.worker.moveToThread(self.thread)

        self.worker.data_ready.connect(self.update_gui_plot)
        self.worker.waterfall_trace_ready.connect(self.add_waterfall_trace)
        self.worker.anomaly_triggered.connect(self.on_anomaly_triggered)

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
        # 1. PRIMA INIZIALIZZA I PLOT (IMPORTANTE: DEVE ESSERE FATTO PRIMA DEI SEGNALI)
        # --- PLOT CAP ---
        self.ui.plt_cap.setBackground(background='#000000')
        self.ui.plt_cap.setAntialiasing(True)
        self.capPLT = self.ui.plt_cap.addPlot()
        self.capPLT.setLabel('left', r"Ampiezza (uV)")
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

        # --- PLOT SEP ---
        self.ui.plt_sep.setBackground(background='#000000')
        self.sepPLT = self.ui.plt_sep.addPlot()
        self.sepPLT.showGrid(x=True, y=True)
        self.cv7_curve = self.sepPLT.plot(self.time * 1000, np.zeros_like(self.time),
                                          pen=pg.mkPen(color='c', width=1), name="Field Potential CV7")
        self.s1_curve = self.sepPLT.plot(self.time * 1000, np.zeros_like(self.time),
                                         pen=pg.mkPen('w', width=1, style=QtCore.Qt.PenStyle.DotLine),
                                         name="Cortical FP (S1)")

        # --- PLOT WATERFALL ---
        self.ui.plt_fall.setBackground(background='#000000')
        self.waterfall_plot = self.ui.plt_fall.addPlot()
        self.waterfall_plot.setLabel('bottom', "Tempo (ms)")
        self.waterfall_plot.getAxis('left').hide()
        self.waterfall_plot.showGrid(x=True, y=False)
        self.waterfall_plot.enableAutoRange(axis='y', enable=False)
        self.waterfall_plot.setYRange(0, self.max_waterfall_traces * 50)

        # --- CURSORI INIZIALI ---
        self.cursor_1 = pg.InfiniteLine(movable=True, pen=pg.mkPen('r', width=2, style=QtCore.Qt.PenStyle.DashLine),
                                        label='T1: {value:.2f} ms', labelOpts={'position': 0.8, 'color': 'r'})
        self.cursor_1.setValue(self.MIN_DISPLAY_DELAY * 1000 + 1)
        self.capPLT.addItem(self.cursor_1)
        self.cursor_2 = pg.InfiniteLine(movable=True, pen=pg.mkPen('b', width=2, style=QtCore.Qt.PenStyle.DashLine),
                                        label='T2: {value:.2f} ms', labelOpts={'position': 0.2, 'color': 'b'})
        self.cursor_2.setValue(self.MIN_DISPLAY_DELAY * 1000 + 5)
        self.capPLT.addItem(self.cursor_2)

        # 2. POI CONFIGURA I CONTROLLI E LE CONNESSIONI

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

        self.ui.comboTrace.addItem('wrist')
        self.ui.comboTrace.addItem('elbow')
        self.ui.comboTrace.addItem('armsit')
        self.ui.comboTrace.addItem('Erb')
        self.ui.comboTrace.addItem('CV7')
        self.ui.comboTrace.addItem('C3/4')
        self.ui.comboTrace.currentIndexChanged.connect(self.handle_traceWaterfall)

        self.ui.comboChn.addItems(['F3', 'F4', 'F3L', 'F4L', 'C3', 'C4', 'C3^', 'C4^'])
        self.ui.comboChn.currentIndexChanged.connect(self.handle_chnSep)
        # Imposta l'indice SOLO DOPO che il plot è stato creato!
        self.ui.comboChn.setCurrentIndex(6)

        self.ui.spinBoxIsi.valueChanged.connect(self.handle_stimIsi)
        self.ui.spinBoxAvg.valueChanged.connect(self.handle_avg)
        self.ui.spinBoxOvl.valueChanged.connect(self.handle_fifo_block)
        self.ui.spinBoxFibers.valueChanged.connect(self.handle_num_fibers)

        self.ui.comboFilter.addItems(['Raw signal', 'Smooth', 'Spline', '5-3000 Hz', '10-1500 Hz'])
        self.ui.comboFilter.currentIndexChanged.connect(self.handle_filter)
        self.ui.comboFilter.setCurrentIndex(0)

        self.cursor_1.sigPositionChanged.connect(self.update_vc_measurement)
        self.cursor_2.sigPositionChanged.connect(self.update_vc_measurement)
        self.update_vc_measurement()

    def setup_choice_buttons(self):
        """Prepara i pulsanti delle alternative leggendoli dal file JSON."""
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

        combo.clear()
        choices = [correct_answer] + distractors
        random.shuffle(choices)
        combo.addItem("Seleziona risposta...")
        combo.addItems(choices)

        try:
            confirm_button.clicked.disconnect()
        except TypeError:
            pass
        confirm_button.clicked.connect(self._handle_choice_confirmation)

    @Slot()
    def on_anomaly_triggered(self):
        """
        Chiamato automaticamente quando l'anomalia si attiva nel worker.
        """
        self.logger.info("GUI: Ricevuta notifica attivazione anomalia.")

        # 1. Attiva visivamente il pulsante o il pannello
        # Esempio: rendi il pulsante 'Anomaly' rosso o lampeggiante per indicare "Attenzione!"
        # O abilita direttamente i pulsanti di risposta se erano disabilitati

        # Se vuoi attivare la modalità "Anomaly Active" automaticamente:
        if not self.anomaly_active:
            self.handle_anomaly()  # Questo metodo attiva la UI dell'anomalia (bottoni, etc.)

        # Opzionale: Feedback sonoro o messaggio status bar
        # self.ui.statusbar.showMessage("ANOMALIA RILEVATA DAL SISTEMA", 5000)

    def _handle_choice_confirmation(self):
        """
        Gestisce la conferma della risposta dell'utente, calcola punteggio, salva nel DB
        e invoca l'Agente AI per le raccomandazioni.
        """
        if self.anomaly_config is None:
            QMessageBox.critical(self, "Errore Configurazione",
                                 "Impossibile verificare la risposta: Il file di configurazione dell'anomalia non è stato caricato correttamente.")
            return

        combo = self.ui.comboAnswers
        confirm_button = self.ui.pushBtConfirm

        # 1. Recupera la scelta
        chosen_action = combo.currentText()
        if combo.currentIndex() == 0:
            QMessageBox.warning(self, "Attenzione", "Seleziona una risposta valida.")
            return

        if not self.learning_manager or not self.anomaly_config:
            self.logger.error("Learning Manager o Configurazione Anomalia mancanti.")
            return

        # 2. Calcolo tempo risposta
        tempo_risposta = 0.0
        if self.response_start_time:
            tempo_risposta = (datetime.now() - self.response_start_time).total_seconds()
            self.logger.info(f"Tempo di risposta calcolato: {tempo_risposta:.2f} secondi.")

        # 3. Determina correttezza della risposta
        assessment_data = self.anomaly_config.get('learning_assessment', {})
        correct_action = assessment_data.get('correct_answer', "IGNORE")

        points_success = assessment_data.get('points', 100)
        points_fail = -20

        is_correct = (chosen_action == correct_action)
        punti_assegnati = points_success if is_correct else points_fail

        # 4. Feedback Visivo Immediato (UI)
        combo.setEnabled(False)
        confirm_button.setEnabled(False)

        if is_correct:
            msg = f"CORRETTO! (+{punti_assegnati} pt)"
            self.ui.textAnswers.append(f"{chosen_action} : {msg}")
            self.ui.pushBtRetray.setEnabled(False)
        else:
            msg = f"ERRATO. Risposta attesa: {correct_action} ({punti_assegnati} pt)"
            self.ui.textAnswers.append(f"Scelto: {chosen_action} -> {msg}")
            self.ui.pushBtRetray.setEnabled(True)

        # 5. Registrazione nel Database
        success = self.learning_manager.record_decision(
            azione_presa=chosen_action,
            esito_corretto=is_correct,
            punti=punti_assegnati,
            tempo_risposta_sec=tempo_risposta
        )

        if not success:
            print("ERRORE CRITICO: Il salvataggio nel DB è fallito. L'AI non potrà attivarsi.")
            return

        # 6. INTEGRAZIONE AI
        if self.ai_agent:
            print("--- AI AGENT: Avvio procedura ---")

            # A. Recupero sicuro dell'ID Soggetto e Sessione
            # Invece di usare self.parent() (fragile), usiamo i dati che abbiamo già nel manager
            session_id = self.learning_manager.current_session_id
            subject_id = None

            try:
                # Query diretta al DB per ottenere l'idcode dalla sessione corrente
                query_subj = "SELECT idcode FROM sessions WHERE idsession = ?"
                self.learning_manager.cur.execute(query_subj, (session_id,))
                result = self.learning_manager.cur.fetchone()
                #
                if result:
                    subject_id = result[0]
                    # print(f"AI AGENT: Trovato Subject ID: {subject_id} per Sessione: {session_id}")
                else:
                    print(f"AI AGENT ERRORE: Nessun subject trovato per session_id {session_id}")

            except Exception as e:
                print(f"AI AGENT SQL ERROR: {e}")

            if subject_id is not None:
                # B. Log Comportamentale
                try:
                    self.ai_agent.log_behavior(
                        user_id=subject_id,
                        action_type="SIMULATION_COMPLETE",
                        details=f"Scenario: {self.ui.comboAnomaly.currentText()} - Score: {punti_assegnati}",
                        duration=tempo_risposta,
                        session_id=session_id
                    )
                except Exception as e:
                    print(f"AI LOGGING ERROR: {e}")

                # C. Generazione Raccomandazione
                try:
                    recommendation = self.ai_agent.get_next_recommendation(subject_id)
                    print(f"AI RECOMMENDATION GENERATA: {recommendation}")
                    self.ai_agent.log_ai_feedback(
                        user_id=subject_id,
                        session_id=session_id,
                        trigger="SIMULATION_COMPLETE",
                        recommendation_data=recommendation
                    )

                    if recommendation.get('message'):
                        msg_box = QMessageBox(self)
                        msg_box.setWindowTitle("AI Tutor Feedback")
                        msg_box.setText(recommendation['message'])

                        diff_adj = recommendation.get('difficulty_adjustment', 'NORMAL')

                        if diff_adj in ['HARD', 'INSANE']:
                            msg_box.setIcon(QMessageBox.Icon.Warning)
                            # Logica opzionale per applicare difficoltà subito
                            if hasattr(self, 'worker') and self.worker:
                                self.worker.noise_level *= 1.2
                                print(f"AI: Difficoltà aumentata. Nuovo noise level: {self.worker.noise_level}")
                        else:
                            msg_box.setIcon(QMessageBox.Icon.Information)

                        msg_box.exec()
                    else:
                        print("AI: Nessun messaggio di raccomandazione generato (forse condizioni non soddisfatte).")
                except Exception as e:
                    print(f"AI RECOMMENDATION ERROR: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("AI: Impossibile procedere senza Subject ID valido.")
        else:
            print("AI Agent non inizializzato (self.ai_agent è None).")

    def _handle_choice_confirmation_old(self):
        """
        Gestisce la conferma della risposta dell'utente, calcola punteggio e salva nel DB.
        """
        combo = self.ui.comboAnswers
        confirm_button = self.ui.pushBtConfirm

        tempo_risposta = 0.0
        if self.response_start_time:
            tempo_risposta = (datetime.now() - self.response_start_time).total_seconds()

        # 1. Recupera la scelta
        chosen_action = combo.currentText()
        if combo.currentIndex() == 0:
            QMessageBox.warning(self, "Attenzione", "Seleziona una risposta valida.")
            return

        if not self.learning_manager or not self.anomaly_config:
            self.logger.error("Learning Manager o Configurazione Anomalia mancanti.")
            return

        # 2. Calcolo tempo risposta
        tempo_risposta = None
        if self.response_start_time:
            tempo_risposta = (datetime.now() - self.response_start_time).total_seconds()
            self.logger.info(f"Tempo di risposta calcolato: {tempo_risposta:.2f} secondi.")

        # 3. Determina correttezza
        assessment_data = self.anomaly_config.get('learning_assessment', {})
        correct_action = assessment_data.get('correct_answer', "IGNORE")

        points_success = assessment_data.get('points', 100)
        points_fail = -20

        is_correct = (chosen_action == correct_action)
        punti_assegnati = points_success if is_correct else points_fail

        # 4. UI Feedback
        combo.setEnabled(False)
        confirm_button.setEnabled(False)

        if is_correct:
            msg = f"CORRETTO! (+{punti_assegnati} pt)"
            self.ui.textAnswers.append(f"{chosen_action} : {msg}")
            # self.ui.textAnswers.setStyleSheet("color: green;")
            self.ui.pushBtRetray.setEnabled(False)
        else:
            msg = f"ERRATO. Risposta attesa: {correct_action} ({punti_assegnati} pt)"
            self.ui.textAnswers.append(f"Scelto: {chosen_action} -> {msg}")
            # self.ui.textAnswers.setStyleSheet("color: red;")
            self.ui.pushBtRetray.setEnabled(True)

        # 5. Registrazione DB
        success = self.learning_manager.record_decision(
            azione_presa=chosen_action,
            esito_corretto=is_correct,
            punti=punti_assegnati,
            tempo_risposta_sec=tempo_risposta
        )
        if not success:
            self.logger.error("Errore nel salvataggio della decisione nel DB.")

        if self.ai_agent:
            # Recupera l'ID utente (dalla sessione o dal parent)
            try:
                # Assumiamo che possiamo risalire all'ID utente
                subject_id = self.parent().dbase_window.current_subject.idcode
                session_id = self.learning_manager.current_session_id

                self.ai_agent.log_behavior(
                    user_id=subject_id,
                    action_type="SIMULATION_COMPLETE",
                    details=f"Scenario: {self.ui.comboAnomaly.currentText()} - Score: {punti_assegnati}",
                    duration=tempo_risposta,
                    session_id=session_id
                )
            except Exception as e:
                self.logger.error(f"Errore log AI: {e}")

        # --- INTERVENTO AGENTE AI ---
        if self.ai_agent and success:
            # Recuperiamo l'ID utente.
            # Nota: learning_manager non ha l'ID utente diretto, ma la sessione.
            # Recuperiamolo dal parent o dalla sessione corrente nel manager se l'hai salvata.
            # Metodo più sicuro se hai accesso alla dbase_window tramite parent:
            try:
                # Assumiamo che il parent sia MainWindow e abbia dbase_window
                subject_id = self.parent().dbase_window.current_subject.idcode
                recommendation = self.ai_agent.get_next_recommendation(subject_id)
                print(f"recommendation msg = {recommendation['message']}")
                # Mostra Feedback AI solo se c'è un messaggio rilevante
                if recommendation['message']:
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("AI Tutor Feedback")
                    msg_box.setText(recommendation['message'])

                    if recommendation['difficulty_adjustment'] in ['HARD', 'INSANE']:
                        msg_box.setIcon(QMessageBox.Icon.Warning)
                        # Esempio di adattamento dinamico del simulatore
                        if self.worker:
                            self.worker.noise_level *= 1.2  # Aumenta rumore del 20%
                            print("AI Tutor: Difficoltà aumentata dinamicamente.")
                    else:
                        msg_box.setIcon(QMessageBox.Icon.Information)
                    msg_box.exec()
            except Exception as e:
                self.logger.error(f"Errore durante feedback AI: {e}")

    def retray_answer(self):
        self.setup_choice_buttons()
        self.ui.pushBtRetray.setEnabled(False)

    def load_json_anomaly(self):
        if self.json_anomaly_path:
            try:
                with open(self.json_anomaly_path, 'r') as f:
                    self.anomaly_config = json.load(f)
                self.logger.info(f"SEP Simulator: Caricata config anomalia da {self.json_anomaly_path}.")
            except json.JSONDecodeError as e:
                self.logger.error(f"Errore decodifica JSON: {e}")
                self.anomaly_config = None

    def handle_anomaly(self):
        if not self.anomaly_active:
            self.ui.pushBtAnomaly.setText('ION-Sim Anomaly: ACTIVE')
            self.setFixedSize(1150, 480)
            self.ui.comboAnomaly.setEnabled(True)
            self.anomaly_active = True
            self.response_start_time = datetime.now()
            self.logger.info(f"Tempo risposta INIZIATO: {self.response_start_time}")
            self.ui.pushBtAnomaly.setStyleSheet("background-color: darkred;")
        else:
            self.ui.pushBtAnomaly.setText('ION-Sim Anomaly -> inactive')
            self.setFixedSize(1000, 480)
            self.ui.comboAnomaly.setEnabled(False)
            self.anomaly_active = False
            self.ui.pushBtAnomaly.setStyleSheet("background-color: darkgreen;")

    def show_report_window(self):
        self.report_window.show()
        self.report_window.raise_()

    def handle_helpSep(self):
        pdf_path = 'help_docs/docs_simula/SEP_tutorial_Gemini.pdf'
        wb.open_new(pdf_path)

    def handle_chnSep(self, new_index):
        eeg_name = self.ui.comboChn.currentText()
        meg_name = self.MEG_EEG_MAP.get(eeg_name, S1_INITIAL_CH_NAME)
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

    @Slot(object)
    def add_waterfall_trace(self, waterfall_data):
        self.main_averages_blocks += 1
        timems = self.time * 1000
        if len(self.waterfall_plots) >= self.max_waterfall_traces:
            oldest_plot = self.waterfall_plots.pop(0)
            self.waterfall_plot.removeItem(oldest_plot)
            self.report_window.remove_oldest_waterfall_entry()

        if self.TRACE_KEY in waterfall_data:
            if self.TRACE_KEY == 'S1':
                data = waterfall_data[self.TRACE_KEY] * -1e3
            else:
                data = waterfall_data[self.TRACE_KEY]
            self.waterfall_offset_step = int(abs(float(np.max(data)) - float(np.min(data))) / 5)
            offset_data = data + self.waterfall_offset
            color = (200, 200, 255, 180)
            new_curve = self.waterfall_plot.plot(timems, offset_data, pen=pg.mkPen(color=color, width=1))
            self.waterfall_plots.append(new_curve)
            self.waterfall_offset += self.waterfall_offset_step

            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.report_window.add_waterfall_entry(current_time_str)
            self.report_window.add_full_log_entry(current_time_str)
            self.waterfall_plot.enableAutoRange(axis='y', enable=True)
        self.ui.lcdWaterFall.display(self.main_averages_blocks)

    def clear_waterfall_traces(self):
        """ CORRETTO: Rimuove gli elementi dal grafico (plot) e svuota la lista (plots) """
        for curve in self.waterfall_plots:
            self.waterfall_plot.removeItem(curve)
        self.waterfall_plots = []
        self.waterfall_offset = 0.0
        self.waterfall_plot.enableAutoRange(axis='y', enable=True)

    def handle_filter(self, new_index):
        QtCore.QMetaObject.invokeMethod(self.worker, "update_filter_state",
                                        QtCore.Qt.ConnectionType.QueuedConnection, QtCore.Q_ARG(int, new_index))

    def update_vc_measurement(self):
        if self.cursor_1 is None or self.cursor_2 is None: return
        t1_ms = self.cursor_1.value();
        t2_ms = self.cursor_2.value()
        delta_t_ms = abs(t2_ms - t1_ms);
        delta_t_s = delta_t_ms / 1000.0
        vc_ms = self.VC_REF_DIST_M / delta_t_s if delta_t_s != 0 else 0.0
        self.ui.labelLatency.setText(f"VC misurata: {vc_ms:.2f} m/s")

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
        self.N_FIBERS = new_value * 100
        self.ui.label_fiber.setText(f"fibers = {self.N_FIBERS}")
        QtCore.QMetaObject.invokeMethod(self.worker, "update_fiber_structure",
                                        QtCore.Qt.ConnectionType.QueuedConnection, QtCore.Q_ARG(int, self.N_FIBERS))
        if self.ui.pushBtStart.text() == 'Stop': self.startStop(); self.startStop()

    def handle_stimIsi(self, new_value):
        new_value = round((1 / new_value) * 1000)
        self.ui.label_isi.setText(f"ISI = {new_value} ms")
        self.interStim = new_value
        if self.ui.pushBtStart.text() == 'Stop': self.startStop(); self.startStop()

    def closeEvent(self, event):
        self.simulation_closed.emit('SEPAS')
        self.closeWin()
        event.accept()

    def closeWin(self):
        QtCore.QMetaObject.invokeMethod(self.worker, "stop_simulation", QtCore.Qt.ConnectionType.QueuedConnection)
        self.worker.setParent(None)
        self.thread.quit()
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.close()

    def startStop(self):
        self.capPLT.enableAutoRange(axis='y', enable=True)
        self.capPLT.enableAutoRange(axis='x', enable=True)
        if self.ui.pushBtStart.text() == 'Start':
            self.ui.pushBtStart.setText('Stop')
            QtCore.QMetaObject.invokeMethod(self.worker, "start_simulation",
                                            QtCore.Qt.ConnectionType.QueuedConnection,
                                            QtCore.Q_ARG(int, self.interStim))
        else:
            self.ui.pushBtStart.setText('Start')
            QtCore.QMetaObject.invokeMethod(self.worker, "stop_simulation", QtCore.Qt.ConnectionType.QueuedConnection)

    @Slot(object, int, float, object)
    def update_gui_plot(self, averaged_data, current_avg_count, noise_level, cervical_fp_data):
        timems = self.time * 1000
        for dist_cm, data in averaged_data.items():
            if dist_cm in self.cap_curves:
                self.cap_curves[dist_cm].setData(timems, data)
                self.cap_curves[dist_cm].setVisible(self.plot_visibility.get(dist_cm, True))

        self.sepPLT.invertY(True)
        if 'CV7' in cervical_fp_data:
            self.cv7_curve.setData(timems, (cervical_fp_data['CV7'] * -10) + 1.5)
        if 'S1' in cervical_fp_data:
            self.s1_curve.setData(timems, cervical_fp_data['S1'] * 10)

        self.capPLT.setTitle(
            rf"CAP Averaging: {current_avg_count} / {self.N_AVERAGES} Campioni | Rumore Max: {noise_level:.1f} $\mu$V")