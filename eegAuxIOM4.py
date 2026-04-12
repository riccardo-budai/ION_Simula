import datetime
import sys
import json
import logging
import time
from pathlib import Path
import webbrowser as wb
import numpy as np
import pandas as pd
import pyqtgraph as pg

import mne
from mne.time_frequency import RawTFR

import scipy.integrate

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout,
                             QPushButton, QCheckBox, QLabel, QGroupBox,
                             QComboBox, QSlider, QGridLayout, QSpinBox, QLCDNumber, QSplitter, QTableWidget,
                             QHeaderView, QTableWidgetItem
                             )

from utils_funcIOM import apply_bandpass_filter, apply_simple_smooth, apply_spectral_whitening, analyze_r_peaks

# --- EEG CHANNELS ---
EEG_CHANNELS = ['FP2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'FP1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5']
# --- AUXILIAR CHANNELS ---
AUX_EOG = ['VEOG', 'HEOG']
AUX_ECG = ['EKG']
AUX_EMG = ['MILO', 'MASST']

N_EEG_CHANNELS = len(EEG_CHANNELS)
AUX_CHANNELS = ['EOG', 'ECG']
ALL_PLOT_CHANNELS = EEG_CHANNELS + AUX_CHANNELS
N_ALL_CHANNELS = len(ALL_PLOT_CHANNELS)

FS_EEG = 512
TIME_DURATION_S = 10.0
TIME_POINTS_EEG = int(FS_EEG * TIME_DURATION_S)
TIME_VECTOR_EEG = np.linspace(0, TIME_DURATION_S, TIME_POINTS_EEG, endpoint=False)


def load_eeg_data_mne_cnt(cnt_filepath: str, target_time_vector: np.ndarray, target_fs: float):
    """ Carica un file .cnt usando MNE """
    print(f"Caricamento da file CNT (Neuroscan) con MNE: {cnt_filepath}")
    try:
        raw = mne.io.read_raw_cnt(cnt_filepath, preload=True, verbose=False)
        events_original, event_id_map = mne.events_from_annotations(raw, verbose=False)
        original_sfreq = raw.info['sfreq']

        if raw.info['sfreq'] != target_fs:
            print(f"Ricampionamento da {raw.info['sfreq']} Hz a {target_fs} Hz...")
            raw.resample(target_fs, npad="auto")

        n_target_samples = len(target_time_vector)
        n_raw_samples = raw.n_times

        all_data, times = raw.get_data(return_times=True)
        all_data = all_data * -1e6  # Conversione a microvolt

        if n_raw_samples >= n_target_samples:
            all_data = all_data[:, :n_target_samples]
            if original_sfreq != target_fs:
                events, event_id_map = mne.events_from_annotations(raw, verbose=False)
            else:
                events = events_original
            events = events[events[:, 0] < n_target_samples]
        else:
            padding_len = n_target_samples - n_raw_samples
            all_data = np.pad(all_data, ((0, 0), (0, padding_len)), 'constant')
            events, event_id_map = mne.events_from_annotations(raw, verbose=False)

        raw_ch_names = raw.ch_names
        data_map = {name.upper(): all_data[i] for i, name in enumerate(raw_ch_names)}

        eeg_list = []
        for ch_name in EEG_CHANNELS:
            key = ch_name.upper()
            if key in data_map:
                eeg_list.append(data_map[key])
            else:
                eeg_list.append(np.zeros(n_target_samples))

        eeg_data_2d = np.array(eeg_list)

        eog_keys = ['EOG', 'VEOG', 'VEO', 'HEO']
        ecg_keys = ['ECG', 'EKG']

        eog_data = np.zeros(n_target_samples)
        for key in eog_keys:
            if key in data_map:
                eog_data = data_map[key]
                break

        ecg_data = np.zeros(n_target_samples)
        for key in ecg_keys:
            if key in data_map:
                ecg_data = data_map[key]
                break

        return {
            'eeg_data_2d': eeg_data_2d,
            'eog_v': eog_data,
            'ecg': ecg_data,
            'template_len': n_target_samples,
            'events': events,
            'event_id_map': event_id_map
        }

    except Exception as e:
        print(f"ERRORE caricamento MNE: {e}")
        len_dummy = len(target_time_vector)
        return {
            'eeg_data_2d': np.random.normal(0, 1, (len(EEG_CHANNELS), len_dummy)),
            'eog_v': np.random.normal(0, 1, len_dummy),
            'ecg': np.random.normal(0, 1, len_dummy),
            'template_len': len_dummy,
            'events': np.array([]),
            'event_id_map': {}
        }


def add_eeg_artifacts_dynamic(eeg_data, time_vector, fs, artifact_state, eog_template=None, ecg_template=None):
    n_channels, time_points = eeg_data.shape
    out_data = eeg_data.copy()

    state_eog = artifact_state.get('EOG_Blink', {'enabled': False, 'amplitude': 0})
    if state_eog['enabled'] and eog_template is not None:
        factor = state_eog['amplitude'] / 100.0
        out_data[0, :] += eog_template * factor
        out_data[1, :] += eog_template * factor

    state_emg = artifact_state.get('EMG_Noise', {'enabled': False, 'amplitude': 0})
    if state_emg['enabled']:
        noise = np.random.normal(0, state_emg['amplitude'], (n_channels, time_points))
        out_data += noise

    return out_data


# --- WORKER CLASS #####################################################################################################
class EEGWorker(QtCore.QObject):
    data_ready = Signal(dict)
    spectra_ready = Signal(np.ndarray)

    def __init__(self, simulator_params, parent=None):
        super().__init__(parent)
        self.FS = simulator_params['FS']
        self.EEG_CHANNELS = simulator_params['eeg_channels']
        self.N_EEG_CHANNELS = len(self.EEG_CHANNELS)
        self.time = simulator_params['time_vector']

        # --- Flag Analisi ECG ---
        self.ecg_enabled = False

        # --- BUFFER ECG ---
        self.ecg_analysis_window_s = 10.0
        self.ecg_buffer_size = int(self.ecg_analysis_window_s * self.FS)
        self.ecg_buffer = np.zeros(self.ecg_buffer_size)

        # --- BUFFER EEG PER SPETTRI ---
        self.eeg_buffer = np.zeros((self.N_EEG_CHANNELS, self.ecg_buffer_size))
        self.samples_since_last_spectra = 0
        self.spectra_interval_samples = int(2.0 * self.FS)

        loaded = load_eeg_data_mne_cnt(simulator_params['cnt_filepath'], self.time, self.FS)
        self.template_eeg_data = loaded['eeg_data_2d']
        self.template_eog_v = loaded['eog_v']
        self.template_ecg = loaded['ecg']

        self.is_running = False
        self.chunk_duration_s = simulator_params['timer_interval_ms'] / 1000.0
        self.chunk_samples = int(self.chunk_duration_s * self.FS)
        self.current_index = 0
        self.current_display_index = 0

        self.artifact_state = {
            'EOG_Blink': {'enabled': False, 'amplitude': 50.0},
            'EMG_Noise': {'enabled': False, 'amplitude': 10.0},
            'Slow_Drift': {'enabled': False, 'amplitude': 0.0}
        }

    @Slot()
    def _init_timer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._run_simulation_step)

    @Slot(int)
    def start_simulation(self, interval_ms):
        self.is_running = True
        self.timer.start(interval_ms)

    @Slot()
    def stop_simulation(self):
        self.is_running = False
        self.timer.stop()

    @Slot(bool)
    def set_ecg_enabled(self, enabled):
        """ Attiva o disattiva il calcolo pesante dell'ECG """
        self.ecg_enabled = enabled

    @Slot(str, bool, float)
    def update_artifact_state(self, artifact_key: str, enabled: bool, amplitude: float):
        if artifact_key in self.artifact_state:
            self.artifact_state[artifact_key]['enabled'] = enabled
            self.artifact_state[artifact_key]['amplitude'] = amplitude

    @Slot()
    def _run_simulation_step(self):
        if not self.is_running: return

        total_samples = self.template_eeg_data.shape[1]
        end_index = self.current_index + self.chunk_samples

        if end_index > total_samples:
            remaining = end_index - total_samples
            chunk_eeg = np.hstack(
                (self.template_eeg_data[:, self.current_index:], self.template_eeg_data[:, :remaining]))
            chunk_eog = np.hstack((self.template_eog_v[self.current_index:], self.template_eog_v[:remaining]))
            chunk_ecg = np.hstack((self.template_ecg[self.current_index:], self.template_ecg[:remaining]))
            self.current_index = remaining
        else:
            chunk_eeg = self.template_eeg_data[:, self.current_index:end_index]
            chunk_eog = self.template_eog_v[self.current_index:end_index]
            chunk_ecg = self.template_ecg[self.current_index:end_index]
            self.current_index = end_index

        # --- PROCESSING ---
        chunk_eog = chunk_eog * 1e-1
        chunk_eeg = apply_bandpass_filter(chunk_eeg, 1.6, 35, self.FS)
        chunk_eog = apply_bandpass_filter(chunk_eog, 0.1, 20, self.FS)
        average_reference = np.mean(chunk_eeg, axis=0)
        chunk_eeg = chunk_eeg - average_reference

        chunk_ecg_filt = apply_bandpass_filter(chunk_ecg, 1.0, 25, self.FS) * 0.5
        chunk_eeg = add_eeg_artifacts_dynamic(
            chunk_eeg, None, self.FS, self.artifact_state,
            eog_template=chunk_eog, ecg_template=chunk_ecg_filt
        )

        # --- GESTIONE BUFFER ---
        num_new_samples = chunk_eeg.shape[1]

        self.ecg_buffer = np.roll(self.ecg_buffer, -num_new_samples)
        self.ecg_buffer[-num_new_samples:] = chunk_ecg

        self.eeg_buffer = np.roll(self.eeg_buffer, -num_new_samples, axis=1)
        self.eeg_buffer[:, -num_new_samples:] = chunk_eeg

        # --- ANALISI ECG CONDIZIONALE ---
        if self.ecg_enabled:
            qrs_stats = analyze_r_peaks(self.ecg_buffer, self.FS)
        else:
            qrs_stats = {'peaks': [], 'bpm': 0, 'rr_intervals_ms': [], 'sdnn': 0, 'rmssd': 0}

        # --- TRIGGER SPETTRI ---
        self.samples_since_last_spectra += num_new_samples
        if self.samples_since_last_spectra >= self.spectra_interval_samples:
            self.samples_since_last_spectra = 0
            self.spectra_ready.emit(self.eeg_buffer.copy())

        # --- PREPARAZIONE DATI GUI ---
        eeg_multichannel_data = {ch: chunk_eeg[i, :] for i, ch in enumerate(self.EEG_CHANNELS)}
        aux_data = {'EOG': chunk_eog, 'ECG': chunk_ecg_filt}
        start_disp = self.current_display_index
        self.current_display_index += self.chunk_samples

        buffer_len = len(self.ecg_buffer)
        chunk_start_in_buffer = buffer_len - num_new_samples

        peaks_in_current_chunk = []
        if self.ecg_enabled:
            for p_idx in qrs_stats['peaks']:
                if p_idx >= chunk_start_in_buffer:
                    local_idx = p_idx - chunk_start_in_buffer
                    peaks_in_current_chunk.append(local_idx)
        qrs_stats['peaks'] = peaks_in_current_chunk

        self.data_ready.emit({
            'new_data_chunk': eeg_multichannel_data,
            'aux_data_chunk': aux_data,
            'ecg_analysis': qrs_stats,
            'start_display_index': start_disp
        })


# --- MAIN WINDOW ---
class EEGControlWindow(QWidget):
    def __init__(self, json_anomaly_path=None, learning_manager=None, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window)
        self.logger = logging.getLogger(__name__)
        self.json_anomaly_path = json_anomaly_path
        self.learning_manager = learning_manager

        self.setWindowTitle(time.strftime('%d %b %Y') + '  EEG - Real Time Processing')
        self.resize(1200, 800)

        # --- INIZIALIZZAZIONE ---
        self.timer_interval_ms = 100
        self.is_simulating = False
        self.display_mode = "Sweep"
        self.current_display_duration_s = 10.0
        self.display_samples = int(FS_EEG * self.current_display_duration_s)

        self.eeg_display_buffer = np.zeros((N_ALL_CHANNELS, self.display_samples))
        self.time_vector_plot = np.linspace(0, self.current_display_duration_s, self.display_samples, endpoint=False)

        # Lista per memorizzare i picchi da disegnare
        self.ecg_peaks_points = []

        self.worker_params = {
            'FS': FS_EEG,
            'time_vector': TIME_VECTOR_EEG,
            'eeg_channels': EEG_CHANNELS,
            'aux_channels': AUX_CHANNELS,
            'cnt_filepath': 'eegDataSets/ALBAEGIT.CNT',
            'timer_interval_ms': self.timer_interval_ms
        }

        self.spectra_window = None

        self._init_ui()
        self._init_plotting()

        self.thread = QtCore.QThread()
        self.worker = EEGWorker(self.worker_params)
        self.worker.moveToThread(self.thread)
        self.worker.data_ready.connect(self.update_eeg_plot)
        self.worker.spectra_ready.connect(self.handle_spectra_update)
        self.thread.start()
        QtCore.QMetaObject.invokeMethod(self.worker, "_init_timer", QtCore.Qt.ConnectionType.QueuedConnection)

        self.load_json_anomaly()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()

        sim_group = QGroupBox("Simulazione")
        sim_layout = QHBoxLayout()
        self.btn_start_stop = QPushButton("Start EEG")
        self.btn_start_stop.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        self.btn_start_stop.clicked.connect(self.start_stop_simulation)

        self.combo_spectra = QComboBox()
        self.combo_spectra.addItem('Time frequency')
        self.combo_spectra.addItem('Channels PSD')
        self.combo_spectra.addItem('Spectra OFF')
        self.combo_spectra.currentIndexChanged.connect(self.run_spectra)

        self.check_filter = QCheckBox("Filtro Attivo (1.6-45Hz)")
        self.check_filter.setChecked(True)

        sim_layout.addWidget(self.btn_start_stop)
        sim_layout.addWidget(self.combo_spectra)
        sim_layout.addWidget(self.check_filter)
        sim_group.setLayout(sim_layout)

        view_group = QGroupBox("Visualizzazione")
        view_layout = QHBoxLayout()
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Sweep (Continuous)", "Page (Paging)"])
        self.combo_mode.currentIndexChanged.connect(self.change_display_mode)

        lbl_dur = QLabel("Sec/Pagina:")
        self.spin_duration = QSpinBox()
        self.spin_duration.setRange(2, 60)
        self.spin_duration.setValue(int(self.current_display_duration_s))
        self.spin_duration.setSuffix(" s")
        self.spin_duration.valueChanged.connect(self.update_display_duration)

        view_layout.addWidget(QLabel("Modo:"))
        view_layout.addWidget(self.combo_mode)
        view_layout.addWidget(lbl_dur)
        view_layout.addWidget(self.spin_duration)
        view_group.setLayout(view_layout)

        self.btn_close = QPushButton("Close")
        self.btn_close.setStyleSheet("background-color: #f44336; color: blue;")
        self.btn_close.clicked.connect(self.close)

        self.btn_help = QPushButton("Help")
        self.btn_help.setStyleSheet("background-color: #f44336; color: blue;")
        self.btn_help.clicked.connect(self.help_context)

        top_layout.addWidget(sim_group)
        top_layout.addWidget(view_group)
        top_layout.addStretch()
        top_layout.addWidget(self.btn_help)
        top_layout.addWidget(self.btn_close)

        main_layout.addLayout(top_layout)

        self.gl_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.gl_widget, stretch=2)

        art_group = QGroupBox("Generatore Artefatti (Controlli Dinamici)")
        art_layout = QHBoxLayout()
        self.artifact_controls = {}
        artifacts_def = [("EOG_Blink", "EOG Blink"), ("EMG_Noise", "EMG Noise")]
        for key, label_text in artifacts_def:
            col_layout = QVBoxLayout()
            lbl_title = QLabel(label_text)
            lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk_active = QCheckBox("Attivo")
            chk_active.stateChanged.connect(lambda state, k=key: self.handle_artifact_toggle(k, state))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 200)
            slider.setValue(50 if key == 'EOG_Blink' else 10)
            slider.valueChanged.connect(lambda val, k=key: self.handle_artifact_slider(k, val))
            lbl_val = QLabel(f"{slider.value()} uV")
            lbl_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col_layout.addWidget(lbl_title)
            col_layout.addWidget(chk_active)
            col_layout.addWidget(slider)
            col_layout.addWidget(lbl_val)
            self.artifact_controls[key] = {'check': chk_active, 'slider': slider, 'label': lbl_val}
            container = QWidget()
            container.setLayout(col_layout)
            container.setStyleSheet("border: 1px solid #ccc; border-radius: 5px; margin: 2px;")
            art_layout.addWidget(container)
        art_group.setLayout(art_layout)

        self.ecg_group = QGroupBox("ECG Analysis (Real-time)")
        self.ecg_group.setCheckable(True)
        self.ecg_group.setChecked(False)
        self.ecg_group.toggled.connect(self.toggle_ecg_analysis)
        ecg_layout = QHBoxLayout()
        self.lbl_bpm = QLabel("BPM: --")
        self.lbl_bpm.setStyleSheet("font-size: 14pt; font-weight: bold; color: #d32f2f;")
        self.lbl_sdnn = QLabel("SDNN: -- ms")
        self.lbl_rmssd = QLabel("RMSSD: -- ms")
        ecg_layout.addWidget(self.lbl_bpm)
        ecg_layout.addWidget(self.lbl_sdnn)
        ecg_layout.addWidget(self.lbl_rmssd)
        ecg_layout.addStretch()
        self.ecg_group.setLayout(ecg_layout)
        main_layout.addWidget(self.ecg_group)

    def _init_plotting(self):
        self.gl_widget.setBackground(background='#EEE5D1')
        self.eeg_plt = self.gl_widget.addPlot()
        self.eeg_plt.setLabel('bottom', "Tempo (s)")
        self.eeg_plt.showGrid(x=True, y=False)
        # update axis and text colors
        pen = pg.mkPen(color=(100, 100, 100))
        self.eeg_plt.getAxis('bottom').setPen(pen)
        self.eeg_plt.getAxis('bottom').setTextPen(pen)
        self.eeg_plt.hideAxis('left')

        self.eeg_plt.enableAutoRange(axis='y', enable=False)
        self.offset_step = 50.0
        total_height = N_ALL_CHANNELS * self.offset_step
        self.eeg_plt.setYRange(-self.offset_step, total_height + self.offset_step)
        self.eeg_plt.setXRange(0, self.current_display_duration_s)
        self.eeg_plots = {}
        current_y_pos = (N_ALL_CHANNELS - 1) * self.offset_step
        colors = ['#101010' for _ in range(N_ALL_CHANNELS)]
        for i, ch_name in enumerate(ALL_PLOT_CHANNELS):
            color = colors[i]
            pen = pg.mkPen(color='#0000AA', width=1.5) if ch_name in AUX_CHANNELS else pg.mkPen(color=color, width=1)
            curve = self.eeg_plt.plot(
                self.time_vector_plot,
                np.zeros_like(self.time_vector_plot) + current_y_pos,
                pen=pen, name=ch_name
            )
            self.eeg_plots[ch_name] = curve
            text_item = pg.TextItem(text=ch_name, color='#333333', anchor=(1, 0.5))
            self.eeg_plt.addItem(text_item)
            text_item.setPos(0, current_y_pos)
            current_y_pos -= self.offset_step

        # --- SCATTER PLOT PER I DOT SPOT ---
        self.peak_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))
        self.eeg_plt.addItem(self.peak_scatter)

    def toggle_ecg_analysis(self, state):
        """ Attiva/Disattiva l'analisi ECG nel Worker """
        QtCore.QMetaObject.invokeMethod(self.worker, "set_ecg_enabled",
                                        QtCore.Qt.ConnectionType.QueuedConnection,
                                        QtCore.Q_ARG(bool, state))
        if not state:
            self.lbl_bpm.setText("BPM: -- (OFF)")
            self.lbl_sdnn.setText("SDNN: --")
            self.lbl_rmssd.setText("RMSSD: --")
            self.peak_scatter.setData([], [])  # Pulisce i punti se spento
            self.ecg_peaks_points = []

    def change_display_mode(self, index):
        if index == 0:
            self.display_mode = "Sweep"
        else:
            self.display_mode = "Page"
            self.eeg_display_buffer[:] = 0
            self.ecg_peaks_points = []  # Reset punti
            self.peak_scatter.setData([], [])
            self._refresh_full_plot()

    def help_context(self):
        help_path = Path('help_docs/paper/Tut_spectraECoG.pdf')
        if help_path.exists():
            wb.open_new(str(help_path))
        else:
            print(f"Help file not found: {help_path}")

    @Slot(np.ndarray)
    def handle_spectra_update(self, eeg_buffer_data):
        if self.spectra_window and self.spectra_window.isVisible():
            self.spectra_window.update_tfr_data(eeg_buffer_data)

    def update_display_duration(self, value_seconds):
        print(f"Cambio durata visualizzazione a: {value_seconds} s")
        self.current_display_duration_s = float(value_seconds)
        new_samples = int(FS_EEG * self.current_display_duration_s)
        self.display_samples = new_samples
        self.eeg_display_buffer = np.zeros((N_ALL_CHANNELS, self.display_samples))
        self.time_vector_plot = np.linspace(0, self.current_display_duration_s, self.display_samples, endpoint=False)
        self.eeg_plt.setXRange(0, self.current_display_duration_s)
        self.ecg_peaks_points = []  # Reset punti al resize
        self._refresh_full_plot()

    def start_stop_simulation(self):
        if not self.is_simulating:
            self.is_simulating = True
            self.btn_start_stop.setText('Stop EEG')
            self.btn_start_stop.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
            QtCore.QMetaObject.invokeMethod(self.worker, "start_simulation",
                                            QtCore.Qt.ConnectionType.QueuedConnection,
                                            QtCore.Q_ARG(int, self.timer_interval_ms))
        else:
            self.is_simulating = False
            self.btn_start_stop.setText('Start EEG')
            self.btn_start_stop.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            QtCore.QMetaObject.invokeMethod(self.worker, "stop_simulation", QtCore.Qt.ConnectionType.QueuedConnection)

    def handle_artifact_toggle(self, key, state):
        enabled = (state == Qt.CheckState.Checked.value)
        ctrl = self.artifact_controls[key]
        val = ctrl['slider'].value()
        QtCore.QMetaObject.invokeMethod(self.worker, "update_artifact_state",
                                        QtCore.Qt.ConnectionType.QueuedConnection,
                                        QtCore.Q_ARG(str, key),
                                        QtCore.Q_ARG(bool, enabled),
                                        QtCore.Q_ARG(float, val))

    def handle_artifact_slider(self, key, value):
        ctrl = self.artifact_controls[key]
        ctrl['label'].setText(f"{value} uV")
        enabled = ctrl['check'].isChecked()
        QtCore.QMetaObject.invokeMethod(self.worker, "update_artifact_state",
                                        QtCore.Qt.ConnectionType.QueuedConnection,
                                        QtCore.Q_ARG(str, key),
                                        QtCore.Q_ARG(bool, enabled),
                                        QtCore.Q_ARG(float, value))

    # --- METODO RUN_SPECTRA REINSERITO ---
    def run_spectra(self):
        selected_mode = self.combo_spectra.currentText()
        if selected_mode == 'Spectra OFF':
            if self.spectra_window:
                self.spectra_window.close()
                self.spectra_window = None
            return

        if self.spectra_window is not None:
            self.spectra_window.close()

        dummy_data = np.zeros((len(EEG_CHANNELS), int(FS_EEG * 10)))

        self.spectra_window = SpectraWindow(
            eeg_data_buffer=dummy_data,
            channels=EEG_CHANNELS,
            fs=FS_EEG,
            mode=selected_mode
        )
        self.spectra_window.show()
        print(f"Finestra Spettri aperta in modalità: {selected_mode}")

    @Slot(dict)
    def update_eeg_plot(self, data_packet):
        if 'new_data_chunk' not in data_packet: return

        eeg_data = data_packet['new_data_chunk']
        aux_data = data_packet.get('aux_data_chunk', {})
        all_new_data = {**eeg_data, **aux_data}
        raw_start_idx = data_packet.get('start_display_index', 0)
        chunk_len = list(eeg_data.values())[0].size
        start_idx = raw_start_idx % self.display_samples
        end_idx = start_idx + chunk_len

        # Reset per Page mode o riavvio Sweep
        reset_required = False
        if start_idx == 0 or (raw_start_idx // self.display_samples) > (
                (raw_start_idx - chunk_len) // self.display_samples):
            reset_required = True

        if reset_required:
            if self.display_mode == "Page":
                self.eeg_display_buffer[:] = 0
            self.ecg_peaks_points = []  # Reset punti all'inizio del ciclo

        # Scrittura Buffer
        for i, ch_name in enumerate(ALL_PLOT_CHANNELS):
            if ch_name not in all_new_data: continue
            new_chunk = all_new_data[ch_name]
            if end_idx <= self.display_samples:
                self.eeg_display_buffer[i, start_idx:end_idx] = new_chunk
            else:
                part1_len = self.display_samples - start_idx
                self.eeg_display_buffer[i, start_idx:] = new_chunk[:part1_len]
                part2_len = chunk_len - part1_len
                self.eeg_display_buffer[i, :part2_len] = new_chunk[part1_len:]

        self._refresh_full_plot()

        # --- GESTIONE PUNTI ROSSI (ECG DOT SPOT) ---
        if 'ecg_analysis' in data_packet:
            stats = data_packet['ecg_analysis']
            if self.ecg_group.isChecked():
                self.lbl_bpm.setText(f"BPM: {stats['bpm']}")
                self.lbl_sdnn.setText(f"SDNN: {stats['sdnn']} ms")
                self.lbl_rmssd.setText(f"RMSSD: {stats['rmssd']} ms")

                if stats['peaks'] and 'ECG' in ALL_PLOT_CHANNELS:
                    ecg_ch_idx = ALL_PLOT_CHANNELS.index('ECG')
                    y_offset = (N_ALL_CHANNELS - 1 - ecg_ch_idx) * self.offset_step

                    for local_p_idx in stats['peaks']:
                        # Calcola posizione globale nel buffer circolare
                        buffer_idx = (start_idx + local_p_idx) % self.display_samples

                        # Tempo X
                        x_pos = self.time_vector_plot[buffer_idx]

                        # Voltaggio Y (esatto dal buffer + offset grafico)
                        y_val = self.eeg_display_buffer[ecg_ch_idx, buffer_idx]
                        y_pos = y_val + y_offset

                        self.ecg_peaks_points.append({'pos': (x_pos, y_pos), 'brush': pg.mkBrush('r')})

                # Disegna i punti
                self.peak_scatter.setData(self.ecg_peaks_points)

    def _refresh_full_plot(self):
        current_offset = (N_ALL_CHANNELS - 1) * self.offset_step
        for i, ch_name in enumerate(ALL_PLOT_CHANNELS):
            offset_data = self.eeg_display_buffer[i, :] + current_offset
            self.eeg_plots[ch_name].setData(self.time_vector_plot, offset_data)
            current_offset -= self.offset_step

    def load_json_anomaly(self):
        if self.json_anomaly_path:
            self.logger.info(f"Caricamento anomalia: {self.json_anomaly_path}")

    def closeEvent(self, event):
        if self.is_simulating:
            QtCore.QMetaObject.invokeMethod(self.worker, "stop_simulation", QtCore.Qt.ConnectionType.QueuedConnection)
        if self.spectra_window:
            self.spectra_window.close()
        self.thread.quit()
        self.thread.wait()
        self.worker.deleteLater()
        event.accept()


# --- SPECTRA WINDOW ---
from collections import OrderedDict
import datetime

CHANNEL_R = 'O2'
CHANNEL_L = 'O1'


class SpectraWindow(QWidget):
    close_signal = Signal()

    def __init__(self, eeg_data_buffer, channels, fs, mode='Time frequency', parent=None):
        super().__init__(parent)
        self.mode = mode
        self.setWindowTitle(f"EEG Analysis - {self.mode}")
        self.resize(1400, 500)
        self.setWindowFlag(Qt.WindowType.Window)

        self.eeg_data_V = eeg_data_buffer * 1e-6
        self.channels = channels
        self.fs = fs
        self.epoch_counter = 0

        # Variabili per memorizzare gli ultimi dati PSD calcolati (per il cursore)
        self.last_psd_data = None
        self.last_freqs = None

        # Configurazione Time Frequency
        self.tfr_channels = [CHANNEL_R, CHANNEL_L]
        self.tfr_indices = {ch: self.channels.index(ch) for ch in self.tfr_channels if ch in self.channels}
        if len(self.tfr_indices) < 2 and len(self.channels) >= 2:
            self.tfr_channels = [self.channels[0], self.channels[1]]
            self.tfr_indices = {self.channels[0]: 0, self.channels[1]: 1}

        self.offset_step = 20.0
        self.n_time_steps_to_plot_tfr = 5
        self.eeg_plots = OrderedDict()

        # --- LAYOUT PRINCIPALE ---
        main_layout = QVBoxLayout(self)

        # 1. TOOLBAR
        toolbar_layout = QHBoxLayout()
        self.chk_whitening = QCheckBox("Applica Whitening (1/f)")
        self.chk_whitening.setChecked(False)
        self.chk_whitening.toggled.connect(self.refresh_view)
        toolbar_layout.addWidget(self.chk_whitening)
        toolbar_layout.addSpacing(20)

        self.lbl_cursor = QLabel("Cursore: Sposta la linea verticale per info")
        self.lbl_cursor.setStyleSheet("font-weight: bold; color: yellow;")
        toolbar_layout.addWidget(self.lbl_cursor)

        toolbar_layout.addStretch()

        lbl_lcd = QLabel("Epoche:")
        lbl_lcd.setStyleSheet("color: #AAA;")
        self.lcd_epochs = QLCDNumber()
        self.lcd_epochs.setDigitCount(4)
        self.lcd_epochs.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.lcd_epochs.setStyleSheet("background-color: #000; color: #00FF00; border: 1px solid #555;")
        self.lcd_epochs.display(0)
        toolbar_layout.addWidget(lbl_lcd)
        toolbar_layout.addWidget(self.lcd_epochs)

        main_layout.addLayout(toolbar_layout)

        # 2. SPLITTER
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)

        # --- SINISTRA: GRAFICO ---
        self.plot_widget = pg.PlotWidget()
        self.splitter.addWidget(self.plot_widget)

        # --- DESTRA: TAB WIDGET ---
        self.tabs_analysis = QtWidgets.QTabWidget()
        self.tabs_analysis.setFixedWidth(480)
        self.splitter.addWidget(self.tabs_analysis)

        # Setup Tab 1: Picchi + Cursore
        self.tab_peaks = QWidget()
        self._setup_tab_peaks()
        self.tabs_analysis.addTab(self.tab_peaks, "Cursore & Picchi")

        # Setup Tab 2: Bande
        self.tab_bands = QWidget()
        self._setup_tab_bands()
        self.tabs_analysis.addTab(self.tab_bands, "Aree Bande")

        # Inizializzazione Grafico
        if self.mode == 'Channels PSD':
            self._init_psd_raster_plotting()
            self._init_cursor()  # Inizializza il cursore solo in modalità PSD
        else:
            self._init_tfr_plotting()

        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)

    def _setup_tab_peaks(self):
        layout = QVBoxLayout(self.tab_peaks)
        self.table_peaks = QTableWidget()
        # Aggiunta colonna "Cursor dB"
        self.table_peaks.setColumnCount(4)
        self.table_peaks.setHorizontalHeaderLabels(["Ch", "Max Freq", "Max dB", "Cursor dB"])
        header = self.table_peaks.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table_peaks)

    def _setup_tab_bands(self):
        layout = QVBoxLayout(self.tab_bands)
        self.table_bands = QTableWidget()
        self.table_bands.setColumnCount(5)
        self.table_bands.setHorizontalHeaderLabels(["Ch", "Delta", "Theta", "Alpha", "Beta"])
        header = self.table_bands.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for i in range(1, 5):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table_bands)

    def _init_cursor(self):
        # Linea verticale infinita
        self.v_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('y', width=2, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.v_line)
        # Posiziona inizialmente a 10Hz
        self.v_line.setPos(10)
        # Collega il segnale di movimento
        self.v_line.sigPositionChanged.connect(self.on_cursor_moved)

    def on_cursor_moved(self):
        """ Chiamato quando l'utente sposta la linea verticale """
        if self.last_psd_data is None or self.last_freqs is None:
            return

        # Ottieni la posizione X (Frequenza) corrente della linea
        cursor_freq = self.v_line.value()

        # Aggiorna label in alto
        self.lbl_cursor.setText(f"Cursore a: {cursor_freq:.2f} Hz")

        # Calcolo indice o interpolazione
        # Per semplicità usiamo interpolazione lineare sui dati dB

        psd_db_matrix = 10 * np.log10(self.last_psd_data + 1e-15)

        # Aggiorna la colonna "Cursor dB" nella tabella
        # Disabilito sorting temporaneo per evitare glitch grafici durante update rapido
        self.table_peaks.setSortingEnabled(False)

        for i, ch_name in enumerate(self.channels):
            # Interpola valore dB alla frequenza del cursore
            val_at_cursor = np.interp(cursor_freq, self.last_freqs, psd_db_matrix[i, :])

            # Aggiorna la cella (colonna 3)
            item = self.table_peaks.item(i, 3)
            if item is None:
                item = QTableWidgetItem()
                self.table_peaks.setItem(i, 3, item)

            item.setText(f"{val_at_cursor:.1f}")

            # Opzionale: Evidenzia se vicino al picco?
            # if abs(val_at_cursor - max_val) < 2: ...

        self.table_peaks.setSortingEnabled(True)

    def refresh_view(self):
        if self.mode == 'Channels PSD':
            self._compute_and_plot_psd()

    def _init_tfr_plotting(self):
        self.plot_widget.clear()
        self.eeg_plots.clear()
        self.plot_widget.setBackground('black')
        self.plot_widget.setLabel('bottom', "Frequency", units='Hz')
        self.plot_widget.setLabel('left', "Time Steps", units='')
        self.plot_widget.showGrid(x=True, y=False)
        # ... (codice TFR esistente)

    def _init_psd_raster_plotting(self):
        self.plot_widget.clear()
        self.eeg_plots.clear()
        self.plot_widget.setBackground('#202020')
        self.plot_widget.setLabel('bottom', "Frequency", units='Hz')
        self.plot_widget.setLabel('left', "Channels", units='')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.hideAxis('left')

        n_channels = len(self.channels)
        self.plot_widget.setYRange(-self.offset_step, n_channels * self.offset_step + self.offset_step)
        current_y_pos = (n_channels - 1) * self.offset_step
        for i, ch_name in enumerate(self.channels):
            color_val = int(255 * (i / n_channels))
            pen = pg.mkPen(color=(200, 255 - color_val, 255), width=1.5)
            curve = self.plot_widget.plot([], [], pen=pen)
            text = pg.TextItem(ch_name, anchor=(1, 0.5), color='w')
            self.plot_widget.addItem(text)
            text.setPos(0, current_y_pos)
            self.eeg_plots[ch_name] = {'curve': curve, 'y_offset': current_y_pos, 'text': text}
            current_y_pos -= self.offset_step

    @Slot(np.ndarray)
    def update_tfr_data(self, new_eeg_data_buffer):
        self.eeg_data_V = new_eeg_data_buffer * 1e-6
        self.epoch_counter += 1
        self.lcd_epochs.display(self.epoch_counter)
        if self.mode == 'Channels PSD':
            self._compute_and_plot_psd()
        else:
            self._compute_and_plot_tfr()

    def _compute_and_plot_psd(self):
        try:
            ch_types = ['eeg'] * len(self.channels)
            info = mne.create_info(ch_names=self.channels, sfreq=self.fs, ch_types=ch_types)
            raw_eeg = mne.io.RawArray(self.eeg_data_V, info, verbose=False)

            n_fft = min(int(self.fs * 2), raw_eeg.n_times)
            spectrum = raw_eeg.compute_psd(method='welch', fmin=1, fmax=45, n_fft=n_fft, verbose=False)
            psd_data, freqs = spectrum.get_data(return_freqs=True)

            # Memorizza dati per il cursore
            self.last_psd_data = psd_data
            self.last_freqs = freqs

            # --- PLOTTING ---
            psd_db = 10 * np.log10(psd_data + 1e-15)

            if self.chk_whitening.isChecked():
                plot_data = apply_spectral_whitening(psd_db, freqs)
                scale_factor = 2.0
            else:
                plot_data = psd_db
                scale_factor = 0.8

            db_min = np.percentile(plot_data, 5)
            db_max = np.percentile(plot_data, 95)
            db_range = db_max - db_min
            if db_range == 0: db_range = 1

            for i, ch_name in enumerate(self.channels):
                if ch_name not in self.eeg_plots: continue
                y_vals = plot_data[i, :]
                y_norm = (y_vals - db_min) / db_range
                y_plot = (y_norm * (self.offset_step * scale_factor)) + self.eeg_plots[ch_name]['y_offset']
                y_plot = apply_simple_smooth(y_plot, window_len=7)
                self.eeg_plots[ch_name]['curve'].setData(freqs, y_plot)
                self.eeg_plots[ch_name]['text'].setPos(freqs[0], self.eeg_plots[ch_name]['y_offset'])

            self.plot_widget.setXRange(freqs.min(), freqs.max())

            # --- UPDATE TABLES ---
            # Aggiorna tabelle con i nuovi dati (Nota: on_cursor_moved aggiornerà la colonna cursore se necessario)
            self._update_analysis_tables(psd_data, freqs)

            # Se il cursore esiste, forza l'aggiornamento dei valori del cursore sui nuovi dati
            if hasattr(self, 'v_line'):
                self.on_cursor_moved()

        except Exception as e:
            print(f"Errore PSD e Analisi: {e}")

    def _update_analysis_tables(self, psd_data, freqs):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        self.table_peaks.setRowCount(len(self.channels))
        self.table_bands.setRowCount(len(self.channels))

        bands_def = {
            'Delta': (1, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30)
        }

        # Calcola dB matrix per i picchi
        psd_db = 10 * np.log10(psd_data + 1e-15)

        for i, ch_name in enumerate(self.channels):
            # --- TAB 1: PICCHI ---
            # Trova Max Freq
            idx_max = np.argmax(psd_db[i, :])
            peak_freq = freqs[idx_max]
            max_val_db = psd_db[i, idx_max]

            self.table_peaks.setItem(i, 0, QTableWidgetItem(ch_name))
            self.table_peaks.setItem(i, 1, QTableWidgetItem(f"{peak_freq:.2f}"))
            self.table_peaks.setItem(i, 2, QTableWidgetItem(f"{max_val_db:.1f}"))
            # La colonna 3 (Cursor) viene gestita da on_cursor_moved, ma la inizializziamo vuota se serve
            if self.table_peaks.item(i, 3) is None:
                self.table_peaks.setItem(i, 3, QTableWidgetItem("--"))

            # --- TAB 2: AREE BANDE ---
            self.table_bands.setItem(i, 0, QTableWidgetItem(ch_name))

            band_col = 1
            for band_name, (low, high) in bands_def.items():
                idx_band = np.logical_and(freqs >= low, freqs <= high)
                if np.any(idx_band):
                    freqs_band = freqs[idx_band]
                    psd_band_linear = psd_data[i, idx_band]  # Integriamo la potenza lineare (uV^2/Hz)
                    band_power = np.trapezoid(psd_band_linear, freqs_band)
                else:
                    band_power = 0.0

                self.table_bands.setItem(i, band_col, QTableWidgetItem(f"{band_power:.2e}"))
                band_col += 1

    def _compute_and_plot_tfr(self):
        pass

    def closeEvent(self, event):
        self.close_signal.emit()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGControlWindow()
    window.show()
    sys.exit(app.exec())