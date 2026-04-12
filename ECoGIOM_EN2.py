########################################################################################################################
# ECoG SIMULATOR - PERSISTENT PATTERN SEARCH EDITION (FAST FFT & SINGLE CH OPTION)
# Features:
# 1. FFT-based Correlation (100x speedup)
# 2. Option to search only on Source Channel
# 3. Averaging of found events
# 4. Integrated Artifact Generator (JSON Driven)
########################################################################################################################

import json
import logging
import sys
import os
from pathlib import Path
import time
import webbrowser as wb
import datetime
import random

import mne
import numpy as np
import scipy.io as sio
import pyqtgraph as pg

from PySide6 import QtWidgets
from PySide6.QtGui import QPixmap, QColor, QBrush
from PySide6.QtUiTools import QUiLoader
from pyqtgraph.Qt import QtCore
from PySide6.QtCore import Qt, Signal, Slot, QFile
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox, QApplication, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView)

from utils_funcIOM import apply_bandpass_filter, fast_normalized_cross_correlation

# [INTEGRAZIONE] Importiamo il generatore di artefatti
from etc.artifact_genIOM import ArtifactGenerator

# --- GLOBAL PHYSICAL PARAMETERS ---
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

MAT_LOCS_KEY = 'locs'
MAT_DATA_KEY = 'data'
MNE_CHANNEL_TYPE = 'ecog'

FS_EEG = 1000
TIME_DURATION_S = 10.0
TIME_POINTS_EEG = int(FS_EEG * TIME_DURATION_S)
TIME_VECTOR_EEG = np.linspace(0, TIME_DURATION_S, TIME_POINTS_EEG, endpoint=False)


def load_ecog_locs_and_create_names(brain_filepath: str):
    print(f"Loading locations: {brain_filepath}")
    if not os.path.exists(brain_filepath):
        print(f"File not found: {brain_filepath}")
        return None, None

    try:
        mat_contents = sio.loadmat(brain_filepath)
    except Exception as e:
        print(f"ERROR loading brain .mat: {e}")
        return None, None

    if MAT_LOCS_KEY not in mat_contents:
        print(f"ERROR: Key '{MAT_LOCS_KEY}' missing.")
        return None, None

    locs_matrix = mat_contents[MAT_LOCS_KEY]
    num_electrodes = locs_matrix.shape[0]
    ch_names = [f'E {i:02d}' for i in range(1, num_electrodes + 1)]
    return locs_matrix, ch_names


def load_ecog_data_and_create_mne(data_filepath: str, locs_matrix: np.ndarray, ch_names: list, sfreq: float):
    print(f"Loading data: {data_filepath}")
    if not os.path.exists(data_filepath):
        print(f"Data file not found: {data_filepath}")
        return None

    num_channels = len(ch_names)
    if locs_matrix is None or num_channels == 0: return None

    try:
        mat_contents = sio.loadmat(data_filepath)
        data_matrix = mat_contents[MAT_DATA_KEY]
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None

    if data_matrix.shape[0] > data_matrix.shape[1] and data_matrix.shape[1] == num_channels:
        data_matrix = data_matrix.T * 1e-8
    elif data_matrix.shape[0] != num_channels:
        if data_matrix.shape[1] == num_channels:
            data_matrix = data_matrix.T * 1e-8
        else:
            print(f"CRITICAL ERROR: Incorrect data shape {data_matrix.shape} vs Channels {num_channels}.")
            return None

    ch_types = [MNE_CHANNEL_TYPE] * num_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data_matrix, info)

    locs_in_meters = locs_matrix / 1000.0
    dig_ch_pos = dict(zip(ch_names, locs_in_meters))
    montage = mne.channels.make_dig_montage(ch_pos=dig_ch_pos, coord_frame='mri')
    raw.set_montage(montage)

    template_len_time = data_matrix.shape[1]
    print("MNE RawArray created successfully.")

    return {
        'eeg_data_2d': data_matrix,
        'eeg_channels': ch_names,
        'template_len': template_len_time,
        'raw_ecog': raw,
        'locs_3d_meters': locs_in_meters,
        'eog_v': None,
        'ecg': None,
    }


def add_eeg_artifacts_dynamic(eeg_data, time_vector, fs, artifact_state):
    # Artefatti manuali aggiunti tramite GUI (slider), separati da quelli clinici del JSON
    return eeg_data


########################################################################################################################
class ECoGWorker(QtCore.QObject):
    data_ready = Signal(dict)
    dataset_loaded = Signal(int)
    static_page_ready = Signal(dict)

    def __init__(self, simulator_params, parent=None):
        super().__init__(parent)
        self.FS = simulator_params['FS']
        self.time = simulator_params['time_vector']
        self.time_points = simulator_params['time_points']
        self.EEG_CHANNELS = simulator_params['eeg_channels']
        self.cnt_filepath = simulator_params['cnt_filepath']
        locs_matrix = simulator_params['locs_matrix']
        self.plot_scale_factor = simulator_params['plot_scale_factor']
        self.lowcut = 2.0
        self.highcut = 300.0

        loaded = load_ecog_data_and_create_mne(self.cnt_filepath, locs_matrix, self.EEG_CHANNELS, self.FS)

        if loaded:
            self.template_eeg_data = loaded['eeg_data_2d']
            template_len = loaded['template_len']
            self.template_eog_v = loaded['eog_v'] if loaded['eog_v'] is not None else np.zeros(template_len)
            self.template_ecg = loaded['ecg'] if loaded['ecg'] is not None else np.zeros(template_len)
        else:
            print("Worker: Using dummy data (loading failed).")
            template_len = self.time_points
            self.template_eeg_data = np.zeros((len(self.EEG_CHANNELS), template_len))
            self.template_eog_v = np.zeros(template_len)
            self.template_ecg = np.zeros(template_len)

        self.is_running = False
        self.chunk_duration_s = simulator_params['timer_interval_ms'] / 1000.0
        self.chunk_samples = int(self.chunk_duration_s * self.FS)
        self.current_index = 0
        self.current_display_index = 0
        self.current_time_s = 0.0
        self.artifact_state = {}

        # [INTEGRAZIONE] Generatore di Artefatti
        self.artifact_gen = ArtifactGenerator(self.FS)
        self.active_anomalies = []  # Lista di anomalie ricevute dal JSON

    # [INTEGRAZIONE] Slot per ricevere la lista di anomalie aggiornata
    @Slot(list)
    def update_anomalies(self, anomalies_list):
        self.active_anomalies = anomalies_list
        # print(f"Worker: Anomalie aggiornate. Totale: {len(self.active_anomalies)}")

    @Slot()
    def _init_timer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._run_simulation_step)
        self.dataset_loaded.emit(self.template_eeg_data.shape[1])

    @Slot(int)
    def start_simulation(self, interval_ms):
        self.is_running = True
        self.timer.start(interval_ms)

    @Slot()
    def stop_simulation(self):
        self.is_running = False
        if hasattr(self, 'timer'):
            self.timer.stop()

    @Slot(str, bool, float)
    def update_artifact_state(self, artifact_key: str, enabled: bool, amplitude: float):
        if artifact_key not in self.artifact_state: self.artifact_state[artifact_key] = {}
        self.artifact_state[artifact_key]['enabled'] = enabled
        self.artifact_state[artifact_key]['amplitude'] = amplitude

    @Slot(float, float)
    def update_filter_settings(self, lowcut, highcut):
        self.lowcut = lowcut
        self.highcut = highcut

    @Slot(float)
    def update_plot_scale_factor(self, new_scale_factor):
        self.plot_scale_factor = new_scale_factor

    @Slot(int)
    def reset_display_window(self, new_time_points):
        self.time_points = new_time_points
        self.current_display_index = 0

    @Slot(int, int)
    def request_static_page(self, page_index, samples_per_page):
        if self.template_eeg_data is None: return

        total_samples = self.template_eeg_data.shape[1]
        start_sample = page_index * samples_per_page
        if start_sample >= total_samples: return
        end_sample = min(start_sample + samples_per_page, total_samples)
        eeg_chunk = self.template_eeg_data[:, start_sample:end_sample]

        eeg_chunk = eeg_chunk * self.plot_scale_factor * 1e6
        carRef = np.mean(eeg_chunk, axis=0)
        eeg_chunk = (eeg_chunk - carRef)

        if self.lowcut > 0 and self.highcut > self.lowcut:
            try:
                eeg_chunk = apply_bandpass_filter(eeg_chunk, self.lowcut, self.highcut, self.FS)
            except:
                pass

        eeg_data_dict = {ch: eeg_chunk[i, :] for i, ch in enumerate(self.EEG_CHANNELS)}
        self.static_page_ready.emit({
            'static_data': eeg_data_dict,
            'page_index': page_index,
            'num_samples_in_page': eeg_chunk.shape[1]
        })

    @Slot()
    def _run_simulation_step(self):
        if not self.is_running: return

        total_samples = self.template_eeg_data.shape[1]
        end_index = self.current_index + self.chunk_samples

        # 1. Estrazione dati dal buffer circolare
        if end_index > total_samples:
            p1 = self.template_eeg_data[:, self.current_index:total_samples]
            rem = end_index - total_samples
            p2 = self.template_eeg_data[:, 0:rem]
            eeg_base = np.hstack((p1, p2))
            self.current_index = rem
        else:
            eeg_base = self.template_eeg_data[:, self.current_index:end_index]
            self.current_index = end_index

        # 2. Scaling e Pre-processing di base
        eeg_base = eeg_base * self.plot_scale_factor * 1e6
        carRef = np.mean(eeg_base, axis=0)
        eeg_base = (eeg_base - carRef)

        # 3. Filtro Bandpass (se attivo)
        if self.lowcut > 0 and self.highcut > self.lowcut:
            try:
                eeg_base = apply_bandpass_filter(eeg_base, self.lowcut, self.highcut, self.FS)
            except:
                pass

        # [INTEGRAZIONE] 4. Applicazione Anomalie (JSON Driven)
        if self.active_anomalies:
            # Creiamo il vettore tempo assoluto per questo chunk (essenziale per seno/coseno continui)
            # self.current_time_s è il tempo di inizio simulazione
            chunk_time_vector = np.linspace(self.current_time_s,
                                            self.current_time_s + self.chunk_duration_s,
                                            self.chunk_samples, endpoint=False)

            for anomaly in self.active_anomalies:
                # 4a. Controllo Target Simulator
                target_sim = anomaly.get('target_simulator', 'ALL').upper()
                if target_sim not in ['ECOG', 'ALL']:
                    continue

                # 4b. Controllo Active
                if not anomaly.get('active', False):
                    continue

                # 4c. Selezione Canali
                target_channels_names = anomaly.get('channels', [])
                apply_to_all = len(target_channels_names) == 0

                for i, ch_name in enumerate(self.EEG_CHANNELS):
                    if apply_to_all or (ch_name in target_channels_names):
                        # 4d. Chiamata al generatore di artefatti
                        eeg_base[i, :] = self.artifact_gen.apply_anomaly(
                            eeg_base[i, :],
                            chunk_time_vector,
                            anomaly
                        )

        # 5. Artefatti manuali (Slider GUI)
        eeg_base = add_eeg_artifacts_dynamic(eeg_base, self.time, self.FS, self.artifact_state)

        # 6. Packaging dei dati
        eeg_data_dict = {ch: eeg_base[i, :] for i, ch in enumerate(self.EEG_CHANNELS)}

        self.current_time_s += (self.chunk_samples / self.FS)
        # if self.current_time_s > (total_samples / self.FS): self.current_time_s = 0

        start_disp = self.current_display_index
        self.current_display_index += self.chunk_samples
        reset = False

        if self.current_display_index >= self.time_points:
            self.current_display_index = 0
            reset = True

        self.data_ready.emit({
            'new_data_chunk': eeg_data_dict,
            'start_display_index': start_disp,
            'reset_required': reset
        })


########################################################################################################################
# CLASS for dedicate window to search patterns ---
class PatternSearchWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("Pattern recognition on recording session")
        self.resize(750, 350)

    def closeEvent(self, event):
        if self.parent() and hasattr(self.parent(), 'ui') and hasattr(self.parent().ui, 'pushBt_search'):
            self.parent().ui.pushBt_search.setChecked(False)
        super().closeEvent(event)


########################################################################################################################
class ECoGControlWindow(QWidget):
    filter_settings_changed = Signal(float, float)
    amplitude_scale_changed = Signal(float)
    # [INTEGRAZIONE] Segnale per inviare le anomalie al worker
    anomalies_updated = Signal(list)

    def __init__(self, json_anomaly_path=None, learning_manager=None, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window)

        self.worker = None
        self.thread = None
        self.search_thread = None

        self.EEG_CHANNELS = EEG_CHANNELS
        self.ALL_PLOT_CHANNELS = ALL_PLOT_CHANNELS
        self.N_ALL_CHANNELS = len(self.ALL_PLOT_CHANNELS)

        self.logger = logging.getLogger(__name__)
        self.json_anomaly_path = json_anomaly_path
        self.learning_manager = learning_manager

        # Variabile per tenere in memoria le anomalie caricate
        self.loaded_anomalies = []

        ui_path = Path('res/ecogPlotRTForm.ui')
        if not ui_path.exists():
            current_dir = Path(__file__).parent
            potential_paths = [
                current_dir / 'res/ecogPlotRTForm.ui',
                current_dir / 'ecogPlotRTForm.ui',
                Path('etc/res/ecogPlotRTForm.ui')
            ]
            for p in potential_paths:
                if p.exists():
                    ui_path = p
                    break

        ui_file = QFile(ui_path)
        if not ui_file.open:
            print(f"Errore: Impossibile aprire il file {ui_file}")
            sys.exit(-1)

        loader = QUiLoader()
        # Registriamo il widget di pyqtgraph per farlo riconoscere al loader
        loader.registerCustomWidget(pg.GraphicsLayoutWidget)
        self.ui = loader.load(ui_file)
        ui_file.close()

        # Creiamo un layout per la nostra finestra principale e ci inseriamo l'interfaccia caricata
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Togliamo i bordi vuoti
        main_layout.addWidget(self.ui)

        self.setWindowTitle(time.strftime('%d %b %Y') + '  ECoG - Persistent Pattern Search')
        self.setFixedSize(1400, 950)

        logo_path = Path('res/logos/cortex_left.png')
        if not logo_path.exists():
            logo_path = Path(__file__).parent / 'res/logos/cortex_left.png'

        if logo_path.exists() and hasattr(self.ui, 'lbl_cortex_L'):
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                self.ui.lbl_cortex_L.setPixmap(
                    pixmap.scaled(self.ui.lbl_cortex_L.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation))

        self.LABEL_X_OFFSET_S = -0.28
        self.current_time_duration_s = TIME_DURATION_S
        self.MIN_TIME_S = 2.0
        self.MAX_TIME_S = 20.0
        self.SCALE_BAR_AMPLITUDE_uV = 200
        self.SCALE_BAR_HEIGHT_UNITS = 5.0
        self.NUM_ELETRODI_PER_RIGA = 8
        self.offset_step = 10.0
        self.timer_interval_ms = 100
        self.is_simulating = False
        self.display_mode = "Sweep"

        self.eeg_plots = {}
        self.eeg_text_items = {}
        self.real_channel_row_map = {}
        self.current_visible_channels = []
        self.ALL_PLOT_CHANNELS = []

        self.current_static_page_index = 0
        self.total_static_pages = 0
        self.total_template_samples = 0
        self.was_simulating_before_paging = False

        self.match_items = []
        self.stored_pattern = None
        self.current_search_source_ch = None
        self.global_search_results = {}
        self.current_pattern_color = QColor(255, 0, 0, 100)

        self._setup_ui()
        self._setup_filter_controls()
        self._setup_amplitude_controls()
        self.all_found_events_indices = []
        self.current_event_nav_index = -1

        self.ui.comboBoxMS.addItem('speech_basic', "ecogDataSets/speech_basic/speech_basic/data")
        self.ui.comboBoxMS.addItem('motor_basic', "ecogDataSets/motor_basic/motor_basic/data")
        self.ui.comboBoxMS.currentIndexChanged.connect(self.on_select_cognitive_function)

        self.directory_target_data = Path("ecogDataSets/speech_basic/speech_basic/data")
        self.directory_target_brains = Path("ecogDataSets/speech_basic/speech_basic/brains")
        self.estensione_file = "*.mat"

        self.popola_combobox()
        self.ui.comboExamples.currentIndexChanged.connect(self.on_selezione_cambiata)

        if self.ui.comboExamples.count() > 0:
            self.on_selezione_cambiata(0)

        # [INTEGRAZIONE] Caricamento del JSON
        self.load_json_anomaly()

    def _setup_ui(self):
        self.ui.btn_start_stop.clicked.connect(self.start_stop_simulation)
        self.ui.pushBtClose.clicked.connect(self.close)
        self.ui.pushBtHelp.clicked.connect(self.helpECoG)
        self.ui.spinBoxRows.valueChanged.connect(self.update_channel_display)
        self.ui.checkBoxChAll.stateChanged.connect(self.update_channel_display)
        self.ui.btn_time_inc.clicked.connect(self._on_time_button_inc_clicked)
        self.ui.btn_time_dec.clicked.connect(self._on_time_button_dec_clicked)
        self.ui.checkScroll.stateChanged.connect(self._on_scroll_mode_changed)

        self.ui.label_time_base.setText(f"{self.current_time_duration_s:.0f} s")
        self.ui.checkScroll.setChecked(True)
        self.ui.btn_time_inc.setEnabled(False)
        self.ui.btn_time_dec.setEnabled(False)

        if hasattr(self.ui, 'pushBt_prev') and hasattr(self.ui, 'pushBt_next'):
            self.ui.pushBt_prev.clicked.connect(self.nav_prev_match)
            self.ui.pushBt_next.clicked.connect(self.nav_next_match)
            self.ui.pushBt_prev.setEnabled(False)
            self.ui.pushBt_next.setEnabled(False)

        if hasattr(self.ui, 'pushBt_search'):
            self.ui.pushBt_search.setCheckable(True)
            self.ui.pushBt_search.clicked.connect(self.toggle_pattern_search_panel)

        self._setup_pattern_search_ui()

    def _setup_pattern_search_ui(self):
        self.search_window = PatternSearchWindow(parent=self)
        layout_main = QVBoxLayout()
        self.search_window.setLayout(layout_main)

        grp_controls = QGroupBox("Controlli")
        layout_controls = QHBoxLayout()
        grp_controls.setLayout(layout_controls)

        self.btn_toggle_roi = QtWidgets.QPushButton("Seleziona ROI")
        self.btn_toggle_roi.setCheckable(True)
        self.btn_toggle_roi.clicked.connect(self.toggle_search_roi)
        layout_controls.addWidget(self.btn_toggle_roi)

        layout_controls.addWidget(QLabel("Ch:"))
        self.combo_source_ch = QtWidgets.QComboBox()
        self.combo_source_ch.setMinimumWidth(60)
        layout_controls.addWidget(self.combo_source_ch)

        self.btn_lock_pattern = QtWidgets.QPushButton("2. Memorizza")
        self.btn_lock_pattern.setStyleSheet("background-color: #e0f7fa;")
        self.btn_lock_pattern.clicked.connect(self.lock_pattern_template)
        layout_controls.addWidget(self.btn_lock_pattern)

        layout_controls.addWidget(QLabel("Soglia:"))
        self.spin_corr_thresh = QtWidgets.QDoubleSpinBox()
        self.spin_corr_thresh.setRange(0.1, 0.99)
        self.spin_corr_thresh.setValue(0.80)
        self.spin_corr_thresh.setSingleStep(0.05)
        self.spin_corr_thresh.setFixedWidth(50)
        layout_controls.addWidget(self.spin_corr_thresh)

        # CHECKBOX PER RICERCA SINGOLO CANALE
        self.chk_search_single_ch = QtWidgets.QCheckBox("Solo Ch Sorgente")
        self.chk_search_single_ch.setChecked(True)  # Default: attivo per velocità
        layout_controls.addWidget(self.chk_search_single_ch)

        self.btn_search_all = QtWidgets.QPushButton("3. Cerca")
        self.btn_search_all.setStyleSheet("background-color: #ffccbc;")
        self.btn_search_all.clicked.connect(self.start_global_search)
        self.btn_search_all.setEnabled(False)
        layout_controls.addWidget(self.btn_search_all)

        self.btn_clear_search = QtWidgets.QPushButton("Reset")
        self.btn_clear_search.clicked.connect(self.clear_all_search_data)
        layout_controls.addWidget(self.btn_clear_search)

        layout_main.addWidget(grp_controls)

        # RIGA 2: INFO VISIVE
        layout_info = QHBoxLayout()

        self.pattern_preview_plot = pg.PlotWidget(title="Pattern Model vs Average")
        self.pattern_preview_plot.setBackground('white')
        self.pattern_preview_plot.setFixedWidth(250)
        self.pattern_preview_plot.showGrid(x=True, y=True, alpha=0.3)
        self.pattern_preview_plot.hideAxis('left')
        layout_info.addWidget(self.pattern_preview_plot)

        self.table_pattern_log = QTableWidget()
        self.table_pattern_log.setColumnCount(7)
        self.table_pattern_log.setHorizontalHeaderLabels([
            "Ora Log", "Sorgente", "Pat. (ms)", "Amp (µV)", "Rec. Dur", "N Ch", "Hits"
        ])

        header = self.table_pattern_log.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)

        self.table_pattern_log.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_pattern_log.verticalHeader().setVisible(False)
        layout_info.addWidget(self.table_pattern_log)

        layout_main.addLayout(layout_info)

        self.search_roi = pg.LinearRegionItem(values=[0, 1], brush=pg.mkBrush(0, 0, 255, 30))
        self.search_roi.setZValue(100)
        self.roi_added_to_plot = False

    def change_display_mode(self, index):
        if index == 0:
            self.display_mode = "Sweep"
        else:
            self.display_mode = "Page"
            self.eeg_display_buffer[:] = 0
            self._refresh_full_plot()

    def _refresh_full_plot(self):
        if not hasattr(self, 'eeg_display_buffer') or self.eeg_display_buffer is None:
            return
        current_offset = 0.0
        for i, ch_name in enumerate(self.ALL_PLOT_CHANNELS):
            if ch_name not in self.eeg_plots:
                continue
            if ch_name not in self.current_visible_channels:
                continue
            raw_data = self.eeg_display_buffer[i, :]
            offset_data = raw_data + current_offset
            self.eeg_plots[ch_name]['curve'].setData(self.time_vector_s, offset_data)
            current_offset += self.offset_step

    def toggle_pattern_search_panel(self):
        if self.ui.pushBt_search.isChecked():
            self.search_window.show()
            self.search_window.raise_()
            self.search_window.activateWindow()
        else:
            self.search_window.hide()
            if self.btn_toggle_roi.isChecked():
                self.btn_toggle_roi.setChecked(False)
                self.toggle_search_roi()

    def toggle_search_roi(self):
        if not hasattr(self, 'eeg_plt'): return
        if not self.roi_added_to_plot:
            self.eeg_plt.addItem(self.search_roi)
            self.roi_added_to_plot = True

        if self.btn_toggle_roi.isChecked():
            vr = self.eeg_plt.viewRange()[0]
            c, w = (vr[0] + vr[1]) / 2, (vr[1] - vr[0]) * 0.1
            self.search_roi.setRegion([c - w / 2, c + w / 2])
            self.search_roi.show()
            self.combo_source_ch.clear()
            self.combo_source_ch.addItems(self.current_visible_channels)
        else:
            self.search_roi.hide()

    def lock_pattern_template(self):
        if not self.btn_toggle_roi.isChecked(): return
        min_x, max_x = self.search_roi.getRegion()

        try:
            idx_start = np.searchsorted(self.time_vector_s, min_x)
            idx_end = np.searchsorted(self.time_vector_s, max_x)
        except:
            return

        source_ch = self.combo_source_ch.currentText()
        if source_ch not in self.eeg_plots: return

        ch_idx = self.eeg_plots[source_ch]['index']
        raw_pattern = self.eeg_display_buffer[ch_idx, idx_start:idx_end]

        if len(raw_pattern) < 5:
            print("Pattern troppo corto.")
            return

        self.stored_pattern = raw_pattern.copy()
        self.current_search_source_ch = source_ch

        h = random.randint(0, 359)
        self.current_pattern_color = QColor.fromHsv(h, 240, 240, 100)
        solid_color = QColor.fromHsv(h, 240, 200)

        self.pattern_preview_plot.clear()
        preview_data = raw_pattern - np.mean(raw_pattern)
        self.pattern_preview_plot.plot(preview_data, pen=pg.mkPen(color=solid_color, width=2), name="Model")

        pattern_duration_ms = (len(raw_pattern) / FS_EEG) * 1000
        amplitude_uv = np.ptp(raw_pattern)
        current_log_time = datetime.datetime.now().strftime("%H:%M:%S")

        if self.worker and self.worker.template_eeg_data is not None:
            total_samples = self.worker.template_eeg_data.shape[1]
            total_seconds = int(total_samples / FS_EEG)
            rec_duration_str = str(datetime.timedelta(seconds=total_seconds))
            num_channels_analyzed = len(self.worker.EEG_CHANNELS)
        else:
            rec_duration_str = "N/A"
            num_channels_analyzed = 0

        row_idx = self.table_pattern_log.rowCount()
        self.table_pattern_log.insertRow(row_idx)

        self.table_pattern_log.setItem(row_idx, 0, QTableWidgetItem(current_log_time))

        item_source = QTableWidgetItem(source_ch)
        item_source.setBackground(solid_color)
        item_source.setForeground(QBrush(Qt.GlobalColor.black))
        self.table_pattern_log.setItem(row_idx, 1, item_source)

        self.table_pattern_log.setItem(row_idx, 2, QTableWidgetItem(f"{pattern_duration_ms:.1f}"))
        self.table_pattern_log.setItem(row_idx, 3, QTableWidgetItem(f"{amplitude_uv:.1f}"))
        self.table_pattern_log.setItem(row_idx, 4, QTableWidgetItem(rec_duration_str))
        self.table_pattern_log.setItem(row_idx, 5, QTableWidgetItem(str(num_channels_analyzed)))

        item_hits = QTableWidgetItem("Ready")
        item_hits.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item_hits.setBackground(QColor("#fff3e0"))
        self.table_pattern_log.setItem(row_idx, 6, item_hits)

        self.table_pattern_log.scrollToBottom()

        self.btn_search_all.setEnabled(True)
        self.btn_lock_pattern.setText(f"OK ({len(raw_pattern)})")
        self.btn_lock_pattern.setStyleSheet("background-color: #a5d6a7;")

    def start_global_search(self):
        if self.stored_pattern is None: return
        if not self.worker: return

        threshold = self.spin_corr_thresh.value()

        # LOGICA FILTRO CANALI (All vs Single)
        if self.chk_search_single_ch.isChecked() and self.current_search_source_ch:
            # Ricerca SOLO sul canale sorgente
            target_channels = [self.current_search_source_ch]

            # FIX: Trova l'indice della riga basandosi sulla lista canali del worker, NON sui plot
            try:
                ch_idx = self.worker.EEG_CHANNELS.index(self.current_search_source_ch)
                # Slice della matrice: prendiamo solo quella riga e la manteniamo 2D (1, Samples)
                target_data = self.worker.template_eeg_data[ch_idx:ch_idx + 1, :]
            except ValueError:
                print(f"Errore: Canale {self.current_search_source_ch} non trovato nei dati raw.")
                return
        else:
            # Ricerca su TUTTI i canali
            target_channels = self.worker.EEG_CHANNELS
            target_data = self.worker.template_eeg_data

        self.ui.label_search.setText("Start searching (FFT)...")
        self.btn_search_all.setEnabled(False)
        self.btn_search_all.setText("Searching...")

        if self.search_thread is not None:
            self.cleanup_search_thread()

        self.search_thread = QtCore.QThread()
        self.search_worker = PatternSearchWorker(
            full_data_matrix=target_data,
            channel_names=target_channels,
            pattern_template=self.stored_pattern,
            threshold=threshold,
            fs=FS_EEG
        )

        self.search_worker.moveToThread(self.search_thread)
        self.search_thread.started.connect(self.search_worker.run_search)
        self.search_worker.search_finished.connect(self.on_global_search_finished)
        self.search_worker.search_finished.connect(self.cleanup_search_thread)
        self.search_worker.error_occurred.connect(lambda e: print(f"Search Error: {e}"))
        self.search_worker.error_occurred.connect(self.cleanup_search_thread)

        self.search_thread.start()

    def cleanup_search_thread(self):
        if hasattr(self, 'search_thread') and self.search_thread is not None:
            self.search_thread.quit()
            self.search_thread.wait()
            self.search_thread = None
            self.search_worker = None
            self.btn_search_all.setEnabled(True)
            self.btn_search_all.setText("3. Cerca")
            self.btn_search_all.setStyleSheet("background-color: #ffccbc;")

    @Slot(dict)
    def on_global_search_finished(self, results):
        # disactive ROI
        self.toggle_search_roi()

        self.global_search_results = results
        all_indices = []
        for ch_name, indices in results.items():
            all_indices.extend(indices)

        all_indices = sorted(list(set(all_indices)))

        if all_indices:
            unique_events = [all_indices[0]]
            min_dist = len(self.stored_pattern) // 2
            for idx in all_indices[1:]:
                if idx - unique_events[-1] > min_dist:
                    unique_events.append(idx)
            self.all_found_events_indices = unique_events
        else:
            self.all_found_events_indices = []

        total_hits = len(self.all_found_events_indices)
        self.current_event_nav_index = -1

        # --- CALCOLO MEDIA (AVERAGING) ---
        # Verifica se abbiamo risultati sul canale sorgente
        if self.current_search_source_ch and self.current_search_source_ch in results:
            hits_on_source = results[self.current_search_source_ch]

            if len(hits_on_source) > 0 and self.worker and self.stored_pattern is not None:
                try:
                    # FIX: Recupera l'indice corretto dalla lista canali del worker (Fonte della verità)
                    if self.current_search_source_ch in self.worker.EEG_CHANNELS:
                        ch_idx_map = self.worker.EEG_CHANNELS.index(self.current_search_source_ch)
                        print(f"channel index from worker = {ch_idx_map}")

                        # Estrai l'intera traccia del canale
                        full_signal = self.worker.template_eeg_data[ch_idx_map, :].copy()
                        full_signal = full_signal * self.plot_scale_factor * 1e6

                        pat_len = len(self.stored_pattern)
                        segments = []

                        for start_idx in hits_on_source:
                            end_idx = start_idx + pat_len
                            # Controllo bounds sicuro
                            if end_idx <= full_signal.size:
                                seg = full_signal[start_idx:end_idx]
                                segments.append(seg)

                        if segments:
                            # Stack e media
                            avg_pattern = np.mean(segments, axis=0)

                            # Debug: Stampa l'ampiezza per verificare che non sia zero
                            amp_avg = np.ptp(avg_pattern)
                            print(f"Media calcolata su {len(segments)} segmenti. Ampiezza media: {amp_avg:.2f} uV")

                            self.pattern_preview_plot.clear()

                            # 1. Template originale (Rosso, Centrato)
                            orig_centered = self.stored_pattern - np.mean(self.stored_pattern)
                            self.pattern_preview_plot.plot(orig_centered,
                                                           pen=pg.mkPen(color=self.current_pattern_color, width=2),
                                                           name="Model")
                            # print(f"orig min/max {np.max(orig_centered)} {np.min(orig_centered)}")

                            # 2. Media trovata (Nero, Spesso, Centrato)
                            avg_centered = avg_pattern - np.mean(avg_pattern)
                            # print(f"avg min/max {np.max(avg_centered)} {np.min(avg_centered)}")
                            #
                            self.pattern_preview_plot.plot(avg_centered,
                                                           pen=pg.mkPen(color='k', width=3,
                                                                        style=Qt.PenStyle.SolidLine),
                                                           name="Average")
                    else:
                        print(
                            f"Errore Averaging: Canale {self.current_search_source_ch} non trovato nella lista Worker.")

                except Exception as e:
                    print(f"Errore calcolo media eventi: {e}")

        self.ui.label_search.setText(f"Ricerca finita: {total_hits} eventi unici.")

        has_results = total_hits > 0
        if hasattr(self.ui, 'pushBt_prev'): self.ui.pushBt_prev.setEnabled(has_results)
        if hasattr(self.ui, 'pushBt_next'): self.ui.pushBt_next.setEnabled(has_results)

        row_count = self.table_pattern_log.rowCount()
        if row_count > 0:
            last_row = row_count - 1
            item_hits = self.table_pattern_log.item(last_row, 6)
            item_hits.setText(str(total_hits))
            if total_hits > 0:
                item_hits.setBackground(QColor("#c8e6c9"))
            else:
                item_hits.setBackground(QColor("#ffcdd2"))
        self.draw_stored_matches_on_current_view()

    def draw_stored_matches_on_current_view(self):
        for item in self.match_items: self.eeg_plt.removeItem(item)
        self.match_items = []

        if not self.global_search_results or self.stored_pattern is None: return

        if not self.ui.checkScroll.isChecked():
            start_abs = self.current_static_page_index * self.display_samples
            end_abs = start_abs + self.display_samples
            pat_len = len(self.stored_pattern)

            for ch_name, hits in self.global_search_results.items():
                if ch_name not in self.current_visible_channels: continue

                hits_in_page = [h for h in hits if start_abs <= h < end_abs]
                if not hits_in_page: continue

                vis_y = self.eeg_text_items[ch_name].pos().y()

                for h_idx in hits_in_page:
                    rel_sample = h_idx - start_abs
                    if rel_sample < len(self.time_vector_s):
                        t_start = self.time_vector_s[rel_sample]
                        t_width = pat_len / FS_EEG

                        rect = QtWidgets.QGraphicsRectItem(t_start, vis_y - self.offset_step * 0.4, t_width,
                                                           self.offset_step * 0.8)

                        rect.setBrush(QBrush(self.current_pattern_color))
                        rect.setPen(pg.mkPen(None))
                        rect.setZValue(50)

                        self.eeg_plt.addItem(rect)
                        self.match_items.append(rect)

    def nav_next_match(self):
        if not self.all_found_events_indices: return
        if self.current_event_nav_index < len(self.all_found_events_indices) - 1:
            self.current_event_nav_index += 1
            self._jump_to_event_index(self.current_event_nav_index)
        else:
            self.ui.label_search.setText("Ultimo evento raggiunto.")

    def nav_prev_match(self):
        if not self.all_found_events_indices: return
        if self.current_event_nav_index > 0:
            self.current_event_nav_index -= 1
            self._jump_to_event_index(self.current_event_nav_index)
        else:
            self.ui.label_search.setText("First event achieved...")

    def _jump_to_event_index(self, list_index):
        target_sample = self.all_found_events_indices[list_index]
        target_page = int(target_sample // self.display_samples)

        if self.ui.checkScroll.isChecked():
            self.ui.checkScroll.setChecked(False)

        self.current_static_page_index = target_page
        self._request_page_from_worker(target_page)
        self._update_button_labels()
        self._update_navigation_state()

        total = len(self.all_found_events_indices)
        time_s = target_sample / FS_EEG
        self.ui.label_search.setText(f"Evento {list_index + 1}/{total} a {time_s:.2f}s")

    def clear_all_search_data(self):
        self.global_search_results = {}
        self.stored_pattern = None
        self.all_found_events_indices = []
        self.current_event_nav_index = -1

        self.btn_search_all.setEnabled(False)
        self.btn_lock_pattern.setText("2. Memorizza")
        self.btn_lock_pattern.setStyleSheet("background-color: #e0f7fa;")

        if hasattr(self.ui, 'pushBt_prev'): self.ui.pushBt_prev.setEnabled(False)
        if hasattr(self.ui, 'pushBt_next'): self.ui.pushBt_next.setEnabled(False)

        for item in self.match_items: self.eeg_plt.removeItem(item)
        self.match_items = []
        self.pattern_preview_plot.clear()
        self.table_pattern_log.setRowCount(0)

    def start_stop_simulation(self):
        if self.worker is None:
            return
        if not self.is_simulating:
            self.is_simulating = True
            self.ui.btn_start_stop.setText('Stop EEG')
            QtCore.QMetaObject.invokeMethod(self.worker, "start_simulation",
                                            QtCore.Qt.ConnectionType.QueuedConnection,
                                            QtCore.Q_ARG(int, self.timer_interval_ms))
        else:
            self.is_simulating = False
            self.ui.btn_start_stop.setText('Start EEG')
            QtCore.QMetaObject.invokeMethod(self.worker, "stop_simulation",
                                            QtCore.Qt.ConnectionType.QueuedConnection)
        self._update_navigation_state()

    def on_select_cognitive_function(self, index):
        file_data = self.ui.comboBoxMS.currentData()
        self.directory_target_data = Path(file_data)
        file_data_brains = file_data.replace('/data', '/brains')
        self.directory_target_brains = Path(file_data_brains)
        self.popola_combobox()

    def popola_combobox(self):
        self.ui.comboExamples.clear()
        file_paths = sorted(self.directory_target_data.glob(self.estensione_file))
        if not file_paths:
            self.ui.comboExamples.addItem("Nessun file", None)
            self.ui.comboExamples.setEnabled(False)
            return
        self.ui.comboExamples.setEnabled(True)
        for fp in file_paths:
            lbl = fp.stem.replace('_', ' ').title()
            self.ui.comboExamples.addItem(lbl, str(fp))

    @Slot(int)
    def on_selezione_cambiata(self, index):
        file_ecog_data = self.ui.comboExamples.currentData()
        if not file_ecog_data: return
        path_data = Path(file_ecog_data)
        prefix = path_data.stem.split('_')[0]
        path_brain = self.directory_target_brains / f"{prefix}_brain.mat"
        if not path_brain.exists():
            print(f"Brain not found: {prefix}")
            return
        self.load_new_dataset(str(file_ecog_data), str(path_brain))

    def load_json_anomaly(self):
        if not self.json_anomaly_path:
            return

        self.logger.info(f"Caricamento anomalia: {self.json_anomaly_path}")
        if os.path.exists(self.json_anomaly_path):
            try:
                with open(self.json_anomaly_path, 'r') as f:
                    data = json.load(f)

                # [INTEGRAZIONE] Navigazione struttura JSON specifica
                # root -> scenario_specific_config -> anomalies
                scenario_config = data.get('scenario_specific_config', {})
                anomalies = scenario_config.get('anomalies', [])

                self.loaded_anomalies = anomalies
                print(f"JSON caricato: trovate {len(anomalies)} anomalie {anomalies}")

                if self.worker:
                    self.anomalies_updated.emit(self.loaded_anomalies)

            except Exception as e:
                print(f"Errore lettura JSON: {e}")
        else:
            print(f"File JSON non trovato: {self.json_anomaly_path}")

    def helpECoG(self):
        help_path = Path('help_docs/paper/Tut_ECoG.pdf')
        if help_path.exists():
            wb.open_new(str(help_path))
        else:
            print(f"Help file not found: {help_path}")

    def _recalculate_time_variables(self):
        self.display_samples = int(FS_EEG * self.current_time_duration_s)
        self.time_vector_s = np.linspace(0, self.current_time_duration_s, self.display_samples, endpoint=False)
        if self.total_template_samples > 0 and self.display_samples > 0:
            self.total_static_pages = int(np.ceil(self.total_template_samples / self.display_samples))
        else:
            self.total_static_pages = 0
        self.current_static_page_index = 0
        self._update_button_labels()
        self._update_navigation_state()
        self.eeg_display_buffer = np.zeros((self.N_ALL_CHANNELS, self.display_samples))

    def _setup_amplitude_controls(self):
        self.ui.slider_amp.setMinimum(20)
        self.ui.slider_amp.setMaximum(500)
        self.ui.slider_amp.setValue(int(self.SCALE_BAR_AMPLITUDE_uV))
        self.ui.label_amp.setText(f"{int(self.SCALE_BAR_AMPLITUDE_uV)} µV")
        self.ui.slider_amp.valueChanged.connect(self._on_amplitude_slider_changed)

    def _update_button_labels(self):
        if self.ui.checkScroll.isChecked():
            self.ui.btn_time_inc.setText("Base T. +")
            self.ui.btn_time_dec.setText("Base T. -")
            self.ui.label_time_base.setText(f"{self.current_time_duration_s:.0f} s")
        else:
            self.ui.btn_time_inc.setText("Page >>")
            self.ui.btn_time_dec.setText("<< Page")
            self.ui.label_time_base.setText(f"Pag. {self.current_static_page_index + 1} / {self.total_static_pages}")

    def _update_navigation_state(self):
        is_scrolling = self.ui.checkScroll.isChecked()
        can_use = is_scrolling or (not is_scrolling and not self.is_simulating)
        self.ui.btn_time_inc.setEnabled(can_use)
        self.ui.btn_time_dec.setEnabled(can_use)

        if not is_scrolling and can_use:
            can_go_back = bool(self.current_static_page_index > 0)
            can_go_fwd = bool(self.current_static_page_index < (self.total_static_pages - 1))
            self.ui.btn_time_dec.setEnabled(can_go_back)
            self.ui.btn_time_inc.setEnabled(can_go_fwd)

    @Slot()
    def _on_time_button_inc_clicked(self):
        if self.ui.checkScroll.isChecked():
            self._change_time_base(1.0)
        else:
            self._navigate_static_page(1)

    @Slot()
    def _on_time_button_dec_clicked(self):
        if self.ui.checkScroll.isChecked():
            self._change_time_base(-1.0)
        else:
            self._navigate_static_page(-1)

    def _navigate_static_page(self, direction):
        if self.is_simulating or self.ui.checkScroll.isChecked(): return
        new_p = int(np.clip(self.current_static_page_index + direction, 0, self.total_static_pages - 1))
        if new_p == self.current_static_page_index: return
        self.current_static_page_index = new_p
        self._request_page_from_worker(new_p)
        self._update_button_labels()
        self._update_navigation_state()

    def _request_page_from_worker(self, page_index):
        if self.worker:
            self.eeg_display_buffer[:] = 0
            QtCore.QMetaObject.invokeMethod(self.worker, "request_static_page",
                                            QtCore.Qt.ConnectionType.QueuedConnection, QtCore.Q_ARG(int, page_index),
                                            QtCore.Q_ARG(int, self.display_samples))

    @Slot(int)
    def _on_dataset_loaded(self, total_samples):
        self.total_template_samples = total_samples
        self.current_static_page_index = 0
        if self.display_samples > 0:
            self.total_static_pages = int(np.ceil(self.total_template_samples / self.display_samples))
        self._update_navigation_state()
        self._update_button_labels()

        if self.ui.checkScroll.isChecked():
            self.start_stop_simulation()
        else:
            self._request_page_from_worker(0)

    @Slot(dict)
    def _on_static_page_received(self, data_packet):
        static_data = data_packet.get('static_data')
        n_samp = data_packet.get('num_samples_in_page', 0)
        if not static_data: return
        self.eeg_display_buffer[:] = 0
        current_offset = 0.0
        for ch in self.current_visible_channels:
            if ch in static_data and ch in self.eeg_plots:
                i = self.eeg_plots[ch]['index']
                self.eeg_display_buffer[i, :n_samp] = static_data[ch]
                self.eeg_plots[ch]['curve'].setData(self.time_vector_s, self.eeg_display_buffer[i, :] + current_offset)
                current_offset += self.offset_step
        self.draw_stored_matches_on_current_view()

    @Slot(int)
    def _on_scroll_mode_changed(self, state):
        is_scrolling = (state == Qt.CheckState.Checked.value)
        self.ui.btn_start_stop.setEnabled(is_scrolling)
        self._update_navigation_state()
        self._update_button_labels()

        if is_scrolling:
            for item in self.match_items:
                self.eeg_plt.removeItem(item)
            self.match_items = []

            if hasattr(self, 'eeg_display_buffer'):
                self.eeg_display_buffer[:] = 0
                self.update_channel_display()

            if self.worker:
                QtCore.QMetaObject.invokeMethod(self.worker, "reset_display_window",
                                                QtCore.Qt.ConnectionType.QueuedConnection,
                                                QtCore.Q_ARG(int, self.display_samples))

            if self.was_simulating_before_paging:
                if not self.is_simulating:
                    self.start_stop_simulation()
                self.was_simulating_before_paging = False
        else:
            self.was_simulating_before_paging = self.is_simulating
            if self.is_simulating:
                self.start_stop_simulation()
            self._request_page_from_worker(self.current_static_page_index)

    @Slot(int)
    def _on_amplitude_slider_changed(self, value):
        self.SCALE_BAR_AMPLITUDE_uV = float(value)
        self.ui.label_amp.setText(f"{value} µV")
        if hasattr(self, 'scale_bar_text'):
            self.scale_bar_text.setText(f"{value} µV")
        self.plot_scale_factor = self.SCALE_BAR_HEIGHT_UNITS / self.SCALE_BAR_AMPLITUDE_uV
        self.amplitude_scale_changed.emit(self.plot_scale_factor)

    @Slot(float)
    def _change_time_base(self, delta_s):
        was_simulating = self.is_simulating
        if was_simulating:
            self.start_stop_simulation()
            QtCore.QThread.msleep(50)

        new_d = np.clip(self.current_time_duration_s + delta_s, self.MIN_TIME_S, self.MAX_TIME_S)
        if new_d == self.current_time_duration_s:
            if was_simulating: self.start_stop_simulation()
            return

        self.current_time_duration_s = round(new_d, 1)
        self._recalculate_time_variables()
        self.eeg_plt.setXRange(self.LABEL_X_OFFSET_S, self.time_vector_s[-1])
        self.update_channel_display()

        if self.worker:
            QtCore.QMetaObject.invokeMethod(self.worker, "reset_display_window",
                                            QtCore.Qt.ConnectionType.QueuedConnection,
                                            QtCore.Q_ARG(int, self.display_samples))
        if was_simulating:
            self.start_stop_simulation()

    def _setup_filter_controls(self):
        self.ui.slider_lowcut.valueChanged.connect(self._on_filter_slider_changed)
        self.ui.slider_highcut.valueChanged.connect(self._on_filter_slider_changed)
        self.ui.label_lowcut.setText("0 Hz")
        self.ui.label_highcut.setText("200 Hz")

    @Slot()
    def _on_filter_slider_changed(self):
        l = self.ui.slider_lowcut.value()
        h = self.ui.slider_highcut.value()
        if l >= h:
            if self.sender() == self.ui.slider_lowcut:
                self.ui.slider_highcut.setValue(l + 1);
                h = l + 1
            else:
                self.ui.slider_lowcut.setValue(h - 1);
                l = h - 1
        self.ui.label_lowcut.setText(f"{l} Hz")
        self.ui.label_highcut.setText(f"{h} Hz")
        self.filter_settings_changed.emit(float(l), float(h))

    def update_channel_display(self):
        all_chk = self.ui.checkBoxChAll.isChecked()
        row_idx = self.ui.spinBoxRows.value()
        self.ui.lblrow.setText(str(row_idx))
        self.ui.spinBoxRows.setEnabled(not all_chk)

        if all_chk:
            target = self.ALL_PLOT_CHANNELS
            if hasattr(self, 'electrode_buttons'):
                for b in self.electrode_buttons: b.setText("-"); b.setEnabled(False)
        else:
            target = self.real_channel_row_map.get(row_idx, [])
            labs = self.channel_map.get(row_idx, [])
            if hasattr(self, 'electrode_buttons'):
                for i, b in enumerate(self.electrode_buttons):
                    if i < len(labs) and i < len(target):
                        b.setText(labs[i])
                        b.setEnabled(True)
                    else:
                        b.setText("-");
                        b.setEnabled(False)

        self.current_visible_channels = target
        curr_off = 0.0

        if hasattr(self, 'eeg_plots'):
            for ch in self.ALL_PLOT_CHANNELS:
                if ch not in self.eeg_plots: continue
                c = self.eeg_plots[ch]['curve']
                t = self.eeg_text_items[ch]
                if ch in target:
                    c.show();
                    t.show()
                    t.setPos(self.LABEL_X_OFFSET_S, curr_off)
                    i = self.eeg_plots[ch]['index']
                    if hasattr(self, 'eeg_display_buffer'):
                        c.setData(self.time_vector_s, self.eeg_display_buffer[i, :] + curr_off)
                    curr_off += self.offset_step
                else:
                    c.hide();
                    t.hide()
            Y_max = self.offset_step * (len(target) + 0.5)
            self.eeg_plt.setYRange(-self.offset_step * 0.5, Y_max, padding=0)
            self.draw_stored_matches_on_current_view()

    def load_new_dataset(self, f_data, f_brain):
        self.clear_all_search_data()

        if self.is_simulating:
            self.is_simulating = False
            self.ui.btn_start_stop.setText('Start EEG')

        if self.thread is not None:
            if self.thread.isRunning():
                self.thread.quit()
                self.thread.wait(2000)
            if self.worker:
                self.worker.deleteLater()
                self.worker = None
            self.thread.deleteLater()
            self.thread = None

        locs, chs = load_ecog_locs_and_create_names(f_brain)
        if not chs:
            print("Errore: Caricamento canali fallito.")
            return

        self.dynamic_eeg_channels = chs
        self.ALL_PLOT_CHANNELS = chs
        self.N_ALL_CHANNELS = len(chs)
        self.NUM_RIGHE = max(1, len(chs) // self.NUM_ELETRODI_PER_RIGA)
        self.channel_map = {r + 1: [f'R{r + 1}-El{i}' for i in range(1, 9)] for r in range(self.NUM_RIGHE)}

        if hasattr(self.ui, 'pushBt_el1'):
            self.electrode_buttons = [
                self.ui.pushBt_el1, self.ui.pushBt_el2, self.ui.pushBt_el3, self.ui.pushBt_el4,
                self.ui.pushBt_el5, self.ui.pushBt_el6, self.ui.pushBt_el7, self.ui.pushBt_el8,
            ]
        else:
            self.electrode_buttons = []

        self.real_channel_row_map = {r + 1: chs[r * 8:(r + 1) * 8] for r in range(self.NUM_RIGHE)}
        self.current_visible_channels = self.real_channel_row_map.get(1, [])
        self.ui.spinBoxRows.setMaximum(self.NUM_RIGHE)
        self.ui.spinBoxRows.setValue(1)
        self.plot_scale_factor = self.SCALE_BAR_HEIGHT_UNITS / self.SCALE_BAR_AMPLITUDE_uV

        params = {
            'FS': FS_EEG, 'time_vector': TIME_VECTOR_EEG, 'time_points': int(FS_EEG * self.current_time_duration_s),
            'eeg_channels': chs, 'aux_channels': [], 'cnt_filepath': f_data,
            'locs_matrix': locs, 'timer_interval_ms': self.timer_interval_ms,
            'plot_scale_factor': self.plot_scale_factor
        }

        self.thread = QtCore.QThread()
        self.worker = ECoGWorker(params)
        self.worker.moveToThread(self.thread)

        self.worker.data_ready.connect(self.update_eeg_plot)
        self.worker.dataset_loaded.connect(self._on_dataset_loaded)
        self.worker.static_page_ready.connect(self._on_static_page_received)
        self.filter_settings_changed.connect(self.worker.update_filter_settings)
        self.amplitude_scale_changed.connect(self.worker.update_plot_scale_factor)

        # [INTEGRAZIONE] Connessione segnale anomalie
        self.anomalies_updated.connect(self.worker.update_anomalies)

        self.thread.started.connect(self.worker._init_timer)

        self.thread.start()

        self._rebuild_plot_area()
        self.update_channel_display()

        # [INTEGRAZIONE] Invia anomalie caricate al nuovo worker
        if self.loaded_anomalies:
            self.anomalies_updated.emit(self.loaded_anomalies)

    @Slot(dict)
    def update_eeg_plot(self, data_packet):
        if 'new_data_chunk' not in data_packet: return

        eeg_data = data_packet['new_data_chunk']
        aux_data = data_packet.get('aux_data_chunk', {})
        all_new_data = {**eeg_data, **aux_data}
        raw_start_idx = data_packet.get('start_display_index', 0)

        vals = list(eeg_data.values())
        if not vals: return
        chunk_len = vals[0].size

        start_idx = raw_start_idx % self.display_samples
        end_idx = start_idx + chunk_len

        reset_required = data_packet.get('reset_required', False)
        if reset_required and self.display_mode == "Page":
            self.eeg_display_buffer[:] = 0

        for i, ch_name in enumerate(self.ALL_PLOT_CHANNELS):
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

        if 'ecg_analysis' in data_packet and hasattr(self.ui, 'lbl_bpm'):
            stats = data_packet['ecg_analysis']
            self.ui.lbl_bpm.setText(f"BPM: {stats['bpm']}")
            self.ui.lbl_sdnn.setText(f"SDNN: {stats['sdnn']} ms")
            self.ui.lbl_rmssd.setText(f"RMSSD: {stats['rmssd']} ms")

    def _rebuild_plot_area(self):
        if hasattr(self, 'eeg_plt'):
            self.eeg_plt.clear()
            self.roi_added_to_plot = False
        else:
            self.eeg_plt = self.ui.pltSignalEeg.addPlot()
            self.ui.pltSignalEeg.setBackground('#EEE5D1')

        self._recalculate_time_variables()
        pen = pg.mkPen(color=(100, 100, 100))
        self.eeg_plt.getAxis('bottom').setPen(pen)
        self.eeg_plt.getAxis('bottom').setTextPen(pen)
        self.eeg_plt.showGrid(x=True, alpha=0.2)
        self.eeg_plt.invertY(True)
        self.eeg_plt.setXRange(self.LABEL_X_OFFSET_S, self.time_vector_s[-1])

        sb_x, sb_y = self.LABEL_X_OFFSET_S * 0.5, self.offset_step * 0.3
        self.eeg_plt.addItem(pg.PlotCurveItem([sb_x, sb_x], [sb_y, sb_y + self.SCALE_BAR_HEIGHT_UNITS],
                                              pen=pg.mkPen('#101010', width=3)))
        self.scale_bar_text = pg.TextItem(f"{int(self.SCALE_BAR_AMPLITUDE_uV)} µV", color='#101010', anchor=(0.5, 0))
        self.eeg_plt.addItem(self.scale_bar_text)
        self.scale_bar_text.setPos(sb_x - 10, sb_y + self.SCALE_BAR_HEIGHT_UNITS)

        self.eeg_plots = {}
        self.eeg_text_items = {}
        self.eeg_offset = 0.0

        for i, ch in enumerate(self.ALL_PLOT_CHANNELS):
            c = self.eeg_plt.plot(self.time_vector_s, np.zeros_like(self.time_vector_s),
                                  pen=pg.mkPen('#101010', width=1))
            self.eeg_plots[ch] = {'curve': c, 'index': i}
            ti = pg.TextItem(ch, color='#101010', anchor=(0, 0.5))
            self.eeg_plt.addItem(ti)
            ti.setPos(self.LABEL_X_OFFSET_S, self.eeg_offset)
            self.eeg_text_items[ch] = ti
            self.eeg_offset += self.offset_step

        if self.btn_toggle_roi.isChecked():
            self.eeg_plt.addItem(self.search_roi)
            self.roi_added_to_plot = True

    def closeEvent(self, e):
        if self.is_simulating:
            if self.worker:
                QtCore.QMetaObject.invokeMethod(self.worker, "stop_simulation",
                                                QtCore.Qt.ConnectionType.QueuedConnection)

        if self.search_thread:
            self.search_thread.quit()
            self.search_thread.wait()

        if self.thread:
            self.thread.quit();
            self.thread.wait()

        super().closeEvent(e)


########################################################################################################################
# (WORKER) for pattern recognition ---
class PatternSearchWorker(QtCore.QObject):
    search_finished = Signal(dict)
    search_progress = Signal(int)
    error_occurred = Signal(str)

    def __init__(self, full_data_matrix, channel_names, pattern_template, threshold, fs, parent=None):
        super().__init__(parent)
        self.data_matrix = full_data_matrix
        self.channel_names = channel_names
        self.template = pattern_template
        self.threshold = threshold
        self.fs = fs
        self._is_interrupted = False

    @Slot()
    def run_search(self):
        print(f"[SearchThread] Avvio scansione su {len(self.channel_names)} canali...")
        results = {}
        try:
            templ_mean = np.mean(self.template)
            templ_centered = self.template - templ_mean
            norm_template = np.sqrt(np.sum(templ_centered ** 2))

            if norm_template == 0:
                self.error_occurred.emit("Template piatto (varianza zero).")
                return
            M = len(self.template)
            num_channels = len(self.channel_names)

            for i, ch_name in enumerate(self.channel_names):
                if self._is_interrupted: break

                self.search_progress.emit(int((i / num_channels) * 100))

                full_signal = self.data_matrix[i, :]
                try:
                    # FIX: Uso la funzione FAST FFT convolve
                    corr = fast_normalized_cross_correlation(full_signal, self.template)
                    matches = np.where(corr >= self.threshold)[0]
                    if len(matches) > 0:
                        clean_matches = []
                        last_idx = -99999
                        for m in matches:
                            if m - last_idx > M:
                                clean_matches.append(m)
                                last_idx = m
                        if clean_matches:
                            results[ch_name] = clean_matches
                except Exception as e:
                    print(f"[SearchThread] Errore su canale {ch_name}: {e}")

            if not self._is_interrupted:
                self.search_progress.emit(100)
                print(f"[SearchThread] Finito. Trovati match in {len(results)} canali.")
                self.search_finished.emit(results)
            else:
                print("[SearchThread] Interrotto dall'utente.")
        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self._is_interrupted = True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ECoGControlWindow(json_anomaly_path='anomaly/ecog_1.json')
    window.show()
    sys.exit(app.exec())