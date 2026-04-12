import json
import logging
import sys
import os
from pathlib import Path
import time
import webbrowser as wb
import datetime
import random
import numpy as np
import scipy.io as sio
import pyqtgraph as pg
import mne

# --- PYSIDE6 IMPORTS ---
from PySide6.QtCore import Qt, Signal, Slot, QThread, QObject, QTimer, QSize, QMetaObject, Q_ARG
from PySide6.QtGui import QPixmap, QColor, QFont, QIcon, QAction, QBrush
from PySide6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
                               QGroupBox, QPushButton, QComboBox, QSpinBox, QCheckBox,
                               QSpacerItem, QSizePolicy, QSplitter, QFrame, QTableWidget,
                               QTableWidgetItem, QHeaderView, QAbstractItemView, QDoubleSpinBox, QGraphicsRectItem)

# Assicurati che utils_funcIOM sia accessibile
try:
    from utils_funcIOM import apply_bandpass_filter, fast_normalized_cross_correlation
except ImportError:
    # Fallback dummy se il file utils non è presente nella stessa cartella durante il test
    def apply_bandpass_filter(data, low, high, fs):
        return data


    def fast_normalized_cross_correlation(sig, pattern):
        return np.zeros_like(sig)

# Imposta regole di logging per pulire la console
os.environ["QT_LOGGING_RULES"] = "qt.qpa.theme.gnome.warning=false;qt.core.qobject.connect.warning=false"

# --- COSTANTI GLOBALI ---
MAT_LOCS_KEY = 'locs'
MAT_DATA_KEY = 'data'
MNE_CHANNEL_TYPE = 'ecog'
EEG_CHANNELS = ['FP2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'FP1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5']
AUX_EOG = ['VEOG', 'HEOG']
AUX_ECG = ['EKG']
AUX_EMG = ['MILO', 'MASST']

N_EEG_CHANNELS = len(EEG_CHANNELS)
AUX_CHANNELS = ['EOG', 'ECG']
ALL_PLOT_CHANNELS = EEG_CHANNELS + AUX_CHANNELS
N_ALL_CHANNELS = len(ALL_PLOT_CHANNELS)

FS_EEG = 1000
TIME_DURATION_S = 10.0
TIME_POINTS_EEG = int(FS_EEG * TIME_DURATION_S)
TIME_VECTOR_EEG = np.linspace(0, TIME_DURATION_S, TIME_POINTS_EEG, endpoint=False)


########################################################################################################################
# CLASSE GENERATORE ARTEFATTI (Dummy o Reale)
class ArtifactGenerator:
    def __init__(self, fs): pass

    def apply_anomaly(self, data, time, anomaly): return data


########################################################################################################################
# FUNZIONI DI CARICAMENTO DATI
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
    raw.set_eeg_reference(projection=True)

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
# WORKER PRINCIPALE (SIMULAZIONE EEG)
class ECoGWorker(QObject):
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

        self.artifact_gen = ArtifactGenerator(self.FS)
        self.active_anomalies = []

    @Slot(list)
    def update_anomalies(self, anomalies_list):
        self.active_anomalies = anomalies_list

    @Slot()
    def _init_timer(self):
        self.timer = QTimer()
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

        if end_index > total_samples:
            p1 = self.template_eeg_data[:, self.current_index:total_samples]
            rem = end_index - total_samples
            p2 = self.template_eeg_data[:, 0:rem]
            eeg_base = np.hstack((p1, p2))
            self.current_index = rem
        else:
            eeg_base = self.template_eeg_data[:, self.current_index:end_index]
            self.current_index = end_index

        eeg_base = eeg_base * self.plot_scale_factor * 1e6
        carRef = np.mean(eeg_base, axis=0)
        eeg_base = (eeg_base - carRef)

        if self.lowcut > 0 and self.highcut > self.lowcut:
            try:
                eeg_base = apply_bandpass_filter(eeg_base, self.lowcut, self.highcut, self.FS)
            except:
                pass

        if self.active_anomalies:
            chunk_time_vector = np.linspace(self.current_time_s,
                                            self.current_time_s + self.chunk_duration_s,
                                            self.chunk_samples, endpoint=False)
            for anomaly in self.active_anomalies:
                target_sim = anomaly.get('target_simulator', 'ALL').upper()
                if target_sim not in ['ECOG', 'ALL']: continue
                if not anomaly.get('active', False): continue
                target_channels_names = anomaly.get('channels', [])
                apply_to_all = len(target_channels_names) == 0

                for i, ch_name in enumerate(self.EEG_CHANNELS):
                    if apply_to_all or (ch_name in target_channels_names):
                        eeg_base[i, :] = self.artifact_gen.apply_anomaly(
                            eeg_base[i, :], chunk_time_vector, anomaly
                        )

        eeg_base = add_eeg_artifacts_dynamic(eeg_base, self.time, self.FS, self.artifact_state)
        eeg_data_dict = {ch: eeg_base[i, :] for i, ch in enumerate(self.EEG_CHANNELS)}

        self.current_time_s += (self.chunk_samples / self.FS)
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
# WORKER PER RICERCA PATTERN (FFT)
class PatternSearchWorker(QObject):
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


########################################################################################################################
# WINDOW POPUP PER I CONTROLLI DI RICERCA
class PatternSearchWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("Pattern recognition on recording session")
        self.resize(750, 350)

    def closeEvent(self, event):
        # Quando chiudo la finestra, rilascio il pulsante "Search Mode" nella main window
        if self.parent() and hasattr(self.parent(), 'pushBt_search'):
            self.parent().pushBt_search.setChecked(False)
        super().closeEvent(event)


########################################################################################################################
# MAIN WINDOW CONTROL
class ECoGControlWindow(QWidget):
    filter_settings_changed = Signal(float, float)
    amplitude_scale_changed = Signal(float)
    anomalies_updated = Signal(list)
    simulation_finished = Signal()
    sig_worker_start = Signal(int)  # Segnale per avviare (passa l'intervallo ms)
    sig_worker_stop = Signal()

    def __init__(self, json_anomaly_path=None, learning_manager=None, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window)

        # Setup base variabili
        self.worker = None
        self.thread = None
        self.search_thread = None
        self.EEG_CHANNELS = EEG_CHANNELS
        self.ALL_PLOT_CHANNELS = ALL_PLOT_CHANNELS
        self.N_ALL_CHANNELS = len(self.ALL_PLOT_CHANNELS)
        self.logger = logging.getLogger(__name__)

        # Configurazione Finestra
        self.setWindowTitle(time.strftime('%d %b %Y') + '  ECoG - Persistent Pattern Search (v2.0)')
        self.resize(1400, 1050)  # Altezza aumentata

        # Parametri grafici
        self.LABEL_X_OFFSET_S = -0.28
        self.current_time_duration_s = TIME_DURATION_S
        self.MIN_TIME_S = 2.0
        self.MAX_TIME_S = 20.0
        self.SCALE_BAR_AMPLITUDE_uV = 200  # Default
        self.SCALE_BAR_HEIGHT_UNITS = 5.0  # Altezza visiva fissa della barra
        self.NUM_ELETRODI_PER_RIGA = 8
        self.offset_step = 10.0
        self.timer_interval_ms = 100
        self.is_simulating = False
        self.display_mode = "Sweep"

        self.eeg_plots = {}
        self.eeg_text_items = {}
        self.eeg_display_buffer = np.zeros((1, 1))  # Placeholder
        self.display_samples = 0
        self.roi_added_to_plot = False
        self.loaded_anomalies = []
        self.match_items = []

        self.current_static_page_index = 0
        self.total_static_pages = 0
        self.total_template_samples = 0
        self.was_simulating_before_paging = False

        self.stored_pattern = None
        self.current_search_source_ch = None
        self.global_search_results = {}
        self.current_pattern_color = QColor(255, 0, 0, 100)
        self.all_found_events_indices = []
        self.current_event_nav_index = -1

        # Oggetti Grafici Scale Bar
        self.scale_bar_line = None
        self.scale_bar_text = None

        # --- COSTRUZIONE INTERFACCIA PROGRAMMATICA ---
        self._init_ui_programmatico()
        self._setup_pattern_search_ui()  # Inizializza la finestra di ricerca

        # --- LOGICA INIZIALE ---
        self.comboBoxMS.addItem('speech_basic', "eeg_data/speech_basic")
        self.comboBoxMS.addItem('motor_basic', "eeg_data/motor_basic")

        self.directory_target_data = Path("eeg_data/motor_basic")
        self.directory_target_brains = Path("eeg_data/motor_basic/brains")
        self.estensione_file = "*.mat"

        # Setup Grafico Pyqtgraph
        self._setup_plot_widget()

        # Popolamento
        self.popola_combobox()

        # Caricamento JSON
        if json_anomaly_path:
            self.load_json_anomaly(json_anomaly_path)

        # Selezione iniziale
        if self.comboExamples.count() > 0:
            self.on_selezione_cambiata(0)

    def _init_ui_programmatico(self):
        """Costruisce l'intera GUI via codice senza file .ui"""

        # Layout Principale (Verticale)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- 1. HEADER (Titolo e Controlli Dataset) ---
        header_group = QGroupBox("Dataset Configuration")
        header_layout = QHBoxLayout(header_group)

        self.lbl_cortex_L = QLabel()
        self.lbl_cortex_L.setFixedSize(60, 60)
        self.lbl_cortex_L.setStyleSheet("border: 1px solid #ccc; background: #fff;")
        self.lbl_cortex_L.setScaledContents(True)
        logo_path = Path('res/logos/cortex_left.png')
        if logo_path.exists():
            self.lbl_cortex_L.setPixmap(QPixmap(str(logo_path)))

        header_layout.addWidget(self.lbl_cortex_L)
        header_layout.addWidget(QLabel("Cognitive Func:"))
        self.comboBoxMS = QComboBox()
        self.comboBoxMS.setMinimumWidth(150)
        self.comboBoxMS.currentIndexChanged.connect(self.on_select_cognitive_function)
        header_layout.addWidget(self.comboBoxMS)

        header_layout.addWidget(QLabel("Example Set:"))
        self.comboExamples = QComboBox()
        self.comboExamples.setMinimumWidth(200)
        self.comboExamples.currentIndexChanged.connect(self.on_selezione_cambiata)
        header_layout.addWidget(self.comboExamples)

        header_layout.addStretch()

        self.btn_start_stop = QPushButton("Start EEG")
        self.btn_start_stop.setMinimumHeight(40)
        self.btn_start_stop.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #333333;")
        self.btn_start_stop.clicked.connect(self.start_stop_simulation)
        header_layout.addWidget(self.btn_start_stop)

        main_layout.addWidget(header_group)

        # --- 2. AREA CENTRALE (Splitter: Controlli SX - Grafico DX) ---
        central_splitter = QSplitter(Qt.Orientation.Horizontal)

        # A. Pannello Sinistro (Controlli)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Gruppo Visualizzazione
        vis_group = QGroupBox("Visualization Control")
        vis_layout = QVBoxLayout(vis_group)

        h_row = QHBoxLayout()
        h_row.addWidget(QLabel("Rows:"))
        self.spinBoxRows = QSpinBox()
        self.spinBoxRows.setRange(1, 10)
        self.spinBoxRows.valueChanged.connect(self.update_channel_display)
        h_row.addWidget(self.spinBoxRows)
        vis_layout.addLayout(h_row)

        self.checkBoxChAll = QCheckBox("Show All Channels")
        self.checkBoxChAll.stateChanged.connect(self.update_channel_display)
        vis_layout.addWidget(self.checkBoxChAll)

        # --- NUOVO CONTROLLO AMPIEZZA (SCALA) ---
        vis_layout.addWidget(QLabel("Amplitude Scale (µV):"))
        self.combo_amp = QComboBox()
        self.combo_amp.addItems(["10", "20", "50", "100", "200", "500", "1000"])
        self.combo_amp.setCurrentText(str(self.SCALE_BAR_AMPLITUDE_uV))
        self.combo_amp.currentTextChanged.connect(self._on_amplitude_changed)
        vis_layout.addWidget(self.combo_amp)
        # ----------------------------------------

        vis_layout.addWidget(QLabel("Time Base:"))
        h_time = QHBoxLayout()
        self.btn_time_dec = QPushButton("-")
        self.btn_time_dec.clicked.connect(self._on_time_button_dec_clicked)
        self.label_time_base = QLabel("10 s")
        self.label_time_base.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_time_inc = QPushButton("+")
        self.btn_time_inc.clicked.connect(self._on_time_button_inc_clicked)
        h_time.addWidget(self.btn_time_dec)
        h_time.addWidget(self.label_time_base)
        h_time.addWidget(self.btn_time_inc)
        vis_layout.addLayout(h_time)

        self.checkScroll = QCheckBox("Scroll Mode")
        self.checkScroll.setChecked(True)
        self.checkScroll.stateChanged.connect(self._on_scroll_mode_changed)
        vis_layout.addWidget(self.checkScroll)

        # Filtri (Placeholder UI)
        self.slider_lowcut = QSpinBox()  # Uso SpinBox invece di Slider per semplicità
        self.slider_highcut = QSpinBox()
        self.slider_highcut.setRange(0, 500);
        self.slider_highcut.setValue(200)
        vis_layout.addWidget(QLabel("Filters (Low/High Hz):"))
        f_lay = QHBoxLayout()
        f_lay.addWidget(self.slider_lowcut)
        f_lay.addWidget(self.slider_highcut)
        vis_layout.addLayout(f_lay)
        # Connessioni dummy
        self.slider_lowcut.valueChanged.connect(self._on_filter_changed)
        self.slider_highcut.valueChanged.connect(self._on_filter_changed)

        left_layout.addWidget(vis_group)

        # Pattern Search Controls
        search_group = QGroupBox("Pattern Search")
        search_layout = QVBoxLayout(search_group)
        self.pushBt_search = QPushButton("Search Mode")
        self.pushBt_search.setCheckable(True)
        self.pushBt_search.clicked.connect(self.toggle_pattern_search_panel)
        search_layout.addWidget(self.pushBt_search)

        self.label_search_status = QLabel("Ready")
        self.label_search_status.setStyleSheet("font-size: 10px; color: gray;")
        search_layout.addWidget(self.label_search_status)

        h_nav = QHBoxLayout()
        self.pushBt_prev = QPushButton("< Prev")
        self.pushBt_next = QPushButton("Next >")
        self.pushBt_prev.clicked.connect(self.nav_prev_match)
        self.pushBt_next.clicked.connect(self.nav_next_match)
        self.pushBt_prev.setEnabled(False)
        self.pushBt_next.setEnabled(False)
        h_nav.addWidget(self.pushBt_prev)
        h_nav.addWidget(self.pushBt_next)
        search_layout.addLayout(h_nav)

        left_layout.addWidget(search_group)

        left_layout.addStretch()

        # Pulsanti Sistema
        self.pushBtHelp = QPushButton("Help")
        self.pushBtHelp.clicked.connect(self.helpECoG)
        self.pushBtClose = QPushButton("Close")
        self.pushBtClose.clicked.connect(self.close)

        left_layout.addWidget(self.pushBtHelp)
        left_layout.addWidget(self.pushBtClose)

        left_panel.setMaximumWidth(250)
        central_splitter.addWidget(left_panel)

        # B. Pannello Destro (Grafico ECoG)
        self.pltSignalEeg = pg.GraphicsLayoutWidget()
        self.pltSignalEeg.setBackground('#EEE5D1')
        central_splitter.addWidget(self.pltSignalEeg)
        self.pltSignalEeg.setMinimumHeight(700)
        central_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(central_splitter)

    def _setup_plot_widget(self):
        """Inizializza l'area di plot specifica"""
        self.search_roi = pg.LinearRegionItem([0, 1], brush=pg.mkBrush(0, 0, 255, 50))
        self.search_roi.setZValue(10)

    def _setup_pattern_search_ui(self):
        """Costruisce la finestra popup per la ricerca pattern"""
        self.search_window = PatternSearchWindow(parent=self)
        layout_main = QVBoxLayout()
        self.search_window.setLayout(layout_main)

        grp_controls = QGroupBox("Controlli")
        layout_controls = QHBoxLayout()
        grp_controls.setLayout(layout_controls)

        self.btn_toggle_roi = QPushButton("Seleziona ROI")
        self.btn_toggle_roi.setCheckable(True)
        self.btn_toggle_roi.clicked.connect(self.toggle_search_roi)
        layout_controls.addWidget(self.btn_toggle_roi)

        layout_controls.addWidget(QLabel("Ch:"))
        self.combo_source_ch = QComboBox()
        self.combo_source_ch.setMinimumWidth(60)
        layout_controls.addWidget(self.combo_source_ch)

        self.btn_lock_pattern = QPushButton("2. Memorizza")
        self.btn_lock_pattern.setStyleSheet("background-color: #e0f7fa;")
        self.btn_lock_pattern.clicked.connect(self.lock_pattern_template)
        layout_controls.addWidget(self.btn_lock_pattern)

        layout_controls.addWidget(QLabel("Soglia:"))
        self.spin_corr_thresh = QDoubleSpinBox()
        self.spin_corr_thresh.setRange(0.1, 0.99)
        self.spin_corr_thresh.setValue(0.80)
        self.spin_corr_thresh.setSingleStep(0.05)
        self.spin_corr_thresh.setFixedWidth(50)
        layout_controls.addWidget(self.spin_corr_thresh)

        # CHECKBOX PER RICERCA SINGOLO CANALE
        self.chk_search_single_ch = QCheckBox("Solo Ch Sorgente")
        self.chk_search_single_ch.setChecked(True)
        layout_controls.addWidget(self.chk_search_single_ch)

        self.btn_search_all = QPushButton("3. Cerca")
        self.btn_search_all.setStyleSheet("background-color: #ffccbc;")
        self.btn_search_all.clicked.connect(self.start_global_search)
        self.btn_search_all.setEnabled(False)
        layout_controls.addWidget(self.btn_search_all)

        self.btn_clear_search = QPushButton("Reset")
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

    # --- METODI LOGICI ---

    def on_select_cognitive_function(self):
        print(f"Cognitive Function Selected: {self.comboBoxMS.currentText()}")

    @Slot(int)
    def on_selezione_cambiata(self, index):
        file_ecog_data = self.comboExamples.currentData()
        if not file_ecog_data: return
        path_data = Path(file_ecog_data)
        prefix = path_data.stem.split('_')[0]
        path_brain = self.directory_target_brains / f"{prefix}_brain.mat"
        if not path_brain.exists():
            print(f"Brain not found at: {path_brain}")
            return
        self.load_new_dataset(str(file_ecog_data), str(path_brain))

    def popola_combobox(self):
        self.comboExamples.clear()
        file_paths = sorted(self.directory_target_data.glob(self.estensione_file))
        if not file_paths:
            self.comboExamples.addItem("Nessun file", None)
            self.comboExamples.setEnabled(False)
            return
        self.comboExamples.setEnabled(True)
        for fp in file_paths:
            lbl = fp.stem.replace('_', ' ').title()
            self.comboExamples.addItem(lbl, str(fp))

    def start_stop_simulation(self):
        if not self.is_simulating:
            # Avvio
            if self.worker:
                self.sig_worker_start.emit(self.timer_interval_ms)
            self.btn_start_stop.setText("Stop EEG")
            self.is_simulating = True
            # Abilita/Disabilita controlli
            self.btn_time_inc.setEnabled(True)
            self.btn_time_dec.setEnabled(True)
        else:
            # Stop
            if self.worker:
                self.sig_worker_stop.emit()
            self.btn_start_stop.setText("Start EEG")
            self.is_simulating = False
            self.btn_time_inc.setEnabled(True)
            self.btn_time_dec.setEnabled(True)

    def draw_stored_matches_on_current_view(self):
        for item in self.match_items: self.eeg_plt.removeItem(item)
        self.match_items = []

        if not self.global_search_results or self.stored_pattern is None: return

        if not self.checkScroll.isChecked():
            start_abs = self.current_static_page_index * self.display_samples
            end_abs = start_abs + self.display_samples
            pat_len = len(self.stored_pattern)

            for ch_name, hits in self.global_search_results.items():
                if ch_name not in self.current_visible_channels: continue

                hits_in_page = [h for h in hits if start_abs <= h < end_abs]
                if not hits_in_page: continue

                if ch_name in self.eeg_text_items:
                    vis_y = self.eeg_text_items[ch_name].pos().y()

                    for h_idx in hits_in_page:
                        rel_sample = h_idx - start_abs
                        if rel_sample < len(self.time_vector_s):
                            t_start = self.time_vector_s[rel_sample]
                            t_width = pat_len / FS_EEG

                            rect = QGraphicsRectItem(t_start, vis_y - self.offset_step * 0.4, t_width,
                                                     self.offset_step * 0.8)

                            rect.setBrush(QBrush(self.current_pattern_color))
                            rect.setPen(pg.mkPen(None))
                            rect.setZValue(50)

                            self.eeg_plt.addItem(rect)
                            self.match_items.append(rect)

    def update_channel_display(self):
        all_chk = self.checkBoxChAll.isChecked()
        row_idx = self.spinBoxRows.value()
        self.spinBoxRows.setEnabled(not all_chk)

        if all_chk:
            target = self.ALL_PLOT_CHANNELS
        else:
            target = self.real_channel_row_map.get(row_idx, [])

        self.current_visible_channels = target
        curr_off = 0.0

        if hasattr(self, 'eeg_plots'):
            for ch in self.ALL_PLOT_CHANNELS:
                if ch not in self.eeg_plots: continue
                c = self.eeg_plots[ch]['curve']
                t = self.eeg_text_items[ch]
                if ch in target:
                    c.show()
                    t.show()
                    t.setPos(self.LABEL_X_OFFSET_S, curr_off)
                    i = self.eeg_plots[ch]['index']
                    if hasattr(self, 'eeg_display_buffer'):
                        c.setData(self.time_vector_s, self.eeg_display_buffer[i, :] + curr_off)
                    curr_off += self.offset_step
                else:
                    c.hide()
                    t.hide()
            Y_max = self.offset_step * (len(target) + 0.5)
            self.eeg_plt.setYRange(-self.offset_step * 0.5, Y_max, padding=0)
            self.draw_stored_matches_on_current_view()

    def _on_time_button_inc_clicked(self):
        # Aumenta di 1 secondo (puoi cambiare il valore, es. 2.0 o 5.0)
        self._change_time_duration(1.0)

    def _on_time_button_dec_clicked(self):
        # Diminuisce di 1 secondo
        self._change_time_duration(-1.0)

    def _change_time_duration(self, change_amount):
        """
        Gestisce il cambio della durata temporale (Zoom asse X)
        sia in modalità Scroll che in modalità Statica.
        """
        # 1. Calcola la nuova durata rispettando i limiti
        new_duration = self.current_time_duration_s + change_amount
        self.current_time_duration_s = np.clip(new_duration, self.MIN_TIME_S, self.MAX_TIME_S)

        # 2. Aggiorna l'etichetta dell'interfaccia
        self.label_time_base.setText(f"{self.current_time_duration_s:.0f} s")

        # 3. Ricalcola i buffer grafici (vettori tempo e matrici display)
        self._recalculate_time_variables()

        # 4. Ricostruisce l'area del grafico (aggiorna X-Range e assi)
        self._rebuild_plot_area()

        # 5. AGGIORNAMENTO WORKER (Fondamentale)
        # Se stiamo simulando, dobbiamo dire al worker che la finestra è cambiata
        # affinché gestisca correttamente il buffer circolare.
        if self.worker:
            # Calcoliamo i nuovi punti totali: Durata * Frequenza di campionamento
            new_time_points = int(FS_EEG * self.current_time_duration_s)

            # Invochiamo il metodo del worker in modo thread-safe
            QMetaObject.invokeMethod(self.worker, "reset_display_window",
                                     Qt.ConnectionType.QueuedConnection,
                                     Q_ARG(int, new_time_points))

        # 6. Rinfresca i dati
        if not self.is_simulating:
            # Se siamo in PAUSA (modalità statica), dobbiamo ricaricare la pagina corrente
            # con la nuova larghezza temporale.
            self._request_page_from_worker(self.current_static_page_index)

            # Ridisegna eventuali match trovati (se la ricerca pattern è attiva)
            self.draw_stored_matches_on_current_view()

    def _navigate_static_page(self, direction):
        if self.is_simulating: return
        new_p = int(np.clip(self.current_static_page_index + direction, 0, self.total_static_pages - 1))
        if new_p == self.current_static_page_index: return
        self.current_static_page_index = new_p
        self._request_page_from_worker(new_p)
        self.label_time_base.setText(f"Pag. {self.current_static_page_index + 1} / {self.total_static_pages}")

    def _on_scroll_mode_changed(self):
        self.display_mode = "Scroll" if self.checkScroll.isChecked() else "Sweep"
        if self.display_mode == "Scroll":
            self.start_stop_simulation()
            pass
        else:
            if self.is_simulating: self.start_stop_simulation()
            self._request_page_from_worker(self.current_static_page_index)

    # --- NUOVO METODO CAMBIO SCALA ---
    def _on_amplitude_changed(self, text):
        try:
            val = float(text)
            self.SCALE_BAR_AMPLITUDE_uV = val
            # Aggiorna il fattore di scala
            self.plot_scale_factor = self.SCALE_BAR_HEIGHT_UNITS / self.SCALE_BAR_AMPLITUDE_uV

            # Invia aggiornamento al worker
            self.amplitude_scale_changed.emit(self.plot_scale_factor)

            # Aggiorna visualmente la Scale Bar
            if self.scale_bar_text:
                self.scale_bar_text.setText(f"{int(val)} µV")

        except ValueError:
            pass

    # ---------------------------------

    def helpECoG(self):
        print("Show Help Dialog")

    def toggle_pattern_search_panel(self):
        if self.pushBt_search.isChecked():
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

        row_idx = self.table_pattern_log.rowCount()
        self.table_pattern_log.insertRow(row_idx)
        self.table_pattern_log.setItem(row_idx, 0, QTableWidgetItem(current_log_time))
        self.table_pattern_log.setItem(row_idx, 1, QTableWidgetItem(source_ch))
        self.table_pattern_log.setItem(row_idx, 2, QTableWidgetItem(f"{pattern_duration_ms:.1f}"))
        self.table_pattern_log.setItem(row_idx, 3, QTableWidgetItem(f"{amplitude_uv:.1f}"))

        item_hits = QTableWidgetItem("Ready")
        item_hits.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table_pattern_log.setItem(row_idx, 6, item_hits)

        self.btn_search_all.setEnabled(True)
        self.btn_lock_pattern.setText(f"OK ({len(raw_pattern)})")
        self.btn_lock_pattern.setStyleSheet("background-color: #a5d6a7;")

    def start_global_search(self):
        if self.stored_pattern is None: return
        if not self.worker: return

        threshold = self.spin_corr_thresh.value()

        if self.chk_search_single_ch.isChecked() and self.current_search_source_ch:
            target_channels = [self.current_search_source_ch]
            try:
                ch_idx = self.worker.EEG_CHANNELS.index(self.current_search_source_ch)
                target_data = self.worker.template_eeg_data[ch_idx:ch_idx + 1, :]
            except ValueError:
                return
        else:
            target_channels = self.worker.EEG_CHANNELS
            target_data = self.worker.template_eeg_data

        self.label_search_status.setText("Start searching (FFT)...")
        self.btn_search_all.setEnabled(False)
        self.btn_search_all.setText("Searching...")

        if self.search_thread is not None:
            self.search_thread.quit()
            self.search_thread.wait()

        self.search_thread = QThread()
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
        self.search_worker.error_occurred.connect(lambda e: print(f"Search Error: {e}"))

        # Cleanup
        self.search_worker.search_finished.connect(lambda: self.btn_search_all.setEnabled(True))
        self.search_worker.search_finished.connect(lambda: self.btn_search_all.setText("3. Cerca"))
        self.search_worker.search_finished.connect(lambda: self.search_thread.quit())

        self.search_thread.start()

    @Slot(dict)
    def on_global_search_finished(self, results):
        self.toggle_search_roi()  # disattiva roi

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

        # --- AVERAGING ---
        if self.current_search_source_ch and self.current_search_source_ch in results:
            hits_on_source = results[self.current_search_source_ch]
            if len(hits_on_source) > 0 and self.worker:
                try:
                    if self.current_search_source_ch in self.worker.EEG_CHANNELS:
                        ch_idx_map = self.worker.EEG_CHANNELS.index(self.current_search_source_ch)
                        full_signal = self.worker.template_eeg_data[ch_idx_map, :].copy()
                        full_signal = full_signal * self.plot_scale_factor * 1e6
                        pat_len = len(self.stored_pattern)
                        segments = []
                        for start_idx in hits_on_source:
                            end_idx = start_idx + pat_len
                            if end_idx <= full_signal.size:
                                segments.append(full_signal[start_idx:end_idx])

                        if segments:
                            avg_pattern = np.mean(segments, axis=0)
                            self.pattern_preview_plot.clear()
                            orig_centered = self.stored_pattern - np.mean(self.stored_pattern)
                            self.pattern_preview_plot.plot(orig_centered,
                                                           pen=pg.mkPen(color=self.current_pattern_color, width=2))
                            avg_centered = avg_pattern - np.mean(avg_pattern)
                            self.pattern_preview_plot.plot(avg_centered,
                                                           pen=pg.mkPen(color='k', width=3))
                except Exception as e:
                    print(f"Errore averaging: {e}")

        self.label_search_status.setText(f"Found: {total_hits} unique events.")
        has_results = total_hits > 0
        self.pushBt_prev.setEnabled(has_results)
        self.pushBt_next.setEnabled(has_results)

        self.draw_stored_matches_on_current_view()

    def nav_next_match(self):
        if not self.all_found_events_indices: return
        if self.current_event_nav_index < len(self.all_found_events_indices) - 1:
            self.current_event_nav_index += 1
            self._jump_to_event_index(self.current_event_nav_index)

    def nav_prev_match(self):
        if not self.all_found_events_indices: return
        if self.current_event_nav_index > 0:
            self.current_event_nav_index -= 1
            self._jump_to_event_index(self.current_event_nav_index)

    def _jump_to_event_index(self, list_index):
        target_sample = self.all_found_events_indices[list_index]
        target_page = int(target_sample // self.display_samples)

        if self.checkScroll.isChecked():
            self.checkScroll.setChecked(False)

        self.current_static_page_index = target_page
        self._request_page_from_worker(target_page)
        self.label_time_base.setText(f"Pag. {self.current_static_page_index + 1} / {self.total_static_pages}")

        time_s = target_sample / FS_EEG
        self.label_search_status.setText(
            f"Event {list_index + 1}/{len(self.all_found_events_indices)} at {time_s:.2f}s")

    def clear_all_search_data(self):
        self.global_search_results = {}
        self.stored_pattern = None
        self.all_found_events_indices = []
        self.btn_search_all.setEnabled(False)
        self.btn_lock_pattern.setText("2. Memorizza")
        self.btn_lock_pattern.setStyleSheet("background-color: #e0f7fa;")
        for item in self.match_items: self.eeg_plt.removeItem(item)
        self.match_items = []
        self.pattern_preview_plot.clear()

    def load_new_dataset(self, f_data, f_brain):
        self.clear_all_search_data()
        if self.is_simulating:
            self.is_simulating = False
            self.btn_start_stop.setText('Start EEG')

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
        if not chs: return

        self.dynamic_eeg_channels = chs
        self.ALL_PLOT_CHANNELS = chs
        self.N_ALL_CHANNELS = len(chs)
        self.NUM_RIGHE = max(1, len(chs) // self.NUM_ELETRODI_PER_RIGA)
        self.channel_map = {r + 1: [f'R{r + 1}-El{i}' for i in range(1, 9)] for r in range(self.NUM_RIGHE)}

        self.real_channel_row_map = {r + 1: chs[r * 8:(r + 1) * 8] for r in range(self.NUM_RIGHE)}
        self.current_visible_channels = self.real_channel_row_map.get(1, [])
        self.spinBoxRows.setMaximum(self.NUM_RIGHE)
        self.spinBoxRows.setValue(1)
        self.plot_scale_factor = self.SCALE_BAR_HEIGHT_UNITS / self.SCALE_BAR_AMPLITUDE_uV

        params = {
            'FS': FS_EEG, 'time_vector': TIME_VECTOR_EEG, 'time_points': int(FS_EEG * self.current_time_duration_s),
            'eeg_channels': chs, 'aux_channels': [], 'cnt_filepath': f_data,
            'locs_matrix': locs, 'timer_interval_ms': self.timer_interval_ms,
            'plot_scale_factor': self.plot_scale_factor
        }

        self.thread = QThread()
        self.worker = ECoGWorker(params)
        self.worker.moveToThread(self.thread)

        self.sig_worker_start.connect(self.worker.start_simulation)
        self.sig_worker_stop.connect(self.worker.stop_simulation)

        self.worker.data_ready.connect(self.update_eeg_plot)
        self.worker.dataset_loaded.connect(self._on_dataset_loaded)
        self.worker.static_page_ready.connect(self._on_static_page_received)
        self.filter_settings_changed.connect(self.worker.update_filter_settings)
        self.amplitude_scale_changed.connect(self.worker.update_plot_scale_factor)

        # Connessione segnale anomalie
        self.anomalies_updated.connect(self.worker.update_anomalies)
        self.thread.started.connect(self.worker._init_timer)
        self.thread.start()

        self._rebuild_plot_area()
        self.update_channel_display()

        if self.loaded_anomalies:
            self.anomalies_updated.emit(self.loaded_anomalies)

    @Slot(dict)
    def update_eeg_plot(self, data_packet):
        if 'new_data_chunk' not in data_packet: return

        eeg_data = data_packet['new_data_chunk']
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
            if ch_name not in eeg_data: continue
            new_chunk = eeg_data[ch_name]

            if end_idx <= self.display_samples:
                self.eeg_display_buffer[i, start_idx:end_idx] = new_chunk
            else:
                part1_len = self.display_samples - start_idx
                self.eeg_display_buffer[i, start_idx:] = new_chunk[:part1_len]
                part2_len = chunk_len - part1_len
                self.eeg_display_buffer[i, :part2_len] = new_chunk[part1_len:]

        self._refresh_full_plot()

    def _rebuild_plot_area(self):
        # 1. Calcola i vettori temporali PRIMA di costruire il grafico
        self._recalculate_time_variables()

        # 2. Pulisce e ricostruisce il layout
        self.pltSignalEeg.clear()
        self.eeg_plt = self.pltSignalEeg.addPlot()
        self.eeg_plt.setMouseEnabled(x=False, y=False)
        self.eeg_plt.showGrid(x=True, alpha=0.2)

        # 3. Imposta i range usando le variabili appena calcolate
        self.eeg_plt.setXRange(self.LABEL_X_OFFSET_S, self.current_time_duration_s)
        self.eeg_plt.invertY(True)

        # --- NASCONDI ASSE Y E AGGIUNGI SCALE BAR ---
        self.eeg_plt.hideAxis('left')

        # Disegna la linea della Scale Bar (angolo in basso a sinistra)
        sb_x = self.LABEL_X_OFFSET_S * 0.5
        sb_y = self.offset_step * 0.3

        # Linea verticale spessa
        self.scale_bar_line = pg.PlotCurveItem(
            [sb_x, sb_x],
            [sb_y, sb_y + self.SCALE_BAR_HEIGHT_UNITS],
            pen=pg.mkPen('k', width=3)
        )
        self.eeg_plt.addItem(self.scale_bar_line)

        # Testo
        self.scale_bar_text = pg.TextItem(
            text=f"{int(self.SCALE_BAR_AMPLITUDE_uV)} µV",
            color='k',
            anchor=(0.5, 0)
        )
        self.scale_bar_text.setPos(sb_x - 0.05, sb_y + self.SCALE_BAR_HEIGHT_UNITS)
        self.eeg_plt.addItem(self.scale_bar_text)
        # ---------------------------------------------

        self.eeg_plots = {}
        self.eeg_text_items = {}
        curr_off = 0.0

        # 4. Ricrea le curve
        for i, ch in enumerate(self.ALL_PLOT_CHANNELS):
            # Crea una curva vuota inizialmente ma con la lunghezza giusta per evitare errori
            dummy_data = np.zeros(len(self.time_vector_s))
            c = self.eeg_plt.plot(self.time_vector_s, dummy_data, pen='k')

            ti = pg.TextItem(ch, color='k')
            self.eeg_plt.addItem(ti)
            self.eeg_plots[ch] = {'curve': c, 'index': i}
            self.eeg_text_items[ch] = ti

            # Posiziona l'etichetta
            ti.setPos(self.LABEL_X_OFFSET_S, curr_off)
            curr_off += self.offset_step

    def _refresh_full_plot(self):
        current_offset = 0.0
        for i, ch_name in enumerate(self.ALL_PLOT_CHANNELS):
            if ch_name not in self.eeg_plots: continue
            if ch_name not in self.current_visible_channels: continue

            raw_data = self.eeg_display_buffer[i, :]
            offset_data = raw_data + current_offset
            self.eeg_plots[ch_name]['curve'].setData(self.time_vector_s, offset_data)
            current_offset += self.offset_step

    def _recalculate_time_variables(self):
        self.time_vector_s = np.linspace(0, self.current_time_duration_s, int(FS_EEG * self.current_time_duration_s))
        self.display_samples = len(self.time_vector_s)
        self.eeg_display_buffer = np.zeros((self.N_ALL_CHANNELS, self.display_samples))

        if self.total_template_samples > 0:
            self.total_static_pages = int(np.ceil(self.total_template_samples / self.display_samples))

    @Slot(int)
    def _on_dataset_loaded(self, total_samples):
        self.total_template_samples = total_samples
        self._recalculate_time_variables()
        self.btn_start_stop.setEnabled(True)
        self.current_static_page_index = 0
        self.label_time_base.setText(f"{self.current_time_duration_s:.0f} s")
        if self.checkScroll.isChecked():
            self.start_stop_simulation()

    @Slot(dict)
    def _on_static_page_received(self, data_packet):
        static_data = data_packet.get('static_data')
        n_samp = data_packet.get('num_samples_in_page', 0)
        if not static_data: return
        self.eeg_display_buffer[:] = 0
        for ch in self.current_visible_channels:
            if ch in static_data:
                i = self.eeg_plots[ch]['index']
                self.eeg_display_buffer[i, :n_samp] = static_data[ch]
        self._refresh_full_plot()
        self.draw_stored_matches_on_current_view()

    def _request_page_from_worker(self, page_index):
        if self.worker:
            QMetaObject.invokeMethod(self.worker, "request_static_page",
                                     Qt.ConnectionType.QueuedConnection, Q_ARG(int, page_index),
                                     Q_ARG(int, self.display_samples))

    def _on_filter_changed(self):
        l = self.slider_lowcut.value()
        h = self.slider_highcut.value()
        self.filter_settings_changed.emit(float(l), float(h))

    def load_json_anomaly(self, path=None):
        if not path: return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                scenario = data.get('scenario_specific_config', {})
                anomalies = scenario.get('anomalies', [])
                self.loaded_anomalies = anomalies
                if self.worker: self.anomalies_updated.emit(self.loaded_anomalies)
        except Exception as e:
            print(f"JSON Error: {e}")

    def closeEvent(self, e):
        self.start_stop_simulation()
        if self.search_thread: self.search_thread.quit()
        self.simulation_finished.emit()
        super().closeEvent(e)


# --- Main Esecuzione ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Esempio: passa il path del json se vuoi testare gli artefatti
    window = ECoGControlWindow()  # json_anomaly_path='anomaly/ecog_1.json')
    window.show()
    sys.exit(app.exec())
    