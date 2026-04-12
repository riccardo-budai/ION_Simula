"""
    BAEP simulator : simple complex of all waves of Brain Stem acoustic Evoked Potential
    stimuli are specified in 3 modalities : condensation, rarefaction and alternating polarity
    noise settable in amplitude
    recording in two channels : omolateral to stimulus (A1 - C) and contralateral to the stimulus (A2 - Cz)
    stimulus : monolateral or bilateral
    possibile anomalies:

    ## UPDATE 2025-11-13 ##
    - "Block Averaging" logic with overlay (Overlay).
    - FIX: Stimulation frequency (Hz) is now accurate.
    - The worker loop has been redesigned for 1 sweep / 1 iteration.
    - "Sweeps per update" is now "UI Update Rate" and no longer blocks the thread.

    ## UPDATE 2025-11-15 ##
    - Added markers (vertical dashes) for Ch1 peaks (I, III, V)
      on the Stack Plot (raster_plot_ch1).

    ## UPDATE 2025-11-15 (v2) ##
    - Extended peak analysis and markers (main plot and stack plot)
      to Channel 2 (Waves III, V) as well.
"""

import sys
import os
import webbrowser as wb
os.environ["QT_API"] = "pyside6"
import numpy as np
import pyqtgraph as pg
import scipy
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QComboBox, QLabel, QSpinBox,
                               QDoubleSpinBox, QMessageBox, QCheckBox, QLineEdit, QFormLayout, QAbstractItemView,
                               QTableWidget, QTabWidget, QTableWidgetItem, QApplication)
from PySide6.QtCore import QTimer, QDateTime
from PySide6.QtCore import QThread, QObject, Signal, Slot, QMutex, Qt


# (Mock LearningManager if not available for testing)
class LearningManager:
    pass


# Basic settings for pyqtgraph (dark background charts)
pg.setConfigOption('background', '#343a40')
pg.setConfigOption('foreground', 'w')


########################################################################################################################
class BaepWorker(QObject):
    """
    Worker that manages the BAEP simulation logic.
    (No changes to the Worker in this version)
    """
    update_plot_signal = Signal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int)
    update_raster_signal = Signal(np.ndarray, np.ndarray, int)
    finished_signal = Signal()

    def __init__(self, time_vector, wave_params, cm_freq, cm_amp, cm_decay, fs,
                 initial_noise: float,
                 initial_ui_update_rate: int,  ## NAME CHANGE ##
                 initial_polarity_mode: str,
                 initial_stim_side: str,
                 initial_amplitude_scale: float,
                 initial_block_sweeps_target: int,
                 initial_overlay_percent: int,
                 initial_stim_rate: int,
                 parent=None):
        super().__init__(parent)

        self.time_vector = time_vector
        self.wave_params = wave_params
        self.cm_freq = cm_freq
        self.cm_amp = cm_amp
        self.cm_decay = cm_decay
        self.fs = fs
        self.param_mutex = QMutex()

        self.current_noise_level = initial_noise
        self.current_ui_update_rate = initial_ui_update_rate  ## NAME CHANGE ##
        self.current_polarity_mode = initial_polarity_mode
        self.current_stim_side = initial_stim_side
        self.current_amplitude_scale = initial_amplitude_scale
        self.current_block_sweeps_target = initial_block_sweeps_target
        self.current_overlay_percent = initial_overlay_percent
        self.current_stim_rate = initial_stim_rate

        self._is_running = False
        self.block_sweep_count = 0
        self.current_block_average_ch1 = np.zeros(len(self.time_vector))
        self.current_block_average_ch2 = np.zeros(len(self.time_vector))

    # --- Signal Generation (Unchanged) ---
    def _generate_gaussian(self, lat, amp, std):
        return amp * np.exp(-((self.time_vector - lat) ** 2) / (2 * std ** 2))

    def _generate_base_baep(self, latency_shift=0.0):
        self.param_mutex.lock()
        scale = self.current_amplitude_scale
        self.param_mutex.unlock()
        signal = np.zeros(len(self.time_vector))
        for wave, params in self.wave_params.items():
            lat = params['lat'] + latency_shift
            amp = params['amp'] * scale
            signal += self._generate_gaussian(lat, amp, params['std'])
        return signal

    def _generate_contra_baep(self, latency_shift=0.0):
        self.param_mutex.lock()
        scale = self.current_amplitude_scale
        self.param_mutex.unlock()
        signal = np.zeros(len(self.time_vector))
        params_iii = self.wave_params['III']
        params_iv = self.wave_params['IV']
        params_v = self.wave_params['V']
        amp_iii = (params_iii['amp'] * 0.3) * scale
        signal += self._generate_gaussian(params_iii['lat'] + latency_shift, amp_iii, params_iii['std'])
        amp_iv = params_iv['amp'] * scale
        signal += self._generate_gaussian(params_iv['lat'] + latency_shift, amp_iv, params_iv['std'])
        amp_v = params_v['amp'] * scale
        signal += self._generate_gaussian(params_v['lat'] + latency_shift, amp_v, params_v['std'])
        return signal

    def _generate_cm(self, polarity: int):
        if polarity == 0:
            return np.zeros(len(self.time_vector))
        decay = np.exp(-self.time_vector / self.cm_decay)
        sine_wave = self.cm_amp * np.sin(2 * np.pi * self.cm_freq * (self.time_vector / 1000))
        cm = sine_wave * decay
        cutoff_index = int(self.fs * (4 / 1000))
        cm[cutoff_index:] = 0
        return cm * polarity

    def _generate_noise(self):
        self.param_mutex.lock()
        noise_level = self.current_noise_level
        self.param_mutex.unlock()
        return np.random.normal(0, noise_level, len(self.time_vector))

    # --- Public slots for UI Thread control (Unchanged) ---

    @Slot(float)
    def set_noise_level(self, noise):
        self.param_mutex.lock()
        self.current_noise_level = noise
        self.param_mutex.unlock()

    @Slot(float)
    def set_amplitude_scale(self, scale):
        self.param_mutex.lock()
        self.current_amplitude_scale = scale
        self.param_mutex.unlock()

    @Slot(int)
    def set_ui_update_rate(self, rate):
        self.param_mutex.lock()
        self.current_ui_update_rate = rate
        self.param_mutex.unlock()

    @Slot(str)
    def set_polarity_mode(self, mode):
        self.param_mutex.lock()
        self.current_polarity_mode = mode
        self.param_mutex.unlock()

    @Slot(str)
    def set_stim_side(self, side):
        self.param_mutex.lock()
        self.current_stim_side = side
        self.param_mutex.unlock()

    @Slot(int)
    def set_block_sweeps_target(self, target_sweeps):
        self.param_mutex.lock()
        old_target = self.current_block_sweeps_target
        self.current_block_sweeps_target = target_sweeps
        if self.block_sweep_count > target_sweeps:
            self.block_sweep_count = 0
            self.current_block_average_ch1 = np.zeros(len(self.time_vector))
            self.current_block_average_ch2 = np.zeros(len(self.time_vector))
        self.param_mutex.unlock()

    @Slot(int)
    def set_overlay_percent(self, percent):
        self.param_mutex.lock()
        self.current_overlay_percent = percent
        self.param_mutex.unlock()

    @Slot(int)
    def set_stim_rate(self, rate):
        self.param_mutex.lock()
        self.current_stim_rate = rate
        self.param_mutex.unlock()

    @Slot()
    def stop_worker(self):
        self._is_running = False

    @Slot()
    def reset_worker(self):
        self.param_mutex.lock()
        self.block_sweep_count = 0
        self.current_block_average_ch1 = np.zeros(len(self.time_vector))
        self.current_block_average_ch2 = np.zeros(len(self.time_vector))
        self.param_mutex.unlock()

    # --- Main Worker execution method (Unchanged) ---

    @Slot()
    def run_simulation(self):
        self._is_running = True

        while self._is_running:
            self.param_mutex.lock()
            polarity_mode = self.current_polarity_mode
            stim_side = self.current_stim_side
            block_target = self.current_block_sweeps_target
            overlay_percent = self.current_overlay_percent
            stim_rate = self.current_stim_rate
            ui_update_rate = self.current_ui_update_rate
            self.param_mutex.unlock()

            if stim_rate <= 0: stim_rate = 1
            delay_ms = int(1000 / stim_rate)

            self.param_mutex.lock()
            if self.block_sweep_count >= block_target:
                self.param_mutex.unlock()
                QThread.msleep(delay_ms)
                continue
            current_sweep_idx = self.block_sweep_count
            self.param_mutex.unlock()

            stim_ear = "Left"
            if stim_side == "Left":
                stim_ear = "Left"
            elif stim_side == "Right":
                stim_ear = "Right"
            elif stim_side == "Alternating L/R":
                stim_ear = "Left" if (current_sweep_idx % 2 == 0) else "Right"
            polarity = 0
            shift = 0.0
            if polarity_mode == "Alternating":
                is_rare = (current_sweep_idx % 2 == 0)
                polarity = 1 if is_rare else -1
                shift = 0.0 if is_rare else 0.08
            elif polarity_mode == "Rarefaction":
                polarity = 1
                shift = 0.0
            elif polarity_mode == "Condensation":
                polarity = -1
                shift = 0.08

            ipsi_signal = self._generate_base_baep(latency_shift=shift)
            contra_signal = self._generate_contra_baep(latency_shift=shift)
            cm = self._generate_cm(polarity=polarity)
            noise1 = self._generate_noise()
            noise2 = self._generate_noise()
            last_single_ch1 = None
            last_single_ch2 = None

            if stim_ear == "Left":
                last_single_ch1 = ipsi_signal + cm + noise1
                last_single_ch2 = contra_signal + noise2
            else:
                last_single_ch1 = contra_signal + noise1
                last_single_ch2 = ipsi_signal + cm + noise2

            self.param_mutex.lock()
            self.block_sweep_count += 1
            n = self.block_sweep_count
            self.current_block_average_ch1 = (self.current_block_average_ch1 * (n - 1) + last_single_ch1) / n
            self.current_block_average_ch2 = (self.current_block_average_ch2 * (n - 1) + last_single_ch2) / n
            avg1_copy = self.current_block_average_ch1.copy()
            avg2_copy = self.current_block_average_ch2.copy()
            sweep_copy = n
            target_copy = block_target
            trigger_raster = False
            if sweep_copy >= block_target:
                trigger_raster = True
                overlay_count = int(block_target * (overlay_percent / 100.0))
                if overlay_count > 0:
                    self.block_sweep_count = overlay_count
                    self.current_block_average_ch1 = avg1_copy
                    self.current_block_average_ch2 = avg2_copy
                else:
                    self.block_sweep_count = 0
                    self.current_block_average_ch1 = np.zeros(len(self.time_vector))
                    self.current_block_average_ch2 = np.zeros(len(self.time_vector))
            self.param_mutex.unlock()

            if trigger_raster:
                self.update_raster_signal.emit(avg1_copy, avg2_copy, sweep_copy)

            if (sweep_copy % ui_update_rate == 0) or trigger_raster:
                self.update_plot_signal.emit(avg1_copy, last_single_ch1, avg2_copy, last_single_ch2, sweep_copy,
                                             target_copy)
            QThread.msleep(delay_ms)

        self.finished_signal.emit()
        print("Worker: Simulation stopped.")
        self._is_running = False


########################################################################################################################
class BaepSimulator(QMainWindow):
    """
    BAEP Simulation Window.
    """
    simulation_finished = Signal()
    def __init__(self, json_anomaly_path: str = None,
                 learning_manager: LearningManager = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("BAEP Simulator (Block Overlay)")
        self.setGeometry(100, 100, 1200, 750)

        # ... (Simulation parameters unchanged) ...
        self.anomaly_path = json_anomaly_path
        self.learning_manager = learning_manager
        self.fs = 50000
        self.duration_ms = 10
        self.time_vector = np.linspace(0, self.duration_ms, int(self.fs * (self.duration_ms / 1000)))
        self.wave_params = {
            'I': {'lat': 1.6, 'amp': 0.15, 'std': 0.15},
            'II': {'lat': 2.5, 'amp': 0.10, 'std': 0.15},
            'III': {'lat': 3.5, 'amp': 0.30, 'std': 0.30},
            'IV': {'lat': 4.8, 'amp': 0.25, 'std': 0.25},
            'V': {'lat': 5.7, 'amp': 0.50, 'std': 0.40},
        }
        self.cm_freq = 1000
        self.cm_amp = 0.2
        self.cm_decay = 3.0

        self.is_running = False
        self.sweep_count = 0
        self.worker_thread = None
        self.baep_worker = None

        # Buffer for raster data
        self.raster_data_ch1 = None
        self.raster_data_ch2 = None

        ## MODIFIED: Added buffers for Ch2 ##
        self.last_completed_avg_ch1 = None
        self.last_completed_avg_ch2 = None

        # Buffer for raster plot *objects*
        self.raster_lines_ch1 = []
        self.raster_lines_ch2 = []

        # Buffer for peak markers in the raster
        self.raster_peaks_ch1 = []
        self.raster_peaks_ch2 = []  ## NEW

        # Buffer for peak *positions* (latencies)
        self.raster_peak_locations_ch1 = []
        self.raster_peak_locations_ch2 = []  ## NEW

        self._load_anomalies()
        self._setup_ui()
        self._reset_raster_buffers()
        self._initialize_worker_thread()

        print(f"BAEP Simulator started. Anomaly: {self.anomaly_path}")

    # --- _reset_raster_buffers (MODIFIED) ---
    def _reset_raster_buffers(self):
        """
        Resets the data buffers and graphic objects for the stack plot.
        ## MODIFIED: Now also handles Ch1 and Ch2.
        """
        if not hasattr(self, 'raster_rows_spinbox'):
            return

        rows = self.raster_rows_spinbox.value()
        num_samples = len(self.time_vector)
        offset = self.raster_offset_spinbox.value()

        # Signal data buffer
        self.raster_data_ch1 = np.zeros((rows, num_samples))
        self.raster_data_ch2 = np.zeros((rows, num_samples))

        # Peak positions buffer (list of lists, one per row)
        self.raster_peak_locations_ch1 = [[] for _ in range(rows)]
        self.raster_peak_locations_ch2 = [[] for _ in range(rows)]  ## NEW

        # Removing old items (Lines)
        for plot in self.raster_lines_ch1:
            self.raster_plot_ch1.removeItem(plot)
        for plot in self.raster_lines_ch2:
            self.raster_plot_ch2.removeItem(plot)

        # Removing old items (Peak Markers)
        for item in self.raster_peaks_ch1:
            self.raster_plot_ch1.removeItem(item)
        for item in self.raster_peaks_ch2:  ## NEW
            self.raster_plot_ch2.removeItem(item)

        # Empty the object lists
        self.raster_lines_ch1 = []
        self.raster_lines_ch2 = []
        self.raster_peaks_ch1 = []
        self.raster_peaks_ch2 = []  ## NEW

        # Pens for the lines
        pen_ch1 = pg.mkPen('c', width=1)
        pen_ch2 = pg.mkPen('#00FF00', width=1)

        # Style for peak markers (vertical dash)
        peak_pen_ch1 = pg.mkPen('r', width=1)  # Red for Ch1
        peak_pen_ch2 = pg.mkPen('r', width=1)  # Red for Ch2
        peak_symbol = '|'
        peak_size = 10

        # Recreate the objects for each row
        for i in range(rows):
            # Signal lines
            plot_ch1 = pg.PlotDataItem(pen=pen_ch1)
            plot_ch2 = pg.PlotDataItem(pen=pen_ch2)
            self.raster_lines_ch1.append(plot_ch1)
            self.raster_lines_ch2.append(plot_ch2)
            self.raster_plot_ch1.addItem(plot_ch1)
            self.raster_plot_ch2.addItem(plot_ch2)

            # Peak marker objects Ch1
            peak_item_ch1 = pg.ScatterPlotItem(
                pen=peak_pen_ch1,
                symbol=peak_symbol,
                size=peak_size,
                name=f"peaks_ch1_{i}"
            )
            self.raster_peaks_ch1.append(peak_item_ch1)
            self.raster_plot_ch1.addItem(peak_item_ch1)

            # Peak marker objects Ch2 ## NEW
            peak_item_ch2 = pg.ScatterPlotItem(
                pen=peak_pen_ch2,
                symbol=peak_symbol,
                size=peak_size,
                name=f"peaks_ch2_{i}"
            )
            self.raster_peaks_ch2.append(peak_item_ch2)
            self.raster_plot_ch2.addItem(peak_item_ch2)

        # Set Y range (unchanged)
        max_y = ((rows - 1) * offset) + 1.5
        min_y = -1.5
        self.raster_plot_ch1.setYRange(min_y, max_y)
        self.raster_plot_ch2.setYRange(min_y, max_y)

    # --- _setup_ui (MODIFIED) ---
    def _setup_ui(self):
        """Creates the user interface."""
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # --- Control Panel (Left) ---
        # (No changes here, code omitted for brevity)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(250)

        btn = QPushButton("Start Simulation")
        btn.setStyleSheet("backgroud-color=blue; color: white;")
        self.start_stop_button = btn
        self.start_stop_button.clicked.connect(self.start_stop)
        self.reset_button = QPushButton("Reset Simulation")
        self.reset_button.clicked.connect(self.reset_simulation)

        self.stim_side_combo = QComboBox()
        self.stim_side_combo.addItems(["Alternating L/R", "Left", "Right"])
        self.polarity_combo = QComboBox()
        self.polarity_combo.addItems(["Alternating", "Rarefaction", "Condensation"])

        self.stim_rate_spinbox = QSpinBox()
        self.stim_rate_spinbox.setRange(5, 70)
        self.stim_rate_spinbox.setValue(12)
        self.stim_rate_spinbox.setSuffix(" /s (Hz)")

        self.block_sweeps_spinbox = QSpinBox()
        self.block_sweeps_spinbox.setRange(100, 20000)
        self.block_sweeps_spinbox.setValue(200)
        self.block_sweeps_spinbox.setSuffix(" sweeps/block")

        self.overlay_percent_spinbox = QSpinBox()
        self.overlay_percent_spinbox.setRange(0, 90)
        self.overlay_percent_spinbox.setValue(75)
        self.overlay_percent_spinbox.setSuffix(" % Overlay")

        self.ui_update_rate_spinbox = QSpinBox()
        self.ui_update_rate_spinbox.setRange(1, 100)
        self.ui_update_rate_spinbox.setValue(1)
        self.ui_update_rate_spinbox.setSuffix(" sweeps/UI update")
        self.ui_update_rate_spinbox.setToolTip("Updates the plot every N sweeps.")

        self.noise_spinbox = QDoubleSpinBox()
        self.noise_spinbox.setRange(0.1, 5.0)
        self.noise_spinbox.setValue(1.0)
        self.noise_spinbox.setSingleStep(0.1)
        self.noise_spinbox.setSuffix(" µV (std)")

        self.amplitude_spinbox = QDoubleSpinBox()
        self.amplitude_spinbox.setRange(0.1, 5.0)
        self.amplitude_spinbox.setValue(3.0)
        self.amplitude_spinbox.setSingleStep(0.1)
        self.amplitude_spinbox.setSuffix(" x (Gain)")

        self.raster_rows_spinbox = QSpinBox()
        self.raster_rows_spinbox.setRange(5, 100)
        self.raster_rows_spinbox.setValue(10)
        self.raster_rows_spinbox.setSuffix(" rows (blocks)")

        self.raster_offset_spinbox = QDoubleSpinBox()
        self.raster_offset_spinbox.setRange(0.1, 5.0)
        self.raster_offset_spinbox.setValue(1.0)
        self.raster_offset_spinbox.setSingleStep(0.1)
        self.raster_offset_spinbox.setSuffix(" µV Offset")

        self.show_single_checkbox = QCheckBox("Show Input signal")
        self.show_single_checkbox.setChecked(False)
        self.show_single_checkbox.stateChanged.connect(self._toggle_single_sweep_visibility)

        self.sweep_counter_label = QLabel(f"Sweeps: 0 / {self.block_sweeps_spinbox.value()}")

        lbl = QLabel("BAEP Simulation Controls")
        lbl.setStyleSheet("background-color: dark-gray; color: white;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setMinimumHeight(31)
        control_layout.addWidget(lbl)
        run_control_layout = QHBoxLayout()
        run_control_layout.addWidget(self.start_stop_button)
        run_control_layout.addWidget(self.reset_button)
        control_layout.addLayout(run_control_layout)
        control_layout.addSpacing(10)
        control_layout.addWidget(QLabel("Stimulus Side:"))
        control_layout.addWidget(self.stim_side_combo)
        control_layout.addWidget(QLabel("Stimulus Polarity:"))
        control_layout.addWidget(self.polarity_combo)
        control_layout.addSpacing(10)
        control_layout.addWidget(QLabel("Stimulation Rate (Hz):"))
        control_layout.addWidget(self.stim_rate_spinbox)
        control_layout.addWidget(QLabel("Block Size (Sweeps):"))
        control_layout.addWidget(self.block_sweeps_spinbox)
        control_layout.addWidget(QLabel("Block Overlay Percent:"))
        control_layout.addWidget(self.overlay_percent_spinbox)
        control_layout.addWidget(QLabel("Noise Level (µV std):"))
        control_layout.addWidget(self.noise_spinbox)
        control_layout.addWidget(QLabel("Amplitude Scale (Gain):"))
        control_layout.addWidget(self.amplitude_spinbox)
        control_layout.addSpacing(10)
        control_layout.addWidget(QLabel("Stack Plot (History):"))
        control_layout.addWidget(self.raster_rows_spinbox)
        control_layout.addWidget(QLabel("Stack Plot Offset (µV):"))
        control_layout.addWidget(self.raster_offset_spinbox)
        control_layout.addSpacing(10)
        control_layout.addWidget(QLabel("UI Update Rate:"))
        control_layout.addWidget(self.ui_update_rate_spinbox)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.show_single_checkbox)
        control_layout.addSpacing(15)
        control_layout.addWidget(self.sweep_counter_label)
        control_layout.addSpacing(10)
        bottom_button_layout = QHBoxLayout()
        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self._show_help)
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        bottom_button_layout.addWidget(self.help_button)
        bottom_button_layout.addWidget(self.exit_button)
        control_layout.addLayout(bottom_button_layout)
        control_layout.addStretch()

        # --- Plots (Center) ---
        self.plot_layout = pg.GraphicsLayoutWidget()

        # Ch1 Plot
        self.plot_ch1 = self.plot_layout.addPlot(row=0, col=0)
        self.plot_ch1.setTitle("Ch 1: A1 - Cz (Current Block Avg)")
        self.plot_ch1.setLabel('left', 'Amplitude', units='µV')
        self.plot_ch1.setYRange(-1.5, 1.5)
        self.plot_ch1.showGrid(x=True, y=True, alpha=0.3)
        self.plot_ch1.setLabel('bottom', 'Time', units='ms')
        self.plot_curve_avg_ch1 = self.plot_ch1.plot(pen=pg.mkPen('c', width=2), name="Avg Ch1")
        self.plot_curve_single_ch1 = self.plot_ch1.plot(
            pen=pg.mkPen('y', width=1, style=pg.QtCore.Qt.PenStyle.DotLine),
            name="Single Ch1")
        self.peak_markers_ch1 = pg.ScatterPlotItem(
            pen=pg.mkPen(None),
            brush=pg.mkBrush('r'),  # Red
            symbol='o',
            size=8,
            name="Peaks Ch1"
        )
        self.plot_ch1.addItem(self.peak_markers_ch1)

        # Ch2 Plot
        self.plot_ch2 = self.plot_layout.addPlot(row=1, col=0)
        self.plot_ch2.setTitle("Ch 2: A2 - Cz (Current Block Avg)")
        self.plot_ch2.setLabel('left', 'Amplitude', units='µV')
        self.plot_ch2.setLabel('bottom', 'Time', units='ms')
        self.plot_ch2.setYRange(-1.5, 1.5)
        self.plot_ch2.setXRange(0, self.duration_ms)
        self.plot_ch2.showGrid(x=True, y=True, alpha=0.3)
        self.plot_curve_avg_ch2 = self.plot_ch2.plot(pen=pg.mkPen('#00FF00', width=2), name="Avg Ch2")
        self.plot_curve_single_ch2 = self.plot_ch2.plot(
            pen=pg.mkPen('r', width=1, style=pg.QtCore.Qt.PenStyle.DotLine),
            name="Single Ch2")

        ## NEW: Peak markers for Ch2
        self.peak_markers_ch2 = pg.ScatterPlotItem(
            pen=pg.mkPen(None),
            brush=pg.mkBrush('r'),  # red
            symbol='o',
            size=8,
            name="Peaks Ch2"
        )
        self.plot_ch2.addItem(self.peak_markers_ch2)

        # Raster Plots (Stack Plot)
        self.raster_plot_ch1 = self.plot_layout.addPlot(row=0, col=1)
        self.raster_plot_ch1.setTitle("Ch 1 Raster (newest at bottom)")
        self.raster_plot_ch1.hideAxis('left')
        self.raster_plot_ch1.setLabel('bottom', 'Time', units='ms')

        self.raster_plot_ch2 = self.plot_layout.addPlot(row=1, col=1)
        self.raster_plot_ch2.setTitle("Ch 2 Raster (newest at bottom)")
        self.raster_plot_ch2.hideAxis('left')
        self.raster_plot_ch2.setLabel('bottom', 'Time', units='ms')

        # Link X-Axes
        self.plot_ch2.setXLink(self.plot_ch1)
        self.raster_plot_ch1.setXLink(self.plot_ch1)
        self.raster_plot_ch2.setXLink(self.plot_ch1)

        ## TAB Panel (Right) ##
        self.info_tab_widget = QTabWidget()
        self.info_tab_widget.setMaximumWidth(350)

        # -- Tab 1: Block Log (Unchanged) --
        self.log_tab = QWidget()
        log_layout = QVBoxLayout(self.log_tab)
        log_layout.addWidget(QLabel("Block Log (newest at top)"))
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(4)
        self.log_table.setHorizontalHeaderLabels(["Time", "Rate (Hz)", "Overlay (%)", "Noise (µV)"])
        self.log_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.log_table.setColumnWidth(0, 70);
        self.log_table.setColumnWidth(1, 60)
        self.log_table.setColumnWidth(2, 75);
        self.log_table.setColumnWidth(3, 75)
        log_layout.addWidget(self.log_table)
        self.info_tab_widget.addTab(self.log_tab, "Log")

        # -- Tab 2: Latencies (MODIFIED) --
        self.latency_tab = QWidget()
        lat_layout = QFormLayout(self.latency_tab)
        lat_layout.addRow(QLabel("Latencies (Last Block)"))  ## MODIFIED

        # Ch1 Fields
        self.lat_I = QLineEdit("N/A")
        self.lat_III = QLineEdit("N/A")
        self.lat_V = QLineEdit("N/A")
        self.ip_I_III = QLineEdit("N/A")
        self.ip_I_V = QLineEdit("N/A")

        ## NEW: Ch2 Fields (only III and V)
        self.lat_III_ch2 = QLineEdit("N/A")
        self.lat_V_ch2 = QLineEdit("N/A")

        all_fields = [self.lat_I, self.lat_III, self.lat_V, self.ip_I_III, self.ip_I_V,
                      self.lat_III_ch2, self.lat_V_ch2]  ## MODIFIED

        for field in all_fields:
            field.setReadOnly(True)
            field.setStyleSheet("background-color: #2E2E2E; color: white;")

        # Add fields to layout
        lat_layout.addRow(QLabel("--- Channel 1 (A1-Cz) ---"))  ## NEW
        lat_layout.addRow("Latency I (ms):", self.lat_I)
        lat_layout.addRow("Latency III (ms):", self.lat_III)
        lat_layout.addRow("Latency V (ms):", self.lat_V)
        lat_layout.addRow("Inter-peak I-III (ms):", self.ip_I_III)
        lat_layout.addRow("Inter-peak I-V (ms):", self.ip_I_V)

        lat_layout.addRow(QLabel("--- Channel 2 (A2-Cz) ---"))  ## NEW
        lat_layout.addRow("Latency III (ms):", self.lat_III_ch2)
        lat_layout.addRow("Latency V (ms):", self.lat_V_ch2)

        self.stats_button = QPushButton("Statistical Analysis")
        self.stats_button.clicked.connect(self._on_stats_button_clicked)
        self.stats_button.setToolTip("Performs analysis on the entire Stack Plot (future work)")
        lat_layout.addRow(self.stats_button)

        # Save the last signal for analysis
        self.last_completed_avg_ch1 = None
        self.last_completed_avg_ch2 = None  ## NEW

        self.info_tab_widget.addTab(self.latency_tab, "Latencies")

        ## Main Layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.plot_layout, 1)
        main_layout.addWidget(self.info_tab_widget)

        self.setCentralWidget(main_widget)

    # --- _initialize_worker_thread (Unchanged) ---
    def _initialize_worker_thread(self):
        if self.noise_spinbox is None:
            print("ERROR: UI widgets have not been created yet.")
            return

        self.worker_thread = QThread()
        self.baep_worker = BaepWorker(
            time_vector=self.time_vector,
            wave_params=self.wave_params,
            cm_freq=self.cm_freq,
            cm_amp=self.cm_amp,
            cm_decay=self.cm_decay,
            fs=self.fs,
            initial_noise=self.noise_spinbox.value(),
            initial_ui_update_rate=self.ui_update_rate_spinbox.value(),
            initial_polarity_mode=self.polarity_combo.currentText(),
            initial_stim_side=self.stim_side_combo.currentText(),
            initial_amplitude_scale=self.amplitude_spinbox.value(),
            initial_block_sweeps_target=self.block_sweeps_spinbox.value(),
            initial_overlay_percent=self.overlay_percent_spinbox.value(),
            initial_stim_rate=self.stim_rate_spinbox.value()
        )
        self.baep_worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.baep_worker.run_simulation)
        self.baep_worker.update_plot_signal.connect(self._update_ui_from_worker)
        self.baep_worker.finished_signal.connect(self._simulation_finished)
        self.baep_worker.update_raster_signal.connect(self._update_raster_plots)
        self.noise_spinbox.valueChanged.connect(self.baep_worker.set_noise_level)
        self.ui_update_rate_spinbox.valueChanged.connect(self.baep_worker.set_ui_update_rate)
        self.amplitude_spinbox.valueChanged.connect(self.baep_worker.set_amplitude_scale)
        self.block_sweeps_spinbox.valueChanged.connect(self.baep_worker.set_block_sweeps_target)
        self.overlay_percent_spinbox.valueChanged.connect(self.baep_worker.set_overlay_percent)
        self.stim_rate_spinbox.valueChanged.connect(self.baep_worker.set_stim_rate)
        self.raster_rows_spinbox.valueChanged.connect(self._reset_raster_buffers)
        self.raster_offset_spinbox.valueChanged.connect(self._reset_raster_buffers)
        self.polarity_combo.currentTextChanged.connect(self.baep_worker.set_polarity_mode)
        self.stim_side_combo.currentTextChanged.connect(self.baep_worker.set_stim_side)

    # --- _show_help (Unchanged) ---
    def _show_help(self):
        QMessageBox.information(self, "BAEP Simulator Help",
                                "Simulates a 2-channel BAEP response (A1-Cz, A2-Cz).\n\n"
                                "- **Stimulation Rate (Hz):** Controls the stimulation frequency (how many stimuli per second).\n"
                                "- **Block Size:** Defines how many sweeps make up an averaging block.\n"
                                "- **Block Overlay Percent:** Defines what percentage (in 'weight') of the previous block is used as the starting point for the next block.\n"
                                "- **Stack Plot:** Shows the final average of each completed block.\n"
                                "- **UI Update Rate:** How many sweeps to process before updating the plot (does not affect stimulation speed).\n"
                                "- **Start/Stop:** Starts and pauses the simulation.\n"
                                "- **Reset:** Resets the current block and the stack plot.")
        pdf_path = 'help_docs/paper/Tut_Baep.pdf'
        wb.open_new(pdf_path)

    # --- _log_block_data (Unchanged) ---
    def _log_block_data(self):
        try:
            timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
            rate = self.stim_rate_spinbox.value()
            overlay = self.overlay_percent_spinbox.value()
            noise = self.noise_spinbox.value()
            self.log_table.insertRow(0)
            time_item = QTableWidgetItem(timestamp)
            rate_item = QTableWidgetItem(f"{rate}")
            overlay_item = QTableWidgetItem(f"{overlay}%")
            noise_item = QTableWidgetItem(f"{noise:.1f}")
            for item in [time_item, rate_item, overlay_item, noise_item]:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.log_table.setItem(0, 0, time_item)
            self.log_table.setItem(0, 1, rate_item)
            self.log_table.setItem(0, 2, overlay_item)
            self.log_table.setItem(0, 3, noise_item)
            if self.log_table.rowCount() > 200:
                self.log_table.removeRow(200)
        except Exception as e:
            print(f"Error while updating log: {e}")

    # --- _analyze_and_display_peaks_ch1 (MODIFIED: Name) ---
    def _analyze_and_display_peaks_ch1(self, signal_data):
        """
        Analyzes peaks on CH1, updates the Latencies table
        and markers on the Ch1 plot.
        Returns the latencies (I, III, V) found for the raster.
        """
        # Clear markers and Ch1 fields
        self.peak_markers_ch1.clear()
        for field in [self.lat_I, self.lat_III, self.lat_V, self.ip_I_III, self.ip_I_V]:
            field.setText("N/A")

        identified_lats_for_raster = []

        if signal_data is None:
            return identified_lats_for_raster

        try:
            min_dist_samples = int(0.5 * self.fs / 1000)
            peaks_indices, properties = scipy.signal.find_peaks(
                signal_data,
                height=0.05,
                distance=min_dist_samples
            )

            if len(peaks_indices) == 0:
                return identified_lats_for_raster

            found_lats_ms = self.time_vector[peaks_indices]
            found_amps = signal_data[peaks_indices]

            # Identify I, III, V
            expected_windows = {
                'I': (1.2, 2.2),
                'III': (3.0, 4.2),
                'V': (5.0, 6.5)
            }
            identified_peaks = {'I': None, 'III': None, 'V': None}
            for lat, amp in zip(found_lats_ms, found_amps):
                if expected_windows['I'][0] <= lat <= expected_windows['I'][1]:
                    if identified_peaks['I'] is None or amp > identified_peaks['I'][1]:
                        identified_peaks['I'] = (lat, amp)
                elif expected_windows['III'][0] <= lat <= expected_windows['III'][1]:
                    if identified_peaks['III'] is None or amp > identified_peaks['III'][1]:
                        identified_peaks['III'] = (lat, amp)
                elif expected_windows['V'][0] <= lat <= expected_windows['V'][1]:
                    if identified_peaks['V'] is None or amp > identified_peaks['V'][1]:
                        identified_peaks['V'] = (lat, amp)

            # Populate the Ch1 table
            lat_I, lat_III, lat_V = None, None, None
            plot_lats = []
            plot_amps = []

            if identified_peaks['I']:
                lat_I = identified_peaks['I'][0]
                self.lat_I.setText(f"{lat_I:.2f}")
                plot_lats.append(lat_I)
                plot_amps.append(identified_peaks['I'][1])
                identified_lats_for_raster.append(lat_I)

            if identified_peaks['III']:
                lat_III = identified_peaks['III'][0]
                self.lat_III.setText(f"{lat_III:.2f}")
                plot_lats.append(lat_III)
                plot_amps.append(identified_peaks['III'][1])
                identified_lats_for_raster.append(lat_III)

            if identified_peaks['V']:
                lat_V = identified_peaks['V'][0]
                self.lat_V.setText(f"{lat_V:.2f}")
                plot_lats.append(lat_V)
                plot_amps.append(identified_peaks['V'][1])
                identified_lats_for_raster.append(lat_V)

            if lat_I and lat_III:
                self.ip_I_III.setText(f"{lat_III - lat_I:.2f}")
            if lat_I and lat_V:
                self.ip_I_V.setText(f"{lat_V - lat_I:.2f}")

            # Draw the "dots" on the Ch1 plot
            if plot_lats:
                self.peak_markers_ch1.setData(x=plot_lats, y=plot_amps)

            return identified_lats_for_raster

        except Exception as e:
            print(f"Error during automatic Ch1 peak analysis: {e}")
            return []

    ## --- NEW FUNCTION FOR CH2 ---
    def _analyze_and_display_peaks_ch2(self, signal_data):
        """
        Analyzes peaks on CH2 (only III, V), updates the Latencies table
        and markers on the Ch2 plot.
        Returns the latencies (III, V) found for the raster.
        """
        # Clear markers and Ch2 fields
        self.peak_markers_ch2.clear()
        self.lat_III_ch2.setText("N/A")
        self.lat_V_ch2.setText("N/A")

        identified_lats_for_raster = []

        if signal_data is None:
            return identified_lats_for_raster

        try:
            min_dist_samples = int(0.5 * self.fs / 1000)
            peaks_indices, properties = scipy.signal.find_peaks(
                signal_data,
                height=0.05,
                distance=min_dist_samples
            )

            if len(peaks_indices) == 0:
                return identified_lats_for_raster

            found_lats_ms = self.time_vector[peaks_indices]
            found_amps = signal_data[peaks_indices]

            # Identify only III, V (contralateral signal)
            expected_windows = {
                'III': (3.0, 4.2),
                'V': (5.0, 6.5)
            }
            identified_peaks = {'III': None, 'V': None}
            for lat, amp in zip(found_lats_ms, found_amps):
                if expected_windows['III'][0] <= lat <= expected_windows['III'][1]:
                    if identified_peaks['III'] is None or amp > identified_peaks['III'][1]:
                        identified_peaks['III'] = (lat, amp)
                elif expected_windows['V'][0] <= lat <= expected_windows['V'][1]:
                    if identified_peaks['V'] is None or amp > identified_peaks['V'][1]:
                        identified_peaks['V'] = (lat, amp)

            # Populate the Ch2 table
            lat_III, lat_V = None, None
            plot_lats = []
            plot_amps = []

            if identified_peaks['III']:
                lat_III = identified_peaks['III'][0]
                self.lat_III_ch2.setText(f"{lat_III:.2f}")
                plot_lats.append(lat_III)
                plot_amps.append(identified_peaks['III'][1])
                identified_lats_for_raster.append(lat_III)

            if identified_peaks['V']:
                lat_V = identified_peaks['V'][0]
                self.lat_V_ch2.setText(f"{lat_V:.2f}")
                plot_lats.append(lat_V)
                plot_amps.append(identified_peaks['V'][1])
                identified_lats_for_raster.append(lat_V)

            # Draw the "dots" on the Ch2 plot
            if plot_lats:
                self.peak_markers_ch2.setData(x=plot_lats, y=plot_amps)

            return identified_lats_for_raster

        except Exception as e:
            print(f"Error during automatic Ch2 peak analysis: {e}")
            return []

    # --- _toggle_single_sweep_visibility (Unchanged) ---
    @Slot(int)
    def _toggle_single_sweep_visibility(self, state):
        is_visible = (state == Qt.CheckState.Checked.value)
        if is_visible:
            self.plot_curve_single_ch1.show()
            self.plot_curve_single_ch2.show()
        else:
            self.plot_curve_single_ch1.hide()
            self.plot_curve_single_ch2.hide()
            self.plot_curve_single_ch1.clear()
            self.plot_curve_single_ch2.clear()

    # --- _update_ui_from_worker (Unchanged) ---
    @Slot(np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int)
    def _update_ui_from_worker(self,
                               avg_ch1: np.ndarray, single_ch1: np.ndarray,
                               avg_ch2: np.ndarray, single_ch2: np.ndarray,
                               sweep_count: int, target_sweeps: int):
        self.sweep_count = sweep_count
        self.plot_curve_avg_ch1.setData(self.time_vector, avg_ch1)
        self.plot_curve_avg_ch2.setData(self.time_vector, avg_ch2)

        if self.show_single_checkbox.isChecked() and single_ch1 is not None:
            self.plot_curve_single_ch1.setData(self.time_vector, single_ch1)
            self.plot_curve_single_ch2.setData(self.time_vector, single_ch2)
        else:
            self.plot_curve_single_ch1.clear()
            self.plot_curve_single_ch2.clear()

        self.sweep_counter_label.setText(f"Sweeps: {self.sweep_count} / {target_sweeps}")

    # --- _update_raster_plots (MODIFIED) ---
    @Slot(np.ndarray, np.ndarray, int)
    def _update_raster_plots(self, avg_ch1, avg_ch2, sweep_count):
        """
        Updates the stack plot (raster) when a block is completed.
        ## MODIFIED: Now analyzes and updates Ch1 and Ch2.
        """

        # Analyze Ch1
        self.last_completed_avg_ch1 = avg_ch1.copy()
        peak_latencies_ch1 = self._analyze_and_display_peaks_ch1(avg_ch1)

        # Analyze Ch2 ## NEW
        self.last_completed_avg_ch2 = avg_ch2.copy()
        peak_latencies_ch2 = self._analyze_and_display_peaks_ch2(avg_ch2)

        # Roll the signal buffers
        self.raster_data_ch1 = np.roll(self.raster_data_ch1, shift=1, axis=0)
        self.raster_data_ch2 = np.roll(self.raster_data_ch2, shift=1, axis=0)
        self.raster_data_ch1[0, :] = avg_ch1
        self.raster_data_ch2[0, :] = avg_ch2

        # Roll the Ch1 peak positions buffer
        if self.raster_peak_locations_ch1:
            self.raster_peak_locations_ch1.pop()
        self.raster_peak_locations_ch1.insert(0, peak_latencies_ch1)

        # Roll the Ch2 peak positions buffer ## NEW
        if self.raster_peak_locations_ch2:
            self.raster_peak_locations_ch2.pop()
        self.raster_peak_locations_ch2.insert(0, peak_latencies_ch2)

        offset_step = self.raster_offset_spinbox.value()
        num_rows = len(self.raster_lines_ch1)

        for i in range(num_rows):
            current_offset = i * offset_step

            # Update the signal line (Ch1 and Ch2)
            plot_data_ch1 = self.raster_data_ch1[i, :] + current_offset
            plot_data_ch2 = self.raster_data_ch2[i, :] + current_offset
            self.raster_lines_ch1[i].setData(self.time_vector, plot_data_ch1)
            self.raster_lines_ch2[i].setData(self.time_vector, plot_data_ch2)

            # Update Ch1 peak markers
            if i < len(self.raster_peak_locations_ch1) and i < len(self.raster_peaks_ch1):
                current_peak_lats_ch1 = self.raster_peak_locations_ch1[i]
                if current_peak_lats_ch1:
                    y_values = [current_offset] * len(current_peak_lats_ch1)
                    self.raster_peaks_ch1[i].setData(x=current_peak_lats_ch1, y=y_values)
                else:
                    self.raster_peaks_ch1[i].clear()

            # Update Ch2 peak markers ## NEW
            if i < len(self.raster_peak_locations_ch2) and i < len(self.raster_peaks_ch2):
                current_peak_lats_ch2 = self.raster_peak_locations_ch2[i]
                if current_peak_lats_ch2:
                    y_values = [current_offset] * len(current_peak_lats_ch2)
                    self.raster_peaks_ch2[i].setData(x=current_peak_lats_ch2, y=y_values)
                else:
                    self.raster_peaks_ch2[i].clear()

        ## Log the data (unchanged)
        self._log_block_data()

    # --- _simulation_finished (Unchanged) ---
    @Slot()
    def _simulation_finished(self):
        self.is_running = False
        self.start_stop_button.setText("Start Simulation")
        self.start_stop_button.setStyleSheet("background-color: blue")
        self.start_stop_button.setEnabled(True)
        if self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()

    # --- start_stop (Unchanged) ---
    def start_stop(self):
        if self.is_running:
            self.start_stop_button.setText("Stopping...")
            self.start_stop_button.setEnabled(False)
            if self.baep_worker:
                self.baep_worker.stop_worker()
        else:
            if self.worker_thread and self.worker_thread.isFinished():
                self.worker_thread.wait()
            self.is_running = True
            self.worker_thread.start()
            self.start_stop_button.setText("Stop Simulation")
            self.start_stop_button.setStyleSheet("background-color: red")

    # --- reset_simulation (MODIFIED) ---
    @Slot()
    def reset_simulation(self):
        """
        Resets the stack plot, current block, logs, and latencies.
        ## MODIFIED: Also clears Ch2 elements.
        """
        print("--- RESETTING STACK PLOT AND CURRENT BLOCK ---")

        # 1. Reset the stack plot buffers and UI (now includes Ch2)
        self._reset_raster_buffers()

        # 2. Reset the worker
        if self.baep_worker:
            self.baep_worker.reset_worker()

        # 3. Reset the counter UI
        self.sweep_count = 0
        current_target = self.block_sweeps_spinbox.value()
        self.sweep_counter_label.setText(f"Sweeps: 0 / {current_target}")

        # 4. Reset the main plots
        empty_data = np.zeros(len(self.time_vector))
        self.plot_curve_avg_ch1.setData(self.time_vector, empty_data)
        self.plot_curve_single_ch1.setData(self.time_vector, empty_data)
        self.plot_curve_avg_ch2.setData(self.time_vector, empty_data)
        self.plot_curve_single_ch2.setData(self.time_vector, empty_data)

        # 5. Clear info tab and markers
        self.log_table.setRowCount(0)

        self.last_completed_avg_ch1 = None
        self.last_completed_avg_ch2 = None  ## NEW

        self.peak_markers_ch1.clear()
        self.peak_markers_ch2.clear()  ## NEW

        # Clear all latency text fields
        all_fields = [self.lat_I, self.lat_III, self.lat_V, self.ip_I_III, self.ip_I_V,
                      self.lat_III_ch2, self.lat_V_ch2]  ## MODIFIED
        for field in all_fields:
            field.setText("N/A")

        print("--- RESET COMPLETE ---")

    # --- _on_stats_button_clicked (Unchanged) ---
    @Slot()
    def _on_stats_button_clicked(self):
        QMessageBox.information(self, "In Development",
                                "This function is reserved for future statistical analysis "
                                "(e.g., average latencies, standard deviation, etc.)")

    # --- _load_anomalies (Unchanged) ---
    def _load_anomalies(self):
        if self.anomaly_path:
            print(f"Loading anomaly from: {self.anomaly_path}")
            pass
        else:
            print("No anomaly specified, using standard BAEP parameters.")

    # --- closeEvent (Unchanged) ---
    def closeEvent(self, event):
        print("Closing BAEP simulator...")
        if self.worker_thread and self.worker_thread.isRunning():
            self.baep_worker.stop_worker()
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.simulation_finished.emit()
        super().closeEvent(event)


# --- Main execution block (for testing) ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mock_manager = LearningManager()
    main_window = BaepSimulator(learning_manager=mock_manager)
    main_window.show()
    sys.exit(app.exec())