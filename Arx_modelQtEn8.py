import sys
import numpy as np

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QTimer
# Import new widgets
from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QMainWindow, QVBoxLayout, QApplication,
    QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QFileDialog,
    QMessageBox  # <-- NEW: For pop-up messages
)

import pyqtgraph as pg
import pandas as pd
from statsmodels.stats import diagnostic  # accor_ljungbox
from vedo.pyplot import histogram, plot
from vedo import settings, show, Text2D, Plotter
from scipy.stats import norm
from scipy.signal import butter, filtfilt



# --- 1. Simulation Parameters (MODIFIED) ---
FS = 2000  # Sampling Frequency (Hz)

# New sweep parameters
STIM_RATE = 20
N_PRE_STIM_SAMPLES = 20  # <--- NEW: Samples before stimulus
N_SIGNAL_SAMPLES = 200  # <--- NEW: Duration of SEP signal (formerly N_POINTS_PER_SWEEP)
N_POST_STIM_SAMPLES = 60  # <--- NEW: Samples after signal

N_TOTAL_SWEEP_SAMPLES = N_PRE_STIM_SAMPLES + N_SIGNAL_SAMPLES + N_POST_STIM_SAMPLES  # 280 samples
SWEEP_DURATION_SEC = N_TOTAL_SWEEP_SAMPLES / FS  # 0.14 seconds

N_TOTAL_SWEEPS = 5000   # Total number of sweeps to simulate
K_BLOCK_SIZE = 10       # Block size for "input" average

SIGNAL_AMPLITUDE = 3.0  # microVolts (our "true" SEP)
NOISE_AMPLITUDE = 15.0  # microVolts (the background EEG noise)

# --- 2. Create a "True" SEP Signal (Template) (MODIFIED) ---

# 2a. Time axis for the *signal* (200 samples, from 0 to 0.1s)
# This is only used to generate the waveform SHAPE
signal_time_axis = np.linspace(0, N_SIGNAL_SAMPLES / FS, N_SIGNAL_SAMPLES, endpoint=False)

# 2b. Create the SEP *signal* (200 samples of pure waveform)
N20_PEAK_TIME_IDEAL = 0.019
P30_PEAK_TIME_IDEAL = 0.026  # This is P25 in your code
N30_PEAK_TIME_IDEAL = 0.040  # This is N40 in your code

n20_shape = -SIGNAL_AMPLITUDE * 1.2 * np.exp(-((signal_time_axis - N20_PEAK_TIME_IDEAL) ** 2) / (2 * (0.0015 ** 2)))
p30_shape = SIGNAL_AMPLITUDE * 0.8 * np.exp(-((signal_time_axis - P30_PEAK_TIME_IDEAL) ** 2) / (2 * (0.003 ** 2)))
n30_shape = -SIGNAL_AMPLITUDE * 1.5 * np.exp(-((signal_time_axis - N30_PEAK_TIME_IDEAL) ** 2) / (2 * (0.007 ** 2)))
vero_sep_shape = n20_shape + p30_shape + n30_shape

# 2c. *Total* time axis (280 samples)
# We center it so that 0.0s is the time of the stimulus
time_axis = np.linspace(
    -N_PRE_STIM_SAMPLES / FS,
    (N_SIGNAL_SAMPLES + N_POST_STIM_SAMPLES) / FS,
    N_TOTAL_SWEEP_SAMPLES,
    endpoint=False
)

# 2d. Create the ideal *total* template (280 samples)
# It will be all zeros, except for the signal window
vero_sep_template = np.zeros(N_TOTAL_SWEEP_SAMPLES)

# Insert the SEP shape in the right place (after the 20 pre-stim samples)
start_index = N_PRE_STIM_SAMPLES
end_index = N_PRE_STIM_SAMPLES + N_SIGNAL_SAMPLES
vero_sep_template[start_index:end_index] = vero_sep_shape

'''
# --- 3. "Real-Time" Simulator Class (UPDATED to accept modifiers) ---
class EEGSimulator:
    def __init__(self):
        self.sweep_count = 0
        self.jitter_max_sec = 0.001
        print(f"EEG Simulator initialized with N20, P25, N40 Jitter: +/- {self.jitter_max_sec * 1000} ms")
        print(f"Total sweep: {N_TOTAL_SWEEP_SAMPLES} samples ({SWEEP_DURATION_SEC:.3f}s)")
        print(f"  Pre-Stimulus: {N_PRE_STIM_SAMPLES} samples")
        print(f"  SEP Signal: {N_SIGNAL_SAMPLES} samples")
        print(f"  Post-Stimulus: {N_POST_STIM_SAMPLES} samples")

    # --- MODIFIED: Method signature changed ---
    def get_next_sweep(self, latency_mod_ms, amplitude_mod_percent):
        if self.sweep_count >= N_TOTAL_SWEEPS:
            return None

        # --- NEW: Get modifiers from spin boxes ---
        latency_mod_sec = latency_mod_ms / 1000.0
        amplitude_mod_factor = amplitude_mod_percent / 100.0
        # --- END NEW ---

        # 1. Calculate random jitter (as before)
        jitter_offset_n20 = np.random.uniform(0.0, self.jitter_max_sec)
        jitter_offset_p30 = np.random.uniform(-self.jitter_max_sec, self.jitter_max_sec)
        jitter_offset_n30 = np.random.uniform(-self.jitter_max_sec, self.jitter_max_sec)

        # --- MODIFIED: Apply latency modifier from spin box ---
        n20_peak_time = N20_PEAK_TIME_IDEAL + jitter_offset_n20 + latency_mod_sec
        p30_peak_time = P30_PEAK_TIME_IDEAL + jitter_offset_p30 + latency_mod_sec
        n30_peak_time = N30_PEAK_TIME_IDEAL + jitter_offset_n30 + latency_mod_sec
        # --- END MODIFIED ---

        # 2. Generate the jittered signal *shape* (200 samples)
        # --- MODIFIED: Apply amplitude modifier from spin box ---
        n20_jittered = -SIGNAL_AMPLITUDE * np.exp(
            -((signal_time_axis - n20_peak_time) ** 2) / (2 * (0.003 ** 2))) * amplitude_mod_factor
        p30_jittered = SIGNAL_AMPLITUDE * 0.8 * np.exp(
            -((signal_time_axis - p30_peak_time) ** 2) / (2 * (0.004 ** 2))) * amplitude_mod_factor
        n30_jittered = -SIGNAL_AMPLITUDE * 1.5 * np.exp(
            -((signal_time_axis - n30_peak_time) ** 2) / (2 * (0.005 ** 2))) * amplitude_mod_factor
        # --- END MODIFIED ---

        # This is the *pure* 200-sample signal with jitter and modifiers
        sep_con_jitter = n20_jittered + p30_jittered + n30_jittered

        # 3. Generate noise for the 3 separate sections
        pre_stim_noise = np.random.randn(N_PRE_STIM_SAMPLES) * NOISE_AMPLITUDE
        signal_noise = np.random.randn(N_SIGNAL_SAMPLES) * NOISE_AMPLITUDE
        post_stim_noise = np.random.randn(N_POST_STIM_SAMPLES) * NOISE_AMPLITUDE

        # 4. Create the noisy signal part (signal + noise)
        rumorous_signal_part = sep_con_jitter + signal_noise  # (200 samples)

        # 5. Concatenate everything for the final sweep (280 samples)
        rumorous_sweep = np.concatenate((
            pre_stim_noise,
            rumorous_signal_part,
            post_stim_noise
        ))

        self.sweep_count += 1
        return rumorous_sweep
'''

# --- 3. "Real-Time" Simulator Class (UPDATED for infinite run) ---
class EEGSimulator:
    def __init__(self):
        self.sweep_count = 0
        self.jitter_max_sec = 0.001
        print(f"EEG Simulator initialized with N20, P25, N40 Jitter: +/- {self.jitter_max_sec * 1000} ms")
        print(f"Total sweep: {N_TOTAL_SWEEP_SAMPLES} samples ({SWEEP_DURATION_SEC:.3f}s)")
        print(f"  Pre-Stimulus: {N_PRE_STIM_SAMPLES} samples")
        print(f"  SEP Signal: {N_SIGNAL_SAMPLES} samples")
        print(f"  Post-Stimulus: {N_POST_STIM_SAMPLES} samples")
        print("--- Simulation running indefinitely (no total sweep limit) ---")


    # --- MODIFIED: Method signature changed ---
    def get_next_sweep(self, latency_mod_ms, amplitude_mod_percent):
        # --- MODIFIED: Removed check for N_TOTAL_SWEEPS ---
        # if self.sweep_count >= N_TOTAL_SWEEPS:
        #     return None

        # --- NEW: Get modifiers from spin boxes ---
        latency_mod_sec = latency_mod_ms / 1000.0
        amplitude_mod_factor = amplitude_mod_percent / 100.0
        # --- END NEW ---

        # 1. Calculate random jitter (as before)
        jitter_offset_n20 = np.random.uniform(-self.jitter_max_sec, self.jitter_max_sec)
        jitter_offset_p30 = np.random.uniform(-self.jitter_max_sec, self.jitter_max_sec)
        jitter_offset_n30 = np.random.uniform(-self.jitter_max_sec, self.jitter_max_sec)

        # --- MODIFIED: Apply latency modifier from spin box ---
        n20_peak_time = N20_PEAK_TIME_IDEAL + jitter_offset_n20 + latency_mod_sec
        p30_peak_time = P30_PEAK_TIME_IDEAL + jitter_offset_p30 + latency_mod_sec
        n30_peak_time = N30_PEAK_TIME_IDEAL + jitter_offset_n30 + latency_mod_sec
        # --- END MODIFIED ---

        # 2. Generate the jittered signal *shape* (200 samples)
        # --- MODIFIED: Apply amplitude modifier from spin box ---
        n20_jittered = -SIGNAL_AMPLITUDE * 1.2 * np.exp(
            -((signal_time_axis - n20_peak_time) ** 2) / (2 * (0.0015 ** 2))) * amplitude_mod_factor
        p30_jittered = SIGNAL_AMPLITUDE * 0.8 * np.exp(
            -((signal_time_axis - p30_peak_time) ** 2) / (2 * (0.003 ** 2))) * amplitude_mod_factor
        n30_jittered = -SIGNAL_AMPLITUDE * 1.5 * np.exp(
            -((signal_time_axis - n30_peak_time) ** 2) / (2 * (0.007 ** 2))) * amplitude_mod_factor
        # --- END MODIFIED ---

        # This is the *pure* 200-sample signal with jitter and modifiers
        sep_con_jitter = n20_jittered + p30_jittered + n30_jittered

        # 3. Generate noise for the 3 separate sections TODO rerify why two sections post stimulus ???
        pre_stim_noise = np.random.randn(N_PRE_STIM_SAMPLES) * NOISE_AMPLITUDE
        signal_noise = np.random.randn(N_SIGNAL_SAMPLES) * NOISE_AMPLITUDE
        post_stim_noise = np.random.randn(N_POST_STIM_SAMPLES) * NOISE_AMPLITUDE

        # 4. Create the noisy signal part (signal + noise)
        rumorous_signal_part = sep_con_jitter + signal_noise  # (200 samples)

        # 5. Concatenate everything for the final sweep (280 samples)
        rumorous_sweep = np.concatenate((
            pre_stim_noise,
            rumorous_signal_part,
            post_stim_noise
        ))
        self.sweep_count += 1
        return rumorous_sweep

# --- 4. ARX Processor Class (Unchanged) ---
class ProcessoreARX:
    def __init__(self):
        self.na_range = range(1, 9)
        self.nb_range = range(1, 9)
        self.ljung_box_lags = 30
        self.p_value_threshold = 0.05
        print(f"ARX Processor (NumPy+Ljung-Box) initialized.")
        print(f"  Order search: na={self.na_range}, nb={self.nb_range}")
        print(f"  Ljung-Box Test: lags={self.ljung_box_lags}, p-value > {self.p_value_threshold}")

    def calcola_aic(self, n, k, err_var):
        if err_var <= 0: return np.inf
        return np.log(err_var) + 2 * k / n

    def build_regressor_matrix(self, y, u, na, nb):
        if nb > 0:
            kb = -(nb - 1) // 2
            nb1 = kb
            nb2 = nb + kb - 1
            lags_u = list(range(nb1, nb2 + 1))
        else:
            lags_u = []
        lags_y = list(range(1, na + 1))
        start_idx = 0
        if lags_y: start_idx = max(start_idx, max(lags_y))
        if lags_u:
            pos_lags_u = [k for k in lags_u if k > 0]
            if pos_lags_u: start_idx = max(start_idx, max(pos_lags_u))
        end_idx = len(y)
        if lags_u:
            neg_lags_u = [k for k in lags_u if k < 0]
            if neg_lags_u: end_idx = min(end_idx, len(y) + min(neg_lags_u))
        N_valid = end_idx - start_idx
        n_params = na + nb
        if N_valid <= n_params or n_params == 0: return None, None, None
        phi = np.zeros((N_valid, n_params))
        y_target = y[start_idx:end_idx]
        col_idx = 0
        for k in lags_y:
            phi[:, col_idx] = y[start_idx - k: end_idx - k]
            col_idx += 1
        for k in lags_u:
            phi[:, col_idx] = u[start_idx - k: end_idx - k]
            col_idx += 1
        return phi, y_target, (start_idx, end_idx)

    def elabora_sweep(self, y_output, X_input):
        best_aic = np.inf
        best_model_params = None
        best_orders = (0, 0)
        best_p_value = 0
        max_possible_lag = max(self.na_range[-1], self.nb_range[-1] // 2 + 2)
        total_padding = max(max_possible_lag, self.ljung_box_lags + 1)
        y_padded = np.pad(y_output, (total_padding, total_padding), 'constant')
        u_padded = np.pad(X_input, (total_padding, total_padding), 'constant')
        for na in self.na_range:
            for nb in self.nb_range:
                if na == 0 and nb == 0: continue
                n_params = na + nb
                phi, y_target, valid_indices = self.build_regressor_matrix(y_padded, u_padded, na, nb)
                if phi is None or phi.shape[0] < (n_params + self.ljung_box_lags): continue
                try:
                    theta, res, rank, s = np.linalg.lstsq(phi, y_target, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                residuals = y_target - (phi @ theta)
                lags_to_test = min(self.ljung_box_lags, len(residuals) // 2 - 2)
                if lags_to_test <= 0: continue
                lb_test_result = diagnostic.acorr_ljungbox(residuals, lags=[lags_to_test], return_df=True)
                p_value = lb_test_result['lb_pvalue'].iloc[0]
                if p_value < self.p_value_threshold: continue
                err_var = np.var(residuals)
                n_err = len(residuals)
                current_aic = self.calcola_aic(n_err, n_params, err_var)
                if current_aic < best_aic:
                    best_aic = current_aic
                    best_model_params = theta
                    best_orders = (na, nb)
                    best_p_value = p_value
        if best_model_params is not None:
            na, nb = best_orders
            phi_full, y_full, valid_indices = self.build_regressor_matrix(y_padded, u_padded, na, nb)
            if phi_full is None: return None, (0, 0), 0, 0
            y_identificato_valid = phi_full @ best_model_params
            y_identificato_padded = np.zeros_like(y_padded, dtype=float)
            start, end = valid_indices
            y_identificato_padded[start:end] = y_identificato_valid
            y_identificato_unpadded = y_identificato_padded[total_padding:-total_padding]
            if len(y_identificato_unpadded) != len(y_output):
                print(f"Dimension error: expected {len(y_output)}, got {len(y_identificato_unpadded)}")
                return None, (0, 0), 0, 0
            return y_identificato_unpadded, best_orders, best_aic, best_p_value
        else:
            return None, (0, 0), 0, 0

class ResultsWindow(QWidget):
    """
    A new window for displaying result box plots.
    It is initialized with latency and amplitude data.
    """

    def __init__(self, n20_latencies, n20_amplitudes, n30_latencies, n30_amplitudes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Latency & Amplitude Analysis Results")
        self.setGeometry(1000, 100, 700, 600)  # Increased height for two plots

        # Main layout
        main_layout = QVBoxLayout()
        plot_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(plot_widget)
        self.setLayout(main_layout)

        # --- Convert Data ---
        latencies_ms_n20 = np.array(n20_latencies) * 1000
        latencies_ms_n30 = np.array(n30_latencies) * 1000
        amps_n20 = np.array(n20_amplitudes)
        amps_n30 = np.array(n30_amplitudes)

        # --- Plot 1: Latency Box Plots ---
        plot1 = plot_widget.addPlot(row=0, col=0)
        plot1.setTitle("Peak Latency Distribution")
        plot1.setLabel('left', "Latency (ms)")
        plot1.getAxis('bottom').setTicks([[(0, "N20 Latency"), (1, "N30/N40 Latency")]])

        # Calculate stats and draw N20 Latency box
        stats_n20_lat = self.get_boxplot_stats(latencies_ms_n20)
        self.draw_boxplot(plot1, 0, stats_n20_lat, pg.mkBrush(255, 100, 100, 150))

        # Calculate stats and draw N30 Latency box
        stats_n30_lat = self.get_boxplot_stats(latencies_ms_n30)
        self.draw_boxplot(plot1, 1, stats_n30_lat, pg.mkBrush(100, 100, 255, 150))

        # --- 3. Plot 2: Amplitude Box Plots ---
        plot2 = plot_widget.addPlot(row=1, col=0)
        plot2.setTitle("Peak Amplitude Distribution")
        plot2.setLabel('left', "Amplitude (uV)")
        plot2.invertY(True)  # Invert Y-axis for negative amplitude peaks
        plot2.getAxis('bottom').setTicks([
            [(0, "N20 Amplitude"), (1, "N30/N40 Amplitude")]
        ])

        # Calculate stats and draw N20 Amplitude box
        stats_n20_amp = self.get_boxplot_stats(amps_n20)
        self.draw_boxplot(plot2, 0, stats_n20_amp, pg.mkBrush(255, 100, 100, 150))

        # Calculate stats and draw N30 Amplitude box
        stats_n30_amp = self.get_boxplot_stats(amps_n30)
        self.draw_boxplot(plot2, 1, stats_n30_amp, pg.mkBrush(100, 100, 255, 150))

        # Link X axes for zooming
        plot2.setXLink(plot1)

        # Manually set Y ranges to be independent (optional but good practice)
        plot1.enableAutoRange(axis='y')
        plot2.enableAutoRange(axis='y')
        plot1.autoRange()
        plot2.autoRange()

    def get_boxplot_stats(self, data):
        """Calculates all necessary statistics for a box plot."""
        if len(data) == 0:
            # Return empty stats if no data
            return {"mean": 0, "std": 0, "median": 0, "q1": 0, "q3": 0,
                    "wh_low": 0, "wh_high": 0, "outliers": []}

        q1 = np.percentile(data, 25)
        median = np.percentile(data, 50)
        q3 = np.percentile(data, 75)
        mean = np.mean(data)
        std = np.std(data)
        iqr = q3 - q1

        # Define whisker limits (standard 1.5 * IQR)
        whisker_min_val = q1 - 1.5 * iqr
        whisker_max_val = q3 + 1.5 * iqr

        # Find the most extreme data points *within* the whisker range
        valid_data_low = data[data >= whisker_min_val]
        actual_whisker_low = np.min(valid_data_low) if len(valid_data_low) > 0 else q1

        valid_data_high = data[data <= whisker_max_val]
        actual_whisker_high = np.max(valid_data_high) if len(valid_data_high) > 0 else q3

        # Find outliers
        outliers = data[(data < whisker_min_val) | (data > whisker_max_val)]

        return {
            "mean": mean, "std": std, "median": median, "q1": q1, "q3": q3,
            "wh_low": actual_whisker_low, "wh_high": actual_whisker_high,
            "outliers": outliers
        }

    def draw_boxplot(self, plot, x_pos, stats, brush):
        """Draws a complete box plot onto the given plot item."""

        # Box (using BarGraphItem for the Q1-Q3 range)
        box = pg.BarGraphItem(x=[x_pos], y=[stats['q1']],
                              height=[stats['q3'] - stats['q1']],
                              width=0.6, brush=brush,
                              pen=pg.mkPen('w'))
        plot.addItem(box)

        # Median Line
        median_line = pg.PlotDataItem(
            x=[x_pos - 0.3, x_pos + 0.3],
            y=[stats['median'], stats['median']],
            pen=pg.mkPen('w', width=3)
        )
        plot.addItem(median_line)

        # Mean Symbol (+)
        mean_dot = pg.ScatterPlotItem(
            x=[x_pos], y=[stats['mean']],
            symbol='+', size=12, pen=pg.mkPen('y', width=2)
        )
        plot.addItem(mean_dot)

        # Whiskers (as two separate lines)
        top_whisker = pg.PlotDataItem(
            x=[x_pos, x_pos],
            y=[stats['q3'], stats['wh_high']],
            pen=pg.mkPen('w', style=QtCore.Qt.PenStyle.DashLine)
        )
        bottom_whisker = pg.PlotDataItem(
            x=[x_pos, x_pos],
            y=[stats['q1'], stats['wh_low']],
            pen=pg.mkPen('w', style=QtCore.Qt.PenStyle.DashLine)
        )
        plot.addItem(top_whisker)
        plot.addItem(bottom_whisker)

        # Whiskers' "caps"
        top_cap = pg.PlotDataItem(x=[x_pos - 0.2, x_pos + 0.2], y=[stats['wh_high'], stats['wh_high']],
                                  pen=pg.mkPen('w'))
        bottom_cap = pg.PlotDataItem(x=[x_pos - 0.2, x_pos + 0.2], y=[stats['wh_low'], stats['wh_low']],
                                     pen=pg.mkPen('w'))
        plot.addItem(top_cap)
        plot.addItem(bottom_cap)

        # Outliers
        if len(stats['outliers']) > 0:
            outliers_dots = pg.ScatterPlotItem(
                x=np.full(len(stats['outliers']), x_pos),
                y=stats['outliers'],
                symbol='o', size=8, pen=pg.mkPen(None),
                brush=pg.mkBrush(255, 0, 0, 100)  # Semi-transparent red
            )
            plot.addItem(outliers_dots)

'''
# --- 5. Main Application Window (MODIFIED with Block-Averaged CUSUM) ---------------------
class SimulationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARX Simulator with pyqtgraph")
        self.setGeometry(50, 100, 1000, 800)

        # This will hold a reference to the results window
        self.results_win = None

        # --- MODIFIED: Main Layout with Buttons and SpinBoxes ---
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Central container and Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, 1)  # Add tabs with stretch factor

        # Control Layout (buttons and spin boxes)
        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)

        # --- Add Spin Boxes ---
        control_layout.addWidget(QLabel("Latency Offset (ms):"))
        self.latency_spinbox = QDoubleSpinBox()
        self.latency_spinbox.setRange(-20.0, 20.0)
        self.latency_spinbox.setValue(0.0)
        self.latency_spinbox.setSingleStep(0.5)
        self.latency_spinbox.setSuffix(" ms")
        control_layout.addWidget(self.latency_spinbox)

        control_layout.addSpacing(20)  # Add some space

        control_layout.addWidget(QLabel("Amplitude Scale (%):"))
        self.amplitude_spinbox = QSpinBox()
        self.amplitude_spinbox.setRange(0, 200)  # 0% to 200%
        self.amplitude_spinbox.setValue(100)
        self.amplitude_spinbox.setSingleStep(10)
        self.amplitude_spinbox.setSuffix("%")
        control_layout.addWidget(self.amplitude_spinbox)

        control_layout.addSpacing(20)  # Add some space

        # --- Button Layout ---
        self.load_button = QPushButton("Load Saved SEPs")
        control_layout.addWidget(self.load_button)

        # --- NEW: Results Button ---
        self.results_button = QPushButton("Results SEPs")
        self.results_button.setEnabled(False)  # Disabled until data is loaded
        control_layout.addWidget(self.results_button)

        # --- NEW: Reset CUSUM Button ---
        self.reset_cusum_button = QPushButton("Reset CUSUM")
        control_layout.addWidget(self.reset_cusum_button)
        # --- END NEW ---

        control_layout.addStretch()  # This pushes buttons to the right

        self.stop_button = QPushButton("Stop Simulation")
        self.exit_button = QPushButton("Exit Application")
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.exit_button)

        # Connect button signals
        self.load_button.clicked.connect(self.load_results_from_csv)
        self.results_button.clicked.connect(self.resultsView)  # <-- NEW
        self.reset_cusum_button.clicked.connect(self.reset_cusum_charts)  # <-- NEW
        self.stop_button.clicked.connect(self.stop_simulation)
        self.exit_button.clicked.connect(self.close)
        # --- END MODIFIED LAYOUT ---

        # Initialize logic classes
        self.simulatore = EEGSimulator()
        self.processore = ProcessoreARX()

        # Initialize results containers
        self.sweep_history = []
        self.risultati_filtrati = []  # Needed for total avg AND block avgs
        self.risultati_rifiutati = []
        self.waterfall_averages_to_save = []  # List to store data for CSV

        # --- NEW: Lists to store loaded latencies/amps ---
        self.loaded_n20_latencies = []
        self.loaded_n20_amplitudes = []
        self.loaded_n30_latencies = []
        self.loaded_n30_amplitudes = []
        # --- END NEW ---

        # --- Variables for Live Raster and Waterfall ---
        self.n_samples = N_TOTAL_SWEEP_SAMPLES
        self.n_plots = N_TOTAL_SWEEPS - K_BLOCK_SIZE  # Max. number of processed sweeps

        # Pre-allocate array for RASTER
        self.raster_data = np.zeros((self.n_samples, self.n_plots))

        # Counter for single sweep (for Raster)
        self.current_sweep_index = 0

        # --- MODIFIED: Variables for Scrolling Waterfall ---
        self.waterfall_spacing = SIGNAL_AMPLITUDE * 1.5  # Space between averages
        self.waterfall_curves = []  # List to hold PlotDataItems
        self.waterfall_max_curves = 30  # Max curves to show
        # --- END MODIFIED ---

        # --- Variables for CUSUM Plot ---
        self.n20_target_lat = N20_PEAK_TIME_IDEAL
        self.n30_target_lat = N30_PEAK_TIME_IDEAL
        self.cusum_n20_lat_data = [0]
        self.cusum_n30_lat_data = [0]

        # --- NEW: CUSUM Alarm variables ---
        self.cusum_threshold = 150.0  # Threshold at +/- 150%
        self.cusum_n20_in_alarm = False
        self.cusum_n30_in_alarm = False

        # Define pens
        self.normal_pen_n20 = pg.mkPen('r')
        self.normal_pen_n30 = pg.mkPen('b')
        self.alarm_pen = pg.mkPen('y', width=3)
        # --- END NEW ---

        # --- Variables for Deviation Plot ---
        self.deviation_n20_data = []  # This will now hold BLOCK deviations
        self.deviation_n30_data = []  # This will now hold BLOCK deviations
        # --- END NEW ---

        # Setup tabs
        self.setup_live_tab()
        self.setup_results_tabs()

        print(f"Starting processing... Stimuli at {STIM_RATE} Hz, Blocks of {K_BLOCK_SIZE} sweeps.")
        print("-" * 30)

        # Timer for the simulation loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_simulation_step)
        self.timer.start(10)

    def setup_live_tab(self):
        """Creates the tab for real-time plots."""
        tab_live = QWidget()
        layout = QVBoxLayout()
        tab_live.setLayout(layout)

        # We use GraphicsLayoutWidget for subplots
        self.win_live = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win_live)
        self.tabs.addTab(tab_live, "Live Simulation")

        # Plot 1: Data
        self.plot_rt_1 = self.win_live.addPlot(row=0, col=0)
        self.plot_rt_1.setLabel('left', "Amplitude (uV)")
        self.plot_rt_1.addLegend()
        self.plot_rt_1.invertY(True)
        self.curve_y = self.plot_rt_1.plot(pen=pg.mkPen('w', width=1), name="Output (y)")
        self.curve_x = self.plot_rt_1.plot(pen=pg.mkPen('b', style=QtCore.Qt.PenStyle.DashLine),
                                           name=f"Input (Avg K={K_BLOCK_SIZE})")
        self.curve_vero = self.plot_rt_1.plot(pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DotLine), name="True SEP")

        # Plot 2: Result
        self.plot_rt_2 = self.win_live.addPlot(row=1, col=0)
        self.plot_rt_2.setLabel('left', "Amplitude (uV)")
        self.plot_rt_2.setLabel('bottom', "Time (s)")
        self.plot_rt_2.addLegend()
        self.plot_rt_2.invertY(True)
        self.curve_ep = self.plot_rt_2.plot(pen=pg.mkPen('g', width=2), name="Identified signal (ep)")
        self.curve_vero_2 = self.plot_rt_2.plot(pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DotLine), name="True SEP")

        # Link X axes
        self.plot_rt_2.setXLink(self.plot_rt_1)

    def setup_results_tabs(self):
        """Creates the empty tabs for the final plots."""
        # Filtered Average Tab
        self.plot_media_filtrata = pg.PlotWidget()
        self.plot_media_filtrata.addLegend()
        self.plot_media_filtrata.invertY(True)
        self.tabs.addTab(self.plot_media_filtrata, "Filtered Average")

        # Rejected Average Tab
        self.plot_media_rifiutata = pg.PlotWidget()
        self.plot_media_rifiutata.addLegend()
        self.plot_media_rifiutata.invertY(True)
        self.tabs.addTab(self.plot_media_rifiutata, "Rejected Average")

        # --- Raster Plot Tab (Unchanged) ---
        self.plot_item_raster = pg.PlotItem()
        self.image_view_raster = pg.ImageView(view=self.plot_item_raster)
        self.plot_item_raster.vb.setAspectLocked(False)
        self.plot_item_raster.setLabel('bottom', "Time (s)")
        self.plot_item_raster.setLabel('left', "Sweep Number (identified)")
        v_line_n20_raster = pg.InfiniteLine(pos=N20_PEAK_TIME_IDEAL, angle=90, movable=False,
                                            pen=pg.mkPen('gray', style=QtCore.Qt.PenStyle.DashLine))
        v_line_p30_raster = pg.InfiniteLine(pos=P30_PEAK_TIME_IDEAL, angle=90, movable=False,
                                            pen=pg.mkPen('gray', style=QtCore.Qt.PenStyle.DotLine))
        v_line_n30_raster = pg.InfiniteLine(pos=N30_PEAK_TIME_IDEAL, angle=90, movable=False,
                                            pen=pg.mkPen('gray', style=QtCore.Qt.PenStyle.DashLine))
        self.plot_item_raster.addItem(v_line_n20_raster)
        self.plot_item_raster.addItem(v_line_p30_raster)
        self.plot_item_raster.addItem(v_line_n30_raster)
        cmap = pg.colormap.get('RdBu_r', 'matplotlib')
        self.image_view_raster.setColorMap(cmap)
        x_origin = time_axis[0]
        dx = time_axis[1] - time_axis[0]
        self.image_view_raster.setImage(
            self.raster_data, autoLevels=False, pos=[x_origin, 0], scale=[dx, 1]
        )
        self.image_view_raster.setLevels(-SIGNAL_AMPLITUDE, SIGNAL_AMPLITUDE)
        self.tabs.addTab(self.image_view_raster, "Raster Plot")
        # --- END RASTER ---

        # --- Waterfall Plot Tab (Unchanged) ---
        self.plot_waterfall = pg.PlotWidget()
        self.plot_item_waterfall = self.plot_waterfall.getPlotItem()
        self.plot_item_waterfall.invertY(True)
        self.plot_item_waterfall.setLabel('left', "Amplitude (uV) + Scrolling Avg Block")
        self.plot_item_waterfall.setLabel('bottom', "Time (s)")
        self.add_waterfall_reference_lines()
        self.waterfall_cmap = pg.colormap.get('viridis')
        self.tabs.addTab(self.plot_waterfall, "Waterfall Plot")
        # --- END WATERFALL ---

        # --- MODIFIED: CUSUM Plot Tab (with Thresholds) ---
        self.cusum_widget = pg.GraphicsLayoutWidget()

        # CUSUM N20 Plot
        plot_cusum_n20 = self.cusum_widget.addPlot(row=0, col=0)
        plot_cusum_n20.setTitle("N20 Latency CUSUM (from Block Averages)")
        plot_cusum_n20.setLabel('left', "Cumulative Sum (%)")
        plot_cusum_n20.setLabel('bottom', "Block Number")  # <-- MODIFIED LABEL
        plot_cusum_n20.showGrid(x=True, y=True, alpha=0.3)
        self.plot_cusum_n20_line = plot_cusum_n20.plot(pen=self.normal_pen_n20)
        # Add threshold lines
        self.cusum_h_n20_pos = pg.InfiniteLine(pos=self.cusum_threshold, angle=0,
                                               pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine))
        self.cusum_h_n20_neg = pg.InfiniteLine(pos=-self.cusum_threshold, angle=0,
                                               pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine))
        plot_cusum_n20.addItem(self.cusum_h_n20_pos)
        plot_cusum_n20.addItem(self.cusum_h_n20_neg)

        # CUSUM N30 Plot
        plot_cusum_n30 = self.cusum_widget.addPlot(row=1, col=0)
        plot_cusum_n30.setTitle("N30/N40 Latency CUSUM (from Block Averages)")
        plot_cusum_n30.setLabel('left', "Cumulative Sum (%)")
        plot_cusum_n30.setLabel('bottom', "Block Number")  # <-- MODIFIED LABEL
        plot_cusum_n30.showGrid(x=True, y=True, alpha=0.3)
        self.plot_cusum_n30_line = plot_cusum_n30.plot(pen=self.normal_pen_n30)
        # Add threshold lines
        self.cusum_h_n30_pos = pg.InfiniteLine(pos=self.cusum_threshold, angle=0,
                                               pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine))
        self.cusum_h_n30_neg = pg.InfiniteLine(pos=-self.cusum_threshold, angle=0,
                                               pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine))
        plot_cusum_n30.addItem(self.cusum_h_n30_pos)
        plot_cusum_n30.addItem(self.cusum_h_n30_neg)

        self.tabs.addTab(self.cusum_widget, "CUSUM Chart")
        # --- END MODIFIED ---

        # --- MODIFIED: Deviation Plot Tab ---
        self.deviation_widget = pg.GraphicsLayoutWidget()

        # Deviation N20 Plot
        plot_dev_n20 = self.deviation_widget.addPlot(row=0, col=0)
        plot_dev_n20.setTitle("N20 Latency Instantaneous Deviation (of Block Averages)")
        plot_dev_n20.setLabel('left', "Deviation (%)")
        plot_dev_n20.setLabel('bottom', "Block Number")  # <-- MODIFIED LABEL
        plot_dev_n20.showGrid(x=True, y=True, alpha=0.3)
        plot_dev_n20.setYRange(-50, 50)  # <-- Locked Y-Axis
        plot_dev_n20.addLegend()
        # This one line will have both dots and a line
        self.plot_dev_n20_line = plot_dev_n20.plot(
            pen=pg.mkPen('r', width=2),
            symbol='o', symbolSize=5,
            symbolBrush='r', name="Block % Deviation"
        )

        # Deviation N30 Plot
        plot_dev_n30 = self.deviation_widget.addPlot(row=1, col=0)
        plot_dev_n30.setTitle("N30/N40 Latency Instantaneous Deviation (of Block Averages)")
        plot_dev_n30.setLabel('left', "Deviation (%)")
        plot_dev_n30.setLabel('bottom', "Block Number")  # <-- MODIFIED LABEL
        plot_dev_n30.showGrid(x=True, y=True, alpha=0.3)
        plot_dev_n30.setYRange(-50, 50)  # <-- Locked Y-Axis
        plot_dev_n30.addLegend()
        # This one line will have both dots and a line
        self.plot_dev_n30_line = plot_dev_n30.plot(
            pen=pg.mkPen('b', width=2),
            symbol='o', symbolSize=5,
            symbolBrush='b', name="Block % Deviation"
        )

        self.tabs.addTab(self.deviation_widget, "Deviation Chart")
        # --- END MODIFIED ---

    # --- NEW HELPER METHOD ---
    def add_waterfall_reference_lines(self):
        """Adds the vertical reference lines to the waterfall plot."""
        v_line_zero_water = pg.InfiniteLine(pos=0.0, angle=90, movable=False,
                                            pen=pg.mkPen('white', style=QtCore.Qt.PenStyle.DashLine))
        v_line_n20_water = pg.InfiniteLine(pos=N20_PEAK_TIME_IDEAL, angle=90, movable=False,
                                           pen=pg.mkPen('gray', style=QtCore.Qt.PenStyle.DashLine))
        v_line_p30_water = pg.InfiniteLine(pos=P30_PEAK_TIME_IDEAL, angle=90, movable=False,
                                           pen=pg.mkPen('gray', style=QtCore.Qt.PenStyle.DotLine))
        v_line_n30_water = pg.InfiniteLine(pos=N30_PEAK_TIME_IDEAL, angle=90, movable=False,
                                           pen=pg.mkPen('gray', style=QtCore.Qt.PenStyle.DashLine))
        self.plot_item_waterfall.addItem(v_line_zero_water)
        self.plot_item_waterfall.addItem(v_line_n20_water)
        self.plot_item_waterfall.addItem(v_line_p30_water)
        self.plot_item_waterfall.addItem(v_line_n30_water)

    # --- END NEW ---

    def apply_simple_smooth(self, data, window_len=15):
        """
        Applies a simple moving average, ensuring
        output length is EQUAL to input length (mode='same').
        """
        # Ensure we have enough data to smooth
        if len(data) < window_len:
            return data  # Return original data if not enough points

        if window_len % 2 == 0:
            window_len += 1
        w = np.ones(window_len, 'd')
        smoothed_data = np.convolve(data, w / w.sum(), mode='same')
        return smoothed_data

    # --- NEW: Reusable Peak Finding Logic ---
    def find_peak_data(self, data_array, time_array):
        """
        Finds the N20 and N30 peak latencies and amplitudes from a data array.
        Returns a dictionary.
        """
        # N20 search window (15ms to 25ms)
        n20_start_idx = np.argmin(np.abs(time_array - 0.015))
        n20_end_idx = np.argmin(np.abs(time_array - 0.025))

        # N30 (N40) search window (30ms to 50ms)
        n30_start_idx = np.argmin(np.abs(time_array - 0.030))
        n30_end_idx = np.argmin(np.abs(time_array - 0.050))

        # N20 Peak
        search_slice_n20 = data_array[n20_start_idx:n20_end_idx]
        if len(search_slice_n20) > 0:
            idx_in_slice_n20 = np.argmin(search_slice_n20)
            idx_full_n20 = n20_start_idx + idx_in_slice_n20
            lat_n20 = time_array[idx_full_n20]
            amp_n20 = data_array[idx_full_n20]
        else:
            lat_n20, amp_n20 = N20_PEAK_TIME_IDEAL, 0

        # N30 Peak
        search_slice_n30 = data_array[n30_start_idx:n30_end_idx]
        if len(search_slice_n30) > 0:
            idx_in_slice_n30 = np.argmin(search_slice_n30)
            idx_full_n30 = n30_start_idx + idx_in_slice_n30
            lat_n30 = time_array[idx_full_n30]
            amp_n30 = data_array[idx_full_n30]
        else:
            lat_n30, amp_n30 = N30_PEAK_TIME_IDEAL, 0

        return {
            "n20_lat": lat_n20, "n20_amp": amp_n20,
            "n30_lat": lat_n30, "n30_amp": amp_n30
        }

    # --- END NEW ---

    def closeEvent(self, event):
        """
        Overrides the main window's close event.
        Ensures that the results window (if it exists) is closed too.
        """
        print("Main window is closing, also closing results window...")

        # Check if the results window exists
        if self.results_win is not None:
            self.results_win.close()  # Explicitly close the results window

        # Accept the close event to allow the main window to close
        event.accept()

    # --- NEW METHOD ---
    def stop_simulation(self):
        """Stops the simulation timer."""
        if self.timer.isActive():
            self.timer.stop()
            print("\n--- Simulation Manually Stopped ---")
            self.generate_final_plots()  # This will also trigger the save
            self.stop_button.setText("Simulation Stopped")
            self.stop_button.setEnabled(False)

    # --- NEW METHOD ---
    def save_results_automatically(self):
        """Automatically saves the collected waterfall averages to a CSV file."""
        if not self.waterfall_averages_to_save:
            print("No waterfall data to save.")
            return

        fileName = "waterfall_averages.csv"  # Hardcoded filename
        try:
            # Transpose the list of 1D arrays into columns
            # Shape becomes (n_samples, n_blocks)
            data_to_save = np.array(self.waterfall_averages_to_save).T

            # Reshape time_axis to be a column vector
            # Shape (n_samples, 1)
            time_col = time_axis.reshape(-1, 1)

            # Stack time column and data columns horizontally
            full_data = np.hstack((time_col, data_to_save))

            # Create headers
            headers = ["Time (s)"] + [f"Avg_Block_{i + 1}" for i in range(len(self.waterfall_averages_to_save))]

            # Create DataFrame and save
            df = pd.DataFrame(full_data, columns=headers)
            df.to_csv(fileName, index=False, float_format='%.6f')

            print(f"Successfully saved waterfall data to {fileName}")

        except Exception as e:
            print(f"Error automatically saving file: {e}")

    # --- MODIFIED METHOD: load_results_from_csv ---
    def load_results_from_csv(self):
        """Loads waterfall averages from a CSV file and plots them."""
        # fileName, _ = QFileDialog.getOpenFileName(self,
        #                                           "Load Waterfall Averages",
        #                                           "",  # Start in default directory
        #                                           "CSV Files (*.csv)")
        # overwritten file
        fileName = "waterfall_averages.csv"
        if fileName:
            try:
                # Load the data
                df = pd.read_csv(fileName)

                # Extract time axis and data samples
                loaded_time_axis = df.iloc[:, 0].values
                loaded_averages = df.iloc[:, 1:].values  # (samples, blocks)
                num_blocks = loaded_averages.shape[1]
                print(f"Loading {num_blocks} blocks from {fileName}...")

                # --- Clear runtime latency data ---
                self.loaded_n20_latencies.clear()
                self.loaded_n20_amplitudes.clear()
                self.loaded_n30_latencies.clear()
                self.loaded_n30_amplitudes.clear()

                # --- Plot Grand Average ---
                grand_average = np.mean(loaded_averages, axis=1)
                # plotWidget
                self.plot_media_filtrata.clear()
                self.plot_media_filtrata.setTitle(
                    f"Loaded Grand Average (N={num_blocks} blocks of {K_BLOCK_SIZE} sweeps each)")
                self.plot_media_filtrata.plot(loaded_time_axis, grand_average, pen='b', name="Loaded Average")

                # Add reference template *if* dimensions match
                if len(loaded_time_axis) == len(vero_sep_template):
                    self.plot_media_filtrata.plot(time_axis, vero_sep_template,
                                                  pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DotLine),
                                                  name="True SEP")
                self.plot_media_filtrata.autoRange()

                # --- Plot Waterfall Widget ---
                self.plot_item_waterfall.clear()
                self.waterfall_curves.clear()
                self.add_waterfall_reference_lines()  # Re-add lines after clearing

                for i in range(num_blocks):
                    block_data = loaded_averages[:, i]

                    # calculate prestim signal offset and subtract to block_data
                    preoffset = np.mean(block_data[0: N_PRE_STIM_SAMPLES])
                    block_data = block_data - preoffset

                    # Use a fixed offset
                    y_offset = (i * self.waterfall_spacing) / 10

                    # Color based on position
                    normalized_index = i / max(1, num_blocks - 1)
                    color = self.waterfall_cmap.mapToQColor(normalized_index)

                    # --- Add the LINE ---
                    curve = pg.PlotDataItem(
                        loaded_time_axis,
                        block_data + y_offset,
                        pen=pg.mkPen(color)
                    )
                    self.plot_item_waterfall.addItem(curve)
                    self.waterfall_curves.append(curve)

                    # --- MODIFIED: Use new helper function ---
                    peak_data = self.find_peak_data(block_data, loaded_time_axis)

                    # Store latencies and amplitudes
                    self.loaded_n20_latencies.append(peak_data["n20_lat"])
                    self.loaded_n20_amplitudes.append(peak_data["n20_amp"])
                    self.loaded_n30_latencies.append(peak_data["n30_lat"])
                    self.loaded_n30_amplitudes.append(peak_data["n30_amp"])

                    # --- Add N20 dot ---
                    dot_n20 = pg.ScatterPlotItem(
                        x=[peak_data["n20_lat"]], y=[peak_data["n20_amp"] + y_offset],
                        symbol='o', size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 100, 100)
                    )
                    self.plot_item_waterfall.addItem(dot_n20)

                    # --- Add N30 dot ---
                    dot_n30 = pg.ScatterPlotItem(
                        x=[peak_data["n30_lat"]], y=[peak_data["n30_amp"] + y_offset],
                        symbol='o', size=8, pen=pg.mkPen(None), brush=pg.mkBrush(100, 100, 255)
                    )
                    self.plot_item_waterfall.addItem(dot_n30)
                    # --- END MODIFIED ---

                self.plot_item_waterfall.autoRange()

                # --- Enable the results button ---
                self.results_button.setEnabled(True)

                # Switch to the waterfall tab to show the result
                self.tabs.setCurrentWidget(self.plot_waterfall.parent())
                print("...Load and plot complete.")

            except Exception as e:
                print(f"Error loading file: {e}")
                QMessageBox.critical(self, "Load Error", f"Failed to load or parse file:\n{e}")

    # --- END MODIFIED METHOD ---

    # --- NEW METHOD ---
    def reset_cusum_charts(self):
        """Resets all CUSUM and Deviation charts and alarms."""
        print("--- CUSUM and Deviation Charts Reset ---")

        # Reset N20 CUSUM
        self.cusum_n20_lat_data = [0]
        self.plot_cusum_n20_line.setData(self.cusum_n20_lat_data)
        self.cusum_n20_in_alarm = False
        self.plot_cusum_n20_line.setPen(self.normal_pen_n20)

        # Reset N30 CUSUM
        self.cusum_n30_lat_data = [0]
        self.plot_cusum_n30_line.setData(self.cusum_n30_lat_data)
        self.cusum_n30_in_alarm = False
        self.plot_cusum_n30_line.setPen(self.normal_pen_n30)

        # Reset Deviation charts
        self.deviation_n20_data = []
        self.deviation_n30_data = []
        self.plot_dev_n20_line.setData(self.deviation_n20_data)  # Clear line
        self.plot_dev_n30_line.setData(self.deviation_n30_data)  # Clear line

    # --- END NEW ---

    def run_simulation_step(self):
        """Runs a single simulation step (replaces the while loop)."""

        # --- NEW: Read values from SpinBoxes ---
        lat_mod = self.latency_spinbox.value()
        amp_mod = self.amplitude_spinbox.value()
        # --- END NEW ---

        # --- MODIFIED: Pass values to simulator ---
        nuova_sweep = self.simulatore.get_next_sweep(lat_mod, amp_mod)

        # 1. Check if the simulation is finished
        if nuova_sweep is None:
            self.stop_simulation()  # Use the stop function
            return

        # print(f"Received Sweep #{self.simulatore.sweep_count}...")

        # 2. Buffering management
        if len(self.sweep_history) < K_BLOCK_SIZE:
            self.sweep_history.append(nuova_sweep)
            # print(f"Buffering... {len(self.sweep_history)}/{K_BLOCK_SIZE}")
            return

        # 3. Data preparation and processing
        y_output = nuova_sweep
        X_input = np.mean(self.sweep_history[-K_BLOCK_SIZE:], axis=0)
        segnale, orders, aic, p_val = self.processore.elabora_sweep(y_output, X_input)
        self.sweep_history.append(nuova_sweep)

        # 4. Update plots and save results
        if segnale is not None and len(segnale) == len(time_axis):
            na, nb = orders
            # print(f"  -> Model Found: na={na}, nb={nb} (AIC={aic:.2f}, Ljung-Box p={p_val:.3f})")
            self.risultati_filtrati.append(segnale)

            # --- LIVE PLOT UPDATE (Unchanged) ---
            self.plot_rt_1.setTitle(f"Data for ARX Model (Sweep #{self.simulatore.sweep_count})")
            self.curve_y.setData(time_axis, y_output)
            # self.curve_x.setData(time_axis, X_input)
            self.curve_vero.setData(time_axis, vero_sep_template)
            self.plot_rt_2.setTitle(f"ARX Model Result (na={na}, nb={nb})")
            self.curve_ep.setData(time_axis, segnale)
            self.curve_vero_2.setData(time_axis, vero_sep_template)

            # --- RASTER Plot Update (Unchanged) ---
            if self.current_sweep_index < self.n_plots:
                self.raster_data[:, self.current_sweep_index] = segnale
                self.image_view_raster.setImage(self.raster_data, autoLevels=False)
                if self.current_sweep_index % 20 == 0:
                    max_abs_val = np.max(np.abs(self.raster_data[:, :self.current_sweep_index + 1]))
                    if max_abs_val == 0: max_abs_val = 1
                    self.image_view_raster.setLevels(-max_abs_val, max_abs_val)

                # --- LOGIC MOVED: This no longer runs every sweep ---
                # --- CUSUM & DEVIATION Plot Update (MOVED) ---

                # --- SCROLLING WATERFALL PLOT LOGIC ---

                # Check if the number of IDENTIFIED sweeps is a multiple of 25
                if len(self.risultati_filtrati) > 0 and len(self.risultati_filtrati) % 25 == 0:

                    # 1. Calculate and smooth the average block
                    media_blocco = np.mean(self.risultati_filtrati[-25:], axis=0)
                    media_blocco = self.apply_simple_smooth(media_blocco)  # , window_len=15)

                    # --- NEW: Save the block for CSV export ---
                    self.waterfall_averages_to_save.append(media_blocco)
                    # --- END NEW ---

                    # --- NEW: CUSUM & DEVIATION LOGIC MOVED HERE ---
                    # Now we run this logic only ONCE per BLOCK

                    # 1. Find peak data on the *smoothed block average*
                    peak_data = self.find_peak_data(media_blocco, time_axis)

                    # 2. Calculate N20 deviation (% and cumulative)
                    dev_n20_perc = ((peak_data["n20_lat"] - self.n20_target_lat) / self.n20_target_lat) * 100
                    self.cusum_n20_lat_data.append(self.cusum_n20_lat_data[-1] + dev_n20_perc)
                    self.deviation_n20_data.append(dev_n20_perc)  # <-- Add to raw list

                    # 3. Calculate N30 deviation (% and cumulative)
                    dev_n30_perc = ((peak_data["n30_lat"] - self.n30_target_lat) / self.n30_target_lat) * 100
                    self.cusum_n30_lat_data.append(self.cusum_n30_lat_data[-1] + dev_n30_perc)
                    self.deviation_n30_data.append(dev_n30_perc)  # <-- Add to raw list

                    # 4. Update CUSUM plot lines
                    self.plot_cusum_n20_line.setData(self.cusum_n20_lat_data)
                    self.plot_cusum_n30_line.setData(self.cusum_n30_lat_data)

                    # 5. Check CUSUM Alarms
                    current_cusum_n20 = self.cusum_n20_lat_data[-1]
                    if abs(current_cusum_n20) > self.cusum_threshold:
                        if not self.cusum_n20_in_alarm:
                            print(f"--- CUSUM N20 ALARM: Threshold breached! ({current_cusum_n20:.1f}%) ---")
                            self.plot_cusum_n20_line.setPen(self.alarm_pen)
                            self.cusum_n20_in_alarm = True

                    current_cusum_n30 = self.cusum_n30_lat_data[-1]
                    if abs(current_cusum_n30) > self.cusum_threshold:
                        if not self.cusum_n30_in_alarm:
                            print(f"--- CUSUM N30 ALARM: Threshold breached! ({current_cusum_n30:.1f}%) ---")
                            self.plot_cusum_n30_line.setPen(self.alarm_pen)
                            self.cusum_n30_in_alarm = True

                    # 6. Update Deviation Plot
                    self.plot_dev_n20_line.setData(self.deviation_n20_data)
                    self.plot_dev_n30_line.setData(self.deviation_n30_data)
                    # --- END CUSUM/DEVIATION UPDATE ---

                    # --- WATERFALL LOGIC (Unchanged) ---
                    color = self.waterfall_cmap.mapToQColor(1.0)
                    curve = pg.PlotDataItem(
                        time_axis,
                        media_blocco,
                        pen=pg.mkPen(color)
                    )
                    self.plot_item_waterfall.addItem(curve)
                    self.waterfall_curves.append(curve)
                    if len(self.waterfall_curves) > self.waterfall_max_curves:
                        old_curve = self.waterfall_curves.pop(0)
                        self.plot_item_waterfall.removeItem(old_curve)
                    for i, curve_item in enumerate(self.waterfall_curves):
                        y_offset = (i * self.waterfall_spacing) / 10
                        curve_item.setPos(0, y_offset)
                        normalized_index = i / self.waterfall_max_curves
                        color = self.waterfall_cmap.mapToQColor(normalized_index)
                        curve_item.setPen(pg.mkPen(color))
                # --- END WATERFALL LOGIC ---

                # Increment the *total* sweep counter (for the raster)
                self.current_sweep_index += 1

        elif segnale is None and self.simulatore.sweep_count > K_BLOCK_SIZE:
            self.risultati_rifiutati.append(y_output)
            print(f"  -> NO VALID MODEL FOUND = {len(self.risultati_rifiutati)}")
            # As requested, we don't plot anything

    def generate_final_plots(self):
        """Populates the results tabs at the end of the simulation."""

        # This function is called when simulation ends (manually or naturally)

        print("\nGenerating final plots (Average and StDev)...")
        dati_filtrati_np = None

        # --- 6. Filtered Average and StDev Plot (Unchanged) ---
        if len(self.risultati_filtrati) > 1:
            dati_filtrati_np = np.array(self.risultati_filtrati)
            media_filtrata = np.mean(dati_filtrati_np, axis=0)
            std_filtrata = np.std(dati_filtrati_np, axis=0)
            std_superiore = media_filtrata + std_filtrata
            std_inferiore = media_filtrata - std_filtrata

            self.plot_media_filtrata.setTitle(f"Average of IDENTIFIED Signals (N={len(self.risultati_filtrati)})")
            self.plot_media_filtrata.setLabel('left', 'Amplitude (uV)')
            self.plot_media_filtrata.setLabel('bottom', 'Time (s)')
            self.plot_media_filtrata.plot(time_axis, media_filtrata, pen='b', name="Filtered Average")
            self.plot_media_filtrata.plot(time_axis, vero_sep_template,
                                          pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DotLine), name="True SEP")
            curve_std_sup = self.plot_media_filtrata.plot(time_axis, std_superiore, pen=None)
            curve_std_inf = self.plot_media_filtrata.plot(time_axis, std_inferiore, pen=None)
            fill = pg.FillBetweenItem(curve_std_inf, curve_std_sup, brush=pg.mkBrush(0, 0, 255, 50))
            self.plot_media_filtrata.addItem(fill)

        elif len(self.risultati_filtrati) == 1:
            dati_filtrati_np = np.array(self.risultati_filtrati)
            print(f"  Found only 1 filtered result.")
        else:
            print(f"  No filtered results saved.")

        # --- 7. Rejected Average and StDev Plot (Unchanged) ---
        if len(self.risultati_rifiutati) > 1:
            dati_rifiutati_np = np.array(self.risultati_rifiutati)
            media_rifiutata = np.mean(dati_rifiutati_np, axis=0)
            std_rifiutata = np.std(dati_rifiutati_np, axis=0)
            std_sup_rif = media_rifiutata + std_rifiutata
            std_inf_rif = media_rifiutata - std_rifiutata

            self.plot_media_rifiutata.setTitle(f"Average of REJECTED Signals (N={len(self.risultati_rifiutati)})")
            self.plot_media_rifiutata.setLabel('left', 'Amplitude (uV)')
            self.plot_media_rifiutata.setLabel('bottom', 'Time (s)')
            self.plot_media_rifiutata.plot(time_axis, media_rifiutata, pen='orange', name="Rejected Average")
            self.plot_media_rifiutata.plot(time_axis, vero_sep_template,
                                           pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DotLine), name="True SEP")
            curve_std_sup_rif = self.plot_media_rifiutata.plot(time_axis, std_sup_rif, pen=None)
            curve_std_inf_rif = self.plot_media_rifiutata.plot(time_axis, std_inf_rif, pen=None)
            fill_rif = pg.FillBetweenItem(curve_std_inf_rif, curve_std_sup_rif, brush=pg.mkBrush(255, 165, 0, 50))
            self.plot_media_rifiutata.addItem(fill_rif)
        else:
            print(f"  {len(self.risultati_rifiutati)} sweeps were rejected. Not enough for an average plot.")

        # --- 8, 9, 10. Raster, Waterfall, CUSUM, Deviation Plots ---
        print("Live-updating charts already generated.")

        if self.plot_item_waterfall.items:  # Check if there are items before auto-ranging
            self.plot_item_waterfall.autoRange()

        # --- NEW: Trigger automatic save ---
        print("Attempting to save waterfall data automatically...")
        self.save_results_automatically()
        # --- END NEW ---

        # Switch to the first results tab
        self.tabs.setCurrentIndex(1)

    # --- NEW METHOD ---
    def resultsView(self):
        """Shows a new window with latency histograms from the loaded data."""
        if not self.loaded_n20_latencies:
            # If no data has been loaded, show a warning
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("No Data")
            msg_box.setText("No results data has been loaded.\nPlease use 'Load Saved SEPs' first.")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            return

        # Create and show the new results window
        self.results_win = ResultsWindow(self.loaded_n20_latencies,
                                         self.loaded_n20_amplitudes,
                                         self.loaded_n30_latencies,
                                         self.loaded_n30_amplitudes
                                         )
        self.results_win.show()


# --- 6. Application Start ---
if __name__ == '__main__':
    # Create the application
    app = QApplication(sys.argv)

    # Create and show the main window
    main_win = SimulationWindow()
    main_win.show()

    # Run the application loop
    sys.exit(app.exec())
'''


# --- 5. Main Application Window (MODIFIED for Infinite Run & Scrolling Raster) ---
class SimulationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARX Simulator with pyqtgraph")
        self.setGeometry(50, 100, 1000, 800)

        # This will hold a reference to the results window
        self.results_win = None

        # --- MODIFIED: Main Layout with Buttons and SpinBoxes ---
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Central container and Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, 1)  # Add tabs with stretch factor

        # Control Layout (buttons and spin boxes)
        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)

        # --- Add Spin Boxes ---
        control_layout.addWidget(QLabel("Latency Offset (ms):"))
        self.latency_spinbox = QDoubleSpinBox()
        self.latency_spinbox.setRange(-20.0, 20.0)
        self.latency_spinbox.setValue(0.0)
        self.latency_spinbox.setSingleStep(0.5)
        self.latency_spinbox.setSuffix(" ms")
        control_layout.addWidget(self.latency_spinbox)

        control_layout.addSpacing(20)  # Add some space

        control_layout.addWidget(QLabel("Amplitude Scale (%):"))
        self.amplitude_spinbox = QSpinBox()
        self.amplitude_spinbox.setRange(0, 200)  # 0% to 200%
        self.amplitude_spinbox.setValue(100)
        self.amplitude_spinbox.setSingleStep(10)
        self.amplitude_spinbox.setSuffix("%")
        control_layout.addWidget(self.amplitude_spinbox)

        control_layout.addSpacing(20)  # Add some space

        # --- Button Layout ---
        self.load_button = QPushButton("Load Saved SEPs")
        control_layout.addWidget(self.load_button)

        self.results_button = QPushButton("Results SEPs")
        self.results_button.setEnabled(False)  # Disabled until data is loaded
        control_layout.addWidget(self.results_button)

        self.reset_cusum_button = QPushButton("Reset CUSUM")
        control_layout.addWidget(self.reset_cusum_button)

        control_layout.addStretch()  # This pushes buttons to the right

        self.stop_button = QPushButton("Stop Simulation")
        self.exit_button = QPushButton("Exit Application")
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.exit_button)

        # Connect button signals
        self.load_button.clicked.connect(self.load_results_from_csv)
        self.results_button.clicked.connect(self.resultsView)
        self.reset_cusum_button.clicked.connect(self.reset_cusum_charts)
        self.stop_button.clicked.connect(self.stop_simulation)
        self.exit_button.clicked.connect(self.close)
        # --- END MODIFIED LAYOUT ---

        # Initialize logic classes
        self.simulatore = EEGSimulator()
        self.processore = ProcessoreARX()

        # Initialize results containers
        self.sweep_history = []
        self.risultati_filtrati = []  # Needed for total avg AND block avgs
        self.risultati_rifiutati = []
        self.waterfall_averages_to_save = []  # List to store data for CSV

        # --- Lists to store loaded latencies/amps ---
        self.loaded_n20_latencies = []
        self.loaded_n20_amplitudes = []
        self.loaded_n30_latencies = []
        self.loaded_n30_amplitudes = []

        # --- Variables for Live Raster and Waterfall ---
        self.n_samples = N_TOTAL_SWEEP_SAMPLES

        # --- MODIFIED: Raster now scrolls ---
        self.raster_max_sweeps = 500  # Max sweeps to show in raster
        self.raster_data = np.zeros((self.n_samples, self.raster_max_sweeps))
        # --- END MODIFIED ---

        # Counter for single sweep (for Raster)
        self.current_sweep_index = 0

        # --- MODIFIED: Variables for Scrolling Waterfall ---
        self.waterfall_spacing = SIGNAL_AMPLITUDE * 1.5  # Space between averages
        self.waterfall_curves = []  # List to hold PlotDataItems
        self.waterfall_max_curves = 30  # Max curves to show
        # --- END MODIFIED ---

        # --- Variables for CUSUM Plot ---
        self.n20_target_lat = N20_PEAK_TIME_IDEAL
        self.n30_target_lat = N30_PEAK_TIME_IDEAL
        self.cusum_n20_lat_data = [0]
        self.cusum_n30_lat_data = [0]

        # --- CUSUM Alarm variables ---
        self.cusum_threshold = 150.0  # Threshold at +/- 150%
        self.cusum_n20_in_alarm = False
        self.cusum_n30_in_alarm = False

        # Define pens
        self.normal_pen_n20 = pg.mkPen('r')
        self.normal_pen_n30 = pg.mkPen('b')
        self.alarm_pen = pg.mkPen('y', width=3)

        # --- Variables for Deviation Plot ---
        self.deviation_n20_data = []  # This will now hold BLOCK deviations
        self.deviation_n30_data = []

        # Progettiamo un filtro Butterworth passa-basso
        # N = Ordine del filtro (più alto = più ripido)
        # Wn = Frequenza di cutoff (dove "tagliare" il rumore)
        # fs = Frequenza di campionamento (dal tuo file)
        filter_order = 6
        cutoff_freq = 300  # Hz (tagliamo il rumore sopra i 100 Hz)

        # 'self.b' e 'self.a' sono i "coefficienti" del filtro
        self.b, self.a = butter(filter_order, cutoff_freq, btype='low', fs=FS)
        print(f"Filtro Zero-Phase (Butterworth) inizializzato: Ordine={filter_order}, Cutoff={cutoff_freq} Hz")

        # Setup tabs
        self.setup_live_tab()
        self.setup_results_tabs()

        print(f"Starting processing... Stimuli at {STIM_RATE} Hz, Blocks of {K_BLOCK_SIZE} sweeps.")
        print("-" * 30)

        # Timer for the simulation loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_simulation_step)
        self.timer.start(10)

    def setup_live_tab(self):
        """Creates the tab for real-time plots."""
        tab_live = QWidget()
        layout = QVBoxLayout()
        tab_live.setLayout(layout)

        # We use GraphicsLayoutWidget for subplots
        self.win_live = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win_live)
        self.tabs.addTab(tab_live, "Live Simulation")

        # Plot 1: Data
        self.plot_rt_1 = self.win_live.addPlot(row=0, col=0)
        self.plot_rt_1.setLabel('left', "Amplitude (uV)")
        self.plot_rt_1.addLegend()
        self.plot_rt_1.invertY(True)
        self.curve_y = self.plot_rt_1.plot(pen=pg.mkPen('w', width=1), name="Output (y)")
        self.curve_x = self.plot_rt_1.plot(pen=pg.mkPen('b', style=QtCore.Qt.PenStyle.DashLine),
                                           name=f"Input (Avg K={K_BLOCK_SIZE})")
        self.curve_vero = self.plot_rt_1.plot(pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DotLine), name="True SEP")

        # Plot 2: Result
        self.plot_rt_2 = self.win_live.addPlot(row=1, col=0)
        self.plot_rt_2.setLabel('left', "Amplitude (uV)")
        self.plot_rt_2.setLabel('bottom', "Time (s)")
        self.plot_rt_2.addLegend()
        self.plot_rt_2.invertY(True)
        self.curve_ep = self.plot_rt_2.plot(pen=pg.mkPen('g', width=2), name="Identified signal (ep)")
        self.curve_vero_2 = self.plot_rt_2.plot(pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DotLine), name="True SEP")

        # Link X axes
        self.plot_rt_2.setXLink(self.plot_rt_1)

    def setup_results_tabs(self):
        """Creates the empty tabs for the final plots."""
        # Filtered Average Tab
        self.plot_media_filtrata = pg.PlotWidget()
        self.plot_media_filtrata.addLegend()
        self.plot_media_filtrata.invertY(True)
        self.tabs.addTab(self.plot_media_filtrata, "Filtered Average")

        # Rejected Average Tab
        self.plot_media_rifiutata = pg.PlotWidget()
        self.plot_media_rifiutata.addLegend()
        self.plot_media_rifiutata.invertY(True)
        self.tabs.addTab(self.plot_media_rifiutata, "Rejected Average")

        # --- MODIFIED: Raster Plot Tab (now scrolling) ---
        self.plot_item_raster = pg.PlotItem()
        self.image_view_raster = pg.ImageView(view=self.plot_item_raster)
        self.plot_item_raster.vb.setAspectLocked(False)
        self.plot_item_raster.setLabel('bottom', "Time (s)")
        self.plot_item_raster.setLabel('left', f"Identified Sweep # (Last {self.raster_max_sweeps})")

        v_line_n20_raster = pg.InfiniteLine(pos=N20_PEAK_TIME_IDEAL, angle=90, movable=False,
                                            pen=pg.mkPen('gray', style=QtCore.Qt.PenStyle.DashLine))
        v_line_p30_raster = pg.InfiniteLine(pos=P30_PEAK_TIME_IDEAL, angle=90, movable=False,
                                            pen=pg.mkPen('gray', style=QtCore.Qt.PenStyle.DotLine))
        v_line_n30_raster = pg.InfiniteLine(pos=N30_PEAK_TIME_IDEAL, angle=90, movable=False,
                                            pen=pg.mkPen('gray', style=QtCore.Qt.PenStyle.DashLine))
        self.plot_item_raster.addItem(v_line_n20_raster)
        self.plot_item_raster.addItem(v_line_p30_raster)
        self.plot_item_raster.addItem(v_line_n30_raster)
        cmap = pg.colormap.get('RdBu_r', 'matplotlib')
        self.image_view_raster.setColorMap(cmap)

        x_origin = time_axis[0]
        dx = time_axis[1] - time_axis[0]

        # Set image to pre-allocated array.
        # We also set the Y-axis scale here to match the max sweeps.
        self.image_view_raster.setImage(
            self.raster_data,
            autoLevels=False,
            pos=[x_origin, 0],
            scale=[dx, 1]  # dx per pixel, 1 sweep per pixel
        )
        # Set the Y-axis range to match the number of sweeps
        self.plot_item_raster.setYRange(0, self.raster_max_sweeps)

        self.image_view_raster.setLevels(-SIGNAL_AMPLITUDE, SIGNAL_AMPLITUDE)
        self.tabs.addTab(self.image_view_raster, "Raster Plot")
        # --- END MODIFIED RASTER ---

        # --- Waterfall Plot Tab (Unchanged) ---
        self.plot_waterfall = pg.PlotWidget()
        self.plot_item_waterfall = self.plot_waterfall.getPlotItem()
        self.plot_item_waterfall.invertY(True)
        self.plot_item_waterfall.setLabel('left', "Amplitude (uV) + Scrolling Avg Block")
        self.plot_item_waterfall.setLabel('bottom', "Time (s)")
        self.add_waterfall_reference_lines()
        self.waterfall_cmap = pg.colormap.get('viridis')
        self.tabs.addTab(self.plot_waterfall, "Waterfall Plot")
        # --- END WATERFALL ---

        # --- CUSUM Plot Tab (Unchanged) ---
        self.cusum_widget = pg.GraphicsLayoutWidget()
        plot_cusum_n20 = self.cusum_widget.addPlot(row=0, col=0)
        plot_cusum_n20.setTitle("N20 Latency CUSUM (from Block Averages)")
        plot_cusum_n20.setLabel('left', "Cumulative Sum (%)")
        plot_cusum_n20.setLabel('bottom', "Block Number")
        plot_cusum_n20.showGrid(x=True, y=True, alpha=0.3)
        self.plot_cusum_n20_line = plot_cusum_n20.plot(pen=self.normal_pen_n20)
        self.cusum_h_n20_pos = pg.InfiniteLine(pos=self.cusum_threshold, angle=0,
                                               pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine))
        self.cusum_h_n20_neg = pg.InfiniteLine(pos=-self.cusum_threshold, angle=0,
                                               pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine))
        plot_cusum_n20.addItem(self.cusum_h_n20_pos)
        plot_cusum_n20.addItem(self.cusum_h_n20_neg)
        plot_cusum_n30 = self.cusum_widget.addPlot(row=1, col=0)
        plot_cusum_n30.setTitle("N30/N40 Latency CUSUM (from Block Averages)")
        plot_cusum_n30.setLabel('left', "Cumulative Sum (%)")
        plot_cusum_n30.setLabel('bottom', "Block Number")
        plot_cusum_n30.showGrid(x=True, y=True, alpha=0.3)
        self.plot_cusum_n30_line = plot_cusum_n30.plot(pen=self.normal_pen_n30)
        self.cusum_h_n30_pos = pg.InfiniteLine(pos=self.cusum_threshold, angle=0,
                                               pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine))
        self.cusum_h_n30_neg = pg.InfiniteLine(pos=-self.cusum_threshold, angle=0,
                                               pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DashLine))
        plot_cusum_n30.addItem(self.cusum_h_n30_pos)
        plot_cusum_n30.addItem(self.cusum_h_n30_neg)
        self.tabs.addTab(self.cusum_widget, "CUSUM Chart")
        # --- END CUSUM ---

        # --- Deviation Plot Tab (Unchanged) ---
        self.deviation_widget = pg.GraphicsLayoutWidget()
        plot_dev_n20 = self.deviation_widget.addPlot(row=0, col=0)
        plot_dev_n20.setTitle("N20 Latency Instantaneous Deviation (of Block Averages)")
        plot_dev_n20.setLabel('left', "Deviation (%)")
        plot_dev_n20.setLabel('bottom', "Block Number")
        plot_dev_n20.showGrid(x=True, y=True, alpha=0.3)
        plot_dev_n20.setYRange(-50, 50)
        plot_dev_n20.addLegend()
        self.plot_dev_n20_line = plot_dev_n20.plot(
            pen=pg.mkPen('r', width=2), symbol='o', symbolSize=5,
            symbolBrush='r', name="Block % Deviation"
        )
        plot_dev_n30 = self.deviation_widget.addPlot(row=1, col=0)
        plot_dev_n30.setTitle("N30/N40 Latency Instantaneous Deviation (of Block Averages)")
        plot_dev_n30.setLabel('left', "Deviation (%)")
        plot_dev_n30.setLabel('bottom', "Block Number")
        plot_dev_n30.showGrid(x=True, y=True, alpha=0.3)
        plot_dev_n30.setYRange(-50, 50)
        plot_dev_n30.addLegend()
        self.plot_dev_n30_line = plot_dev_n30.plot(
            pen=pg.mkPen('b', width=2), symbol='o', symbolSize=5,
            symbolBrush='b', name="Block % Deviation"
        )
        self.tabs.addTab(self.deviation_widget, "Deviation Chart")
        # --- END Deviation Tab ---

    # --- NEW HELPER METHOD ---
    def add_waterfall_reference_lines(self):
        """Adds the vertical reference lines to the waterfall plot."""
        v_line_zero_water = pg.InfiniteLine(pos=0.0, angle=90, movable=False,
                                            pen=pg.mkPen('white', style=QtCore.Qt.PenStyle.DashLine))
        v_line_n20_water = pg.InfiniteLine(pos=N20_PEAK_TIME_IDEAL, angle=90, movable=False,
                                           pen=pg.mkPen('gray', style=QtCore.Qt.PenStyle.DashLine))
        v_line_p30_water = pg.InfiniteLine(pos=P30_PEAK_TIME_IDEAL, angle=90, movable=False,
                                           pen=pg.mkPen('gray', style=QtCore.Qt.PenStyle.DotLine))
        v_line_n30_water = pg.InfiniteLine(pos=N30_PEAK_TIME_IDEAL, angle=90, movable=False,
                                           pen=pg.mkPen('gray', style=QtCore.Qt.PenStyle.DashLine))
        self.plot_item_waterfall.addItem(v_line_zero_water)
        self.plot_item_waterfall.addItem(v_line_n20_water)
        self.plot_item_waterfall.addItem(v_line_p30_water)
        self.plot_item_waterfall.addItem(v_line_n30_water)

    # --- END NEW ---

    def apply_simple_smooth(self, data, window_len=15):
        """
        Applies a simple moving average, ensuring
        output length is EQUAL to input length (mode='same').
        """
        # Ensure we have enough data to smooth
        # if len(data) < window_len:
        #    return data  # Return original data if not enough points

        # if window_len % 2 == 0:
        #    window_len += 1
        # w = np.ones(window_len, 'd')
        # smoothed_data = np.convolve(data, w / w.sum(), mode='same')
        #return smoothed_data
        try:
            # Applica il filtro progettato in __init__
            # 'self.b' e 'self.a' sono i coefficienti che abbiamo già calcolato
            smoothed_data = filtfilt(self.b, self.a, data)
            return smoothed_data
        except Exception as e:
            # Gestione di un errore comune se i dati sono troppo corti per il filtro
            print(f"Errore durante l'applicazione di filtfilt (dati troppo corti?): {e}")
            return data  # Ritorna i dati originali in caso di errore

    # --- NEW: Reusable Peak Finding Logic ---
    def find_peak_data(self, data_array, time_array):
        """
        Finds the N20 and N30 peak latencies and amplitudes from a data array.
        Returns a dictionary.
        """
        # N20 search window (15ms to 25ms)
        n20_start_idx = np.argmin(np.abs(time_array - 0.015))
        n20_end_idx = np.argmin(np.abs(time_array - 0.025))

        # N30 (N40) search window (30ms to 50ms)
        n30_start_idx = np.argmin(np.abs(time_array - 0.030))
        n30_end_idx = np.argmin(np.abs(time_array - 0.050))

        # N20 Peak
        search_slice_n20 = data_array[n20_start_idx:n20_end_idx]
        if len(search_slice_n20) > 0:
            idx_in_slice_n20 = np.argmin(search_slice_n20)
            idx_full_n20 = n20_start_idx + idx_in_slice_n20
            lat_n20 = time_array[idx_full_n20]
            amp_n20 = data_array[idx_full_n20]
        else:
            lat_n20, amp_n20 = N20_PEAK_TIME_IDEAL, 0

        # N30 Peak
        search_slice_n30 = data_array[n30_start_idx:n30_end_idx]
        if len(search_slice_n30) > 0:
            idx_in_slice_n30 = np.argmin(search_slice_n30)
            idx_full_n30 = n30_start_idx + idx_in_slice_n30
            lat_n30 = time_array[idx_full_n30]
            amp_n30 = data_array[idx_full_n30]
        else:
            lat_n30, amp_n30 = N30_PEAK_TIME_IDEAL, 0

        return {
            "n20_lat": lat_n20, "n20_amp": amp_n20,
            "n30_lat": lat_n30, "n30_amp": amp_n30
        }

    # --- END NEW ---

    def closeEvent(self, event):
        """
        Overrides the main window's close event.
        Ensures that the results window (if it exists) is closed too.
        """
        print("Main window is closing, also closing results window...")

        # Check if the results window exists
        if self.results_win is not None:
            self.results_win.close()  # Explicitly close the results window

        # Accept the close event to allow the main window to close
        event.accept()

    # --- NEW METHOD ---
    def stop_simulation(self):
        """Stops the simulation timer."""
        if self.timer.isActive():
            self.timer.stop()
            print("\n--- Simulation Manually Stopped ---")
            self.generate_final_plots()  # This will also trigger the save
            self.stop_button.setText("Simulation Stopped")
            self.stop_button.setEnabled(False)

    # --- NEW METHOD ---
    def save_results_automatically(self):
        """Automatically saves the collected waterfall averages to a CSV file."""
        if not self.waterfall_averages_to_save:
            print("No waterfall data to save.")
            return

        fileName = "waterfall_averages.csv"  # Hardcoded filename
        try:
            # Transpose the list of 1D arrays into columns
            # Shape becomes (n_samples, n_blocks)
            data_to_save = np.array(self.waterfall_averages_to_save).T

            # Reshape time_axis to be a column vector
            # Shape (n_samples, 1)
            time_col = time_axis.reshape(-1, 1)

            # Stack time column and data columns horizontally
            full_data = np.hstack((time_col, data_to_save))

            # Create headers
            headers = ["Time (s)"] + [f"Avg_Block_{i + 1}" for i in range(len(self.waterfall_averages_to_save))]

            # Create DataFrame and save
            df = pd.DataFrame(full_data, columns=headers)
            df.to_csv(fileName, index=False, float_format='%.6f')

            print(f"Successfully saved waterfall data to {fileName}")

        except Exception as e:
            print(f"Error automatically saving file: {e}")

    # --- MODIFIED METHOD: load_results_from_csv ---
    def load_results_from_csv(self):
        """Loads waterfall averages from a CSV file and plots them."""
        # fileName, _ = QFileDialog.getOpenFileName(self,
        #                                           "Load Waterfall Averages",
        #                                           "",  # Start in default directory
        #                                           "CSV Files (*.csv)")
        # overwritten file
        fileName = "waterfall_averages.csv"
        if fileName:
            try:
                # Load the data
                df = pd.read_csv(fileName)

                # Extract time axis and data samples
                loaded_time_axis = df.iloc[:, 0].values
                loaded_averages = df.iloc[:, 1:].values  # (samples, blocks)
                num_blocks = loaded_averages.shape[1]
                print(f"Loading {num_blocks} blocks from {fileName}...")

                # --- Clear runtime latency data ---
                self.loaded_n20_latencies.clear()
                self.loaded_n20_amplitudes.clear()
                self.loaded_n30_latencies.clear()
                self.loaded_n30_amplitudes.clear()

                # --- Plot Grand Average ---
                grand_average = np.mean(loaded_averages, axis=1)
                # plotWidget
                self.plot_media_filtrata.clear()
                self.plot_media_filtrata.setTitle(
                    f"Loaded Grand Average (N={num_blocks} blocks of {K_BLOCK_SIZE} sweeps each)")
                self.plot_media_filtrata.plot(loaded_time_axis, grand_average, pen='b', name="Loaded Average")

                # Add reference template *if* dimensions match
                if len(loaded_time_axis) == len(vero_sep_template):
                    self.plot_media_filtrata.plot(time_axis, vero_sep_template,
                                                  pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DotLine),
                                                  name="True SEP")
                self.plot_media_filtrata.autoRange()

                # --- Plot Waterfall Widget ---
                self.plot_item_waterfall.clear()
                self.waterfall_curves.clear()
                self.add_waterfall_reference_lines()  # Re-add lines after clearing

                for i in range(num_blocks):
                    block_data = loaded_averages[:, i]

                    # calculate prestim signal offset and subtract to block_data
                    preoffset = np.mean(block_data[0: N_PRE_STIM_SAMPLES])
                    block_data = block_data - preoffset

                    # Use a fixed offset
                    y_offset = (i * self.waterfall_spacing) / 10

                    # Color based on position
                    normalized_index = i / max(1, num_blocks - 1)
                    color = self.waterfall_cmap.mapToQColor(normalized_index)

                    # --- Add the LINE ---
                    curve = pg.PlotDataItem(
                        loaded_time_axis,
                        block_data + y_offset,
                        pen=pg.mkPen(color)
                    )
                    self.plot_item_waterfall.addItem(curve)
                    self.waterfall_curves.append(curve)

                    # --- MODIFIED: Use new helper function ---
                    peak_data = self.find_peak_data(block_data, loaded_time_axis)

                    # Store latencies and amplitudes
                    self.loaded_n20_latencies.append(peak_data["n20_lat"])
                    self.loaded_n20_amplitudes.append(peak_data["n20_amp"])
                    self.loaded_n30_latencies.append(peak_data["n30_lat"])
                    self.loaded_n30_amplitudes.append(peak_data["n30_amp"])

                    # --- Add N20 dot ---
                    dot_n20 = pg.ScatterPlotItem(
                        x=[peak_data["n20_lat"]], y=[peak_data["n20_amp"] + y_offset],
                        symbol='o', size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 100, 100)
                    )
                    self.plot_item_waterfall.addItem(dot_n20)

                    # --- Add N30 dot ---
                    dot_n30 = pg.ScatterPlotItem(
                        x=[peak_data["n30_lat"]], y=[peak_data["n30_amp"] + y_offset],
                        symbol='o', size=8, pen=pg.mkPen(None), brush=pg.mkBrush(100, 100, 255)
                    )
                    self.plot_item_waterfall.addItem(dot_n30)
                    # --- END MODIFIED ---

                self.plot_item_waterfall.autoRange()

                # --- Enable the results button ---
                self.results_button.setEnabled(True)

                # Switch to the waterfall tab to show the result
                self.tabs.setCurrentWidget(self.plot_waterfall.parent())
                print("...Load and plot complete.")

            except Exception as e:
                print(f"Error loading file: {e}")
                QMessageBox.critical(self, "Load Error", f"Failed to load or parse file:\n{e}")

    # --- END MODIFIED METHOD ---

    # --- NEW METHOD ---
    def reset_cusum_charts(self):
        """Resets all CUSUM and Deviation charts and alarms."""
        print("--- CUSUM and Deviation Charts Reset ---")

        # Reset N20 CUSUM
        self.cusum_n20_lat_data = [0]
        self.plot_cusum_n20_line.setData(self.cusum_n20_lat_data)
        self.cusum_n20_in_alarm = False
        self.plot_cusum_n20_line.setPen(self.normal_pen_n20)

        # Reset N30 CUSUM
        self.cusum_n30_lat_data = [0]
        self.plot_cusum_n30_line.setData(self.cusum_n30_lat_data)
        self.cusum_n30_in_alarm = False
        self.plot_cusum_n30_line.setPen(self.normal_pen_n30)

        # Reset Deviation charts
        self.deviation_n20_data = []
        self.deviation_n30_data = []
        self.plot_dev_n20_line.setData(self.deviation_n20_data)  # Clear line
        self.plot_dev_n30_line.setData(self.deviation_n30_data)  # Clear line

    # --- END NEW ---

    def run_simulation_step(self):
        """Runs a single simulation step (replaces the while loop)."""

        # --- NEW: Read values from SpinBoxes ---
        lat_mod = self.latency_spinbox.value()
        amp_mod = self.amplitude_spinbox.value()
        # --- END NEW ---

        # --- MODIFIED: Pass values to simulator ---
        # The simulator no longer returns None
        nuova_sweep = self.simulatore.get_next_sweep(lat_mod, amp_mod)

        # 1. Check if the simulation is finished
        #    (This is now only controlled by the stop button)
        # if nuova_sweep is None:
        #     self.stop_simulation()  # Use the stop function
        #     return

        # print(f"Received Sweep #{self.simulatore.sweep_count}...")

        # 2. Buffering management
        if len(self.sweep_history) < K_BLOCK_SIZE:
            self.sweep_history.append(nuova_sweep)
            # print(f"Buffering... {len(self.sweep_history)}/{K_BLOCK_SIZE}")
            return

        # 3. Data preparation and processing
        y_output = nuova_sweep
        X_input = np.mean(self.sweep_history[-K_BLOCK_SIZE:], axis=0)
        segnale, orders, aic, p_val = self.processore.elabora_sweep(y_output, X_input)
        self.sweep_history.append(nuova_sweep)

        # 4. Update plots and save results
        if segnale is not None and len(segnale) == len(time_axis):
            na, nb = orders
            # print(f"  -> Model Found: na={na}, nb={nb} (AIC={aic:.2f}, Ljung-Box p={p_val:.3f})")
            self.risultati_filtrati.append(segnale)

            # --- LIVE PLOT UPDATE (Unchanged) ---
            self.plot_rt_1.setTitle(f"Data for ARX Model (Sweep #{self.simulatore.sweep_count})")
            self.curve_y.setData(time_axis, y_output)
            # self.curve_x.setData(time_axis, X_input)
            self.curve_vero.setData(time_axis, vero_sep_template)
            self.plot_rt_2.setTitle(f"ARX Model Result (na={na}, nb={nb})")
            self.curve_ep.setData(time_axis, segnale)
            self.curve_vero_2.setData(time_axis, vero_sep_template)

            # --- MODIFIED: SCROLLING RASTER Plot Update ---
            # 1. Calculate the column to write to (circular buffer)
            col_index = self.current_sweep_index % self.raster_max_sweeps

            # 2. Insert the new signal data into that column
            self.raster_data[:, col_index] = segnale

            # 3. Update the image
            # 'autoLevels=False' is crucial for performance
            self.image_view_raster.setImage(self.raster_data, autoLevels=False)

            # 4. Update levels periodically (as before)
            if self.current_sweep_index % 20 == 0:
                max_abs_val = np.max(np.abs(self.raster_data))  # Check all data
                if max_abs_val == 0: max_abs_val = 1
                self.image_view_raster.setLevels(-max_abs_val, max_abs_val)
            # --- END MODIFIED RASTER ---

            # --- SCROLLING WATERFALL PLOT LOGIC ---

            # Check if the number of IDENTIFIED sweeps is a multiple of 25
            if len(self.risultati_filtrati) > 0 and len(self.risultati_filtrati) % 25 == 0:

                # 1. Calculate and smooth the average block
                media_blocco = np.mean(self.risultati_filtrati[-25:], axis=0)
                media_blocco = self.apply_simple_smooth(media_blocco)  # , window_len=15)

                # --- NEW: Save the block for CSV export ---
                self.waterfall_averages_to_save.append(media_blocco)
                # --- END NEW ---

                # --- NEW: CUSUM & DEVIATION LOGIC MOVED HERE ---
                # Now we run this logic only ONCE per BLOCK

                # 1. Find peak data on the *smoothed block average*
                peak_data = self.find_peak_data(media_blocco, time_axis)

                # 2. Calculate N20 deviation (% and cumulative)
                dev_n20_perc = ((peak_data["n20_lat"] - self.n20_target_lat) / self.n20_target_lat) * 100
                self.cusum_n20_lat_data.append(self.cusum_n20_lat_data[-1] + dev_n20_perc)
                self.deviation_n20_data.append(dev_n20_perc)  # <-- Add to raw list

                # 3. Calculate N30 deviation (% and cumulative)
                dev_n30_perc = ((peak_data["n30_lat"] - self.n30_target_lat) / self.n30_target_lat) * 100
                self.cusum_n30_lat_data.append(self.cusum_n30_lat_data[-1] + dev_n30_perc)
                self.deviation_n30_data.append(dev_n30_perc)  # <-- Add to raw list

                # 4. Update CUSUM plot lines
                self.plot_cusum_n20_line.setData(self.cusum_n20_lat_data)
                self.plot_cusum_n30_line.setData(self.cusum_n30_lat_data)

                # 5. Check CUSUM Alarms
                current_cusum_n20 = self.cusum_n20_lat_data[-1]
                if abs(current_cusum_n20) > self.cusum_threshold:
                    if not self.cusum_n20_in_alarm:
                        print(f"--- CUSUM N20 ALARM: Threshold breached! ({current_cusum_n20:.1f}%) ---")
                        self.plot_cusum_n20_line.setPen(self.alarm_pen)
                        self.cusum_n20_in_alarm = True
                else:
                    self.plot_cusum_n20_line.setPen(self.normal_pen_n20)
                    self.cusum_n20_in_alarm = False

                current_cusum_n30 = self.cusum_n30_lat_data[-1]
                if abs(current_cusum_n30) > self.cusum_threshold:
                    if not self.cusum_n30_in_alarm:
                        print(f"--- CUSUM N30 ALARM: Threshold breached! ({current_cusum_n30:.1f}%) ---")
                        self.plot_cusum_n30_line.setPen(self.alarm_pen)
                        self.cusum_n30_in_alarm = True
                else:
                    self.plot_cusum_n30_line.setPen(self.normal_pen_n30)
                    self.cusum_n30_in_alarm = False

                # 6. Update Deviation Plot
                self.plot_dev_n20_line.setData(self.deviation_n20_data)
                self.plot_dev_n30_line.setData(self.deviation_n30_data)
                # --- END CUSUM/DEVIATION UPDATE ---

                # --- WATERFALL LOGIC (Unchanged) ---
                color = self.waterfall_cmap.mapToQColor(1.0)
                curve = pg.PlotDataItem(
                    time_axis,
                    media_blocco,
                    pen=pg.mkPen(color)
                )
                self.plot_item_waterfall.addItem(curve)
                self.waterfall_curves.append(curve)
                if len(self.waterfall_curves) > self.waterfall_max_curves:
                    old_curve = self.waterfall_curves.pop(0)
                    self.plot_item_waterfall.removeItem(old_curve)
                for i, curve_item in enumerate(self.waterfall_curves):
                    y_offset = (i * self.waterfall_spacing) / 10
                    curve_item.setPos(0, y_offset)
                    normalized_index = i / self.waterfall_max_curves
                    color = self.waterfall_cmap.mapToQColor(normalized_index)
                    curve_item.setPen(pg.mkPen(color))
            # --- END WATERFALL LOGIC ---

            # Increment the *total* sweep counter (for the raster)
            self.current_sweep_index += 1

        elif segnale is None and self.simulatore.sweep_count > K_BLOCK_SIZE:
            self.risultati_rifiutati.append(y_output)
            print(f"  -> NO VALID MODEL FOUND = {len(self.risultati_rifiutati)}")
            # As requested, we don't plot anything

    def generate_final_plots(self):
        """Populates the results tabs at the end of the simulation."""

        # This function is called when simulation ends (manually or naturally)

        print("\nGenerating final plots (Average and StDev)...")
        dati_filtrati_np = None

        # --- 6. Filtered Average and StDev Plot (Unchanged) ---
        if len(self.risultati_filtrati) > 1:
            dati_filtrati_np = np.array(self.risultati_filtrati)
            media_filtrata = np.mean(dati_filtrati_np, axis=0)
            std_filtrata = np.std(dati_filtrati_np, axis=0)
            std_superiore = media_filtrata + std_filtrata
            std_inferiore = media_filtrata - std_filtrata

            self.plot_media_filtrata.setTitle(f"Average of IDENTIFIED Signals (N={len(self.risultati_filtrati)})")
            self.plot_media_filtrata.setLabel('left', 'Amplitude (uV)')
            self.plot_media_filtrata.setLabel('bottom', 'Time (s)')
            self.plot_media_filtrata.plot(time_axis, media_filtrata, pen='b', name="Filtered Average")
            self.plot_media_filtrata.plot(time_axis, vero_sep_template,
                                          pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DotLine), name="True SEP")
            curve_std_sup = self.plot_media_filtrata.plot(time_axis, std_superiore, pen=None)
            curve_std_inf = self.plot_media_filtrata.plot(time_axis, std_inferiore, pen=None)
            fill = pg.FillBetweenItem(curve_std_inf, curve_std_sup, brush=pg.mkBrush(0, 0, 255, 50))
            self.plot_media_filtrata.addItem(fill)

        elif len(self.risultati_filtrati) == 1:
            dati_filtrati_np = np.array(self.risultati_filtrati)
            print(f"  Found only 1 filtered result.")
        else:
            print(f"  No filtered results saved.")

        # --- 7. Rejected Average and StDev Plot (Unchanged) ---
        if len(self.risultati_rifiutati) > 1:
            dati_rifiutati_np = np.array(self.risultati_rifiutati)
            media_rifiutata = np.mean(dati_rifiutati_np, axis=0)
            std_rifiutata = np.std(dati_rifiutati_np, axis=0)
            std_sup_rif = media_rifiutata + std_rifiutata
            std_inf_rif = media_rifiutata - std_rifiutata

            self.plot_media_rifiutata.setTitle(f"Average of REJECTED Signals (N={len(self.risultati_rifiutati)})")
            self.plot_media_rifiutata.setLabel('left', 'Amplitude (uV)')
            self.plot_media_rifiutata.setLabel('bottom', 'Time (s)')
            self.plot_media_rifiutata.plot(time_axis, media_rifiutata, pen='orange', name="Rejected Average")
            self.plot_media_rifiutata.plot(time_axis, vero_sep_template,
                                           pen=pg.mkPen('r', style=QtCore.Qt.PenStyle.DotLine), name="True SEP")
            curve_std_sup_rif = self.plot_media_rifiutata.plot(time_axis, std_sup_rif, pen=None)
            curve_std_inf_rif = self.plot_media_rifiutata.plot(time_axis, std_inf_rif, pen=None)
            fill_rif = pg.FillBetweenItem(curve_std_inf_rif, curve_std_sup_rif, brush=pg.mkBrush(255, 165, 0, 50))
            self.plot_media_rifiutata.addItem(fill_rif)
        else:
            print(f"  {len(self.risultati_rifiutati)} sweeps were rejected. Not enough for an average plot.")

        # --- 8, 9, 10. Raster, Waterfall, CUSUM, Deviation Plots ---
        print("Live-updating charts already generated.")

        if self.plot_item_waterfall.items:  # Check if there are items before auto-ranging
            self.plot_item_waterfall.autoRange()

        # --- NEW: Trigger automatic save ---
        print("Attempting to save waterfall data automatically...")
        self.save_results_automatically()
        # --- END NEW ---

        # Switch to the first results tab
        self.tabs.setCurrentIndex(1)

    # --- NEW METHOD ---
    def resultsView(self):
        """Shows a new window with latency histograms from the loaded data."""
        if not self.loaded_n20_latencies:
            # If no data has been loaded, show a warning
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("No Data")
            msg_box.setText("No results data has been loaded.\nPlease use 'Load Saved SEPs' first.")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            return

        # Create and show the new results window
        self.results_win = ResultsWindow(self.loaded_n20_latencies,
                                         self.loaded_n20_amplitudes,
                                         self.loaded_n30_latencies,
                                         self.loaded_n30_amplitudes
                                         )
        self.results_win.show()


# --- 6. Application Start ---
if __name__ == '__main__':
    # Create the application
    app = QApplication(sys.argv)

    # Create and show the main window
    main_win = SimulationWindow()
    main_win.show()

    # Run the application loop
    sys.exit(app.exec())