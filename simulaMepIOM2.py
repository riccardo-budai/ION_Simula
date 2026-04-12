import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel
)
from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtGui import QFont
from typing import Dict, Any, Tuple
import random

# ==============================================================================
# 1. GENERAL PARAMETERS AND CONCEPTUAL MODELING (Traduzione: Parametri e Modello)
# ==============================================================================

# Simulation Parameters
SIM_DURATION_MS = 50.0  # Total simulation duration
DT_MS = 0.05  # Time step (high resolution)
TIME = np.arange(0, SIM_DURATION_MS, DT_MS)

# 1.1 Modello di Stimolazione Corticale (Volley D/I)
# La tES con 3-4 impulsi evoca una salva D seguita da onde I (I1, I2, I3)[cite: 70].
# I tempi sono approssimativi, con un intervallo inter-onda di ~1.5 ms[cite: 253].
# 1.1 Cortical Stimulation Model (D/I Volley)
CORTICAL_SPIKE_TIMES = [
    2.0,  # 5.0 Onda D (attivazione assonale diretta) [cite: 72]
    3.5,  # 6.5 Onda I1 (sinapsi dendritica prossimale) [cite: 78]
    5.0,  # 8.0 Onda I2
    6.5,  # 9.5 Onda I3 (input sinaptici con ritardi maggiori) [cite: 79]
]

# 1.2 Conduction Model (CST)
CONDUCTION_DELAY_MS = 5.0  # (5 ms per upper limb) (19 ms per lower limb)
                            # 15.0 Ritardo medio per raggiungere il midollo spinale
NUM_CST_FIBERS = 10000 # Popolazione CST (circa 1 milione in totale [cite: 112])
DISPERSION_STD_MS = 0.5  # 0.5 Deviazione standard per la dispersione temporale [cite: 113]

# 1.3 Spinal Integration Model (Alpha Motoneuron)
MOTONEURON_THRESHOLD_MV = 20.0      # Activation threshold (symbolic)
EPSP_PEAK_MS = 1.0                  # Time to EPSP peak
EPSP_DECAY_TAU_MS = 3.0             # EPSP decay time constant
EPSP_AMPLITUDE_MV = 7.0             # Amplitude of a single sub-threshold EPSP


# ==============================================================================
# 2. SIMULATION FUNCTIONS (Traduzione: Funzioni di Simulazione)
# ==============================================================================

def generate_epsp(time_array: np.ndarray, time_peak: float, tau_decay: float, amplitude: float,
                  arrival_time: float) -> np.ndarray:
    """Generates an EPSP (Excitatory Postsynaptic Potential) using an alpha function model."""
    t_shifted = time_array - arrival_time
    epsp = np.zeros_like(t_shifted)
    valid_t = t_shifted > 0
    epsp[valid_t] = amplitude * (t_shifted[valid_t] / time_peak) * np.exp(-t_shifted[valid_t] / tau_decay)
    # Normalize height
    epsp_max = np.max(epsp)
    if epsp_max > 0:
        epsp *= (amplitude / epsp_max)
    return epsp

def simulate_cst_conduction(cortical_times: list, num_fibers: int, delay_ms: float,
                            dispersion_std_ms: float) -> Dict[str, np.ndarray]:
    """Simulates propagation along the CST, applying delay and temporal dispersion."""
    arrival_times = {}

    # Dispersion distribution (simulates fiber diameter distribution)
    dispersion_population = np.random.normal(0, dispersion_std_ms, num_fibers)

    for volley_index, cortical_time in enumerate(cortical_times):
        mean_arrival = cortical_time + delay_ms
        final_arrivals = mean_arrival + dispersion_population

        # Simplify to 100 fibers for integration model clarity
        arrival_times[f'V{volley_index}'] = final_arrivals[::num_fibers]  # // 100)]

    return arrival_times


def simulate_spinal_integration(cst_arrival_times: Dict[str, np.ndarray]) -> Tuple[np.ndarray, list]:
    """Simulates temporal summation of EPSPs and Alpha Motoneuron activation."""
    motoneuron_potential = np.zeros_like(TIME)
    motoneuron_spike_times = []
    synaptic_delay = 1.0  # Monosynaptic synaptic latency

    for volley_key, arrival_times_ms in cst_arrival_times.items():
        mean_arrival = np.mean(arrival_times_ms)
        epsp_arrival_time = mean_arrival + synaptic_delay

        # Generate EPSP (scaled amplitude simulates spatial summation)
        single_epsp = generate_epsp(
            TIME,
            EPSP_PEAK_MS,
            EPSP_DECAY_TAU_MS,
            EPSP_AMPLITUDE_MV,
            epsp_arrival_time
        )
        motoneuron_potential += single_epsp

        # Threshold Control (all-or-nothing activation)
        if not motoneuron_spike_times:
            if np.max(motoneuron_potential) >= MOTONEURON_THRESHOLD_MV:
                spike_time_index = np.argmax(motoneuron_potential)
                motoneuron_spike_times.append(TIME[spike_time_index])
                # Reset potential after spike (refractoriness)
                motoneuron_potential[spike_time_index:] = 0.0

    return motoneuron_potential, motoneuron_spike_times


def generate_cmap_waveform(motoneuron_spike_times: list) -> np.ndarray:
    """Simulates CMAP (Compound Muscle Action Potential) synthesis."""
    cmap = np.zeros_like(TIME)
    PERIPHERAL_DELAY = 5.0  # Peripheral conduction + NMJ delay

    def generate_muap(t: np.ndarray, arrival_t: float) -> np.ndarray:
        """Simplified triphasic Motor Unit Action Potential (MUAP) model."""
        t_rel = t - arrival_t
        muap = 20 * np.exp(-((t_rel - 1.5) ** 2) / 0.5)
        muap -= 40 * np.exp(-((t_rel - 3.0) ** 2) / 0.8)
        muap += 10 * np.exp(-((t_rel - 5.0) ** 2) / 1.0)
        return muap

    for spike_time in motoneuron_spike_times:
        mfap_arrival_time = spike_time + PERIPHERAL_DELAY
        muap_wave = generate_muap(TIME, mfap_arrival_time)

        # Spatial summation of MUAPs to form CMAP
        cmap += muap_wave * 0.5

    return cmap


# ==============================================================================
# 3. EXECUTION AND PYQTGRAPH VISUALIZATION CLASS
# ==============================================================================

class CMAPSimulationViewer(QWidget):
    def __init__(self, potential: np.ndarray, cmap: np.ndarray, spikes: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Conceptual Motor Pathway Simulation (tES -> CMAP)")
        self.setGeometry(100, 100, 1000, 700)
        self.potential = potential
        self.cmap = cmap
        self.spikes = spikes

        self._setup_ui()
        self._plot_data()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # Title
        title = QLabel("Conceptual Motor Pathway Simulation (tES -> CMAP)")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        main_layout.addWidget(title)

        # Plots Container
        plots_container = QVBoxLayout()

        # Plot 1: Spinal Integration (Motoneuron)
        self.plot_mn = pg.PlotWidget()
        self.plot_mn.setBackground('w')
        # Labels: "Potential (mV)", "Time from Stimulation (ms)"
        self.plot_mn.setLabel('left', "Potential (mV)")
        self.plot_mn.setLabel('bottom', "Time from Stimulation (ms)")
        self.plot_mn.showGrid(x=True, y=True, alpha=0.5)
        self.plot_mn.setTitle("Synaptic Integration (Spinal Alpha Motoneuron)")
        self.plot_mn_item = self.plot_mn.getPlotItem()
        plots_container.addWidget(self.plot_mn)

        # Plot 2: CMAP (Muscle Response)
        self.plot_cmap = pg.PlotWidget()
        self.plot_cmap.setBackground('w')
        # Labels: "Amplitude (µV) - Symbolic", "Time from Stimulation (ms)"
        self.plot_cmap.setLabel('left', "Amplitude (µV) - Symbolic")
        self.plot_cmap.setLabel('bottom', "Time from Stimulation (ms)")
        self.plot_cmap.showGrid(x=True, y=True, alpha=0.5)
        self.plot_cmap.setTitle("Compound Muscle Action Potential (CMAP) Recorded")
        self.plot_cmap_item = self.plot_cmap.getPlotItem()
        plots_container.addWidget(self.plot_cmap)

        main_layout.addLayout(plots_container)

    def _plot_data(self):
        # --- Plot 1: Spinal Integration (Motoneuron) ---

        # Plot EPSP Summation (blue line)
        self.plot_mn.plot(TIME, self.potential, pen=pg.mkPen('b', width=2), name='EPSP Sum')

        # Plot Threshold (red dashed line)
        self.plot_mn.addLine(y=MOTONEURON_THRESHOLD_MV, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine),
                             name='Activation Threshold')

        # Plot Motoneuron Spike Time (green dotted line)
        if self.spikes:
            for spike_time in self.spikes:
                self.plot_mn.addLine(x=spike_time, pen=pg.mkPen('g', style=Qt.PenStyle.DotLine), name='MN Spike')

        self.plot_mn_item.setXRange(min(CORTICAL_SPIKE_TIMES) - 2, SIM_DURATION_MS)
        self.plot_mn_item.addLegend()

        # --- Plot 2: CMAP (Muscle Response) ---

        # Plot CMAP Waveform (black line)
        self.plot_cmap.plot(TIME, self.cmap, pen=pg.mkPen('k', width=3), name='Simulated CMAP')

        # Invert CMAP Y-axis (Neurophysiological convention)
        self.plot_cmap_item.invertY(True)
        self.plot_cmap_item.addLegend()
        self.plot_cmap_item.setXLink(self.plot_mn_item)  # Share X-axis

        # Add CMAP Onset Latency info (if activated)
        if self.spikes:
            onset_latency = self.spikes[0] + 5.0
            self.plot_cmap.addLine(x=onset_latency, pen=pg.mkPen('g', style=Qt.PenStyle.DotLine))
            self.plot_cmap.setTitle(f"CMAP Recorded (Onset Latency: {onset_latency:.2f} ms)")


def run_simulation() -> Tuple[np.ndarray, np.ndarray, list]:
    """Executes the entire simulation pipeline and prints summary."""

    cst_arrival_times = simulate_cst_conduction(
        CORTICAL_SPIKE_TIMES,
        NUM_CST_FIBERS,
        CONDUCTION_DELAY_MS,
        DISPERSION_STD_MS
    )

    motoneuron_potential, motoneuron_spike_times = simulate_spinal_integration(
        cst_arrival_times
    )

    cmap_waveform = generate_cmap_waveform(motoneuron_spike_times)

    # --- Report (Tradotto) ---
    print("-" * 50)
    print(f"CORTICAL VOLLEY (D/I) SENT AT: {CORTICAL_SPIKE_TIMES} ms")
    print(f"MEAN CST DELAY: {CONDUCTION_DELAY_MS} ms")
    print(f"TEMPORAL DISPERSION CST: {DISPERSION_STD_MS} ms")
    print("-" * 50)

    if motoneuron_spike_times:
        print(f"✅ ALPHA MOTONEURON ACTIVATED AT: {motoneuron_spike_times[0]:.2f} ms")
        print(f"CMAP ONSET LATENCY (approx.): {motoneuron_spike_times[0] + 5.0:.2f} ms")
    else:
        print("❌ ALPHA MOTONEURON NOT ACTIVATED (SUB-THRESHOLD).")

    print("-" * 50)

    return motoneuron_potential, cmap_waveform, motoneuron_spike_times


# --- Script Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    #
    potential, cmap, spikes = run_simulation()
    viewer = CMAPSimulationViewer(potential, cmap, spikes)
    viewer.show()

    sys.exit(QCoreApplication.exec())
