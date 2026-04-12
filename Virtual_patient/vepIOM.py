import sys
import numpy as np
import pyqtgraph as pg
import webbrowser as wb

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QComboBox, QLabel, QSpinBox,
                               QDoubleSpinBox, QMessageBox, QCheckBox)  # <--- Aggiunto QCheckBox
from PySide6.QtCore import QTimer, Qt, Signal
from learning_manager import LearningManager

# Impostazioni di base per pyqtgraph (grafici con sfondo scuro)
pg.setConfigOption('background', '#343a40')
pg.setConfigOption('foreground', 'w')


class VepSimulator(QMainWindow):
    """
    Finestra di simulazione VEP (Potenziali Evocati Visivi)
    Simula 2 canali con controllo frequenza e visualizzazione opzionale dell'input.
    """
    simulation_finished = Signal()

    def __init__(self, json_anomaly_path: str = None,
                 learning_manager: LearningManager = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("VEP Simulator (Visual Evoked Potentials)")
        self.setGeometry(150, 150, 950, 700)

        # Riferimenti
        self.anomaly_path = json_anomaly_path
        self.learning_manager = learning_manager

        # Parametri di simulazione
        self.fs = 10000  # Frequenza di campionamento (10 kHz)
        self.duration_ms = 300  # Durata della traccia (300 ms)
        self.time_vector = np.linspace(0, self.duration_ms, int(self.fs * (self.duration_ms / 1000)))

        # Riduzione ampiezza per il canale ipsilaterale (stimolo mono)
        self.ipsi_scale_factor = 0.6

        # Parametri base delle onde VEP
        self.wave_params = {
            'N75': {'lat': 75, 'amp': -3.0, 'std': 10},
            'P100': {'lat': 100, 'amp': 5.0, 'std': 15},
            'N145': {'lat': 145, 'amp': -4.0, 'std': 20},
        }

        # Stato della simulazione
        self.is_running = False
        self.sweep_count = 0
        self.total_sweeps_target = 200

        # Buffer di media per DUE canali
        self.current_average_ch1 = np.zeros(len(self.time_vector))
        self.current_average_ch2 = np.zeros(len(self.time_vector))

        # Timer per la simulazione
        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self._run_simulation_sweep)

        self._load_anomalies()
        self._setup_ui()

        print(f"VEP Simulator started. Anomaly: {self.anomaly_path}")

    def _setup_ui(self):
        """Crea l'interfaccia utente."""
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # --- Pannello di Controllo (Sinistra) ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(350)

        # Pulsante Start/Stop
        self.start_stop_button = QPushButton("Start Simulation")
        self.start_stop_button.clicked.connect(self.start_stop)
        self.start_stop_button.setStyleSheet("font-weight: bold; padding: 5px;")

        # Selezione Stimolo
        self.stim_combo = QComboBox()
        self.stim_combo.addItems(["Bilateral", "Monolateral Left", "Monolateral Right"])
        self.stim_combo.currentTextChanged.connect(self.reset_simulation)

        # Numero di Sweeps
        self.sweeps_spinbox = QSpinBox()
        self.sweeps_spinbox.setRange(1, 5000)
        self.sweeps_spinbox.setValue(self.total_sweeps_target)
        self.sweeps_spinbox.setSuffix(" sweeps")
        self.sweeps_spinbox.valueChanged.connect(lambda v: setattr(self, 'total_sweeps_target', v))

        # Selettore Frequenza (Hz)
        self.freq_spinbox = QDoubleSpinBox()
        self.freq_spinbox.setRange(0.1, 50.0)
        self.freq_spinbox.setValue(3.1)
        self.freq_spinbox.setSingleStep(0.1)
        self.freq_spinbox.setSuffix(" Hz")
        self.freq_spinbox.valueChanged.connect(self._update_timer_frequency)

        # Livello di Rumore
        self.noise_spinbox = QDoubleSpinBox()
        self.noise_spinbox.setRange(0.0, 20.0)
        self.noise_spinbox.setValue(3.0)
        self.noise_spinbox.setSingleStep(0.1)
        self.noise_spinbox.setSuffix(" µV (std)")

        # ### MODIFICA: Checkbox per mostrare/nascondere input
        self.show_input_checkbox = QCheckBox("Show Single Sweep (Input)")
        self.show_input_checkbox.setChecked(True)  # Default acceso
        self.show_input_checkbox.setStyleSheet("color: #FFFF00;")  # Giallo per richiamare la linea
        self.show_input_checkbox.toggled.connect(self._toggle_input_visibility)

        # Label per contatore
        self.sweep_counter_label = QLabel(f"Sweeps: 0 / {self.total_sweeps_target}")
        self.sweep_counter_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #17a2b8;")

        # Aggiunta widget al layout
        control_layout.addWidget(QLabel("VEP Simulation Controls"))
        control_layout.addWidget(self.start_stop_button)
        control_layout.addSpacing(20)

        control_layout.addWidget(QLabel("Stimulus Mode:"))
        control_layout.addWidget(self.stim_combo)
        control_layout.addSpacing(10)

        control_layout.addWidget(QLabel("Number of Averages (Sweeps):"))
        control_layout.addWidget(self.sweeps_spinbox)
        control_layout.addSpacing(10)

        control_layout.addWidget(QLabel("Stimulation Rate (Frequency):"))
        control_layout.addWidget(self.freq_spinbox)
        control_layout.addSpacing(10)

        control_layout.addWidget(QLabel("Noise Level (µV std):"))
        control_layout.addWidget(self.noise_spinbox)
        control_layout.addSpacing(10)

        # Aggiungo il checkbox
        control_layout.addWidget(self.show_input_checkbox)
        control_layout.addSpacing(10)

        control_layout.addWidget(self.sweep_counter_label)

        control_layout.addSpacing(15)
        self.visual_pathway_label = QLabel()
        self.visual_pathway_label.setFixedSize(280, 200)
        self.visual_pathway_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Gestione Pixmap
        pixmap_path = 'res/logos/visual_pathway.png'
        pixmap = QPixmap(pixmap_path)
        if pixmap.isNull():
            self.visual_pathway_label.setText(f"Immagine non trovata:\n{pixmap_path}")
            self.visual_pathway_label.setStyleSheet("border: 1px dashed gray; color: gray;")
        else:
            scaled_pixmap = pixmap.scaled(
                self.visual_pathway_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.visual_pathway_label.setPixmap(scaled_pixmap)

        control_layout.addWidget(self.visual_pathway_label)

        control_layout.addStretch()
        bottom_button_layout = QHBoxLayout()
        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self._show_help)
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        bottom_button_layout.addWidget(self.help_button)
        bottom_button_layout.addWidget(self.exit_button)
        control_layout.addLayout(bottom_button_layout)

        # --- Grafici (Destra) ---
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)

        # Grafico Canale 1
        self.plot_widget_ch1 = pg.PlotWidget()
        self.plot_widget_ch1.setTitle("Ch 1: Left Occipital (LO - Fz)")
        self.plot_widget_ch1.setLabel('bottom', 'Time (ms)')
        self.plot_widget_ch1.setLabel('left', 'Amplitude (µV)')
        self.plot_widget_ch1.setXRange(0, self.duration_ms)
        self.plot_widget_ch1.setYRange(-15, 15)
        self.plot_widget_ch1.showGrid(x=True, y=True, alpha=0.3)
        self.plot_curve_avg_ch1 = self.plot_widget_ch1.plot(pen=pg.mkPen('#00FFFF', width=2), name="Average Ch1")
        self.plot_curve_single_ch1 = self.plot_widget_ch1.plot(
            pen=pg.mkPen('#FFFF00', width=1, style=Qt.PenStyle.DotLine), name="Single Ch1")

        # Grafico Canale 2
        self.plot_widget_ch2 = pg.PlotWidget()
        self.plot_widget_ch2.setTitle("Ch 2: Right Occipital (RO - Fz)")
        self.plot_widget_ch2.setLabel('bottom', 'Time (ms)')
        self.plot_widget_ch2.setLabel('left', 'Amplitude (µV)')
        self.plot_widget_ch2.setXRange(0, self.duration_ms)
        self.plot_widget_ch2.setYRange(-15, 15)
        self.plot_widget_ch2.showGrid(x=True, y=True, alpha=0.3)
        self.plot_curve_avg_ch2 = self.plot_widget_ch2.plot(pen=pg.mkPen('#00FFFF', width=2), name="Average Ch2")
        self.plot_curve_single_ch2 = self.plot_widget_ch2.plot(
            pen=pg.mkPen('#FFFF00', width=1, style=Qt.PenStyle.DotLine), name="Single Ch2")

        self.plot_widget_ch2.setXLink(self.plot_widget_ch1)

        plot_layout.addWidget(self.plot_widget_ch1)
        plot_layout.addWidget(self.plot_widget_ch2)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(plot_panel)

        self.setCentralWidget(main_widget)

    def _show_help(self):
        pdf_path = 'help_docs/VEP_Simulator_Help.pdf'
        try:
            wb.open_new(pdf_path)
        except Exception:
            QMessageBox.information(self, "VEP Simulator Help", "Simulazione VEP.\nUsa i controlli a sinistra.")

    def _generate_gaussian(self, lat, amp, std):
        return amp * np.exp(-((self.time_vector - lat) ** 2) / (2 * std ** 2))

    def _generate_base_vep(self, amplitude_scale=1.0):
        signal = np.zeros(len(self.time_vector))
        for wave, params in self.wave_params.items():
            scaled_amp = params['amp'] * amplitude_scale
            signal += self._generate_gaussian(params['lat'], scaled_amp, params['std'])
        return signal

    def _generate_noise(self):
        noise_level = self.noise_spinbox.value()
        return np.random.normal(0, noise_level, len(self.time_vector))

    def _update_timer_frequency(self):
        freq = self.freq_spinbox.value()
        if freq <= 0: freq = 0.1
        interval_ms = int(1000 / freq)
        if self.is_running:
            self.sim_timer.setInterval(interval_ms)

    # ### MODIFICA: Funzione per nascondere/mostrare le linee gialle
    def _toggle_input_visibility(self, checked):
        """Nasconde o mostra le linee del singolo sweep (input)."""
        self.plot_curve_single_ch1.setVisible(checked)
        self.plot_curve_single_ch2.setVisible(checked)

    def _run_simulation_sweep(self):
        if not self.is_running or self.sweep_count >= self.total_sweeps_target:
            self.start_stop()
            return

        mode = self.stim_combo.currentText()
        self.sweep_count += 1

        noise1 = self._generate_noise()
        noise2 = self._generate_noise()

        base_ch1 = None
        base_ch2 = None

        if mode == "Bilateral":
            base_ch1 = self._generate_base_vep(amplitude_scale=1.0)
            base_ch2 = self._generate_base_vep(amplitude_scale=1.0)
        elif mode == "Monolateral Left":
            base_ch1 = self._generate_base_vep(amplitude_scale=self.ipsi_scale_factor)
            base_ch2 = self._generate_base_vep(amplitude_scale=1.0)
        elif mode == "Monolateral Right":
            base_ch1 = self._generate_base_vep(amplitude_scale=1.0)
            base_ch2 = self._generate_base_vep(amplitude_scale=self.ipsi_scale_factor)

        last_single_sweep_ch1 = base_ch1 + noise1
        last_single_sweep_ch2 = base_ch2 + noise2

        # Media Mobile cumulativa
        self.current_average_ch1 = (self.current_average_ch1 * (
                    self.sweep_count - 1) + last_single_sweep_ch1) / self.sweep_count
        self.current_average_ch2 = (self.current_average_ch2 * (
                    self.sweep_count - 1) + last_single_sweep_ch2) / self.sweep_count

        self.plot_curve_avg_ch1.setData(self.time_vector, self.current_average_ch1)
        self.plot_curve_avg_ch2.setData(self.time_vector, self.current_average_ch2)

        # Aggiorna il singolo sweep solo se necessario (anche se setVisible gestisce la visibilità, aggiornare i dati è ok)
        self.plot_curve_single_ch1.setData(self.time_vector, last_single_sweep_ch1)
        self.plot_curve_single_ch2.setData(self.time_vector, last_single_sweep_ch2)

        self.sweep_counter_label.setText(f"Sweeps: {self.sweep_count} / {self.total_sweeps_target}")

    def start_stop(self):
        if self.is_running:
            self.is_running = False
            self.sim_timer.stop()
            self.start_stop_button.setText("Start Simulation")
            self.start_stop_button.setStyleSheet("font-weight: bold; padding: 5px;")
        else:
            self.reset_simulation()
            self.is_running = True
            self._update_timer_frequency()
            self.sim_timer.start()
            self.start_stop_button.setText("Stop Simulation")
            self.start_stop_button.setStyleSheet(
                "background-color: #dc3545; color: white; font-weight: bold; padding: 5px;")

    def reset_simulation(self):
        self.sweep_count = 0
        self.total_sweeps_target = self.sweeps_spinbox.value()
        self.current_average_ch1 = np.zeros(len(self.time_vector))
        self.current_average_ch2 = np.zeros(len(self.time_vector))
        self.sweep_counter_label.setText(f"Sweeps: 0 / {self.total_sweeps_target}")

        self.plot_curve_avg_ch1.setData(self.time_vector, self.current_average_ch1)
        self.plot_curve_single_ch1.setData(self.time_vector, self.current_average_ch1)
        self.plot_curve_avg_ch2.setData(self.time_vector, self.current_average_ch2)
        self.plot_curve_single_ch2.setData(self.time_vector, self.current_average_ch2)

    def _load_anomalies(self):
        if self.anomaly_path:
            pass

    def closeEvent(self, event):
        self.sim_timer.stop()
        self.simulation_finished.emit()
        super().closeEvent(event)


if __name__ == '__main__':
    app = pg.mkQApp()
    window = VepSimulator()
    window.show()
    sys.exit(app.exec())