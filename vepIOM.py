import sys
import numpy as np
import pyqtgraph as pg
import webbrowser as wb

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QComboBox, QLabel, QSpinBox,
                             QDoubleSpinBox, QMessageBox)
from PySide6.QtCore import QTimer, Qt
from learning_manager import LearningManager  # Importa il tuo learning manager

# Impostazioni di base per pyqtgraph (grafici con sfondo scuro)
pg.setConfigOption('background', '#343a40')
pg.setConfigOption('foreground', 'w')


class VepSimulator(QMainWindow):
    """
    Finestra di simulazione VEP (Potenziali Evocati Visivi)
    Simula 2 canali (emisfero sx/dx) e l'effetto
    di stimoli monolaterali o bilaterali.
    """

    def __init__(self, json_anomaly_path: str = None,
                 learning_manager: LearningManager = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("VEP Simulator (Visual Evoked Potentials)")
        self.setGeometry(150, 150, 900, 700)  # Finestra un po' più grande

        # Riferimenti
        self.anomaly_path = json_anomaly_path
        self.learning_manager = learning_manager

        # Parametri di simulazione
        self.fs = 10000  # Frequenza di campionamento (10 kHz)
        self.duration_ms = 300  # Durata della traccia (300 ms)
        self.time_vector = np.linspace(0, self.duration_ms, int(self.fs * (self.duration_ms / 1000)))

        # Riduzione ampiezza per il canale ipsilaterale (stimolo mono)
        self.ipsi_scale_factor = 0.6

        # Parametri base delle onde VEP (N75, P100, N145)
        # Questi sono i parametri che potrai modificare con i file JSON
        self.wave_params = {
            'N75': {'lat': 75, 'amp': -3.0, 'std': 10},
            'P100': {'lat': 100, 'amp': 5.0, 'std': 15},
            'N145': {'lat': 145, 'amp': -4.0, 'std': 20},
        }

        # Stato della simulazione
        self.is_running = False
        self.sweep_count = 0
        self.total_sweeps_target = 200  # I VEP richiedono meno medie dei BAEP

        # Buffer di media per DUE canali
        self.current_average_ch1 = np.zeros(len(self.time_vector))
        self.current_average_ch2 = np.zeros(len(self.time_vector))

        # Timer per la simulazione
        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self._run_simulation_sweep)

        self._load_anomalies()  # Funzione per caricare il JSON
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

        # Selezione Stimolo
        self.stim_combo = QComboBox()
        self.stim_combo.addItems(["Bilateral", "Monolateral Left", "Monolateral Right"])
        self.stim_combo.currentTextChanged.connect(self.reset_simulation)

        # Numero di Sweeps (medie)
        self.sweeps_spinbox = QSpinBox()
        self.sweeps_spinbox.setRange(1, 5000)
        self.sweeps_spinbox.setValue(self.total_sweeps_target)
        self.sweeps_spinbox.setSuffix(" sweeps")
        self.sweeps_spinbox.valueChanged.connect(lambda v: setattr(self, 'total_sweeps_target', v))

        # Velocità simulazione
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(1, 100)
        self.speed_spinbox.setValue(5)  # 5 sweep per tick

        # Livello di Rumore
        self.noise_spinbox = QDoubleSpinBox()
        self.noise_spinbox.setRange(0.0, 10.0)
        self.noise_spinbox.setValue(3.0)  # Rumore di base
        self.noise_spinbox.setSingleStep(0.1)
        self.noise_spinbox.setSuffix(" µV (std)")

        # Label per contatore
        self.sweep_counter_label = QLabel(f"Sweeps: 0 / {self.total_sweeps_target}")

        # Aggiungi widget al pannello di controllo
        control_layout.addWidget(QLabel("VEP Simulation Controls"))
        control_layout.addWidget(self.start_stop_button)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Stimulus Mode:"))
        control_layout.addWidget(self.stim_combo)
        control_layout.addSpacing(10)
        control_layout.addWidget(QLabel("Number of Averages (Sweeps):"))
        control_layout.addWidget(self.sweeps_spinbox)
        control_layout.addSpacing(10)
        control_layout.addWidget(QLabel("Noise Level (µV std):"))
        control_layout.addWidget(self.noise_spinbox)
        control_layout.addSpacing(10)
        control_layout.addWidget(QLabel("Sweeps per update:"))
        control_layout.addWidget(self.speed_spinbox)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.sweep_counter_label)

        control_layout.addSpacing(15)
        self.visual_pathway_label = QLabel()
        self.visual_pathway_label.setFixedSize(280, 300)  # Dimensione fissa come richiesto
        self.visual_pathway_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Carica e imposta il Pixmap
        # (Assicurati che questo file esista! Sostituisci con il percorso corretto)
        pixmap_path = 'res/logos/visual_pathway.png'  # <-- ATTENZIONE: Percorso d'esempio!
        pixmap = QPixmap(pixmap_path)
        if pixmap.isNull():
            print(f"Attenzione: Immagine non trovata in {pixmap_path}")
            self.visual_pathway_label.setText(f"Immagine non trovata:\n{pixmap_path}")
            # Stile per far capire che è un segnaposto
            self.visual_pathway_label.setStyleSheet("border: 1px dashed gray; color: gray;")
        else:
            scaled_pixmap = pixmap.scaled(
                self.visual_pathway_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.visual_pathway_label.setPixmap(scaled_pixmap)

        control_layout.addWidget(self.visual_pathway_label)

        # Layout pulsanti di uscita
        control_layout.addStretch()  # Spinge i pulsanti in fondo
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
        plot_layout = QVBoxLayout(plot_panel)  # Layout verticale per i due grafici

        # Grafico Canale 1 (Sinistro)
        self.plot_widget_ch1 = pg.PlotWidget()
        self.plot_widget_ch1.setTitle("Channel 1: Left Occipital (LO - Fz)")
        self.plot_widget_ch1.setLabel('bottom', 'Time (ms)')
        self.plot_widget_ch1.setLabel('left', 'Amplitude (µV)')
        self.plot_widget_ch1.setXRange(0, self.duration_ms)
        self.plot_widget_ch1.setYRange(-10, 10)
        self.plot_widget_ch1.showGrid(x=True, y=True, alpha=0.3)
        self.plot_curve_avg_ch1 = self.plot_widget_ch1.plot(pen=pg.mkPen('c', width=2), name="Average Ch1")
        self.plot_curve_single_ch1 = self.plot_widget_ch1.plot(
            pen=pg.mkPen('y', width=1, style=pg.QtCore.Qt.PenStyle.DotLine), name="Single Ch1")

        # Grafico Canale 2 (Destro)
        self.plot_widget_ch2 = pg.PlotWidget()
        self.plot_widget_ch2.setTitle("Channel 2: Right Occipital (RO - Fz)")
        self.plot_widget_ch2.setLabel('bottom', 'Time (ms)')
        self.plot_widget_ch2.setLabel('left', 'Amplitude (µV)')
        self.plot_widget_ch2.setXRange(0, self.duration_ms)
        self.plot_widget_ch2.setYRange(-10, 10)
        self.plot_widget_ch2.showGrid(x=True, y=True, alpha=0.3)
        self.plot_curve_avg_ch2 = self.plot_widget_ch2.plot(pen=pg.mkPen('c', width=2), name="Average Ch2")
        self.plot_curve_single_ch2 = self.plot_widget_ch2.plot(
            pen=pg.mkPen('y', width=1, style=pg.QtCore.Qt.PenStyle.DotLine), name="Single Ch2")

        # Collega gli assi X dei due grafici
        self.plot_widget_ch2.setXLink(self.plot_widget_ch1)

        plot_layout.addWidget(self.plot_widget_ch1)
        plot_layout.addWidget(self.plot_widget_ch2)

        # Aggiungi pannello di controllo e grafici al layout principale
        main_layout.addWidget(control_panel)
        main_layout.addWidget(plot_panel)  # Aggiungi il pannello dei grafici

        self.setCentralWidget(main_widget)

    def _show_help(self):
        """Mostra un file di aiuto o un messaggio."""
        pdf_path = 'help_docs/VEP_Simulator_Help.pdf'  # Percorso d'esempio
        try:
            wb.open_new(pdf_path)
        except Exception as e:
            QMessageBox.information(self, "VEP Simulator Help",
                                    "Simula la risposta VEP da 2 canali occipitali.\n\n"
                                    "- **Bilateral:** Stimola entrambi gli occhi. Risposta simmetrica.\n"
                                    "- **Monolateral (Left/Right):** Stimola un occhio. "
                                    "La risposta è più forte (100%) sull'emisfero controlaterale "
                                    f"e più debole ({self.ipsi_scale_factor * 100}%) sull'emisfero ipsilaterale.\n\n"
                                    f"(Impossibile aprire il file di aiuto: {pdf_path})")

    def _generate_gaussian(self, lat, amp, std):
        """Genera un singolo picco Gaussiano."""
        return amp * np.exp(-((self.time_vector - lat) ** 2) / (2 * std ** 2))

    def _generate_base_vep(self, amplitude_scale=1.0):
        """Genera il segnale VEP di base sommando le onde."""
        signal = np.zeros(len(self.time_vector))
        for wave, params in self.wave_params.items():
            # Applica lo scaling per la simulazione ipsi/contra
            scaled_amp = params['amp'] * amplitude_scale
            signal += self._generate_gaussian(params['lat'], scaled_amp, params['std'])
        return signal

    def _generate_noise(self):
        """Genera rumore di fondo."""
        noise_level = self.noise_spinbox.value()
        return np.random.normal(0, noise_level, len(self.time_vector))

    def _run_simulation_sweep(self):
        """Esegue un singolo ciclo di aggiornamento della simulazione."""
        if not self.is_running or self.sweep_count >= self.total_sweeps_target:
            self.start_stop()  # Ferma la simulazione
            return

        mode = self.stim_combo.currentText()
        sweeps_per_tick = self.speed_spinbox.value()

        last_single_sweep_ch1 = None
        last_single_sweep_ch2 = None

        for _ in range(sweeps_per_tick):
            if self.sweep_count >= self.total_sweeps_target:
                break

            self.sweep_count += 1

            noise1 = self._generate_noise()
            noise2 = self._generate_noise()

            base_ch1 = None
            base_ch2 = None

            if mode == "Bilateral":
                # Segnale pieno (100%) su entrambi i canali
                base_ch1 = self._generate_base_vep(amplitude_scale=1.0)
                base_ch2 = self._generate_base_vep(amplitude_scale=1.0)

            elif mode == "Monolateral Left":
                # Ch1 (Ipsilaterale) = segnale ridotto
                base_ch1 = self._generate_base_vep(amplitude_scale=self.ipsi_scale_factor)
                # Ch2 (Controlaterale) = segnale pieno
                base_ch2 = self._generate_base_vep(amplitude_scale=1.0)

            elif mode == "Monolateral Right":
                # Ch1 (Controlaterale) = segnale pieno
                base_ch1 = self._generate_base_vep(amplitude_scale=1.0)
                # Ch2 (Ipsilaterale) = segnale ridotto
                base_ch2 = self._generate_base_vep(amplitude_scale=self.ipsi_scale_factor)

            # Crea sweep singoli
            last_single_sweep_ch1 = base_ch1 + noise1
            last_single_sweep_ch2 = base_ch2 + noise2

            # Calcola la media mobile per entrambi i canali
            self.current_average_ch1 = (self.current_average_ch1 * (
                        self.sweep_count - 1) + last_single_sweep_ch1) / self.sweep_count
            self.current_average_ch2 = (self.current_average_ch2 * (
                        self.sweep_count - 1) + last_single_sweep_ch2) / self.sweep_count

        # Aggiorna i grafici
        self.plot_curve_avg_ch1.setData(self.time_vector, self.current_average_ch1)
        self.plot_curve_avg_ch2.setData(self.time_vector, self.current_average_ch2)

        if last_single_sweep_ch1 is not None:
            self.plot_curve_single_ch1.setData(self.time_vector, last_single_sweep_ch1)
            self.plot_curve_single_ch2.setData(self.time_vector, last_single_sweep_ch2)

        # Aggiorna il contatore
        self.sweep_counter_label.setText(f"Sweeps: {self.sweep_count} / {self.total_sweeps_target}")

    def start_stop(self):
        """Avvia o ferma la simulazione."""
        if self.is_running:
            self.is_running = False
            self.sim_timer.stop()
            self.start_stop_button.setText("Start Simulation")
        else:
            self.reset_simulation()  # Resetta prima di iniziare
            self.is_running = True
            self.sim_timer.start(20)  # Timer un po' più lento del BAEP
            self.start_stop_button.setText("Stop Simulation")

    def reset_simulation(self):
        """Resetta la media e il contatore per entrambi i canali."""
        self.sweep_count = 0
        self.total_sweeps_target = self.sweeps_spinbox.value()
        self.current_average_ch1 = np.zeros(len(self.time_vector))
        self.current_average_ch2 = np.zeros(len(self.time_vector))
        self.sweep_counter_label.setText(f"Sweeps: 0 / {self.total_sweeps_target}")

        # Pulisci entrambi i grafici
        self.plot_curve_avg_ch1.setData(self.time_vector, self.current_average_ch1)
        self.plot_curve_single_ch1.setData(self.time_vector, self.current_average_ch1)
        self.plot_curve_avg_ch2.setData(self.time_vector, self.current_average_ch2)
        self.plot_curve_single_ch2.setData(self.time_vector, self.current_average_ch2)

    def _load_anomalies(self):
        """
        Carica i dati di anomalia dal file JSON.
        Modificherà self.wave_params (es. aumentando latenza P100)
        """
        if self.anomaly_path:
            print(f"Loading VEP anomaly from: {self.anomaly_path}")
            # TODO: Aggiungi logica per leggere il JSON
            # Esempio:
            # try:
            #     with open(self.anomaly_path, 'r') as f:
            #         anomaly_data = json.load(f)
            #
            #     # Esempio anomalia: ritardo P100 (neurite ottica)
            #     if 'p100_latency_delay' in anomaly_data:
            #         self.wave_params['P100']['lat'] += anomaly_data['p100_latency_delay']
            #
            # except Exception as e:
            #     print(f"Error loading VEP anomaly: {e}")
            pass
        else:
            print("No anomaly specified, using standard VEP parameters.")

    def closeEvent(self, event):
        """Gestisce la chiusura della finestra."""
        print("Closing VEP simulator...")
        self.sim_timer.stop()
        super().closeEvent(event)


# Codice per testare il modulo singolarmente
if __name__ == '__main__':
    app = pg.mkQApp()  # Funzione helper di PyQtGraph
    window = VepSimulator()
    window.show()
    sys.exit(app.exec())