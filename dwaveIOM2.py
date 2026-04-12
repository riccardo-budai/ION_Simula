import sys
import datetime
import numpy as np
import pyqtgraph as pg
import webbrowser as wb
import logging
import json

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QComboBox, QLabel, QSpinBox, QTabWidget, QTextEdit,
                             QDoubleSpinBox, QMessageBox, QTableWidget, QTableWidgetItem, QAbstractItemView, QApplication)
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QPixmap, QColor
from numpy import random

# Assicurati che questo file sia nella stessa cartella o nel tuo PYTHONPATH
from signalTrakerIOM_EN import SignalTrackerWidget
from etc.anomaly_manager import AnomalyInjector  # Import the anomaly manager

# from learning_manager import LearningManager # Temporarily commented out if not available

# Basic settings for pyqtgraph (dark background graphs)
pg.setConfigOption('background', '#343a40')
pg.setConfigOption('foreground', 'w')

# --- Global Parameters Dictionary ---
## --- Valori 'std' ridotti per curve più ripide ---
ALL_site_PARAMS = {
    'Proximal': {
        'lat': 5, 'amp': 800, 'std': 0.6,
        # Fase 1 (Neg-Piccolo): 30% Amp, 100% Larghezza
        'neg1_scale': 0.3, 'neg1_std_scale': 1.0,
        # Fase 2 (Pos-Ampio): 1.0ms dopo, 100% Amp, 100% Larghezza
        'pos_shift': 1.0, 'pos_scale': 1.0, 'pos_std_scale': 1.0,
        # Fase 3 (Neg-Piccolo,Allungato): 2.5ms dopo, 25% Amp, 150% Larghezza
        'neg2_shift': 2.5, 'neg2_scale': 0.25, 'neg2_std_scale': 1.5,
    },
    'Distal': {
        'lat': 10, 'amp': 600, 'std': 0.6,
        'neg1_scale': 0.3, 'neg1_std_scale': 1.0,
        'pos_shift': 1.0, 'pos_scale': 1.0, 'pos_std_scale': 1.0,
        'neg2_shift': 2.5, 'neg2_scale': 0.25, 'neg2_std_scale': 1.5,
    },
}

# --- Global Muscle Color Dictionary ---
ALL_site_COLORS = {
    # spinal cord sites : proximal to operative site and distal to operative site
    'Proximal': 'cyan', 'Distal': 'gold'
}

# --- Scenario Definitions ---
SCENARIO_CONFIG = {
    "spinalcord_sites": {
        "title": "DWAVE Simulator (Spinal Cord - tcMEP)",
        "cord_sites": ['Proximal', 'Distal'],
        "duration_ms": 40,
    }
}


# (Mock LearningManager if not available for testing)
# class LearningManager:
#     pass

########################################################################################################################
class DwaveSimulator(QWidget):
    """
    DWAVE (Direct wave Evoked Potentials) Simulator
    Displays 1 stacked graph and 1 scrolling raster graph.
    channel one on proximal site and second channel on distal site
    """

    # Segnale emesso alla chiusura della simulazione
    simulation_closed = Signal(str)

    def __init__(self,
                 scenario_name: str = "spinalcord_sites",
                 json_anomaly_path: str = None,
                 learning_manager=None,
                 ai_agent=None,
                 parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window)

        self.logger = logging.getLogger(__name__)

        # --- Scenario Loading Logic ---
        default_config = SCENARIO_CONFIG["spinalcord_sites"]
        config = SCENARIO_CONFIG.get(scenario_name, default_config)

        self.scenario_name = scenario_name
        self.site_list = config["cord_sites"]
        self.num_sites = len(self.site_list)
        self.all_site_params = ALL_site_PARAMS
        self.all_site_colors = ALL_site_COLORS

        for site in self.site_list:
            if site not in self.all_site_params:
                raise ValueError(f"Missing parameters in ALL_site_PARAMS for: {site}")
            if site not in self.all_site_colors:
                raise ValueError(f"Missing color in ALL_site_COLORS for: {site}")

        print(f"Simulator started for scenario: '{scenario_name}' with sites: {self.site_list}")

        self.setWindowTitle(config["title"] + " - Raster View")
        self.setGeometry(100, 100, 1400, 800)

        self.anomaly_path = json_anomaly_path
        self.learning_manager = learning_manager
        self.ai_agent = ai_agent

        if self.ai_agent is None:
            self.logger.warning("AI Agent not passed to DwaveSimulator. Adaptive functions disabled.")

        # Simulation parameters
        self.fs = 20000
        self.duration_ms = config["duration_ms"]
        self.time_points = int(self.fs * (self.duration_ms / 1000))
        self.time_vector = np.linspace(0, self.duration_ms, self.time_points)

        self.facilitation_map = {
            1: 0.05, 2: 0.25, 3: 0.60,
            4: 0.85, 5: 1.00, 6: 1.05,
        }
        self.artifact_amp = 1500.0
        self.artifact_std = 0.1
        self.artifact_isi_ms = 0.5

        # Display parameters
        self.stacked_v_offset_uV = 500
        self.raster_gain_uV = 100
        self.raster_trace_spacing = 1.0
        self.raster_max_traces = 15

        # Simulation state
        self.is_running = False
        self.stim_timer_interval_ms = 1000

        # References to plots and curves
        self.stacked_plot = None
        self.stacked_curves = {}
        self.stacked_labels = {}
        self.stacked_peak_markers = {}

        self.raster_plot = None
        self.raster_curves = {m: [] for m in self.site_list}
        self.raster_sector_lines = []

        self.raster_peak_lines = {}

        self.tracker_widgets = {}
        self.marker_log_widget = None
        self.stim_log_widget = None
        self.peak_log_table = None

        # Timer for simulation
        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self._run_simulation_sweep)

        # Inizializzazione Anomaly Manager
        self.injector = None
        self.anomaly_config = None
        self.anomaly_active = False
        self.response_start_time = None

        if self.anomaly_path:
            try:
                self.injector = AnomalyInjector(self.anomaly_path)
                self.injector.reset()
                self.injector.arm_trigger(0.0)  # Armiamo subito o su start?

                # Carica la configurazione per usarla nella UI (risposte)
                with open(self.anomaly_path, 'r') as f:
                    self.anomaly_config = json.load(f)
                print(f"Anomaly Injector initialized with: {self.anomaly_path}")
            except Exception as e:
                print(f"Failed to init AnomalyInjector: {e}")

        self._setup_ui()
        self.setup_choice_buttons()  # Setup pulsanti risposta

        # Simulation time tracker for anomaly
        self.simulation_elapsed_time = 0.0
        self.prev_anomaly_active = False

        print(f"MEP Simulator ('{self.scenario_name}') started. Anomaly: {self.anomaly_path}")

    def _setup_ui(self):
        """Creates the user interface."""
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # --- Control Panel (Left) ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(300)  # Aumentato un po' per i bottoni

        self.start_stop_button = QPushButton("Start Stimulation")
        self.start_stop_button.setStyleSheet("background-color: blue;")
        self.start_stop_button.clicked.connect(self.start_stop)

        # --- Anomaly Controls ---
        self.anomaly_group = QWidget()
        anomaly_layout = QVBoxLayout(self.anomaly_group)

        self.anomaly_label = QLabel("Anomaly Response:")
        self.comboAnswers = QComboBox()
        self.comboAnswers.setEnabled(False)  # Abilitato solo quando scatta anomalia

        self.pushBtConfirm = QPushButton("Confirm Answer")
        self.pushBtConfirm.setEnabled(False)
        self.pushBtConfirm.clicked.connect(self._handle_choice_confirmation)

        anomaly_layout.addWidget(self.anomaly_label)
        anomaly_layout.addWidget(self.comboAnswers)
        anomaly_layout.addWidget(self.pushBtConfirm)
        control_layout.addWidget(self.anomaly_group)

        # --------------------------------------------------------------------------------------------------------------
        control_layout.addWidget(QLabel("Train Pulses (1-6):"))
        self.pulse_spinbox = QSpinBox()
        self.pulse_spinbox.setRange(1, 6)
        self.pulse_spinbox.setValue(3)
        control_layout.addWidget(self.pulse_spinbox)

        control_layout.addWidget(QLabel("Stimulus Intensity (%):"))
        self.intensity_spinbox = QSpinBox()
        self.intensity_spinbox.setRange(0, 100)
        self.intensity_spinbox.setValue(80)
        self.intensity_spinbox.setSuffix(" %")
        control_layout.addWidget(self.intensity_spinbox)

        control_layout.addWidget(QLabel("Stimulation Interval (ms):"))
        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setRange(500, 10000)
        self.interval_spinbox.setValue(self.stim_timer_interval_ms)
        self.interval_spinbox.setSuffix(" ms")
        self.interval_spinbox.valueChanged.connect(self._update_timer_interval)
        control_layout.addWidget(self.interval_spinbox)

        control_layout.addWidget(QLabel("Noise Level (µV std):"))
        self.noise_spinbox = QDoubleSpinBox()
        self.noise_spinbox.setRange(0.0, 50.0)
        self.noise_spinbox.setValue(15.0)
        self.noise_spinbox.setSuffix(" µV")
        control_layout.addWidget(self.noise_spinbox)

        control_layout.addWidget(QLabel("Inter stimulus ISI (ms):"))
        self.isi_spinbox = QDoubleSpinBox()
        self.isi_spinbox.setRange(0.010, 10.0)
        self.isi_spinbox.setValue(self.artifact_isi_ms)
        self.isi_spinbox.setSingleStep(0.1)
        self.isi_spinbox.setSuffix(" ms")
        control_layout.addWidget(self.isi_spinbox)

        control_layout.addWidget(QLabel("Stacked Plot Offset (µV):"))
        self.stacked_offset_spinbox = QSpinBox()
        self.stacked_offset_spinbox.setRange(100, 5000)
        self.stacked_offset_spinbox.setValue(self.stacked_v_offset_uV)
        self.stacked_offset_spinbox.setSingleStep(100)
        self.stacked_offset_spinbox.setSuffix(" µV")
        self.stacked_offset_spinbox.valueChanged.connect(self._update_stacked_offset)
        control_layout.addWidget(self.stacked_offset_spinbox)

        control_layout.addWidget(QLabel("Raster Gain (µV per traccia):"))
        self.raster_gain_spinbox = QSpinBox()
        self.raster_gain_spinbox.setRange(10, 5000)
        self.raster_gain_spinbox.setValue(self.raster_gain_uV)
        self.raster_gain_spinbox.setSingleStep(10)
        self.raster_gain_spinbox.setSuffix(" µV")
        control_layout.addWidget(self.raster_gain_spinbox)

        control_layout.addWidget(QLabel("Traces per Sector (N):"))
        self.raster_traces_spinbox = QSpinBox()
        self.raster_traces_spinbox.setRange(5, 100)
        self.raster_traces_spinbox.setValue(self.raster_max_traces)
        self.raster_traces_spinbox.valueChanged.connect(self._clear_raster_plot)
        control_layout.addWidget(self.raster_traces_spinbox)

        self.clear_raster_button = QPushButton("Clear Raster")
        self.clear_raster_button.clicked.connect(self._clear_raster_plot)
        control_layout.addWidget(self.clear_raster_button)
        control_layout.addWidget(self.start_stop_button)
        control_layout.addStretch(10)

        self.motor_homunculus_label = QLabel()
        self.motor_homunculus_label.setFixedSize(250, 250)
        self.motor_homunculus_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap_path = 'res/logos/motor_homunculus.png'
        pixmap = QPixmap(pixmap_path)
        if pixmap.isNull():
            self.motor_homunculus_label.setText(f"Image not found:\n{pixmap_path}")
            self.motor_homunculus_label.setStyleSheet("border: 1px dashed gray; color: gray;")
        else:
            scaled_pixmap = pixmap.scaled(self.motor_homunculus_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.motor_homunculus_label.setPixmap(scaled_pixmap)
        control_layout.addWidget(self.motor_homunculus_label)
        control_layout.addStretch()

        bottom_button_layout = QHBoxLayout()
        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self._show_help)
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        bottom_button_layout.addWidget(self.help_button)
        bottom_button_layout.addWidget(self.exit_button)
        control_layout.addLayout(bottom_button_layout)
        # --- Fine Pannello di Controllo -------------------------------------------------------------------------------

        # --- Plots (Right) ---
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        monitoring_widget = QWidget()
        monitoring_layout = QHBoxLayout(monitoring_widget)

        # --- 1a. Stacked Plot ---
        self.stacked_plot = pg.PlotWidget()
        self.stacked_plot.setTitle("Dwave Traces (Stacked)")
        self.stacked_plot.setLabel('bottom', 'Time (ms)')
        self.stacked_plot.setXRange(0, self.duration_ms)
        self.stacked_plot.showGrid(x=True, y=True, alpha=0.1)
        self.stacked_plot.getAxis('left').hide()
        self.stacked_plot.invertY(True)

        for i, site_name in enumerate(self.site_list):
            offset = (i + 1) * self.stacked_v_offset_uV
            color = self.all_site_colors.get(site_name, 'w')
            pen = pg.mkPen(color, width=2)
            curve = self.stacked_plot.plot(pen=pen, name=site_name)
            self.stacked_curves[site_name] = curve

            # Aggiungi i "dot" al grafico
            peak_marker = pg.ScatterPlotItem(
                pen=pg.mkPen(None),  # Nessun contorno
                brush=pg.mkBrush('r'),  # Colore rosso
                size=8,
                symbol='t'  # Cerchio
            )
            self.stacked_plot.addItem(peak_marker)
            self.stacked_peak_markers[site_name] = peak_marker

            # Label of single stack traces ---
            content_html = (
                f'<div style="text-align: center">'
                f'<span style="color: #FFF;">{site_name}</span>'
                f'</div>'
            )
            label = pg.TextItem(html=content_html, anchor=(0, 1.5), angle=0, border='w', fill=(0, 0, 255, 100))
            label.setZValue(1000)
            label_x_pos = self.duration_ms * 0.88
            label.setPos(label_x_pos, offset)
            self.stacked_plot.addItem(label)
            self.stacked_labels[site_name] = label

        self.stacked_plot.setYRange(0, (self.num_sites + 1) * self.stacked_v_offset_uV)
        self.stacked_plot.hideButtons()

        # --- 1b. RASTER Plot (Invariato) ---
        self.raster_plot = pg.PlotWidget()
        self.raster_plot.setTitle("Raster History (Newest on Bottom)")
        self.raster_plot.setLabel('bottom', 'Time (ms)')
        self.raster_plot.setXRange(0, self.duration_ms)
        self.raster_plot.getAxis('left').setLabel("Trace N.")
        self.raster_plot.invertY(True)
        self.raster_plot.showGrid(x=True, y=True, alpha=0.1)
        self.raster_plot.hideButtons()
        line_pen = pg.mkPen(color='w', style=Qt.PenStyle.DashLine, width=1)
        for i in range(1, self.num_sites):
            y_pos = i * self.raster_max_traces
            line = pg.InfiniteLine(pos=y_pos, angle=0, pen=line_pen)
            self.raster_plot.addItem(line)
            self.raster_sector_lines.append(line)
        total_height = self.num_sites * self.raster_max_traces
        self.raster_plot.setYRange(0, total_height)
        self.raster_plot.setXLink(self.stacked_plot)

        for site_name in self.site_list:
            color = self.all_site_colors.get(site_name, 'w')
            # Linea per il picco P1 (Peak Lat) - Linea tratteggiata sottile
            peak_line = pg.InfiniteLine(
                pos=0, angle=90,
                pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DotLine),
                # label=f'{site_name} P1',
                # labelOpts={'position': 0.9, 'color': color, 'movable': False}
            )
            # Fissa i limiti Y per farla alta quanto il plot
            # peak_line.setSpan((0, total_height))

            self.raster_plot.addItem(peak_line)
            self.raster_peak_lines[site_name] = peak_line

        monitoring_layout.addWidget(self.stacked_plot)
        monitoring_layout.addWidget(self.raster_plot)
        plot_layout.addWidget(monitoring_widget)

        # --- Analysis Area (Invariato) ---
        analysis_area_widget = QWidget()
        analysis_area_layout = QHBoxLayout(analysis_area_widget)
        analysis_area_widget.setFixedHeight(300)
        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.setStyleSheet("""
            QTabWidget::pane { border-top: 2px solid #666; }
            QTabBar::tab { background-color: #343a40; padding: 6px 12px;
                           border: 1px solid #444; border-bottom: none; margin-right: 6px; }
            QTabBar::tab:selected { background-color: #555; border: 1px solid #666;
                                    border-bottom: 2px solid #555; margin-bottom: -2px; }
            QTabBar::tab:!selected:hover { background-color: #444; }
        """)
        for site_name in self.site_list:
            initial_y = np.zeros_like(self.time_vector)
            color_code = self.all_site_colors.get(site_name, 'w')
            tracker = SignalTrackerWidget(
                x_data=self.time_vector,
                y_data=initial_y,
                label_x="Time (ms)",
                label_y=f"{site_name} (uV)",
                pen=color_code
            )
            initial_amplitude = self.stacked_offset_spinbox.value()
            tracker.set_amplitude_scale(initial_amplitude)
            self.tracker_widgets[site_name] = tracker
            tracker.marker_added.connect(
                lambda marker_data, m=site_name: self._on_marker_added(m, marker_data)
            )
            tab_index = self.analysis_tabs.addTab(tracker, site_name)
            q_color = pg.mkColor(color_code)
            self.analysis_tabs.tabBar().setTabTextColor(tab_index, q_color)
        analysis_area_layout.addWidget(self.analysis_tabs, stretch=7)

        # --- Tabs Log e Picchi ---
        self.marker_tabs = QTabWidget()
        self.marker_log_widget = QTextEdit()
        self.marker_log_widget.setReadOnly(True)
        self.marker_log_widget.setFontFamily("Monospace")
        self.marker_log_widget.setHtml(
            "<span style='color: gray;'><i>No markers added. Press 'A' at selected position...</i></span>"
        )
        self.marker_tabs.addTab(self.marker_log_widget, "Marker Log")

        self.stim_log_widget = QTextEdit()
        self.stim_log_widget.setReadOnly(True)
        self.stim_log_widget.setFontFamily("Monospace")
        self.stim_log_widget.setHtml(
            "<span style='color: gray;'><i>Start stimulation to log events...</i></span>"
        )
        self.marker_tabs.addTab(self.stim_log_widget, "Stimulation Log")

        # -- Log Risposte Utente (Nuovo) --
        self.user_log_widget = QTextEdit()
        self.user_log_widget.setReadOnly(True)
        self.marker_tabs.addTab(self.user_log_widget, "User Decisions")

        # Tabella "Automatic Peaks" aggiornata
        self.peak_log_table = QTableWidget()
        self.peak_log_table.setRowCount(self.num_sites)
        self.peak_log_table.setColumnCount(4)  # 4 colonne

        self.peak_log_table.setVerticalHeaderLabels(self.site_list)
        self.peak_log_table.setHorizontalHeaderLabels(
            ["Time", "Onset Lat (ms)", "Peak Lat (ms)", "Amplitude (µV)"]
        )

        self.peak_log_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.peak_log_table.setColumnWidth(0, 70)  # Colonna Time
        self.peak_log_table.setColumnWidth(1, 100)  # Colonna Onset Lat
        self.peak_log_table.setColumnWidth(2, 100)  # Colonna Peak Lat
        self.peak_log_table.horizontalHeader().setStretchLastSection(True)  # Colonna Amplitude

        self.marker_tabs.addTab(self.peak_log_table, "Automatic Peaks")

        analysis_area_layout.addWidget(self.marker_tabs, stretch=7)
        plot_layout.addWidget(analysis_area_widget)

        # --- Finalizzazione Layout ---
        main_layout.addWidget(control_panel)
        main_layout.addWidget(plot_panel)
        # self.setCentralWidget(main_widget)

    def setup_choice_buttons(self):
        """Prepara i pulsanti delle alternative leggendoli dal file JSON."""
        if not self.anomaly_config or 'learning_assessment' not in self.anomaly_config:
            return

        assessment_data = self.anomaly_config['learning_assessment']
        correct_answer = assessment_data.get('correct_answer')
        distractors = assessment_data.get('distractors', [])

        if not correct_answer:
            return

        self.comboAnswers.clear()
        choices = [correct_answer] + distractors
        random.shuffle(choices)
        self.comboAnswers.addItem("Seleziona risposta...")
        self.comboAnswers.addItems(choices)

    def _handle_choice_confirmation(self):
        """
        Gestisce la conferma della risposta dell'utente.
        """
        if self.anomaly_config is None:
            QMessageBox.critical(self, "Errore Configurazione",
                                 "Impossibile verificare la risposta: Il file di configurazione dell'anomalia non è stato caricato correttamente.")
            return
        # 1. Recupera la scelta
        self.marker_tabs.setCurrentIndex(2)
        chosen_action = self.comboAnswers.currentText()
        if self.comboAnswers.currentIndex() == 0:
            QMessageBox.warning(self, "Attenzione", "Seleziona una risposta valida.")
            return

        # 2. Tempo di risposta
        tempo_risposta = 0.0
        if self.response_start_time:
            tempo_risposta = (datetime.datetime.now() - self.response_start_time).total_seconds()

        # 3. Verifica correttezza
        correct_action = self.anomaly_config['learning_assessment'].get('correct_answer', '')
        points_success = self.anomaly_config['learning_assessment'].get('points', 100)
        points_fail = -20

        is_correct = (chosen_action == correct_action)
        punti_assegnati = points_success if is_correct else points_fail

        # 4. UI Feedback
        self.comboAnswers.setEnabled(False)
        self.pushBtConfirm.setEnabled(False)

        msg = ""
        if is_correct:
            msg = f"CORRETTO! (+{punti_assegnati} pt)"
            self.user_log_widget.append(f"<span style='color: green;'>{chosen_action}: {msg}</span>")
        else:
            msg = f"ERRATO. Risposta attesa: {correct_action}"
            self.user_log_widget.append(f"<span style='color: red;'>{chosen_action}: {msg}</span>")

        # 5. Registrazione DB
        if self.learning_manager:
            success = self.learning_manager.record_decision(
                azione_presa=chosen_action,
                esito_corretto=is_correct,
                punti=punti_assegnati,
                tempo_risposta_sec=tempo_risposta
            )

            # 6. AI Feedback
            if self.ai_agent and success:
                self._invoke_ai_feedback(punti_assegnati, tempo_risposta)

    def _invoke_ai_feedback(self, score, duration):
        # Logica AI (simile a SepSimulator)
        session_id = self.learning_manager.current_session_id
        subject_id = None

        try:
            query_subj = "SELECT idcode FROM sessions WHERE idsession = ?"
            self.learning_manager.cur.execute(query_subj, (session_id,))
            res = self.learning_manager.cur.fetchone()
            if res:
                subject_id = res[0]
        except Exception as e:
            print(f"Error getting subject ID: {e}")

        if subject_id is not None:
            try:
                self.ai_agent.log_behavior(
                    user_id=subject_id,
                    action_type="SIMULATION_COMPLETE",
                    details=f"DWAVE Scenario - Score: {score}",
                    duration=duration,
                    session_id=session_id
                )

                recommendation = self.ai_agent.get_next_recommendation(subject_id)
                self.ai_agent.log_ai_feedback(
                    user_id=subject_id,
                    session_id=session_id,
                    trigger="SIMULATION_COMPLETE",
                    recommendation_data=recommendation
                )

                if recommendation.get('message'):
                    QMessageBox.information(self, "AI Tutor Feedback", recommendation['message'])

            except Exception as e:
                print(f"AI Error: {e}")

    def _on_marker_added(self, site, marker_data):
        # ... (Invariato) ...
        marker_data['site'] = site
        print(f"Marker received from [{site}]: {marker_data}")
        if self.marker_log_widget:
            if "No markers added" in self.marker_log_widget.toPlainText():
                self.marker_log_widget.clear()
            try:
                ts = marker_data['timestamp'].split('T')[1].split('.')[0]
            except IndexError:
                ts = "??:??:??"
            pg_color_code = self.all_site_colors.get(site, 'w')
            css_color_map = {
                'c': 'cyan', 'y': 'yellow', 'g': 'lime', 'm': 'magenta',
                'r': 'red', 'b': 'blue', 'w': 'white', 'k': 'black'
            }
            site_color = css_color_map.get(pg_color_code, 'white')
            log_entry_html = (
                f"<b>[<span style='color: {site_color};'>{site}</span>]</b> "
                f"time: {marker_data['x_value']:.1f} ms, "
                f"amp: {marker_data['y_value']:.1f} µV "
                f"<span style='color: gray;'>({ts})</span>"
            )
            self.marker_log_widget.append(log_entry_html)

    def _update_stacked_offset(self, new_offset_uV):
        self.stacked_v_offset_uV = new_offset_uV
        for i, site_name in enumerate(self.site_list):
            if site_name in self.stacked_labels:
                offset = (i + 1) * self.stacked_v_offset_uV
                label_x_pos = self.duration_ms * 0.80
                self.stacked_labels[site_name].setPos(label_x_pos, offset)
        self.stacked_plot.setYRange(0, (self.num_sites + 1) * self.stacked_v_offset_uV)
        for tracker in self.tracker_widgets.values():
            tracker.set_amplitude_scale(new_offset_uV)

    def _clear_raster_plot(self):
        # ... (Invariato) ...
        print("Clearing Raster Plot...")
        self.raster_max_traces = self.raster_traces_spinbox.value()
        for site_name in self.site_list:
            curve_list = self.raster_curves[site_name]
            for curve in curve_list:
                self.raster_plot.removeItem(curve)
            curve_list.clear()
        for line in self.raster_sector_lines:
            self.raster_plot.removeItem(line)
        self.raster_sector_lines.clear()
        line_pen = pg.mkPen(color='w', style=Qt.PenStyle.DashLine, width=1)
        total_height = 0
        for i in range(1, self.num_sites):
            y_pos = i * self.raster_max_traces
            line = pg.InfiniteLine(pos=y_pos, angle=0, pen=line_pen)
            self.raster_plot.addItem(line)
            self.raster_sector_lines.append(line)
        total_height = self.num_sites * self.raster_max_traces
        self.raster_plot.setYRange(0, total_height)

    def _show_help(self):
        # ... (Invariato) ...
        pdf_path = 'help_docs/paper/Tut_Dwave.pdf'
        try:
            wb.open_new(pdf_path)
        except Exception as e:
            QMessageBox.information(self, "MEP Simulator Help",
                                    f"MEP Simulator for {self.scenario_name}.\n\n"
                                    "...")

    def _generate_gaussian(self, lat, amp, std):
        # ... (Invariato) ...
        return amp * np.exp(-((self.time_vector - lat) ** 2) / (2 * std ** 2))

    def _generate_mep_signal(self, params, amplitude_scale=1.0):
        """
        Genera un segnale MEP/DWAVE trifasico (Neg-Pos-Neg)
        basato sui parametri del dizionario.
        Permette di scalare ampiezza e larghezza di ogni fase.
        """

        # 1. Estrai i parametri di base
        # 'lat' è la latenza del *primo* picco (Neg1)
        lat_base = params['lat']
        # 'amp' è l'ampiezza *massima* (il picco Positivo)
        amp_base = params['amp'] * amplitude_scale
        # 'std' è la larghezza *base* (del picco Positivo)
        std_base = params['std']

        # 2. Estrai i parametri per le 3 fasi (con fallback sicuri)

        # --- Fase 1 (Negativo 1) ---
        neg1_scale = params.get('neg1_scale', 0.3)
        neg1_std_scale = params.get('neg1_std_scale', 1.0)

        # --- Fase 2 (Positivo) ---
        pos_shift = params.get('pos_shift', 1.0)
        pos_scale = params.get('pos_scale', 1.0)
        pos_std_scale = params.get('pos_std_scale', 1.0)

        # --- Fase 3 (Negativo 2) ---
        neg2_shift = params.get('neg2_shift', 2.5)
        neg2_scale = params.get('neg2_scale', 0.25)
        neg2_std_scale = params.get('neg2_std_scale', 1.5)

        # --- Genera le 3 onde (Gaussiana) ---

        # Fase 1: Picco Negativo 1 (Piccolo)
        # Latenza: lat_base
        # Ampiezza: -amp_base * neg1_scale (Negativa -> verso l'alto)
        # Std: std_base * neg1_std_scale
        wave1_neg = self._generate_gaussian(
            lat_base,
            -amp_base * neg1_scale,
            std_base * neg1_std_scale
        )

        # Fase 2: Picco Positivo (Ampio)
        # Latenza: lat_base + pos_shift
        # Ampiezza: +amp_base * pos_scale (Positiva -> verso il basso)
        # Std: std_base * pos_std_scale
        wave2_pos = self._generate_gaussian(
            lat_base + pos_shift,
            amp_base * pos_scale,
            std_base * pos_std_scale
        )

        # Fase 3: Picco Negativo 2 (Piccolo, Allungato)
        # Latenza: lat_base + neg2_shift
        # Ampiezza: -amp_base * neg2_scale (Negativa -> verso l'alto)
        # Std: std_base * neg2_std_scale (più largo)
        wave3_neg = self._generate_gaussian(
            lat_base + neg2_shift,
            -amp_base * neg2_scale,
            std_base * neg2_std_scale
        )

        # Somma le tre fasi per ottenere la forma d'onda finale
        return wave1_neg + wave2_pos + wave3_neg

    def _generate_noise(self):
        # ... (Invariato) ...
        noise_level = self.noise_spinbox.value()
        return np.random.normal(0, noise_level, len(self.time_vector))

    def _generate_stim_artifact(self, lat, amp, std):
        # ... (Invariato) ...
        lat1 = lat
        amp1 = amp
        std1 = std
        lat2 = lat1 + (std1 * 2.5)
        amp2 = -amp * 0.7
        std2 = std
        artifact1 = self._generate_gaussian(lat1, amp1, std1)
        artifact2 = self._generate_gaussian(lat2, amp2, std2)
        return artifact1 + artifact2

    def _analyze_cmap_peaks(self, signal_data, site_name):
        """
        Analizza un segnale DWAVE per trovare il primo picco Negativo (N1)
        e il primo picco Positivo (P1) successivo, e calcola l'ampiezza Peak-to-Peak (N1-P1).
        Usa intervalli di ricerca fissi basati sulla latenza attesa.
        """
        try:
            params = self.all_site_params[site_name]
            expected_n1_lat_base = params['lat']  # 5 ms per Proximal

            # --- 1. Definizione Finestre di Ricerca Ottimizzate ---

            # Finestra N1: Centrata sulla latenza attesa (e.g., 5 ms)
            n1_search_start_ms = max(0, expected_n1_lat_base - 1.5)  # Es. 3.5 ms
            n1_search_end_ms = min(self.duration_ms, expected_n1_lat_base + 1.0)  # Es. 6.0 ms

            n1_search_start_idx = int(n1_search_start_ms / self.duration_ms * self.time_points)
            n1_search_end_idx = int(n1_search_end_ms / self.duration_ms * self.time_points)

            if n1_search_start_idx >= n1_search_end_idx:
                return 0.0, 0.0, 0.0, 0.0, 0.0

            # --- 2. Trova Picco Negativo (N1: La prima deflessione Negativa) ---

            n1_search_roi = signal_data[n1_search_start_idx:n1_search_end_idx]

            # Trova il minimo assoluto (N1) nella finestra N1
            n1_local_idx = np.argmin(n1_search_roi)
            n1_global_idx = n1_local_idx + n1_search_start_idx

            n1_amp = signal_data[n1_global_idx]
            n1_lat_ms = self.time_vector[n1_global_idx]

            # --- 3. Trova Picco Positivo (P1: La prima deflessione Positiva) ---

            # Finestra P1: Inizia subito dopo il picco N1 e si estende per 3-4 ms
            p1_search_start_idx = n1_global_idx
            # Estendiamo la ricerca P1 fino a 8 ms (per catturare 6.2 ms)
            p1_search_end_ms = min(self.duration_ms, expected_n1_lat_base + 4.0)  # Es. 9.0 ms
            p1_search_end_idx = int(p1_search_end_ms / self.duration_ms * self.time_points)

            if p1_search_start_idx >= p1_search_end_idx:
                return 0.0, n1_amp, 0.0, 0.0, 0.0  # Trovato solo N1

            p1_search_roi = signal_data[p1_search_start_idx:p1_search_end_idx]

            # Trova il massimo assoluto (P1) nella finestra P1
            p1_local_idx = np.argmax(p1_search_roi)
            p1_global_idx = p1_local_idx + p1_search_start_idx

            p1_amp = signal_data[p1_global_idx]
            p1_lat_ms = self.time_vector[p1_global_idx]

            # --- 4. Calcola Ampiezza Peak-to-Peak (N1-P1) ---
            final_amplitude_pp = p1_amp - n1_amp

            # --- 5. Rilevamento Soglia (Filtro Anti-Rumore) ---
            noise_std_dev_setting = self.noise_spinbox.value()
            if final_amplitude_pp < (noise_std_dev_setting * 2):
                return 0.0, 0.0, 0.0, 0.0, 0.0  # Sotto soglia di rumore P-P

            # --- 6. Risultati di output per la UI ---
            # Onset Lat: Lat. Picco N1 (come latenza d'inizio risposta)
            # Peak Lat: Lat. Picco P1
            return (
                n1_lat_ms,  # Onset Lat (Lat. Picco N1)
                n1_amp,  # Onset Amp at Baseline (Amp. Picco N1)
                p1_lat_ms,  # Peak Lat (Lat. Picco P1)
                p1_amp,  # Peak Amp (Amp. Picco P1)
                final_amplitude_pp  # Final Amplitude (Peak-to-Peak N1-P1)
            )

        except Exception as e:
            # print(f"Analisi picco DWAVE fallita per {site_name}: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0

    def _analyze_cmap_peaks2(self, signal_data, site_name):
        """
        Analizza un segnale DWAVE per trovare il primo picco Negativo (N1)
        e il primo picco Positivo (P1) successivo, e calcola l'ampiezza Peak-to-Peak (N1-P1).
        """
        try:
            params = self.all_site_params[site_name]
            expected_lat_base = params['lat']
            noise_std_dev_setting = self.noise_spinbox.value()

            # --- 1. Definizione Finestra di Ricerca (ROI) ---
            # Cerchiamo in un intervallo stretto intorno alla latenza attesa
            search_start_ms = max(0, expected_lat_base - 1.0)
            search_end_ms = min(self.duration_ms, expected_lat_base + 3.0)  # Finestra stretta per N1-P1

            # Converti in indici
            search_start_idx = int(search_start_ms / self.duration_ms * self.time_points)
            search_end_idx = int(search_end_ms / self.duration_ms * self.time_points)

            if search_start_idx >= search_end_idx:
                return 0.0, 0.0, 0.0, 0.0, 0.0

            signal_roi = signal_data[search_start_idx:search_end_idx]

            # --- 2. Trova Picco Negativo (N1: La prima grande deflessione verso l'alto) ---

            # Calcolo del rumore per la soglia di rilevamento
            # Pre-search window per baseline (e.g., 1-2 ms prima dell'inizio della ricerca)
            base_end_idx = search_start_idx
            base_start_idx = max(0, base_end_idx - int(2.0 / self.duration_ms * self.time_points))

            baseline_window = signal_data[base_start_idx:base_end_idx]
            baseline_mean = np.mean(baseline_window)
            baseline_std = np.std(baseline_window)

            if baseline_std < 1e-6:
                baseline_std = noise_std_dev_setting

            # Soglia per N1 (2.5 volte il rumore, negativo)
            n1_threshold = baseline_mean - (2.5 * baseline_std)

            n1_global_idx = -1
            n1_amp = 0.0

            # Cerca il primo punto che supera la soglia negativa (Onset N1)
            onset_n1_idx = -1
            for i in range(search_start_idx, search_end_idx):
                if signal_data[i] < n1_threshold:
                    onset_n1_idx = i
                    break

            if onset_n1_idx == -1:
                # N1 non trovato, il segnale è probabilmente troppo piccolo
                return 0.0, 0.0, 0.0, 0.0, 0.0

                # Cerca il minimo (picco N1) dopo l'onset
            search_n1_window = signal_data[onset_n1_idx:search_end_idx]
            n1_local_idx_relative = np.argmin(search_n1_window)
            n1_global_idx = n1_local_idx_relative + onset_n1_idx
            n1_amp = signal_data[n1_global_idx]
            n1_lat_ms = self.time_vector[n1_global_idx]

            # --- 3. Trova Picco Positivo (P1: La prima grande deflessione verso il basso) ---

            # Soglia per P1 (2.5 volte il rumore, positivo)
            p1_threshold = baseline_mean + (2.5 * baseline_std)

            # Cerca il massimo (picco P1) DOPO il picco N1
            p1_global_idx = -1
            p1_amp = 0.0

            # Limita la ricerca P1 per non catturare deflessioni troppo tardive
            p1_search_end_idx = min(search_end_idx, n1_global_idx + int(2.0 / self.duration_ms * self.time_points))

            p1_search_roi = signal_data[n1_global_idx:p1_search_end_idx]

            if len(p1_search_roi) == 0:
                # Intervallo troppo piccolo o N1 troppo tardi
                return 0.0, 0.0, 0.0, 0.0, 0.0

            # Trova l'indice del primo picco che supera la soglia positiva
            p1_local_idx_start = -1
            for i in range(len(p1_search_roi)):
                if p1_search_roi[i] > p1_threshold:
                    p1_local_idx_start = i
                    break

            if p1_local_idx_start == -1:
                # P1 non trovato oltre la soglia
                return 0.0, 0.0, 0.0, 0.0, 0.0

            # Ora cerca il picco massimo nell'intervallo P1
            p1_search_peak_roi = p1_search_roi[p1_local_idx_start:]
            p1_local_idx_relative = np.argmax(p1_search_peak_roi)

            p1_global_idx = p1_local_idx_start + p1_local_idx_relative + n1_global_idx
            p1_amp = signal_data[p1_global_idx]
            p1_lat_ms = self.time_vector[p1_global_idx]

            # --- 4. Calcola Ampiezza Peak-to-Peak (N1-P1) ---
            final_amplitude_pp = p1_amp - n1_amp

            # --- 5. Risultati di output per la UI ---
            return (
                n1_lat_ms,  # Onset Lat (Lat. Picco N1)
                n1_amp,  # Onset Amp at Baseline (Amp. Picco N1)
                p1_lat_ms,  # Peak Lat (Lat. Picco P1)
                p1_amp,  # Peak Amp (Amp. Picco P1)
                final_amplitude_pp  # Final Amplitude (Peak-to-Peak N1-P1)
            )

        except Exception as e:
            print(f"Analisi picco DWAVE fallita per {site_name}: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0

    def _analyze_cmap_peaks_old(self, signal_data, site_name):
        """
        Analizza un segnale CMAP per trovare latenza di onset,
        latenza di picco e ampiezza (onset-to-peak).
        """
        try:
            params = self.all_site_params[site_name]
            expected_lat_ms = params['lat']
            noise_std_dev_setting = self.noise_spinbox.value()

            # --- 1. Definestra Finestra di Baseline (rumore) ---
            base_start_ms = max(2.0, expected_lat_ms - 10)
            base_end_ms = max(base_start_ms + 1, expected_lat_ms - 2)

            base_start_idx = int(base_start_ms / self.duration_ms * self.time_points)
            base_end_idx = int(base_end_ms / self.duration_ms * self.time_points)

            if base_start_idx >= base_end_idx:
                base_start_idx = 0
                base_end_idx = 10

            baseline_window = signal_data[base_start_idx:base_end_idx]
            baseline_mean = np.mean(baseline_window)
            baseline_std = np.std(baseline_window)

            if baseline_std < 1e-6:
                baseline_std = noise_std_dev_setting

            onset_threshold = baseline_mean - (1.5 * baseline_std)

            # --- 2. Definestra Finestra di Ricerca Picco ---
            search_start_ms = max(0, expected_lat_ms - 5)
            search_end_ms = min(self.duration_ms, expected_lat_ms + 15)
            search_start_idx = int(search_start_ms / self.duration_ms * self.time_points)
            search_end_idx = int(search_end_ms / self.duration_ms * self.time_points)

            if search_start_idx >= search_end_idx:
                return 0.0, 0.0, 0.0, 0.0, 0.0

            signal_roi = signal_data[search_start_idx:search_end_idx]

            # --- 3. Trova Picco Negativo ---
            neg_peak_local_idx = np.argmin(signal_roi)
            neg_peak_global_idx = neg_peak_local_idx + search_start_idx

            peak_lat_ms = self.time_vector[neg_peak_global_idx]
            peak_amp = signal_data[neg_peak_global_idx]

            # --- 4. Trova Onset Latency (lavorando all'indietro dal picco) ---
            onset_idx = neg_peak_global_idx
            for i in range(neg_peak_global_idx, base_start_idx, -1):
                if signal_data[i] > onset_threshold:
                    onset_idx = i
                    break

            if onset_idx == neg_peak_global_idx:
                onset_idx = search_start_idx

            onset_lat_ms = self.time_vector[onset_idx]
            onset_amp_at_baseline = baseline_mean

            # --- 5. Calcola Ampiezza Finale (Baseline-to-Peak) ---
            final_amplitude = onset_amp_at_baseline - peak_amp

            if final_amplitude < (2 * baseline_std) or final_amplitude < (noise_std_dev_setting * 2):
                return 0.0, 0.0, 0.0, 0.0, 0.0

            return onset_lat_ms, onset_amp_at_baseline, peak_lat_ms, peak_amp, final_amplitude

        except Exception as e:
            # print(f"Analisi picco fallita per {site_name}: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0

    def _update_peak_log_table(self, site_index, timestamp_str, onset_lat_ms, peak_lat_ms, amplitude):
        """
        Aggiorna una riga specifica nella tabella dei picchi con i nuovi 4 valori.
        """

        data = [
            timestamp_str,
            f"{onset_lat_ms:.2f}",
            f"{peak_lat_ms:.2f}",
            f"{amplitude:.1f}"
        ]

        for col_idx, text in enumerate(data):
            item = self.peak_log_table.item(site_index, col_idx)
            if not item:
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.peak_log_table.setItem(site_index, col_idx, item)

            if col_idx == 3:  # Colonna Ampiezza
                if amplitude < 100.0 and amplitude > 1e-6:
                    item.setForeground(QColor('red'))
                else:
                    item.setForeground(QColor('white'))

            item.setText(text)

    def _run_simulation_sweep(self):
        """
        Executes ONE single MEP stimulation sweep.
        """
        if not self.is_running:
            return

        # --- 0. Gestione Anomalia Tempo (Update tempo simulazione) ---
        current_time_ms = self.simulation_elapsed_time
        interval_s = self.stim_timer_interval_ms / 1000.0
        self.simulation_elapsed_time += interval_s

        # --- 1. Read values from controls
        num_pulses = self.pulse_spinbox.value()
        intensity_pct = self.intensity_spinbox.value()
        artifact_isi_ms = self.isi_spinbox.value()
        raster_gain_uV = self.raster_gain_spinbox.value()
        max_traces_per_sector = self.raster_traces_spinbox.value()
        trace_spacing = self.raster_trace_spacing
        stim_interval_ms = self.interval_spinbox.value()

        # --- Verifica attivazione Anomalia ---
        is_active_now = False
        if self.injector:
            is_active_now = self.injector.is_active
            if not is_active_now:
                # Check trigger (se non già attiva, prova ad attivarla)
                # Nota: apply_anomaly lo fa internamente se gli passiamo i dati,
                # ma qui dobbiamo farlo "a monte" se vogliamo attivare la UI.

                # TODO Simuliamo un passaggio dati fittizio per triggerare lo stato ####################################
                _ = self.injector.apply_anomaly(self.simulation_elapsed_time, np.zeros(10), np.zeros(10))
                is_active_now = self.injector.is_active

        # SEGNALE VISIVO: Se l'anomalia è appena partita
        if is_active_now and not self.prev_anomaly_active:
            print("DWAVE: Anomalia attivata! Abilito controlli.")
            self.comboAnswers.setEnabled(True)
            self.pushBtConfirm.setEnabled(True)
            self.anomaly_label.setStyleSheet("color: red; font-weight: bold;")
            self.anomaly_label.setText("ANOMALY DETECTED - ACTION REQUIRED")
            self.response_start_time = datetime.datetime.now()  # Start timer risposta

        self.prev_anomaly_active = is_active_now

        ## --- [Ciclo di simulazione] ---

        # Prendi il timestamp UNA VOLTA per sweep
        timestamp_str = datetime.datetime.now().strftime("%H:%M:%S")

        # Aggiorna il log di stimolazione
        if self.stim_log_widget:
            if "Start stimulation" in self.stim_log_widget.toPlainText():
                self.stim_log_widget.clear()
            log_entry = (
                f"<b><span style='color: white;'>Stim @ {timestamp_str}:</span></b> "
                f"<span style='color: gray;'>Pulses=</span>{num_pulses}, "
                f"<span style='color: gray;'>Int=</span>{intensity_pct}%, "
                f"<span style='color: gray;'>ISI=</span>{artifact_isi_ms:.2f}ms, "
                f"<span style='color: gray;'>Interval=</span>{stim_interval_ms}ms"
            )
            self.stim_log_widget.append(log_entry)

        # 2. Calculate scaling factors (invariato)
        facilitation_factor = self.facilitation_map.get(num_pulses, 0.0)
        intensity_factor = intensity_pct / 100.0
        final_amplitude_scale = facilitation_factor * intensity_factor

        # 3. Generate artifact train (invariato)
        artifact_scale_factor = intensity_factor
        total_stim_artifact = np.zeros_like(self.time_vector)
        for i in range(num_pulses):
            current_lat_ms = i * artifact_isi_ms
            if current_lat_ms < self.duration_ms:
                one_artifact = self._generate_stim_artifact(
                    current_lat_ms,
                    self.artifact_amp * artifact_scale_factor,
                    self.artifact_std
                )
                total_stim_artifact += one_artifact

        # 4. Generate and draw signals
        for i, site_name in enumerate(self.site_list):
            params = self.all_site_params[site_name]

            base_signal = self._generate_mep_signal(params, amplitude_scale=final_amplitude_scale)
            noise = self._generate_noise()

            # [1] Crea il segnale pulito per l'analisi
            clean_mep_signal = base_signal + noise

            # --- INIEZIONE ANOMALIA ---
            if self.injector:
                # Applica l'anomalia ai dati fisici se attiva
                clean_mep_signal = self.injector.apply_anomaly(
                    self.simulation_elapsed_time,
                    clean_mep_signal,
                    self.time_vector
                )

            # [2] Esegui l'analisi automatica dei picchi
            (onset_lat, onset_amp_baseline,
             peak_lat, peak_amp,
             final_amp) = self._analyze_cmap_peaks(clean_mep_signal, site_name)

            # [3] Aggiorna la tabella dei picchi
            self._update_peak_log_table(i, timestamp_str, onset_lat, peak_lat, final_amp)

            # [4] Aggiungi l'artefatto DOPO l'analisi
            final_signal = (clean_mep_signal + total_stim_artifact) # * - 1.0

            # --- A. Update Stacked Plot (Left) ---
            offset = (i + 1) * self.stacked_v_offset_uV
            stacked_signal = final_signal + offset
            self.stacked_curves[site_name].setData(self.time_vector, stacked_signal)

            # --- B. Update Peak Markers (Dots) ---
            if final_amp > 0:  # Solo se l'analisi ha trovato un picco valido
                # Dot 1: Onset
                # Dot 2: Negative Peak
                x_dots = [onset_lat, peak_lat]
                # Aggiungi l'offset alle coordinate Y
                # !! QUESTA È LA RIGA CORRETTA !!
                y_dots_with_offset = [onset_amp_baseline + offset, peak_amp + offset]
                self.stacked_peak_markers[site_name].setData(x_dots, y_dots_with_offset)
                if site_name in self.raster_peak_lines:
                    # peak_lat contiene la latenza del Picco Positivo (P1)
                    self.raster_peak_lines[site_name].setValue(peak_lat)
            else:
                # Se non ci sono picchi, cancella i "dot" vecchi
                self.stacked_peak_markers[site_name].clear()
                # Nascondi le linee se il picco non è rilevato
                if site_name in self.raster_peak_lines:
                    self.raster_peak_lines[site_name].setValue(0)  # Sposta a 0 (o fuori schermo se necessario)

            # --- C. Update Raster Plot (Right) (Invariato) ---
            curve_list = self.raster_curves[site_name]

            if len(curve_list) >= max_traces_per_sector:
                old_curve = curve_list.pop(0)
                self.raster_plot.removeItem(old_curve)

            scaled_data = final_signal / raster_gain_uV

            color = self.all_site_colors.get(site_name, 'w')
            raster_pen = pg.mkPen(color, width=1)

            new_curve = pg.PlotDataItem(self.time_vector, scaled_data, pen=raster_pen)
            self.raster_plot.addItem(new_curve)
            curve_list.append(new_curve)

            y_base_offset = i * max_traces_per_sector
            for j, curve in enumerate(curve_list):
                y_scroll_offset = j * trace_spacing
                curve.setPos(0, y_base_offset + y_scroll_offset)

            if site_name in self.tracker_widgets:
                tracker = self.tracker_widgets[site_name]
                tracker.update_data(self.time_vector, final_signal)
                # Imposta automaticamente la posizione della linea del tracker
                if final_amp > 0:
                    tracker.set_auto_v_line(onset_lat)
                else:
                    tracker.set_auto_v_line(0)

    def _update_timer_interval(self, interval_ms):
        self.stim_timer_interval_ms = interval_ms
        if self.is_running:
            self.sim_timer.setInterval(self.stim_timer_interval_ms)

    def start_stop(self):
        if self.is_running:
            self.is_running = False
            self.sim_timer.stop()
            self.start_stop_button.setText("Start Stimulation")
            self.start_stop_button.setStyleSheet("background-color: blue;")
            # Reset anomaly timer if needed or keep running? Usually reset on stop.
            # self.simulation_elapsed_time = 0
        else:
            self.is_running = True

            # Arm Anomaly Trigger on Start
            if self.injector:
                self.injector.reset()
                self.injector.arm_trigger(0.0)
                self.simulation_elapsed_time = 0.0  # Reset time counter
                self.prev_anomaly_active = False

            self.sim_timer.start(self.stim_timer_interval_ms)
            self.start_stop_button.setText("Stop Stimulation")
            self.start_stop_button.setStyleSheet("background-color: red;")
            self._run_simulation_sweep()

    def _load_anomalies(self):
        # ... (Invariato) ...
        if self.anomaly_path:
            print(f"Loading MEP anomaly from: {self.anomaly_path}")
            pass
        else:
            print("No anomaly specified, using standard MEP parameters.")

    def closeEvent(self, event):
        print("Closing DWAVE simulator...")
        self.simulation_closed.emit('DWAVE')  # Emetti segnale chiusura
        self.sim_timer.stop()
        super().closeEvent(event)


# --- Blocco di esecuzione principale (per test) ---
if __name__ == '__main__':
    # Assicurati che l'applicazione PyQt esista
    app = QApplication(sys.argv)

    # Crea la finestra principale
    # Scegli uno scenario: "upper_limbs", "lower_limbs", o "cranial_nerves"
    main_window = DwaveSimulator(scenario_name="spinalcord_sites")

    # Mostra la finestra
    main_window.show()

    # Esegui l'applicazione
    sys.exit(app.exec())