import sys
import time
import numpy as np
from pathlib import Path
import webbrowser as wb
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel,
                               QPushButton, QLineEdit, QMessageBox,
                               QComboBox, QFrame, QHBoxLayout, QSpinBox, QTabWidget, QScrollArea, QTableWidget,
                               QHeaderView, QGridLayout, QTableWidgetItem)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot, QThread
import pyqtgraph as pg

# -----------------------------------------------------------------------------
# Costanti di Simulazione
# -----------------------------------------------------------------------------
NUM_CHANNELS = 16
SAMPLES_PER_CHUNK = 10
INTERVAL_MS = 20
SAMPLE_RATE = SAMPLES_PER_CHUNK / (INTERVAL_MS / 1000.0)  # 500 Hz
VIRTUAL_PLOT_WIDTH_MM = 300

SWEEP_SPEEDS = {"15 mm/s": 15, "30 mm/s": 30, "60 mm/s": 60}
AMPLITUDE_SCALES = {
    "20 µV": 20, "50 µV": 50, "100 µV": 100,
    "200 µV": 200, "500 µV": 500, "1000 µV": 1000
}

# Tipi di Pattern EMG
ANOMALY_TYPES = ["A-Train (Neurotonic)", "B-Burst (Rhythmic)", "B-Spike (Intermittent)", "C-Train (Complex)"]


# -----------------------------------------------------------------------------
# WORKER (Generatore Dati + Pattern EMG Avanzati)
# -----------------------------------------------------------------------------
class EmgWorker(QObject):
    newData = Signal(object)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.running = False

        # Stato Anomalia
        self.anomaly_active = False
        self.anomaly_type = None
        self.anomaly_start_time = 0
        self.anomaly_channel = 0

    def trigger_anomaly(self, anomaly_type, channel_idx=0):
        """Attiva un pattern specifico."""
        self.anomaly_active = True
        self.anomaly_type = anomaly_type
        self.anomaly_start_time = time.perf_counter()
        self.anomaly_channel = channel_idx
        print(f"Injecting: {anomaly_type} on Ch{channel_idx + 1}")

    def stop_anomaly(self):
        self.anomaly_active = False

    @Slot()
    def run(self):
        self.running = True
        while self.running:
            t0 = time.perf_counter()

            # 1. Genera Rumore di Fondo (Free Running)
            data_chunk = np.random.normal(0, 2.0, (NUM_CHANNELS, SAMPLES_PER_CHUNK))

            # 2. Iniezione Pattern (se attivo)
            if self.anomaly_active:
                elapsed = t0 - self.anomaly_start_time
                t_chunk = np.linspace(elapsed, elapsed + (SAMPLES_PER_CHUNK / SAMPLE_RATE), SAMPLES_PER_CHUNK)
                signal_to_inject = np.zeros(SAMPLES_PER_CHUNK)

                # --- LOGICA DEI PATTERN ---
                if "A-Train" in self.anomaly_type:
                    freq = 150
                    amp = 150
                    envelope = np.minimum(elapsed * 10, 1.0)
                    signal_to_inject = np.sin(2 * np.pi * freq * t_chunk) * amp * envelope

                elif "B-Burst" in self.anomaly_type:
                    burst_rate = 0.2
                    cycle_pos = elapsed % burst_rate
                    if cycle_pos < 0.05:
                        noise_burst = np.random.normal(0, 50, SAMPLES_PER_CHUNK)
                        signal_to_inject = noise_burst

                elif "B-Spike" in self.anomaly_type:
                    if np.random.rand() < 0.05:
                        signal_to_inject = np.array([0, 50, 150, -100, -50, 0, 0, 0, 0, 0])
                        if len(signal_to_inject) != SAMPLES_PER_CHUNK:
                            signal_to_inject = np.resize(signal_to_inject, SAMPLES_PER_CHUNK)

                elif "C-Train" in self.anomaly_type:
                    high_noise = np.random.normal(0, 80, SAMPLES_PER_CHUNK)
                    slow_wave = np.sin(2 * np.pi * 30 * t_chunk) * 50
                    signal_to_inject = high_noise + slow_wave

                # Somma al canale specifico
                if self.anomaly_channel < NUM_CHANNELS:
                    data_chunk[self.anomaly_channel, :] += signal_to_inject

            self.newData.emit(data_chunk)

            process_time = time.perf_counter() - t0
            wait_time = (INTERVAL_MS / 1000.0) - process_time
            if wait_time > 0:
                time.sleep(wait_time)

        self.finished.emit()

    def stop(self):
        self.running = False


# -----------------------------------------------------------------------------
# WIDGET SNAPSHOT (Invariato)
# -----------------------------------------------------------------------------
class EventSnapshotWidget(QFrame):
    def __init__(self, title, data_snapshot, timestamp, trigger_idx=None, tresh_val=50):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet("background-color: #222; border: 1px solid #444; margin-bottom: 2px;")
        self.setFixedHeight(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        lbl_header = QLabel(f"<b>[{timestamp}]</b> {title}")
        lbl_header.setStyleSheet("color: #ff5555; border: none;")
        layout.addWidget(lbl_header)

        plot = pg.PlotWidget()
        plot.setBackground('#111')
        plot.setMouseEnabled(x=True, y=False)
        plot.hideAxis('left')

        plot.plot(data_snapshot, pen=pg.mkPen('r', width=1.0))

        if trigger_idx is not None:
            line_trg = pg.InfiniteLine(pos=trigger_idx, angle=90, pen=pg.mkPen('y', style=Qt.PenStyle.DashLine))
            plot.addItem(line_trg)
            line_tresh = pg.InfiniteLine(pos=tresh_val, angle=0, pen=pg.mkPen('y', style=Qt.PenStyle.DashLine))
            plot.addItem(line_tresh)

        layout.addWidget(plot)


# -----------------------------------------------------------------------------
# GUI PRINCIPALE
# -----------------------------------------------------------------------------
class EmgControl(QMainWindow):
    simulation_finished = Signal()

    def __init__(self, json_anomaly_path=None):
        super().__init__()
        self.setWindowTitle("EMG Pattern Generator & Monitor")
        self.resize(1280, 900)

        # --- Dati Simulazione ---
        self.worker = None
        self.thread = None
        self.is_running = False

        self.buffer_size = 5000
        self.data_buffer = np.zeros((NUM_CHANNELS, self.buffer_size))

        # Buffer dedicato ai marker (eventi)
        self.marker_buffer = np.zeros(self.buffer_size)

        self.ptr = 0
        self.curves = []
        self.channel_offsets = []
        self.sample_rate = SAMPLE_RATE

        # --- Trigger Logic ---
        self.trigger_threshold = 50.0
        self.last_trigger_time = 0
        self.pending_captures = []
        self.pre_trigger_sec = 0.3
        self.post_trigger_sec = 1.2
        self.last_trigger_active = False  # Flag per il marker grafico

        # --- Timer ---
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.update_sim_clock)
        self.sim_seconds = 0

        # --- GUI ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QHBoxLayout(main_widget)

        self.setup_left_panel()
        self.setup_right_panel()

        self.init_plots()
        self.start_thread()

    def setup_left_panel(self):
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        # --- 1. HEADER ROW ---
        header_layout = QHBoxLayout()
        self.lbl_timer = QLabel("00:00:00")
        self.lbl_timer.setStyleSheet("""
            QLabel { font-family: Consolas; font-size: 24px; color: #00ff00; 
            border: 2px solid #444; border-radius: 5px; padding: 5px; background-color: black; }
        """)
        self.lbl_timer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_timer.setFixedWidth(150)
        header_layout.addWidget(self.lbl_timer)
        header_layout.addStretch()

        btn_help = QPushButton("Help")
        btn_help.clicked.connect(self.show_help)
        header_layout.addWidget(btn_help)

        btn_exit = QPushButton("Exit")
        btn_exit.clicked.connect(self.close)
        btn_exit.setStyleSheet("background-color: #aa0000; color: white; font-weight: bold;")
        header_layout.addWidget(btn_exit)

        left_layout.addLayout(header_layout)

        # --- 2. CONTROLS ROW ---
        ctrl_layout = QHBoxLayout()

        self.combo_speed = QComboBox()
        self.combo_speed.addItems(SWEEP_SPEEDS.keys())
        self.combo_speed.setCurrentText("30 mm/s")
        self.combo_speed.currentTextChanged.connect(self.update_sweep_speed)
        ctrl_layout.addWidget(QLabel("Speed:"))
        ctrl_layout.addWidget(self.combo_speed)

        self.combo_amp = QComboBox()
        self.combo_amp.addItems(AMPLITUDE_SCALES.keys())
        self.combo_amp.setCurrentText("100 µV")
        self.combo_amp.currentTextChanged.connect(self.update_amplitude)
        ctrl_layout.addWidget(QLabel("Gain:"))
        ctrl_layout.addWidget(self.combo_amp)

        self.spin_thresh = QSpinBox()
        self.spin_thresh.setRange(10, 1000)
        self.spin_thresh.setValue(int(self.trigger_threshold))
        self.spin_thresh.setSingleStep(10)
        self.spin_thresh.valueChanged.connect(self.update_threshold)
        ctrl_layout.addWidget(QLabel("Thresh:"))
        ctrl_layout.addWidget(self.spin_thresh)

        left_layout.addLayout(ctrl_layout)

        # --- 3. ANOMALY CONTROL ROW ---
        anom_layout = QHBoxLayout()
        anom_layout.addWidget(QLabel("<b>Pattern Generator:</b>"))

        self.combo_anomaly = QComboBox()
        self.combo_anomaly.addItems(ANOMALY_TYPES)
        self.combo_anomaly.setStyleSheet("background-color: #444; color: white;")
        anom_layout.addWidget(self.combo_anomaly)

        self.btn_inject = QPushButton("INJECT")
        self.btn_inject.setCheckable(True)
        self.btn_inject.setStyleSheet("background-color: #d4aa00; color: black; font-weight: bold;")
        self.btn_inject.toggled.connect(self.toggle_anomaly_injection)
        anom_layout.addWidget(self.btn_inject)

        self.btn_freeze = QPushButton("FREEZE")
        self.btn_freeze.setCheckable(True)
        self.btn_freeze.clicked.connect(self.toggle_acquisition)
        self.btn_freeze.setStyleSheet("background-color: #008800; color: white; font-weight: bold;")
        anom_layout.addWidget(self.btn_freeze)

        left_layout.addLayout(anom_layout)

        # --- 4. GRAPHS AREA (MAIN + MARKER) ---
        # Usiamo un layout verticale per impilare i due plot
        graph_layout = QVBoxLayout()

        # A. Main EMG Plot
        self.plot_widget = pg.PlotWidget(title="Live EMG")
        self.plot_widget.showGrid(x=True, y=False, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setBackground('#000000')
        self.plot_widget.setMouseEnabled(x=False, y=False)

        # [MODIFICA]: Riabilitiamo l'asse sinistro per usarlo come etichette
        left_axis = self.plot_widget.getAxis('left')
        left_axis.show()
        left_axis.setPen('w')  # Colore linea asse (bianco)
        left_axis.setTextPen('w')  # Colore testo etichette (bianco)
        left_axis.setWidth(60)  # Larghezza fissa per allineamento

        self.scan_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('gray', width=2))
        self.plot_widget.addItem(self.scan_line)

        self.thresh_line_ui = pg.InfiniteLine(
            pos=self.trigger_threshold, angle=0, movable=True,
            pen=pg.mkPen('r', style=Qt.PenStyle.DashLine, width=1.0),
            label='Thresh={value:0.0f}uV', labelOpts={'position': 0.95, 'color': (200, 50, 50), 'movable': True}
        )
        self.thresh_line_ui.sigPositionChanged.connect(self.on_line_dragged)
        self.plot_widget.addItem(self.thresh_line_ui)

        # B. Marker Plot (Sync X)
        self.marker_plot = pg.PlotWidget()
        self.marker_plot.setMaximumHeight(80)  # Altezza fissa ridotta
        self.marker_plot.setBackground('#000000')
        self.marker_plot.setMouseEnabled(x=False, y=False)
        self.marker_plot.setXLink(self.plot_widget)  # Sincronizza asse X
        self.marker_plot.getAxis('left').hide()
        self.marker_plot.getAxis('left').setRange(0, 1.5)  # Scala fissa 0-1
        self.marker_plot.setYRange(0, 1.2)

        # Etichetta laterale per i marker
        lbl = pg.TextItem("Markers", anchor=(0, 0.5), color='y')
        lbl.setPos(0, 0.5)
        # self.marker_plot.addItem(lbl) # Opzionale

        self.marker_curve = self.marker_plot.plot(
            pen=pg.mkPen('y', width=1, stepMode=True))  # Step mode per segnali digitali
        self.marker_scan_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=1))
        self.marker_plot.addItem(self.marker_scan_line)

        graph_layout.addWidget(self.plot_widget)
        graph_layout.addWidget(self.marker_plot)

        left_layout.addLayout(graph_layout, stretch=10)
        self.main_layout.addWidget(left_container, stretch=7)

    def setup_right_panel(self):
        """Pannello destro con Tabs."""
        right_container = QFrame()
        right_container.setFrameStyle(QFrame.Shape.StyledPanel)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.right_tabs = QTabWidget()
        self.right_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #aaa; padding: 8px; }
            QTabBar::tab:selected { background: #555; color: white; border-bottom: 2px solid #00ccff; }
        """)

        # Tab 1: Events
        self.tab_events = QWidget()
        layout_events = QVBoxLayout(self.tab_events)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: #1e1e1e; border: none;")
        self.events_inner = QWidget()
        self.events_layout = QVBoxLayout(self.events_inner)
        self.events_layout.addStretch()
        self.events_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.events_inner)
        layout_events.addWidget(self.scroll_area)
        self.right_tabs.addTab(self.tab_events, "Auto Events")

        # Tab 2: Annotations
        self.tab_notes = QWidget()
        self.setup_annotations_tab()
        self.right_tabs.addTab(self.tab_notes, "Annotations")

        right_layout.addWidget(self.right_tabs)
        self.main_layout.addWidget(right_container, stretch=3)

    def setup_annotations_tab(self):
        layout = QVBoxLayout(self.tab_notes)
        self.note_table = QTableWidget(0, 2)
        self.note_table.setHorizontalHeaderLabels(["Time", "Note"])
        self.note_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.note_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.note_table.verticalHeader().setVisible(False)
        self.note_table.setStyleSheet("background-color: #222; color: white; gridline-color: #444;")
        layout.addWidget(self.note_table)

        input_group = QFrame()
        input_layout = QVBoxLayout(input_group)
        input_layout.setContentsMargins(0, 5, 0, 0)

        row_input = QHBoxLayout()
        self.txt_note = QLineEdit()
        self.txt_note.setPlaceholderText("Type annotation...")
        self.txt_note.setStyleSheet("padding: 5px; color: white; background-color: #333; border: 1px solid #555;")
        self.txt_note.returnPressed.connect(self.add_manual_note)
        btn_add = QPushButton("Add")
        btn_add.setStyleSheet("background-color: #006600; padding: 5px; font-weight: bold;")
        btn_add.clicked.connect(self.add_manual_note)
        row_input.addWidget(self.txt_note)
        row_input.addWidget(btn_add)
        input_layout.addLayout(row_input)

        grid_quick = QGridLayout()
        quick_notes = ["Baseline Stable", "Artifact", "Neurotonic", "EMG Silence", "Cautery", "Drill Noise"]
        row, col = 0, 0
        for note in quick_notes:
            btn = QPushButton(note)
            btn.setStyleSheet("background-color: #444; padding: 4px; font-size: 10px;")
            btn.clicked.connect(lambda checked, n=note: self.add_annotation(n))
            grid_quick.addWidget(btn, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        input_layout.addLayout(grid_quick)
        layout.addWidget(input_group)

    def add_manual_note(self):
        text = self.txt_note.text().strip()
        if text:
            self.add_annotation(text)
            self.txt_note.clear()

    def add_annotation(self, text):
        sim_time = self.lbl_timer.text()
        row = self.note_table.rowCount()
        self.note_table.insertRow(row)
        item_time = QTableWidgetItem(sim_time)
        item_time.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item_time.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item_note = QTableWidgetItem(text)
        self.note_table.setItem(row, 0, item_time)
        self.note_table.setItem(row, 1, item_note)
        self.note_table.scrollToBottom()
        self.plot_widget.setFocus()

        # Aggiunge un marker "pulso" al grafico marker
        # Inseriamo un valore alto nel buffer corrente
        # Poiché il buffer è aggiornato nel thread, modifichiamo una flag o iniettiamo direttamente
        # Qui usiamo un approccio grafico semplificato: modifichiamo l'ultimo chunk nel prossimo update
        self.last_trigger_active = True  # Questo farà scattare un '1' nel prossimo update_data

    def init_plots(self):
        self.plot_widget.clear()
        self.plot_widget.addItem(self.scan_line)
        self.plot_widget.addItem(self.thresh_line_ui)
        self.curves = []
        # Rimossa lista etichette testo, usiamo l'asse

        # Colori e Nomi per i 16 Canali
        # 0-7 Left (Cyan), 8-15 Right (Magenta)
        for i in range(NUM_CHANNELS):
            is_right = i >= 8
            color = '#FF00FF' if is_right else '#00FFFF'  # Magenta vs Cyan

            # Curva
            c = self.plot_widget.plot(pen=pg.mkPen(color, width=1))
            self.curves.append(c)

        self.update_amplitude(self.combo_amp.currentText())
        self.update_sweep_speed(self.combo_speed.currentText())

    def start_thread(self):
        self.thread = QThread()
        self.worker = EmgWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.newData.connect(self.update_data)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()
        self.is_running = True
        self.sim_timer.start(1000)

    # --- LOGICA INIEZIONE PATTERN ---
    def toggle_anomaly_injection(self, checked):
        if checked:
            pattern = self.combo_anomaly.currentText()
            chn = np.random.randint(0, NUM_CHANNELS)
            self.worker.trigger_anomaly(pattern, chn)
            self.btn_inject.setText(f"STOP {pattern.split()[0]}")
            self.btn_inject.setStyleSheet("background-color: #ff5500; color: white; font-weight: bold;")
            self.add_annotation(f"Injecting: {pattern} (Ch{chn + 1})")
        else:
            self.worker.stop_anomaly()
            self.btn_inject.setText("INJECT")
            self.btn_inject.setStyleSheet("background-color: #d4aa00; color: black; font-weight: bold;")
            self.add_annotation("Injection Stopped")

    # --- LOGICA TRIGGER & BUFFER ---
    @Slot(object)
    def update_data(self, data_chunk):
        if not self.is_running: return
        chunk_len = data_chunk.shape[1]

        # 1. Update Buffer EMG
        if self.ptr + chunk_len > self.buffer_size:
            part1 = self.buffer_size - self.ptr
            self.data_buffer[:, self.ptr:] = data_chunk[:, :part1]
            part2 = chunk_len - part1
            self.data_buffer[:, :part2] = data_chunk[:, part1:]
            self.ptr = part2
        else:
            self.data_buffer[:, self.ptr: self.ptr + chunk_len] = data_chunk
            self.ptr += chunk_len

        # 2. Update Buffer Markers (Sincronizzato col ptr)
        # Creiamo un chunk di marker
        m_chunk = np.zeros(chunk_len)

        # Logica: Se c'è iniezione attiva, marker = 0.5. Se trigger o annotazione, marker = 1.0
        base_val = 0.5 if self.btn_inject.isChecked() else 0.0
        m_chunk[:] = base_val

        # Se è scattato un trigger o annotazione (flag last_trigger_active)
        if self.last_trigger_active:
            m_chunk[:] = 1.0
            self.last_trigger_active = False  # Reset flag

        # Scrittura buffer circolare marker (stessa logica di data_buffer)
        prev_ptr = (self.ptr - chunk_len) % self.buffer_size
        if prev_ptr + chunk_len > self.buffer_size:
            part1 = self.buffer_size - prev_ptr
            self.marker_buffer[prev_ptr:] = m_chunk[:part1]
            part2 = chunk_len - part1
            self.marker_buffer[:part2] = m_chunk[part1:]
        else:
            self.marker_buffer[prev_ptr: prev_ptr + chunk_len] = m_chunk

        # 3. Aggiornamento Grafici Curve
        t_axis = np.arange(self.buffer_size) / self.sample_rate

        # Aggiorna Curve EMG
        for i in range(NUM_CHANNELS):
            self.curves[i].setData(t_axis, self.data_buffer[i] + self.channel_offsets[i])

        # Aggiorna Curva Marker
        # Per stepMode=True, x deve essere len(y)+1
        # t_axis_markers = np.append(t_axis, t_axis[-1] + (1/self.sample_rate))
        # Oppure usiamo standard plot per semplicità
        self.marker_curve.setData(t_axis, self.marker_buffer)

        # Aggiorna Scan Lines
        pos = self.ptr / self.sample_rate
        self.scan_line.setPos(pos)
        self.marker_scan_line.setPos(pos)

        # Trigger Check
        self.check_trigger(data_chunk)
        self.process_pending_captures(chunk_len)

    def check_trigger(self, chunk):
        if time.time() - self.last_trigger_time < 2.0: return
        max_val = np.max(np.abs(chunk))
        if max_val > self.trigger_threshold:
            self.last_trigger_time = time.time()
            self.last_trigger_active = True  # Attiva marker visivo

            chan_idx, _ = np.unravel_index(np.argmax(np.abs(chunk)), chunk.shape)
            wait_samps = int(self.post_trigger_sec * self.sample_rate)
            self.pending_captures.append({
                "channel": chan_idx, "trigger_ptr": self.ptr, "wait_samples": wait_samps
            })

    def process_pending_captures(self, samples_added):
        for task in self.pending_captures[:]:
            task["wait_samples"] -= samples_added
            if task["wait_samples"] <= 0:
                self.finalize_capture(task)
                self.pending_captures.remove(task)

    def finalize_capture(self, task):
        chn, trig_ptr = task["channel"], task["trigger_ptr"]
        pre = int(self.pre_trigger_sec * self.sample_rate)
        post = int(self.post_trigger_sec * self.sample_rate)
        indices = np.arange(trig_ptr - pre, trig_ptr + post)
        snapshot = np.take(self.data_buffer[chn], indices, mode='wrap')

        if self.btn_inject.isChecked():
            lbl = f"Evt {self.combo_anomaly.currentText().split()[0]} Ch{chn + 1}"
        else:
            lbl = f"Spont. Ch{chn + 1} (> {self.trigger_threshold:.0f}uV)"

        self.add_event_to_log(lbl, snapshot, trigger_idx=pre)

    # --- UI UPDATES ---
    def update_sweep_speed(self, text):
        if text not in SWEEP_SPEEDS: return
        speed = SWEEP_SPEEDS[text]
        time_s = VIRTUAL_PLOT_WIDTH_MM / speed
        self.buffer_size = int(time_s * self.sample_rate)

        # Reset Buffers
        self.data_buffer = np.zeros((NUM_CHANNELS, self.buffer_size))
        self.marker_buffer = np.zeros(self.buffer_size)
        self.ptr = 0

        self.plot_widget.setXRange(0, time_s, padding=0)
        self.marker_plot.setXRange(0, time_s, padding=0)

    def update_amplitude(self, text):
        val = AMPLITUDE_SCALES.get(text, 100)
        spacing = val * 2.5
        self.channel_offsets = [-i * spacing for i in range(NUM_CHANNELS)]

        # Aggiorna range Y
        self.plot_widget.setYRange(-(NUM_CHANNELS) * spacing, spacing)

        # [MODIFICA]: Rigenera le TICKS dell'asse sinistro
        ticks = []
        for i in range(NUM_CHANNELS):
            is_right = i >= 8
            side_lbl = "R" if is_right else "L"
            ch_num = (i % 8) + 1
            label_text = f"{side_lbl}-M{ch_num}"
            # Aggiungi alla lista: (valore, etichetta)
            ticks.append((self.channel_offsets[i], label_text))

        # setTicks richiede una lista di livelli: [major_ticks, minor_ticks, ...]
        self.plot_widget.getAxis('left').setTicks([ticks])

    def update_threshold(self, val):
        self.trigger_threshold = float(val)
        self.thresh_line_ui.setPos(self.trigger_threshold)

    def on_line_dragged(self):
        val = self.thresh_line_ui.value()
        if val < 0: val = 0
        self.trigger_threshold = val
        self.spin_thresh.blockSignals(True)
        self.spin_thresh.setValue(int(val))
        self.spin_thresh.blockSignals(False)

    def toggle_acquisition(self):
        self.is_running = not self.is_running
        if not self.is_running:  # FREEZE
            self.btn_freeze.setText("RUN")
            self.btn_freeze.setStyleSheet("background-color: #880000; color: white; font-weight: bold;")
            self.sim_timer.stop()
            self.add_annotation("Acquisition Paused")
        else:  # RUN
            self.btn_freeze.setText("FREEZE")
            self.btn_freeze.setStyleSheet("background-color: #008800; color: white; font-weight: bold;")
            self.sim_timer.start(1000)
            self.add_annotation("Acquisition Resumed")

    def add_event_to_log(self, title, data, trigger_idx=None):
        ts = datetime.now().strftime("%H:%M:%S")
        w = EventSnapshotWidget(title, data, ts, trigger_idx, self.trigger_threshold)
        self.events_layout.insertWidget(0, w)
        if self.events_layout.count() > 15:
            it = self.events_layout.takeAt(self.events_layout.count() - 2)
            if it.widget(): it.widget().deleteLater()

    def update_sim_clock(self):
        self.sim_seconds += 1
        h, rem = divmod(self.sim_seconds, 3600)
        m, s = divmod(rem, 60)
        self.lbl_timer.setText(f"{h:02}:{m:02}:{s:02}")

    def show_help(self):
        help_path = Path('help_files/Tut_Emg.pdf')
        if help_path.exists():
            wb.open_new(str(help_path))
        else:
            # print(f"Help file not found: {help_path}")
            QMessageBox.information(self, "Help file not found !!",
                                "Controls:\n- INJECT: Starts the selected pattern.\n- FREEZE: Stops monitoring.\n- Annotations: Use the right tab to log events.")

    def closeEvent(self, event):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        self.simulation_finished.emit()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    win = EmgControl()
    win.show()
    sys.exit(app.exec())