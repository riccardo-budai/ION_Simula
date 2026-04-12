import sys
import time
import json
import zmq
from collections import deque
import numpy as np

# PySide6
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel,
                             QHBoxLayout, QFrame, QScrollArea, QPushButton, QGroupBox,
                             QLCDNumber, QTextEdit, QCheckBox, QMessageBox, QInputDialog)
from PySide6.QtCore import QThread, Signal, QTimer, Qt, Slot
import pyqtgraph as pg
from pylsl import resolve_streams, StreamInlet, StreamInfo

from simula_localIOM import LocalGeneratorWorker

# NETWORK CONFIGURATION
TUTOR_IP = "localhost"       #"192.168.50.100"  # Assicurati che questo IP sia quello del PC Tutor
CMD_PORT = 5555
FEEDBACK_PORT = 5556
DB_PORT = 5557
CLIENT_ID = "STUDENT_TABLET"


# --- CLASSE CLIENT DATABASE (Per inviare risultati) ---
class DatabaseClient:
    def __init__(self, server_ip):
        self.server_ip = server_ip
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 2000)
        self.socket.setsockopt(zmq.LINGER, 0)
        try:
            self.socket.connect(f"tcp://{server_ip}:{DB_PORT}")
        except:
            pass

    def submit_result(self, user_id, score, details):
        try:
            req = {"cmd": "SUBMIT_RESULT", "user_id": user_id, "score": score, "details": details}
            self.socket.send_json(req)
            return self.socket.recv_json()
        except:
            return {"status": "ERROR", "msg": "Timeout"}

# --- ZMQ LISTENER (Solo Network Mode) ---
class ZMQListener(QThread):
    command_received = Signal(str, dict)

    def __init__(self, server_ip):
        super().__init__()
        self.server_ip = server_ip

    def run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        try:
            sock.connect(f"tcp://{TUTOR_IP}:{CMD_PORT}")
            sock.subscribe("CLIENT/")
            sock.subscribe("SYS/")
            while True:
                if sock.poll(100):
                    msg_string = sock.recv_string()
                    topic, json_str = msg_string.split(' ', 1)
                    data = json.loads(json_str)['data']
                    self.command_received.emit(topic, data)
        except Exception:
            pass


class FeedbackSender(QThread):
    def __init__(self, server_ip):
        super().__init__()
        self.server_ip = server_ip
        self.active_streams = []

    def set_active_streams(self, streams_list):
        self.active_streams = streams_list

    def run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUSH)
        try:
            sock.connect(f"tcp://{TUTOR_IP}:{FEEDBACK_PORT}")
            while True:
                report = {"sender": CLIENT_ID, "type": "CLIENT_REPORT", "timestamp": time.time(),
                          "data": {"viewing": self.active_streams}}
                sock.send_json(report)
                time.sleep(2.0)
        except Exception: pass


# --- WORKER LSL STANDARD (Network Mode) ---
class StreamWorker(QThread):
    data_signal = Signal(object)

    def __init__(self, stream_info):
        super().__init__()
        self.stream_info = stream_info
        self.running = True

    def run(self):
        inlet = StreamInlet(self.stream_info)
        while self.running:
            try:
                sample, timestamp = inlet.pull_sample(timeout=0.5)
                if timestamp:
                    self.data_signal.emit(sample)
                else:
                    time.sleep(0.01)
            except Exception:
                break

    def stop(self):
        self.running = False
        self.wait()


# --- WIDGET BASE ---
class BaseStreamWidget(QGroupBox):
    def __init__(self, stream_info, auto_start_lsl=True):
        super().__init__(f"{stream_info.name()} ({stream_info.type()})")
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; } QGroupBox::title { color: #00ccff; }")
        self.stream_info = stream_info
        self.layout_main = QHBoxLayout()
        self.setLayout(self.layout_main)

        self.ctrl_panel = QFrame()
        self.ctrl_panel.setFixedWidth(120)
        self.ctrl_layout = QVBoxLayout(self.ctrl_panel)
        self.ctrl_layout.addWidget(QLabel(f"{int(stream_info.nominal_srate())} Hz"))
        self.btn_inspect = QPushButton("Inspect")
        self.btn_inspect.setStyleSheet("background-color: #444; color: white;")
        self.ctrl_layout.addWidget(self.btn_inspect)
        self.ctrl_layout.addStretch()
        self.layout_main.addWidget(self.ctrl_panel)

        self.worker = None
        if auto_start_lsl:
            self.worker = StreamWorker(stream_info)
            self.worker.data_signal.connect(self.update_data)
            self.worker.start()

    def update_data(self, sample):
        pass

    def close_stream(self):
        if self.worker: self.worker.stop()


# --- WIDGET SPECIFICI ---
class GraphStreamWidget(BaseStreamWidget):
    def __init__(self, stream_info, auto_start_lsl=True):
        super().__init__(stream_info, auto_start_lsl)
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#111')
        self.curve = self.plot.plot(pen='y')
        self.layout_main.addWidget(self.plot)
        self.buffer = deque([0] * 500, maxlen=500)

    def update_data(self, sample):
        val = sample[0] if isinstance(sample, list) else sample
        self.buffer.append(val)
        self.curve.setData(self.buffer)


class NumericStreamWidget(BaseStreamWidget):
    def __init__(self, stream_info, auto_start_lsl=True):
        super().__init__(stream_info, auto_start_lsl)
        self.lcd = QLCDNumber()
        self.lcd.setStyleSheet("color: red; border: 1px solid red;")
        self.layout_main.addWidget(self.lcd)

    def update_data(self, sample):
        val = sample[0] if isinstance(sample, list) else sample
        self.lcd.display(int(val))


class TextStreamWidget(BaseStreamWidget):
    def __init__(self, stream_info, auto_start_lsl=True):
        super().__init__(stream_info, auto_start_lsl)
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.layout_main.addWidget(self.txt)

    def update_data(self, sample):
        self.txt.append(str(sample[0]))


# --- SCANNER LSL ---
class LSLScanner(QThread):
    stream_found = Signal(StreamInfo)

    def __init__(self):
        super().__init__()
        self.known_uids = set()
        self.running = True

    def run(self):
        while self.running:
            streams = resolve_streams(wait_time=1.0)
            for s in streams:
                if s.source_id() not in self.known_uids:
                    self.known_uids.add(s.source_id())
                    self.stream_found.emit(s)
            time.sleep(2.0)

    def stop(self):
        self.running = False
        self.wait()


# --- MAIN DASHBOARD ---
class ClientDashboard(QMainWindow):
    simulation_finished = Signal()

    def __init__(self, mode="NETWORK", extra_data=None, server_ip="localhost"):
        super().__init__()
        self.mode = mode
        self.extra_data = extra_data if extra_data else {}
        self.server_ip = server_ip # Salviamo l'IP
        self.active_widgets = {}

        self.setWindowTitle(f"ION-Sim Dashboard - [{self.mode}]")
        self.resize(1000, 800)
        self.setStyleSheet("QMainWindow { background-color: #222; color: white; } QLabel { color: white; }")

        # Setup UI
        central = QWidget();
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Header Row
        header_layout = QHBoxLayout()
        self.lbl_status = QLabel(f"Status: Initializing {mode} mode...")
        self.lbl_status.setStyleSheet("font-size: 14px; color: #00ccff; border: 1px solid #444; padding: 5px;")
        header_layout.addWidget(self.lbl_status)

        # PULSANTE SUBMIT (Solo in locale)
        if self.mode == "LOCAL":
            btn_finish = QPushButton("✅ Finish & Submit")
            btn_finish.setStyleSheet("background-color: #006600; color: white; font-weight: bold; padding: 5px;")
            btn_finish.clicked.connect(self.submit_and_close)
            header_layout.addWidget(btn_finish)

        main_layout.addLayout(header_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.streams_container = QWidget()
        self.streams_layout = QVBoxLayout(self.streams_container)
        self.streams_layout.addStretch()
        self.scroll_area.setWidget(self.streams_container)
        main_layout.addWidget(self.scroll_area)

        # LOGICA DI AVVIO
        if self.mode == "NETWORK":
            self.lbl_status.setText("Status: Scanning for Tutor streams...")
            self.scanner = LSLScanner()
            self.scanner.stream_found.connect(self.add_stream_widget)
            self.scanner.start()

            self.zmq = ZMQListener(self.server_ip)
            self.zmq.command_received.connect(self.handle_cmd)
            self.zmq.start()
            self.feedback = FeedbackSender(self.server_ip)
            self.feedback.start()

        elif self.mode == "LOCAL":
            mod_type = self.extra_data.get("module", "EEG")
            self.lbl_status.setText(f"Status: Local Training - {mod_type} (Generating Data...)")
            self.start_local_simulation(mod_type)

        # Sovrapposizione Blackout (Inizialmente nascosta)
        self.overlay = QLabel("⬛ BLACKOUT ⬛", self)
        self.overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay.setStyleSheet("background-color: black; color: #333; font-size: 40px;")
        self.overlay.hide()

    def resizeEvent(self, event):
        self.overlay.resize(self.size())
        super().resizeEvent(event)

    def start_local_simulation(self, module_type):
        dummy_info = StreamInfo(f"Local_{module_type}", module_type, 8, 500, 'float32', 'loc_id')
        if module_type == "VITALS":
            widget = NumericStreamWidget(dummy_info, auto_start_lsl=False)
        else:
            widget = GraphStreamWidget(dummy_info, auto_start_lsl=False)

        local_worker = LocalGeneratorWorker(module_type, self.extra_data)
        local_worker.data_signal.connect(widget.update_data)
        widget.worker = local_worker
        local_worker.start()

        self.streams_layout.insertWidget(0, widget)
        self.active_widgets["local"] = widget

    @Slot(StreamInfo)
    def add_stream_widget(self, info):
        stype = info.type()
        if stype in ['EEG', 'SEP', 'MEP']:
            w = GraphStreamWidget(info)
        elif stype == 'VITALS':
            w = NumericStreamWidget(info)
        elif stype == 'Markers':
            w = TextStreamWidget(info)
        else:
            w = GraphStreamWidget(info)

        self.streams_layout.insertWidget(0, w)
        self.active_widgets[info.source_id()] = w
        if hasattr(self, 'feedback'):
            names = [wid.stream_info.name() for wid in self.active_widgets.values()]
            self.feedback.set_active_streams(names)

    def handle_cmd(self, topic, data):
        # GESTIONE COMANDI REMOTI
        if topic == "CLIENT/MSG":
            QMessageBox.information(self, "Tutor Message", data.get("text"))

        elif topic == "CLIENT/UX":
            action = data.get("action")
            print(f"UX Command Received: {action}")  # Debug
            if action == "BLACKOUT":
                self.overlay.show()
                self.overlay.raise_()
            elif action == "RESET":
                self.overlay.hide()

    def submit_and_close(self):
        """Invia i risultati al Tutor e chiude."""
        is_sandbox = self.extra_data.get("is_sandbox", False)

        # Simula un punteggio (in un'app reale, questo verrebbe dal calcolo dell'errore sui marker)
        score, ok = QInputDialog.getInt(self, "Self-Evaluation", "Estimate your interpretation confidence (0-100):", 80,
                                        0, 100)
        if not ok: return

        if not is_sandbox:
            db_client = DatabaseClient(self.server_ip)
            uid = self.extra_data.get("user_id")
            mod = self.extra_data.get("module", "Unknown")

            resp = db_client.submit_result(uid, score, f"Module: {mod} (Local)")

            if resp.get("status") == "OK":
                QMessageBox.information(self, "Success", "Result saved to Tutor Database!")
            else:
                QMessageBox.warning(self, "Warning", "Could not save result (Connection Lost).")
        else:
            QMessageBox.information(self, "Sandbox", "Simulation ended. (No data saved)")

        self.close()  # Questo attiverà closeEvent -> simulation_finished

    def closeEvent(self, event):
        if hasattr(self, 'scanner'): self.scanner.stop()
        for w in self.active_widgets.values(): w.close_stream()
        event.accept()
        self.simulation_finished.emit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ClientDashboard(server_ip="localhost")
    win.show()
    sys.exit(app.exec())
