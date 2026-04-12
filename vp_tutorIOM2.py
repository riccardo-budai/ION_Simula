"""
    vp_tutorIOM.py
    Virtual Patient Tutor Module to be imported in mainIOM_EN2.py.
    Includes Non-Blocking ZMQ Polling for clean shutdown.
"""

import sys
import json
import time
import zmq
import threading
from datetime import datetime

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QTableWidget,
                             QTableWidgetItem, QHeaderView, QGroupBox, QLineEdit,
                             QTextEdit, QMessageBox)
from PySide6.QtCore import Signal, QObject, Slot, Qt

from database_managerIOM_sqlt3 import DatabaseManager
from ai_tutorIOM_EN import LearningAgent

########################################################################################################################
class TutorDBService:
    def __init__(self, db_path, port=5557):
        self.db_path = db_path
        self.port = port
        self.running = False
        self.thread = None

        # Inizializza DB e AI (locale al PC Tutor)
        self.db_mgr = DatabaseManager(db_path)
        self.ai_agent = LearningAgent(self.db_mgr)
        self.is_simulation_active = False  # NUOVA VARIABILE
        self.current_scenario_info = "Waiting for Tutor..."

    def set_simulation_status(self, active: bool, info: str = ""):
        """Chiamato dal Main quando si avvia/ferma uno scenario"""
        self.is_simulation_active = active
        self.current_scenario_info = info
        print(f"[DB SERVICE] Simulation Status: {'ACTIVE' if active else 'IDLE'} ({info})")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"[DB SERVICE] Avviato su porta {self.port}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _run_server(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)  # REP = Risponde alle richieste
        socket.bind(f"tcp://*:{self.port}")

        while self.running:
            try:
                # 1. Ricevi Richiesta dal Tablet
                # Usa poll per permettere shutdown pulito
                if socket.poll(500):
                    msg = socket.recv_json()
                    response = self._process_request(msg)
                    socket.send_json(response)
            except Exception as e:
                print(f"[DB SERVICE ERROR] {e}")

    def _process_request(self, req):
        """Gestisce la logica di business"""
        cmd = req.get("cmd")

        # --- LOGIN STUDENTE ---
        if cmd == "LOGIN":
            # Cerca lo studente per nome/cognome
            users = self.db_mgr.get_all_subjects()
            found = None
            for u in users:
                if u.firstname.upper() == req["name"].upper() and u.lastname.upper() == req["surname"].upper():
                    found = u
                    break
            if found:
                return {
                    "status": "OK",
                    "user_id": found.idcode,
                    "level": found.level,
                    # AGGIUNGIAMO LO STATO DELLA SIMULAZIONE ALLA RISPOSTA
                    "sim_active": self.is_simulation_active,
                    "sim_info": self.current_scenario_info
                }
            else:
                return {"status": "ERROR", "msg": "Utente non trovato"}

        elif cmd == "CHECK_STATUS":
            return {
                "status": "OK",
                "sim_active": self.is_simulation_active,
                "sim_info": self.current_scenario_info
            }

        # --- RICHIESTA AIUTO/ESERCIZIO (AI TUTOR) ---
        elif cmd == "GET_AI_TASK":
            user_id = req["user_id"]

            # Chiedi all'AI cosa fare
            rec = self.ai_agent.get_next_recommendation(user_id)

            # Se l'AI suggerisce uno scenario specifico, recuperiamo i dettagli
            scenario_config = {}
            if rec["suggested_scenario_id"]:
                # Esempio: recupera i parametri JSON dell'anomalia dal DB
                # Qui simulo un config di ritorno per il generatore locale
                scenario_config = {
                    "module": "SEP",
                    "anomaly_type": rec["suggested_scenario_id"],
                    "difficulty": rec["difficulty_adjustment"],
                    "noise_level": 0.5 if rec["difficulty_adjustment"] == "HARD" else 0.1
                }

            return {
                "status": "OK",
                "message": rec["message"],
                "config": scenario_config
            }

        # --- SALVATAGGIO RISULTATO ESERCITAZIONE ---
        elif cmd == "SUBMIT_RESULT":
            user_id = req["user_id"]
            score = req["score"]
            details = req["details"]

            # Salva nel DB (Tabella behavior logs o session_decisions)
            self.ai_agent.log_behavior(user_id, "EXERCISE_COMPLETE", details)

            # Qui potresti inserire logica per aggiornare session_decisions se necessario

            return {"status": "OK"}

        return {"status": "ERROR", "msg": "Comando sconosciuto"}


########################################################################################################################
# --- ZMQ BACKEND (Non-Blocking Version) ---
class TutorBackend(QObject):
    report_received = Signal(str, dict)
    log_message = Signal(str)

    def __init__(self, cmd_port=5555, feedback_port=5556):
        super().__init__()
        self.cmd_port = cmd_port
        self.feedback_port = feedback_port
        self.ctx = zmq.Context()
        self.running = False
        self.thread = None
        self.pub = None
        self.pull = None

    def start(self):
        """Avvia il backend ZMQ se non è già attivo."""
        if self.running: return

        try:
            # Setup Sockets
            self.pub = self.ctx.socket(zmq.PUB)
            self.pub.bind(f"tcp://*:{self.cmd_port}")

            self.pull = self.ctx.socket(zmq.PULL)
            self.pull.bind(f"tcp://*:{self.feedback_port}")

            # Flag di controllo
            self.running = True

            # Avvia thread in modalità Daemon
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()
            self.log_message.emit("ZMQ Backend Started.")

        except Exception as e:
            self.log_message.emit(f"ZMQ Start Error: {e}")

    def stop(self):
        """Ferma il thread e pulisce il contesto ZMQ in modo sicuro."""
        self.running = False  # Segnale al loop di fermarsi

        # Attendi che il thread finisca il ciclo corrente
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)  # Aspetta max 1 secondo

        # Chiudi le socket esplicitamente
        try:
            if self.pub:
                self.pub.close(linger=0)
            if self.pull:
                self.pull.close(linger=0)
            if self.ctx:
                self.ctx.term()
        except Exception as e:
            print(f"ZMQ Cleanup Error: {e}")

    def _listen_loop(self):
        """
        Ciclo di ascolto NON BLOCCANTE usando Poller.
        Permette di uscire rapidamente dal loop quando self.running diventa False.
        """
        poller = zmq.Poller()
        poller.register(self.pull, zmq.POLLIN)

        while self.running:
            try:
                # Controlla se ci sono dati per 100ms
                socks = dict(poller.poll(100))

                if self.pull in socks and socks[self.pull] == zmq.POLLIN:
                    # C'è un messaggio, leggilo (non bloccherà perché poll ha detto ok)
                    msg = self.pull.recv_json()
                    sender = msg.get("sender", "?")
                    data = msg.get("data", {})

                    self.report_received.emit(sender, data)
                    log_str = f"<< RECV from [{sender}]: {json.dumps(data)}"
                    self.log_message.emit(log_str)

            except zmq.ContextTerminated:
                # Il contesto è stato chiuso, usciamo
                break
            except Exception as e:
                # Errori temporanei o di polling
                pass

        print("TutorBackend thread stopped clean.")

    def send_cmd(self, topic, payload):
        """Invia un comando JSON sul canale PUB."""
        if not self.running or not self.pub: return
        try:
            msg = {"ts": time.time(), "data": payload}
            json_str = json.dumps(msg)
            self.pub.send_string(f"{topic} {json_str}")
            self.log_message.emit(f">> SENT [{topic}]: {payload}")
        except zmq.ZMQError:
            pass


# --- TUTOR GUI WINDOW ---
class TutorControlWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ION-Sim: TUTOR CONTROL STATION (Integrated)")
        self.resize(700, 850)
        self.setStyleSheet("""
            QMainWindow { background-color: #2e2e2e; color: white; }
            QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QTextEdit { background-color: #1e1e1e; color: #00ff00; font-family: Consolas, Monospace; font-size: 11px; }
            QPushButton { border-radius: 4px; font-weight: bold; }
        """)

        # Inizializza Backend
        self.backend = TutorBackend()
        self.backend.report_received.connect(self.update_dashboard)
        self.backend.log_message.connect(self.append_log)

        self.devices = {}

        # Costruisce l'interfaccia
        self.setup_ui()

        # Avvia il backend ZMQ
        self.backend.start()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        # ==========================================
        # ZONE 1: PATIENT CONTROL (GENERATOR)
        # ==========================================
        box_gen = QGroupBox("🚑 PATIENT CONTROL (Virtual Patient)")
        box_gen.setStyleSheet("QGroupBox { border: 1px solid #00ff00; color: #00ff00; }")
        layout_gen = QHBoxLayout()

        btn_norm = QPushButton("Status: Normal")
        btn_norm.clicked.connect(lambda: self.backend.send_cmd("GEN/VITALS", {"hr": 70, "spo2": 99, "map": 85}))
        btn_norm.setStyleSheet("background-color: #004400; padding: 15px; color: white;")

        btn_brady = QPushButton("⚡ Induce Bradycardia")
        btn_brady.clicked.connect(lambda: self.backend.send_cmd("GEN/VITALS", {"hr": 35, "spo2": 92, "map": 50}))
        btn_brady.setStyleSheet("background-color: #664400; padding: 15px; color: white;")

        btn_arrest = QPushButton("☠️ CARDIAC ARREST")
        btn_arrest.clicked.connect(lambda: self.backend.send_cmd("GEN/VITALS", {"hr": 0, "spo2": 0, "map": 0}))
        btn_arrest.setStyleSheet("background-color: #880000; padding: 15px; font-weight: bold; color: white;")

        layout_gen.addWidget(btn_norm)
        layout_gen.addWidget(btn_brady)
        layout_gen.addWidget(btn_arrest)
        box_gen.setLayout(layout_gen)
        main_layout.addWidget(box_gen)

        # --- NEW: STIMULATION CONTROLS ---
        # Aggiungiamo pulsanti rapidi per testare i trigger SEP/MEP
        box_stim = QGroupBox("⚡ STIMULATION TRIGGERS")
        box_stim.setStyleSheet("QGroupBox { border: 1px solid #ffaa00; color: #ffaa00; }")
        layout_stim = QHBoxLayout()

        btn_stim_sep = QPushButton("Trigger SEP")
        btn_stim_sep.clicked.connect(lambda: self.backend.send_cmd("STIM/SEP", {"intensity": 20, "loc": "UL"}))
        btn_stim_sep.setStyleSheet("background-color: #553300; color: white; padding: 10px;")

        btn_stim_mep = QPushButton("Trigger MEP")
        btn_stim_mep.clicked.connect(lambda: self.backend.send_cmd("STIM/MEP", {"intensity": 100, "loc": "C3-C4"}))
        btn_stim_mep.setStyleSheet("background-color: #553300; color: white; padding: 10px;")

        layout_stim.addWidget(btn_stim_sep)
        layout_stim.addWidget(btn_stim_mep)
        box_stim.setLayout(layout_stim)
        main_layout.addWidget(box_stim)

        # ==========================================
        # ZONE 2: CLASSROOM CONTROL (CLIENT)
        # ==========================================
        box_class = QGroupBox("🎓 STUDENT COMMUNICATION (Client)")
        box_class.setStyleSheet("QGroupBox { border: 1px solid #00ccff; color: #00ccff; }")
        layout_class = QVBoxLayout()

        # Riga Messaggi
        row_msg = QHBoxLayout()
        self.txt_msg = QLineEdit()
        self.txt_msg.setPlaceholderText("Type message for students...")
        self.txt_msg.setStyleSheet("padding: 5px; color: #00ccff; border: 1px solid #555; background-color: #222;")

        btn_send_msg = QPushButton("Send Message")
        btn_send_msg.clicked.connect(self.send_student_message)
        btn_send_msg.setStyleSheet("background-color: #005577; padding: 5px; color: white;")

        row_msg.addWidget(self.txt_msg)
        row_msg.addWidget(btn_send_msg)
        layout_class.addLayout(row_msg)

        # Riga Comandi UX
        row_ux = QHBoxLayout()
        btn_blackout = QPushButton("Screen Blackout")
        btn_blackout.clicked.connect(lambda: self.backend.send_cmd("CLIENT/UX", {"action": "BLACKOUT"}))
        btn_blackout.setStyleSheet("background-color: #444; color: white; padding: 8px;")

        btn_reset_view = QPushButton("Reset View")
        btn_reset_view.clicked.connect(lambda: self.backend.send_cmd("CLIENT/UX", {"action": "RESET"}))
        btn_reset_view.setStyleSheet("background-color: #444; color: white; padding: 8px;")

        row_ux.addWidget(btn_blackout)
        row_ux.addWidget(btn_reset_view)
        layout_class.addLayout(row_ux)
        box_class.setLayout(layout_class)
        main_layout.addWidget(box_class)

        # ==========================================
        # ZONE 3: NETWORK MONITORING
        # ==========================================
        main_layout.addWidget(QLabel("📡 NETWORK STATUS"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "Type", "Activity", "Last Seen"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet("background-color: #222; gridline-color: #444; color: white;")
        main_layout.addWidget(self.table)

        # ==========================================
        # ZONE 4: SYSTEM LOG & CONTROLS
        # ==========================================
        box_log = QGroupBox("📝 SYSTEM LOG & CONTROLS")
        box_log.setStyleSheet("QGroupBox { border: 1px solid #aaaaaa; color: #aaaaaa; }")
        layout_log = QVBoxLayout()

        # Area Log
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMinimumHeight(150)
        layout_log.addWidget(self.txt_log)

        # Bottoni Controllo
        layout_btns = QHBoxLayout()

        btn_help = QPushButton("❓ Help")
        btn_help.clicked.connect(self.show_help)
        btn_help.setStyleSheet("background-color: #555; color: white; padding: 8px;")

        btn_clear = QPushButton("🧹 Clear Log")
        btn_clear.clicked.connect(self.txt_log.clear)
        btn_clear.setStyleSheet("background-color: #555; color: white; padding: 8px;")

        # Tasto Hide (NON Exit) perché siamo integrati
        btn_hide = QPushButton("🔽 Hide Window")
        btn_hide.clicked.connect(self.hide)
        btn_hide.setStyleSheet("background-color: #880000; color: white; padding: 8px;")

        layout_btns.addWidget(btn_help)
        layout_btns.addWidget(btn_clear)
        layout_btns.addStretch()
        layout_btns.addWidget(btn_hide)

        layout_log.addLayout(layout_btns)
        box_log.setLayout(layout_log)

        main_layout.addWidget(box_log)

    # --- FUNZIONI GUI ---
    def send_student_message(self):
        msg = self.txt_msg.text()
        if msg:
            self.backend.send_cmd("CLIENT/MSG", {"text": msg, "level": "info"})
            self.txt_msg.clear()

    # --- METODO NUOVO: CARICA CONFIGURAZIONE ---
    def load_simulation_config(self, config_data):
        """
        Invia la configurazione JSON al Virtual Generator per avviare gli stream corretti.
        Chiamato da mainIOM_EN2.py quando l'utente conferma il Setup.
        """
        # Invia il comando SYS/LOAD_CONFIG con il dizionario di setup
        self.backend.send_cmd("SYS/LOAD_CONFIG", config_data)

        # Aggiorna il log visivo del Tutor
        surg_type = config_data.get('surgery_type', 'Unknown')
        mods = len(config_data.get('active_modules', []))
        self.append_log(f"-> CONFIG SENT: {surg_type} ({mods} modules)")

        QMessageBox.information(self, "System",
                                "Setup sent to Virtual Patient Generator.\nWait for 'STATUS_REPORT' in the log.")

    @Slot(str, dict)
    def update_dashboard(self, sender, data):
        self.devices[sender] = {"data": data, "ts": datetime.now().strftime("%H:%M:%S")}
        self.refresh_table()

    @Slot(str)
    def append_log(self, message):
        """Appends a timestamped message to the log window."""
        ts = datetime.now().strftime("[%H:%M:%S]")
        self.txt_log.append(f"{ts} {message}")
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def refresh_table(self):
        self.table.setRowCount(len(self.devices))
        for i, (sender, info) in enumerate(self.devices.items()):
            role = "GENERATOR" if "active_lsl" in info["data"] else "STUDENT"
            # Se siamo il generatore, mostra quanti stream LSL sono attivi
            status = str(info["data"].get("viewing", "Running"))
            if "active_lsl" in info["data"]:
                n_streams = len(info["data"]["active_lsl"])
                status = f"Active ({n_streams} Streams)"
                # print(f"{time.time()}: active_lsl = {status}")

            self.table.setItem(i, 0, QTableWidgetItem(sender))
            self.table.setItem(i, 1, QTableWidgetItem(role))
            self.table.setItem(i, 2, QTableWidgetItem(status))
            self.table.setItem(i, 3, QTableWidgetItem(info["ts"]))

    def show_help(self):
        msg = """
        <b>ION-Sim Tutor Control Station</b><br><br>
        1. <b>Patient Control:</b> Send Vitals commands.<br>
        2. <b>Stimulation:</b> Trigger Evoked Potentials.<br>
        3. <b>Student Communication:</b> Send messages/blackout.<br>
        4. <b>Config:</b> Automatically loads setup from Main App.<br>
        """
        QMessageBox.information(self, "Help", msg)

    # --- GESTIONE CHIUSURA SICURA ---
    def force_shutdown(self):
        """Metodo chiamato ESPLICITAMENTE dal Main Window (mainIOM_EN2.py)"""
        if self.backend:
            print("Forcing Tutor Backend Shutdown...")
            self.backend.stop()

    def closeEvent(self, event):
        """
        Intercetta la chiusura della finestra.
        In modalità integrata, la 'X' nasconde solo la finestra.
        """
        event.ignore()
        self.hide()


# --- BLOCCO TEST INDIPENDENTE ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Se lanciato da solo, modifichiamo closeEvent per chiudere davvero
    win = TutorControlWindow()
    win.closeEvent = lambda event: (win.force_shutdown(), event.accept())
    win.show()
    sys.exit(app.exec())
