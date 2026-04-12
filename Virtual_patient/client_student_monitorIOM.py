import sys
import zmq
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel,
                             QPushButton, QLineEdit, QMessageBox,
                             QGroupBox, QComboBox, QFrame, QHBoxLayout)
from PySide6.QtCore import Qt, QTimer

# --- IMPORT MODULI GENERICI ---
from client_lsl_monitorIOM import ClientDashboard

from emgIOM import EmgControl
from eegAuxIOM4 import EEGControlWindow
from mepIOM_EN2 import MepasSimulator
from baepIOM_EN import BaepSimulator
from vepIOM import VepSimulator
from anesthesia_IOM_EN import AnesthesiaSimulator
from ECoGIOM3 import ECoGControlWindow

# Aggiungi qui gli altri import man mano che porti i file sul tablet
# from sepIOMmeg import SepSimulator

# from eegAuxIOM4 import EEGControlWindow

# CONFIGURAZIONE RETE
TUTOR_IP = "localhost"  # "192.168.50.100"
DB_PORT = 5557


class DatabaseClient:
    def __init__(self, server_ip):
        self.server_ip = server_ip
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 2000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.connected = False
        try:
            self.socket.connect(f"tcp://{server_ip}:{DB_PORT}")
            self.connected = True
        except:
            self.connected = False

    def login(self, name, surname):
        try:
            self.socket.send_json({"cmd": "LOGIN", "name": name, "surname": surname})
            return self.socket.recv_json()
        except zmq.error.Again:
            self._reset_socket()
            return {"status": "OFFLINE"}
        except Exception as e:
            return {"status": "ERROR", "msg": str(e)}

    def get_ai_recommendation(self, user_id):
        try:
            self.socket.send_json({"cmd": "GET_AI_TASK", "user_id": user_id})
            return self.socket.recv_json()
        except:
            return None

    def _reset_socket(self):
        self.socket.close()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 2000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.server_ip}:{DB_PORT}")

########################################################################################################################
class IonSimLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db_client = DatabaseClient(TUTOR_IP)
        self.current_user_id = None
        self.is_sandbox = False

        self.setWindowTitle("ION-Sim Learning Hub")
        self.resize(500, 450)
        self.setStyleSheet("""
            QMainWindow { background-color: #2e2e2e; color: white; }
            QLineEdit { padding: 8px; border: 1px solid #555; border-radius: 4px; background: #1e1e1e; color: white; }
            QPushButton { padding: 10px; border-radius: 5px; font-weight: bold; }
            QGroupBox { border: 1px solid #444; margin-top: 10px; padding-top: 15px; }
            QGroupBox::title { color: #aaa; subcontrol-origin: margin; left: 10px; padding: 0 3px; }
        """)

        cw = QWidget()
        self.setCentralWidget(cw)
        self.main_layout = QVBoxLayout(cw)

        # HEADER
        lbl_title = QLabel("ION-Sim User Access")
        lbl_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #00ccff;")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(lbl_title)

        # 1. LOGIN
        self.group_login = QGroupBox("1. Database Connection")
        lay_login = QVBoxLayout()
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("Surname")
        self.txt_surname = QLineEdit()
        self.txt_surname.setPlaceholderText("Name")
        self.btn_login = QPushButton("Login to Tutor Server")
        self.btn_login.setStyleSheet("background-color: #005577; color: white;")
        self.btn_login.clicked.connect(self.perform_login)
        lay_login.addWidget(self.txt_name)
        lay_login.addWidget(self.txt_surname)
        lay_login.addWidget(self.btn_login)
        self.group_login.setLayout(lay_login)
        self.main_layout.addWidget(self.group_login)

        # 2. TRAINING
        self.group_train = QGroupBox("2. Training Area")
        self.group_train.setEnabled(False)
        lay_train = QVBoxLayout()
        self.lbl_ai_info = QLabel("Waiting for AI...")
        self.lbl_ai_info.setStyleSheet("color: #00ff00; font-style: italic;")
        self.btn_ai_start = QPushButton("Start AI Recommended Scenario")
        self.btn_ai_start.setStyleSheet("background-color: #006600;")
        self.btn_ai_start.clicked.connect(self.start_ai_mode)

        self.frame_manual = QFrame()
        lay_manual = QVBoxLayout(self.frame_manual)
        lay_manual.addWidget(QLabel("Manual Select (Specific Module):"))

        self.combo_modules = QComboBox()
        # Nomi devono corrispondere alla logica nel metodo launch()
        self.combo_modules.addItems(["EEG",
                                     "ECoG",
                                     "EMG",
                                     "SEP UPPER LIMBS",
                                     "SEP LOWER LIMBS",
                                     "MEP UPPER LIMBS",
                                     "MEP LOWER LIMBS",
                                     "MEP CRANIAL NERVES",
                                     "BAEP",
                                     "VEP",
                                     "ANESTHESIA"])
        self.combo_modules.setStyleSheet("padding: 5px; background: #444444;")

        self.btn_manual = QPushButton("Start Simulation")
        self.btn_manual.clicked.connect(self.start_manual_mode)
        lay_manual.addWidget(self.combo_modules)
        lay_manual.addWidget(self.btn_manual)

        lay_train.addWidget(self.lbl_ai_info)
        lay_train.addWidget(self.btn_ai_start)
        lay_train.addSpacing(10)
        lay_train.addWidget(self.frame_manual)
        self.group_train.setLayout(lay_train)
        self.main_layout.addWidget(self.group_train)

        # 3. LIVE
        self.btn_live = QPushButton("3. Join Live Classroom")
        self.btn_live.setStyleSheet("background-color: #884400; color: gray;")
        self.btn_live.setEnabled(False)
        self.btn_live.clicked.connect(self.start_live_mode)
        self.main_layout.addWidget(self.btn_live)
        self.main_layout.addStretch()

        footer_layout = QHBoxLayout()
        self.btn_help = QPushButton("Help")
        self.btn_help.setStyleSheet("background-color: #444; color: white; padding: 10px;")
        self.btn_help.clicked.connect(self.show_help)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.setStyleSheet("background-color: #880000; color: white; padding: 10px; font-weight: bold;")
        self.btn_exit.clicked.connect(self.close)

        footer_layout.addWidget(self.btn_help)
        footer_layout.addStretch()  # Spinge l'Exit a destra (opzionale)
        footer_layout.addWidget(self.btn_exit)

        self.main_layout.addLayout(footer_layout)

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_class_status)

    def show_help(self):
        msg = """
        <h3>ION-Sim Student Hub</h3>
        <p><b>1. Login:</b> Connect to the Tutor Database to save progress.</p>
        <p><b>2. AI Mode:</b> Follow the recommended learning path.</p>
        <p><b>3. Manual Mode:</b> Practice specific modules (EMG, MEP, SEP).</p>
        <p><b>4. Live Class:</b> Join a synchronized session led by the Tutor.</p>
        <p><i>Note: In Offline/Sandbox mode, progress is NOT saved.</i></p>
        """
        QMessageBox.information(self, "Help", msg)

    def perform_login(self):
        name = self.txt_name.text().strip();
        surname = self.txt_surname.text().strip()
        if not name or not surname: return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        resp = self.db_client.login(name, surname)
        QApplication.restoreOverrideCursor()

        if resp.get("status") == "OK":
            self.is_sandbox = False
            self.current_user_id = resp.get("user_id")
            QMessageBox.information(self, "Success", f"Logged in as {name}.")
            sim_active = resp.get("sim_active", False)
            sim_info = resp.get("sim_info", "")
            self._setup_online(resp.get("level"), self.current_user_id, sim_active, sim_info)
            self.status_timer.start(5000)

        elif resp.get("status") == "OFFLINE":
            msg = "Cannot reach Tutor Server.\nStart in SANDBOX mode?\n(Progress NOT saved)"
            if QMessageBox.question(self, "Offline", msg) == QMessageBox.StandardButton.Yes:
                self.is_sandbox = True
                self._setup_sandbox()

    def _setup_online(self, level, uid, sim_active=False, sim_info=""):
        self.group_login.setEnabled(False)
        self.group_train.setEnabled(True)
        self.btn_live.setEnabled(True)
        self.btn_live.setStyleSheet("background-color: #aa5500; color: white;")
        self.update_live_button(sim_active, sim_info)

        ai_resp = self.db_client.get_ai_recommendation(uid)
        if ai_resp:
            self.ai_config = ai_resp.get("config", {})
            self.lbl_ai_info.setText(f"AI Suggests: {ai_resp.get('message')}")
            self.btn_ai_start.setVisible(True)
        self.frame_manual.setVisible(True)

    def _setup_sandbox(self):
        self.group_login.setEnabled(False)
        self.group_train.setEnabled(True)
        self.group_train.setTitle("2. Training Area (SANDBOX)")
        self.group_train.setStyleSheet("QGroupBox { border: 1px solid red; color: red; }")
        self.btn_ai_start.setVisible(False)
        self.lbl_ai_info.setText("AI Unavailable")
        self.btn_live.setEnabled(False)

    def update_live_button(self, active, info):
        if active:
            self.btn_live.setEnabled(True)
            self.btn_live.setText(f"3. Join Live: {info}")
            self.btn_live.setStyleSheet("background-color: #008800; color: white; font-weight: bold;")
        else:
            self.btn_live.setEnabled(False)
            self.btn_live.setText("3. Waiting for Tutor to start scenario...")
            self.btn_live.setStyleSheet("background-color: #555; color: #aaa;")

    def check_class_status(self):
        if not self.db_client.connected: return
        try:
            self.db_client.socket.send_json({"cmd": "CHECK_STATUS"})
            resp = self.db_client.socket.recv_json()
            self.update_live_button(resp.get("sim_active", False), resp.get("sim_info", ""))
        except:
            pass

    def start_ai_mode(self):
        self.ai_config["user_id"] = self.current_user_id
        # L'AI potrebbe suggerire "SEP" o "MEP". Launch userà la logica di dispatch.
        self.launch("LOCAL", config=self.ai_config)

    def start_manual_mode(self):
        mod = self.combo_modules.currentText()
        config = {"module": mod, "user_id": self.current_user_id if not self.is_sandbox else None}
        self.launch("LOCAL", config=config)

    def start_live_mode(self):
        self.launch("NETWORK")

    def launch(self, mode, config=None):
        """
        Metodo centrale che decide QUALE finestra lanciare in base al modulo.
        """
        self.hide()
        if config is None: config = {}
        config["is_sandbox"] = self.is_sandbox

        # Estrai il nome del modulo (es. "EMG", "SEP")
        module_name = config.get("module", "Generic").upper()

        self.dash = None

        # --- DISPATCHER: Sceglie la finestra giusta ---
        if mode == "LOCAL":
            if module_name == "EMG" and EmgControl is not None:
                print("Launching Specific EMG Simulator...")
                # Passiamo il path anomalia se presente, altrimenti None
                # Nota: EmgControl deve accettare i parametri o essere adattato
                self.dash = EmgControl(json_anomaly_path=config.get("anomaly_json", None))

            elif "EEG" in module_name:
                print("Launching EEG simulator...")
                self.dash = EEGControlWindow()
                self.dash.start_stop_simulation()

            elif "ECOG" in module_name:
                print("Launcing ECoG simulator...")
                self.dash = ECoGControlWindow()
                self.dash.start_stop_simulation()

            elif "MEP UPPER LIMBS" in module_name:
                print("Launching MEP upper limb simulator...")
                self.dash = MepasSimulator(scenario_name="upper_limbs")
                self.dash.start_stop()

            elif "MEP LOWER LIMBS" in module_name:
                print("Launching MEP lower limb simulator...")
                self.dash = MepasSimulator(scenario_name="lower_limbs")
                self.dash.start_stop()

            elif "MEP CRANIAL NERVES" in module_name:
                print("Launching MEP cranial nerves simulator...")
                self.dash = MepasSimulator(scenario_name="cranial_nerves")
                self.dash.start_stop()

            elif "BAEP" in module_name:
                self.dash = BaepSimulator()
                self.dash.start_stop()

            elif "VEP" in module_name:
                self.dash = VepSimulator()
                self.dash.start_stop()

            elif "ANESTHESIA" in module_name:
                self.dash = AnesthesiaSimulator()
                self.dash.show()

            # --- AGGIUNGI QUI GLI ALTRI MODULI QUANDO PRONTI ---
            # elif module_name == "SEP" and SepSimulator is not None:
            #     self.dash = SepSimulator(...)

            else:
                print(f"Module {module_name} specific GUI not found. Using Generic Dashboard.")
                self.dash = ClientDashboard(mode=mode, extra_data=config, server_ip=TUTOR_IP)

        else:
            # NETWORK MODE usa sempre la Dashboard Generica LSL
            self.dash = ClientDashboard(mode=mode, extra_data=config, server_ip=TUTOR_IP)

        # CONNESSIONE SEGNALE DI CHIUSURA
        # Importante: Assicurati che TUTTE le classi (EmgControl, SepSimulator...)
        # abbiano il segnale 'simulation_finished = pyqtSignal()' e lo emettano in closeEvent.
        if hasattr(self.dash, 'simulation_finished'):
            self.dash.simulation_finished.connect(self.on_dashboard_finished)
        else:
            print("WARNING: The launched window does not support 'simulation_finished' signal.")
            # Fallback per non bloccare l'app se il modulo non è aggiornato
            # (In produzione, tutti i moduli devono averlo)
            self.dash.destroyed.connect(self.on_dashboard_finished)

        self.dash.show()

    def on_dashboard_finished(self):
        """Chiamato quando la Dashboard o il Modulo specifico si chiude."""
        self.dash = None
        self.show()  # Riapre il Launcher
        self.reset_to_initial_state()

    def reset_to_initial_state(self):
        self.group_login.setEnabled(True)
        self.group_train.setEnabled(False)
        self.group_train.setTitle("2. Training Area")
        self.group_train.setStyleSheet("")
        self.btn_live.setEnabled(False)
        self.btn_live.setStyleSheet("background-color: #884400; color: gray;")
        self.current_user_id = None
        self.is_sandbox = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = IonSimLauncher()
    win.show()
    sys.exit(app.exec())