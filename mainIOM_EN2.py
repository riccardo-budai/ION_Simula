"""
mainIOM is the main class to control all simulation functions
"""

import sys
import gettext
import time
from datetime import datetime
import webbrowser as wb
import logging
import platform

# --- PYQT6 IMPORTS ---
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import Qt, QTimer, QDateTime, Slot, QFile, QIODevice
from PySide6.QtGui import QPalette, QColor, QPixmap
from PySide6.QtWidgets import QMainWindow, QApplication, QLabel, QFrame, QVBoxLayout, QTabWidget

# --- VEDO / MNE / VTK IMPORTS (LOGO) ---
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import mne
from vedo import Plotter, Mesh, settings, Text3D

# --- PROJECT IMPORTS ---
from helpPdfRB import MyDialogHelp
from dbaseIOM_EN_sqlt3 import dbaseMD
from learning_manager2 import LearningManager
from ai_tutorIOM_EN import LearningAgent

# project simula modules files ---------------------------------
from geminiIOM2 import gemini_controlIOM
from capIOM import MultiDistanceCAPSimulator
from sepIOMmeg import SepSimulator
from sepAiIOMmeg import SepSimulatorAI
from eegAuxIOM4 import EEGControlWindow
from ECoGIOM_EN2 import ECoGControlWindow
from baepIOM_EN import BaepSimulator
from vepIOM import VepSimulator
from mepIOM_EN2 import MepasSimulator
from emgIOM import EmgControl
from dwaveIOM2 import DwaveSimulator
from anesthesia_IOM_EN import AnesthesiaSimulator
#
from setup_managerIOM5 import IOMSetupDialog
from vp_tutorIOM2 import TutorControlWindow, TutorDBService



# ----------------------------------------------------------------------------------------------------------------------
# 1. SETUP LOGO WIDGET CLASS
# ----------------------------------------------------------------------------------------------------------------------
if platform.system() == "Darwin":
    settings.default_backend = "vtk"

# widget logo r&b-Lab --------------------------------------------------------------------------------------------------
class LogoWidget(QFrame):
    """
    Widget personalizzato che ospita la scena 3D (Logo Rotante).
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # Layout
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        # 1. Creiamo esplicitamente il widget VTK
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.vtkWidget)

        # 2. Passiamo il widget creato al Plotter
        self.plt = Plotter(qt_widget=self.vtkWidget, bg='white', axes=0)

        # Variabili mesh
        self.lh_pial = None
        self.rh_pial = None
        self.txt_mesh = None

        # Timer animazione
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotate_step)

        self.is_ready = False

    def load_data_and_create_mesh(self):
        # Evita log pesanti qui se non necessari
        try:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        except Exception as e:
            print(f"Errore download logo: {e}")
            return

        lh_path = fs_dir / 'surf' / 'lh.pial'
        rh_path = fs_dir / 'surf' / 'rh.pial'

        # Emisfero Sinistro
        p_lh, f_lh = mne.read_surface(lh_path)
        self.lh_pial = Mesh([p_lh, f_lh]).c("bisque").lighting("plastic")
        self.lh_pial.name = "lh_pial"

        # Emisfero Destro
        p_rh, f_rh = mne.read_surface(rh_path)
        self.rh_pial = Mesh([p_rh, f_rh]).c("bisque").lighting("shiny")
        self.rh_pial.name = "rh_pial"

        # Testo 3D
        self.txt_mesh = Text3D("R&B-Lab 2025", pos=(10, 5, 70), s=15, c="aqua", alpha=0.5, justify='centered',
                               depth=0.5)
        self.txt_mesh.rotate_x(90)

        # Aggiungiamo TUTTO al plotter
        self.plt.add(self.lh_pial, self.rh_pial, self.txt_mesh)

        # Camera iniziale (bg='navy' come da tua richiesta)
        self.plt.show(zoom=1.5, viewup='z', bg='navy', interactive=False)

        self.is_ready = True

    def start_animation(self):
        if not self.is_ready:
            self.load_data_and_create_mesh()
        # Avvia timer (50ms = 20 fps circa)
        # Avvia il timer solo se non è già attivo
        if not self.timer.isActive():
            self.timer.start(50)

    def stop_animation(self):
        """Ferma il timer per risparmiare risorse."""
        if self.timer.isActive():
            self.timer.stop()

    def rotate_step(self):
        if self.is_ready:
            # Ruota attorno all'asse Z
            self.lh_pial.rotate(1.0, axis=(0, 0, 1))
            self.rh_pial.rotate(1.0, axis=(0, 0, 1))
            self.plt.render()


# ----------------------------------------------------------------------------------------------------------------------
# 2. LOG HANDLER CLASS
# ----------------------------------------------------------------------------------------------------------------------
class QTextEditLogHandler(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        msg = self.format(record)
        try:
            if self.text_edit.isWidgetType():
                self.text_edit.append(msg)
        except RuntimeError:
            pass


# ----------------------------------------------------------------------------------------------------------------------
# 3. MAIN WINDOW CLASS
# ----------------------------------------------------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- SETUP TRADUZIONI ---
        _ = gettext.gettext
        LOCALE_DIR = "locale"
        DOMAIN = "IOM_Simula"
        try:
            t = gettext.translation(DOMAIN, LOCALE_DIR, languages=['it', 'en'])
            _ = t.gettext
        except FileNotFoundError:
            _ = gettext.gettext

        # --- CARICAMENTO UI ---
        ui_file_path = "res/mainIOMForm_EN.ui"
        ui_file = QFile(ui_file_path)
        if not ui_file:
            print(f"Errore: Impossibile aprire il file {ui_file_path}")
            sys.exit(-1)
        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        ui_file.close()

        self.setCentralWidget(self.ui)
        self.setWindowTitle(time.strftime('%d %b %Y') + '  R&B-Lab / ION-Sim / 2025')
        self.setFixedSize(800, 300)
        self.move(50, 50)

        # UI Setup di base
        self.ui.labelDateTime.setText(_(' R&B-Lab  ' + time.strftime('%d %b %Y  %H:%M:%S') + ' - session started'))
        self.ui.pushBtExit.clicked.connect(self.closeWin)
        self.ui.pushBtHelp.clicked.connect(self.helpContext)

        self.ui.pushBtSetup.clicked.connect(self.open_setup_manager)

        self.active_windows = []

        # Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('Application Ready...')
        self.datetime_label = QLabel(QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss"))
        self.status_bar.addPermanentWidget(self.datetime_label)

        # Timer Orologio: aggiornamento ogni 5 secondi
        self.timerTime = QTimer(self)
        self.timerTime.timeout.connect(self.update_time)
        self.timerTime.start(5000)
        self.update_time()

        # Tab IOM default position
        self.ui.tabWidget.setCurrentWidget(self.ui.tab_IOM)

        # --- STYLESHEET ---
        default_bg = "rgb(64, 64, 80)"
        default_text = "rgb(255, 255, 255)"
        hover_bg = "rgb(80, 80, 95)"
        selected_bg = "rgb(53, 132, 228)"
        selected_text = "rgb(0, 0, 0)"

        tab_style = f"""
            QTabBar::tab {{ background-color: {default_bg}; color: {default_text}; padding: 10px; }}
            QTabBar::tab:hover {{ background-color: {hover_bg}; }}
            QTabBar::tab:selected {{ background-color: {selected_bg}; color: {selected_text}; }}
        """
        self.ui.tabWidget.setStyleSheet(tab_style)

        # Mappatura Loghi
        logo_map = {
            self.ui.labelLogo: 'res/logos/logo2.png',
            self.ui.labelLogoAnesth: 'res/logos/anesthesia1.png',
            self.ui.labelLogoSep: 'res/logos/logoSEP1.png',
            self.ui.labelLogoSepai: 'res/logos/logoSEP2.png',
            self.ui.labelLogoMepcb: 'res/logos/logoMep1.png',
            self.ui.labelLogoMep: 'res/logos/logoMep1.png',
            self.ui.labelLogoMepai: 'res/logos/logoMep1.png',
            self.ui.labelLogoVep: 'res/logos/logoVEP2.png',
            self.ui.labelLogoBaep: 'res/logos/logoBAEP2.png',
            self.ui.labelLogoDCS: 'res/logos/logoEcog4.png',
            self.ui.labelLogoEeg: 'res/logos/logoEEG1.png',
            self.ui.labelLogoEcog: 'res/logos/logoEcoG2.png',
            self.ui.labelLogoEmg: 'res/logos/logoEMG1.png',
            self.ui.labelLogoCap: 'res/logos/logoCap1.png',
            self.ui.labelLogoDwave: 'res/logos/logoMep1.png',
        }
        for label_widget, logo_path in logo_map.items():
            self._set_logo_to_label(label_widget, logo_path)

        # 1. Recupero il widget originale e il suo genitore
        original_text_edit = self.ui.textEdit
        parent_widget = original_text_edit.parentWidget()

        # 2. Copiamo la geometria (X, Y, Larghezza, Altezza) settata nel file .ui
        original_geometry = original_text_edit.geometry()

        # 3. Creiamo il TabWidget assegnandogli lo STESSO genitore
        self.log_tab_widget = QTabWidget(parent_widget)
        self.log_tab_widget.setStyleSheet(tab_style)
        self.log_tab_widget.setTabShape(QTabWidget.TabShape.Triangular)

        # 4. Applichiamo la geometria copiata al nuovo TabWidget
        self.log_tab_widget.setGeometry(original_geometry)

        # 5. Creiamo il Widget Logo 3D
        self.logo3d_widget = LogoWidget()

        # 6. Aggiungiamo i Tab
        self.log_tab_widget.addTab(self.logo3d_widget, "Logo")
        self.log_tab_widget.addTab(original_text_edit, "System Logs")

        # Colleghiamo il segnale che scatta quando l'utente cambia tab
        self.log_tab_widget.currentChanged.connect(self.on_log_tab_changed)
        self.log_tab_widget.show()

        # Avvio animazione ritardata
        QTimer.singleShot(50, self.logo3d_widget.start_animation)

        # ----------------------------------------------------------------
        # LOGGING SETUP
        # ----------------------------------------------------------------
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Il textEdit ora è dentro il tab, ma il riferimento self.ui.textEdit è ancora valido
        log_handler = QTextEditLogHandler(self.ui.textEdit)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(formatter)

        if not root_logger.handlers:
            logFName = 'logs/' + time.strftime('%d-%b-%Y-%H-%M-%S') + 'logIOM.log'
            log_handler_file = logging.FileHandler(logFName, mode='a')
            log_handler_file.setFormatter(formatter)

            root_logger.addHandler(log_handler)
            root_logger.addHandler(log_handler_file)

        self.logger = logging.getLogger(__name__)
        self.logger.info(_("IOM-Simula App started with logging handler."))

        # --- VARIE ---
        # Variabili di classe per i simulatori (snake_case coerente)
        self.gemini_window = None
        self.eeg_window = None
        self.ecog_window = None
        self.sep_simula = None
        self.sep_simula_ai = None
        self.cap_simula = None
        self.baep_simula = None
        self.vep_simula = None
        self.mepas_simula = None
        self.emg_simula = None
        self.dwave_simula = None

        self.tutor_window = None

        self.app_start_time = datetime.now()

        # Database ---
        self.dbase_window = None
        if self.dbase_window is None:
            self.dbase_window = dbaseMD(self)
            self.ai_agent = LearningAgent(self.dbase_window.db_manager)
            if self.dbase_window.current_subject:
                self.ai_agent.log_behavior(self.dbase_window.current_subject.idcode, "LOGIN")
            self.dbase_window.show()
            self.dbase_window.simulation_start_requested.connect(self._handle_simulation_launch)

        self.learning_manager = LearningManager(self.dbase_window.db_manager)
        self.ui.pushBtGemini.clicked.connect(self.promptGemini)
        self.ui.pushBtSimVP.clicked.connect(self.open_tutor_window)

        self.db_service = TutorDBService("sqlt3_dbase/IOM_Simula.db")
        self.db_service.start()

    def _set_logo_to_label(self, label: QLabel, path: str):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            # Usa print se il logger non è ancora pronto o usa root logger
            print(f"Warning: Failed to load logo: {path}")
            return
        scaled_pixmap = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    @Slot(int)
    def on_log_tab_changed(self, index):
        """
        Gestisce l'attivazione/disattivazione dell'animazione 3D
        basandosi sul tab visibile.
        index 0 = Logo 3D
        index 1 = TextEdit (Log)
        """
        if index == 0:
            # Siamo sul tab del Logo: riattiva la rotazione
            self.logger.debug("Tab Logo selezionato: ripresa animazione.")
            self.logo3d_widget.start_animation()
        else:
            # Siamo su altri tab (es. Log): ferma la rotazione per CPU saving
            self.logger.debug("Tab Log selezionato: pausa animazione.")
            self.logo3d_widget.stop_animation()

    @Slot(str, str, dict)
    def _handle_simulation_launch(self, simulator_key: str, anomaly_json_path: str, anomaly_data: dict):
        if hasattr(self, 'log_tab_widget'):
            # set the tab widget logs and automatically stops the logo animation
            self.log_tab_widget.setCurrentIndex(1)
        self.logger.info(f"Received start request: {simulator_key}, JSON: {anomaly_json_path}")
        tab_mapping = {
            'SEPAS': self.ui.tab_SEPas,
            'SEPAI': self.ui.tab_SEPai,
            'CAP': self.ui.tab_CAP,
            'BAEP': self.ui.tab_BAEP,
            'VEP': self.ui.tab_VEP,
            'EEG': self.ui.tab_EEG,
            'ECOG': self.ui.tab_EcoG,
            'CCEP': self.ui.tab_EcoG,
            'MEPCN': self.ui.tab_MEPcn,
            'MEPAS': self.ui.tab_MEPas,
            'MEPAI': self.ui.tab_MEPai,
            'DCS': self.ui.tab_DCS,
            'EMG': self.ui.tab_EMG,
            'EMGTR': self.ui.tab_EMG,
            'EMGCN': self.ui.tab_EMG,
            'DWAVE': self.ui.tab_DWAVE
        }

        if simulator_key in tab_mapping:
            self.ui.tabWidget.setCurrentWidget(tab_mapping[simulator_key])
        else:
            self.logger.error(f"Simulator {simulator_key} not mapped to a tab.")
            return

        current_session = self.dbase_window.current_session
        if not current_session:
            self.logger.error("Attempting to start simulation without a current session.")
            return

        if anomaly_data:
            self.logger.info(f"Active anomaly: {anomaly_data.get('label_id')}")
            self.learning_manager.start_learning_session(
                session_id=current_session.idsession,
                anomaly_id=anomaly_data['anomaly_id']
            )
        else:
            self.logger.info("No active anomaly for this session. Starting in normal mode.")

        # Launch Logic
        if simulator_key == 'SEPAS':
            self._launch_sep_simula(anomaly_json_path, self.learning_manager)
        elif simulator_key == 'SEPAI':
            self._launch_sepai_simula(anomaly_json_path, self.learning_manager)
        elif simulator_key == 'CAP':
            self._launch_cap_simula(anomaly_json_path, self.learning_manager)
        elif simulator_key == 'EEG':
            self._launch_eeg_simula(anomaly_json_path, self.learning_manager)
        elif simulator_key == 'ECOG':
            self._launch_ecog_simula(anomaly_json_path, self.learning_manager)
        elif simulator_key == 'BAEP':
            self._launch_baep_simula(anomaly_json_path, self.learning_manager)
        elif simulator_key == 'VEP':
            self._launch_vep_simula(anomaly_json_path, self.learning_manager)
        elif simulator_key in ['MEPAS', 'MEPAI', 'MEPCN']:
            self._launch_mepas_simula(anomaly_json_path, self.learning_manager, simulator_key)
        elif simulator_key in ['EMG', 'EMGTR', 'EMGCN']:
            self._launch_emg_simula(anomaly_json_path, self.learning_manager)
        elif simulator_key == 'DWAVE':
            self._launch_dwave_simula(anomaly_json_path, self.learning_manager)

    # --- SIMULATION LAUNCHERS (Variable names corrected) ---
    def open_eeg_window(self):
        if self.eeg_window is None:
            self.eeg_window = EEGControlWindow(parent=self)
        self.eeg_window.show()
        self.eeg_window.raise_()
        self.eeg_window.start_stop_simulation()

    def open_setup_manager(self):
        # change tab index to stop logo animation
        self.log_tab_widget.setCurrentIndex(1)
        self.logger.info("Opening Setup Manager...")
        self.setup_dialog = IOMSetupDialog(self)
        self.setup_dialog.montage_confirmed.connect(self._launch_montage)
        # open dialog in modal madality
        self.setup_dialog.exec()

    def open_ecog_window(self):
        if self.ecog_window is None:
            self.ecog_window = ECoGControlWindow(parent=self)
        self.ecog_window.show()
        self.ecog_window.raise_()

    def open_tutor_window(self):
        """
        Apre la Tutor Control Station per la gestione del Virtual Patient.
        """
        # Se la finestra non esiste, creala
        if self.tutor_window is None:
            self.logger.info("Initializing Virtual Patient Tutor Station...")
            self.tutor_window = TutorControlWindow(parent=self)  # Parent=self la tiene legata alla main
            self.active_windows.append(self.tutor_window)  # Per chiuderla se chiudi la main app

        # Mostra la finestra
        if not self.tutor_window.isVisible():
            self.tutor_window.show()
            self.tutor_window.raise_()
            self.logger.info("Tutor Station Opened.")
        else:
            self.tutor_window.activateWindow()

    def promptGemini(self):
        if self.gemini_window is None or not self.gemini_window.isVisible():
            self.gemini_window = gemini_controlIOM(self)
            self.gemini_window.show()

    def _launch_montage(self, config):
        self.logger.info(f"Launching Montage for: {config['surgery_type']} - {config['location']}")

        # 1. NOTIFICA IL SERVIZIO CHE LA SIMULAZIONE E' ATTIVA
        sim_name = f"{config['surgery_type']} ({config['location']})"
        self.db_service.set_simulation_status(True, sim_name)

        modules = config['active_modules']

        if "VIRTUAL_PATIENT" in modules:
            self.open_tutor_window()
            # Invia il JSON di setup al Tutor, che lo girerà al Generatore via ZMQ
            if self.tutor_window:
                self.tutor_window.load_simulation_config(config)
        else:
            # 1. LANCIO SEMPRE PRIMA L'ANESTESIA (Fondamentale per TCI)
            if "ANESTHESIA" in modules:
                self._launch_anesthesia_tci()

            # 2. Lancio gli altri moduli con un leggero ritardo o posizionamento
            # Nota: Qui assumiamo che non stiamo caricando anomalie specifiche JSON per ora,
            # ma inizializzando il setup standard. Se serve anomaly, va passata qui.

            if "EEG" in modules:
                self._launch_eeg_simula(None, self.learning_manager)

            if "ECOG" in modules:
                self._launch_ecog_simula(None, self.learning_manager)

            if "SEP_UL" in modules or "SEP_LL" in modules:
                # SepSimulator gestisce entrambi o bisogna configurarlo
                self._launch_sep_simula(None, self.learning_manager)

            if "MEP_CONTRA" in modules or "MEP_FOUR_LIMB" in modules:
                # Qui potresti passare un parametro extra al simulatore MEP per dirgli "Upper" o "Four Limb"
                self._launch_mepas_simula(None, self.learning_manager, "MEPAS")

            if "DWAVE" in modules:
                self._launch_dwave_simula(None, self.learning_manager)

        # ... Aggiungi gli altri casi ...

        self.logger.info("Montage Launched Successfully.")

    def _launch_anesthesia_tci(self):
        if hasattr(self, 'anesthesia_window') and self.anesthesia_window is not None:
            self.anesthesia_window.show()
            return
        self.logger.info("Starting TCI Anesthesia Monitor...")
        self.anesthesia_window = AnesthesiaSimulator()  # Classe dal file precedente
        self.anesthesia_window.show()
        # Posizionala in un punto specifico se vuoi
        self.anesthesia_window.move(100, 100)
        self.active_windows.append(self.anesthesia_window)

    def _launch_eeg_simula(self, path, mgr):
        if self.eeg_window and self.eeg_window.isVisible(): return
        self.logger.info(f"Starting EEG: {path}")
        self.eeg_window = EEGControlWindow(json_anomaly_path=path, learning_manager=mgr, parent=self)
        self.eeg_window.destroyed.connect(lambda: self._reset_simulator_button('EEG'))
        self.eeg_window.show()
        self.eeg_window.start_stop_simulation()

    def _launch_ecog_simula(self, path, mgr):
        if self.ecog_window and self.ecog_window.isVisible(): return
        self.logger.info(f"Starting EcoG: {path}")
        self.ecog_window = ECoGControlWindow(json_anomaly_path=path, learning_manager=mgr, parent=self)
        self.ecog_window.destroyed.connect(lambda: self._reset_simulator_button('ECoG'))
        self.ecog_window.show()
        self.ecog_window.start_stop_simulation()

    def _launch_dwave_simula(self, path, mgr):
        if self.dwave_simula and self.dwave_simula.isVisible(): return
        self.logger.info(f"Starting DWAVE: {path}")
        self.dwave_simula = DwaveSimulator(json_anomaly_path=path, learning_manager=mgr, ai_agent=self.ai_agent,
                                          parent=self)
        self.dwave_simula.destroyed.connect(lambda: self._reset_simulator_button('DWAVE'))
        self.dwave_simula.show()
        self.dwave_simula.start_stop()

    def _launch_sep_simula(self, path, mgr):
        if self.sep_simula and self.sep_simula.isVisible(): return
        self.logger.info(f"Starting SEP: {path}")
        self.sep_simula = SepSimulator(json_anomaly_path=path, learning_manager=mgr, ai_agent=self.ai_agent, parent=self)
        self.sep_simula.destroyed.connect(lambda: self._reset_simulator_button('SEPAS'))
        self.sep_simula.show()
        self.sep_simula.startStop()

    def _launch_sepai_simula(self, path, mgr):
        if self.sep_simula_ai and self.sep_simula_ai.isVisible(): return
        self.sep_simula_ai = SepSimulatorAI(json_anomaly_path=path, learning_manager=mgr, parent=self)
        self.sep_simula_ai.destroyed.connect(lambda: self._reset_simulator_button('SEPAI'))
        self.sep_simula_ai.show()
        self.sep_simula_ai.startStop()

    def _launch_cap_simula(self, path, mgr):
        if self.cap_simula and self.cap_simula.isVisible(): return
        self.logger.info(f"Starting CAP: {path}")
        self.cap_simula = MultiDistanceCAPSimulator(json_anomaly_path=path, learning_manager=mgr, parent=self)
        self.cap_simula.destroyed.connect(lambda: self._reset_simulator_button('CAP'))
        self.cap_simula.show()
        self.cap_simula.startStop()

    def _launch_baep_simula(self, path, mgr):
        if self.baep_simula and self.baep_simula.isVisible(): return
        self.baep_simula = BaepSimulator(json_anomaly_path=path, learning_manager=mgr, parent=self)
        self.baep_simula.destroyed.connect(lambda: self._reset_and_clear_simula('BAEP'))
        self.baep_simula.show()
        self.baep_simula.start_stop()

    def _launch_vep_simula(self, path, mgr):
        if self.vep_simula and self.vep_simula.isVisible(): return
        self.vep_simula = VepSimulator(json_anomaly_path=path, learning_manager=mgr, parent=self)
        self.vep_simula.destroyed.connect(lambda: self._reset_and_clear_simula('VEP'))
        self.vep_simula.show()
        self.vep_simula.start_stop()

    def _launch_emg_simula(self, path, mgr):
        if self.emg_simula and self.emg_simula.isVisible(): return
        self.emg_simula = EmgControl(json_anomaly_path=path, learning_manager=mgr, parent=self)
        self.emg_simula.destroyed.connect(lambda: self._reset_and_clear_simula('EMG'))
        self.emg_simula.show()
        self.emg_simula.toggleSimulation()

    def _launch_mepas_simula(self, path, mgr, key):
        if self.mepas_simula and self.mepas_simula.isVisible(): return
        scenario = {'MEPAS': 'upper_limbs', 'MEPAI': 'lower_limbs', 'MEPCN': 'cranial_nerves'}.get(key, 'upper_limbs')
        self.mepas_simula = MepasSimulator(scenario_name=scenario, json_anomaly_path=path, learning_manager=mgr,
                                           ai_agent=self.ai_agent, parent=self)
        self.mepas_simula.destroyed.connect(lambda: self._reset_and_clear_simula(key))
        self.mepas_simula.show()
        self.mepas_simula.start_stop()

    # --- HELPERS ---
    def helpContext(self):
        if self.dbase_window.current_subject:
            self.ai_agent.log_behavior(self.dbase_window.current_subject.idcode, "HELP_READ", "ION-Sim.pdf")
        wb.open_new('help_docs/paper/ION-Sim.pdf')

    def selectmanualapage(self):
        dialog = MyDialogHelp()
        result = dialog.exec()

    def update_time(self):
        self.datetime_label.setText("R&B-Lab " + QDateTime.currentDateTime().toString("dd MMM yyyy HH:mm:ss"))

    def add_item(self, info):
        self.logger.debug(time.strftime('%H:%M:%S ') + info + '\n')

    @Slot(str)
    def _reset_simulator_button(self, simulator_key: str):
        # Placeholder per logica futura se serve aggiornare la UI
        pass

    @Slot(str)
    def _reset_and_clear_simula(self, simulator_key: str):
        self._reset_simulator_button(simulator_key)
        # Pulizia referenze variabili usando i nomi corretti (snake_case)
        if simulator_key == 'BAEP':
            self.baep_simula = None
        elif simulator_key == 'VEP':
            self.vep_simula = None
        elif simulator_key == 'EMG':
            self.emg_simula = None
        elif simulator_key in ['MEPAS', 'MEPAI', 'MEPCN']:
            self.mepas_simula = None
        elif simulator_key == 'SEPAS':
            self.sep_simula = None
        elif simulator_key == 'SEPAI':
            self.sep_simula_ai = None
        elif simulator_key == 'DWAVE':
            self.dwave_simula = None
        elif simulator_key == 'EEG':
            self.eeg_window = None
        elif simulator_key == 'ECOG':
            self.ecog_window = None
        elif simulator_key == 'CAP':
            self.cap_simula = None
        elif simulator_key == 'TUTOR':
            self.tutor_window = None

        self.logger.info(f"Reference {simulator_key} deleted.")

    def closeWin(self):
        self.logger.info("Closing application sequence initiated...")

        # 1. Stop logo animation e plotter
        if hasattr(self, 'logo3d_widget'):
            try:
                self.logo3d_widget.stop_animation()
                self.logo3d_widget.plt.close()
            except Exception as e:
                self.logger.error(f"Error closing 3D Logo: {e}")

        # 2. CHIUSURA TUTOR (CORRETTO)
        if self.tutor_window is not None:
            self.logger.info("Stopping Tutor Backend...")
            try:
                # Chiamiamo il metodo esplicito che ferma il thread ZMQ
                self.tutor_window.force_shutdown()
                # Ora possiamo permettere a Qt di distruggere il widget
                self.tutor_window.deleteLater()
                self.tutor_window = None
            except Exception as e:
                self.logger.error(f"Error shutting down Tutor: {e}")

        # 3. Chiusura altre finestre attive
        for win in self.active_windows:
            try:
                if win.isVisible():
                    win.close()
            except:
                pass

        # ... (Logica Logout DB invariata) ...
        if self.dbase_window and self.dbase_window.current_subject:
            # ... codice db logout ...
            pass

        self.db_service.set_simulation_status(False, "Tutor Closed")

        self.logger.info("Bye Bye.")

        # 5. Chiusura Main Window
        self.close()

        # 6. FORZA USCITA PYTHON (Nuclear Option)
        # Se ZMQ ha lasciato qualche handle aperto in C++, questo lo uccide.
        # È sicuro farlo qui perché siamo alla fine di tutto.
        QApplication.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    paletteDarkBlue = QPalette()
    paletteDarkBlue.setColor(QPalette.ColorRole.Window, QColor(64, 64, 80))
    paletteDarkBlue.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    app.setPalette(paletteDarkBlue)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
