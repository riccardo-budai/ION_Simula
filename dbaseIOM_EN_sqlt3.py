# file: dbaseIOM.py
import sys
import logging
import webbrowser as wb
from PySide6.QtWidgets import QWidget, QApplication, QVBoxLayout
from PySide6.QtCore import Slot, QSize, Qt, Signal, QFile, QIODevice
from PySide6.QtGui import QPalette, QColor, QPixmap
from PySide6.QtUiTools import QUiLoader

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import os
import webbrowser

# Import necessary components
from database_managerIOM_sqlt3 import DatabaseManager
from dbase_dialogs_sqlt3 import appendRecord, appendSession, AnomalyDialog

# These are your dialog windows for adding/editing records.
# Ensure they are importable from another file.
# from dialogs import appendRecord, appendSession

########################################################################################################################
class dbaseMD(QWidget):
    simulation_start_requested = Signal(str, str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.logger = logging.getLogger(__name__)
        self.setWindowFlag(Qt.WindowType.Window)

        # --- UI Loading and Initialization ---
        # Load the .ui file created with Qt Designer
        # self.ui = uic.loadUi('res/dbaseForm.ui', self)
        ui_file_path = "res/dbaseForm.ui"
        ui_file = QFile(ui_file_path)

        if not ui_file:
            print(f"Errore: Impossibile aprire il file {ui_file_path}")
            sys.exit(-1)

        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        ui_file.close()
        layout = QVBoxLayout(self)
        layout.addWidget(self.ui)  # Aggiungi l'interfaccia caricata al layout
        layout.setContentsMargins(0, 0, 0, 0)  # (Opzionale) Rimuovi bordi extra
        self.setLayout(layout)

        # Store currently selected data
        self.current_subject = None
        self.current_session = None

        logo_path = 'res/logos/logo1.png'
        pixmap = QPixmap(logo_path)
        scaled_pixmap = pixmap.scaled(self.ui.labelLogo.size(),
                                      Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        self.ui.labelLogo.setPixmap(scaled_pixmap)

        # [MODIFICATION] Create the database manager instance
        # passing the SQLite file path
        try:
            # 1. Define the database file name
            DB_FILENAME = "sqlt3_dbase/IOM_Simula.db"
            # 2. Find the correct path (next to executable or script)
            db_path = self._get_resource_path(DB_FILENAME)
            print(f"SQLite database path: {db_path}")

            # 3. Initialize manager with file path
            self.db_manager = DatabaseManager(db_path=db_path)

        except Exception as e:
            # If DB connection fails, the app cannot work.
            print(f"Cannot start application: {e}")
            sys.exit(1)

        self._setup_ui()
        self._connect_signals()

        # Populate subject list on startup
        self.populate_subjects_combo()


    def _get_resource_path(self, relative_path):
        """
        Gets the absolute path of a resource.
        Works both in development and when compiled with Nuitka/PyInstaller.
        """
        try:
            # If compiled, sys.frozen is True and base is the executable dir
            if getattr(sys, 'frozen', False):
                base_path = os.path.dirname(sys.executable)
            else:
                # In dev, base is this script's dir
                base_path = os.path.dirname(os.path.abspath(__file__))
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)


    def _setup_ui(self):
        """Sets up the initial user interface."""
        self.ui.setFixedSize(QSize(420, 368))
        self.ui.move(50 + 800 + 7, 50)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor('gray'))
        self.ui.setPalette(palette)
        self.ui.comboBoxSbj.addItems([' User ', 'Edit user data', 'Append new user'])
        self.ui.comboBoxSes.addItems([' Session ', 'Edit Session data', 'Append new Session'])
        self.ui.comboAnomaly.addItems(['Anomalies', 'Select anomaly', 'Edit anomaly', 'Append new anomaly'])

    def _connect_signals(self):
        """Connects all signals and slots."""
        self.ui.comboSbj.currentIndexChanged.connect(self.on_subject_selected)
        self.ui.comboSes.currentIndexChanged.connect(self.on_session_selected)
        self.ui.comboBoxSbj.currentIndexChanged.connect(self.on_subject_action_selected)
        self.ui.comboBoxSes.currentIndexChanged.connect(self.on_session_action_selected)
        self.ui.comboAnomaly.currentIndexChanged.connect(self.on_anomaly_action_selected)
        self.ui.pushBtHelp.clicked.connect(self.context_help)
        self.ui.pushBtStartSim.clicked.connect(self._launch_simulation_request)
        self.ui.pushBtLearning.clicked.connect(self.show_subject_progress)

    def populate_subjects_combo(self, select_id=None):
        """
        Populates the subject combobox. If `select_id` is provided,
        attempts to select that subject after population.
        """
        self.ui.comboSbj.blockSignals(True)
        self.ui.comboSbj.clear()
        subjects = self.db_manager.get_all_subjects()
        index_to_select = 0
        for i, subject in enumerate(subjects):
            display_text = f"{subject.firstname} {subject.lastname}"
            self.ui.comboSbj.addItem(display_text, userData=subject)
            if subject.idcode == select_id:
                index_to_select = i
        self.ui.comboSbj.setCurrentIndex(index_to_select)
        self.ui.comboSbj.blockSignals(False)
        if self.ui.comboSbj.count() > 0:
            self.on_subject_selected(index_to_select)

    def populate_sessions_combo(self, subject_id: int):
        """Populates the session combobox for the selected subject."""
        self.ui.comboSes.blockSignals(True)
        self.ui.comboSes.clear()
        sessions = self.db_manager.get_sessions_for_subject(subject_id)
        if not sessions:
            self.ui.comboSes.addItem("No sessions recorded.")
        else:
            for session in sessions:
                # [CORRECTION] Handling null or non-datetime datetime
                dt_str = "Unknown Date"
                if isinstance(session.datetime, str):
                    # SQLite returns strings, try parsing if needed
                    try:
                        dt_str = datetime.strptime(session.datetime, '%Y-%m-%d %H:%M:%S').strftime('%d-%m-%Y')
                    except (ValueError, TypeError):
                        dt_str = session.datetime.split(" ")[0]  # Try taking only the date part
                elif isinstance(session.datetime, datetime):
                    dt_str = session.datetime.strftime('%d-%m-%Y')

                display_text = f"{session.session} / {session.taskname} ({dt_str})"
                self.ui.comboSes.addItem(display_text, userData=session)
        self.ui.comboSes.blockSignals(False)
        if self.ui.comboSes.count() > 0:
            self.on_session_selected(0)

    @Slot(int)
    def on_subject_selected(self, index: int):
        """Called when a user selects a subject from the combobox."""
        # ... (no changes here) ...
        if index < 0: return
        subject = self.ui.comboSbj.itemData(index)
        if subject:
            self.current_subject = subject
            self._update_subject_display()
            self.populate_sessions_combo(subject.idcode)

    def _launch_simulation_request(self):
        """Emits signal to start simulation based on current session."""
        if not self.current_session:
            print("ERROR: Select a valid session to start simulation.")
            return
        task_name = self.current_session.taskname
        if 'SEPAS' in task_name.upper():
            simulator_key = 'SEPAS'
        elif 'SEPAI' in task_name.upper():
            simulator_key = 'SEPAI'
        elif 'CAP' in task_name.upper():
            simulator_key = 'CAP'
        elif 'EEG' in task_name.upper():
            simulator_key = 'EEG'
        elif 'EcoG' in task_name:
            simulator_key = 'ECOG'
        elif 'CCEP' in task_name.upper():
            simulator_key = 'CCEP'
        elif 'BAEP' in task_name.upper():
            simulator_key = 'BAEP'
        elif 'VEP' in task_name.upper():
            simulator_key = 'VEP'
        elif 'MEPAS' in task_name.upper():
            simulator_key = 'MEPAS'
        elif 'MEPAI' in task_name.upper():
            simulator_key = 'MEPAI'
        elif 'MEPCN' in task_name.upper():
            simulator_key = 'MEPCN'
        elif 'EMG' in task_name.upper():
            simulator_key = 'EMG'
        elif 'DWAVE' in task_name.upper():
            simulator_key = 'DWAVE'

        else:
            print(f"ERROR: Simulation type not recognized in taskname: {task_name}")
            return

        anomaly_label_id = self.current_session.simulacode
        anomaly_json_path = None
        selected_anomaly_data = {}
        label_simula_code = self.current_session.taskname
        print(f"anomaly_label_id = {anomaly_label_id} {label_simula_code}")

        all_scenarios = self.db_manager.get_all_scenarios()
        current_scenario = next((s for s in all_scenarios if s.label_id == self.current_session.taskname), None)
        print(f"current_scenario = {current_scenario}")

        if current_scenario:
            all_anomalies = self.db_manager.get_anomalies_for_scenario(current_scenario.scenario_id)
            found_anomaly = next((a for a in all_anomalies if a.label_id == anomaly_label_id), None)
            print(f"found_anomaly = {found_anomaly}")
            if found_anomaly:
                self.logger.info(f"Specified anomaly: {anomaly_label_id}. Found in DB.")
                selected_anomaly_data = found_anomaly.__dict__
                anomaly_json_path = f"{selected_anomaly_data['json_anomaly_id']}"
            else:
                self.logger.warning(
                    f"No specific anomaly found for {anomaly_label_id}. Starting in normal mode.")
        else:
            self.logger.error(f"Scenario {self.current_session.taskname} not found!")
        self.simulation_start_requested.emit(simulator_key, anomaly_json_path, selected_anomaly_data)
        self.logger.info(f"Simulation start requested {simulator_key}. Anomaly: {selected_anomaly_data}")

    def _update_subject_display(self):
        """Updates UI labels with current subject data."""
        # ... (no changes here) ...
        if self.current_subject:
            self.ui.lblFirst.setText(self.current_subject.firstname)
            self.ui.lblLast.setText(self.current_subject.lastname)
            self.ui.lblAge.setText(str(self.current_subject.age))
            self.ui.lblGender.setText(self.current_subject.gender)

    @Slot(int)
    def on_session_selected(self, index: int):
        """Called when a user selects a session."""
        # ... (no changes here) ...
        if index < 0: return
        session = self.ui.comboSes.itemData(index)
        if session:
            self.current_session = session
            self._update_session_display()

    def _update_session_display(self):
        """Updates labels with current session data."""
        # ... (no changes here) ...
        if self.current_session:
            text = f"ID: {self.current_session.idsession}, Task: {self.current_session.taskname}, Sim: {self.current_session.simulacode}"
            self.ui.lblSession.setText(text)

    @Slot(int)
    def on_subject_action_selected(self, index: int):
        """Handles subject-related actions (edit, add)."""
        # ... (no changes here) ...
        if index == 0: return
        if index == 1:
            self._edit_subject()
        elif index == 2:
            self._add_new_subject()
        self.ui.comboBoxSbj.setCurrentIndex(0)

    def _edit_subject(self):
        """Opens dialog to edit current subject."""
        # ... (no changes here) ...
        if not self.current_subject:
            print("No subject selected to edit.")
            return
        subject_dict = self.current_subject.__dict__
        self.logger.info(f"Editing subject ID = {self.current_subject.idcode}")
        self.edit_subject_dialog = appendRecord(recMode=1, recRow=subject_dict, parent=self)
        self.edit_subject_dialog.newDataRec.connect(self._handle_update_subject)
        self.edit_subject_dialog.show()
        print(f"Opening edit window for: {subject_dict['firstname']} {subject_dict['lastname']}")

    def _add_new_subject(self):
        """Opens dialog to add a new subject."""
        # ... (no changes here) ...
        self.logger.info(f"Adding a new subject!")
        empty_record = {}
        self.add_subject_dialog = appendRecord(recMode=0, recRow=empty_record, parent=self)
        self.add_subject_dialog.newDataRec.connect(self._handle_append_subject)
        self.add_subject_dialog.show()
        print("Opening window to add a new subject.")

    @Slot(object, bool)
    def _handle_update_subject(self, updated_data: dict, success: bool):
        """Slot to receive updated data from dialog."""
        # ... (no changes here, 'age' logic was already commented out) ...
        if success:
            if self.db_manager.update_subject(updated_data):
                print("Update successful. Reloading list.")
                self.populate_subjects_combo(select_id=updated_data['idcode'])
            else:
                print("Update failed.")

    @Slot(object, bool)
    def _handle_append_subject(self, new_data: dict, success: bool):
        """Slot to receive new subject data."""
        # ... (no changes here) ...
        if success:
            new_id = self.db_manager.add_new_subject(new_data)
            if new_id is not None:
                self.logger.info(f"User added successfully. New ID: {new_id}.")
                self.logger.info(f"Attempting to create default 'EEG' session for user {new_id}...")
                all_scenarios = self.db_manager.get_all_scenarios()
                eeg_scenario = next((s for s in all_scenarios if s.label_id.upper() == 'EEG'), None)
                if eeg_scenario:
                    new_session_data = {
                        'idcode': new_id,
                        'sesnote': "Default session created automatically.",
                        'fname': f"{eeg_scenario.label_id}_{new_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'dbasename': 'ION-Sim',
                        'datetime': datetime.now(),
                        'session': eeg_scenario.nome_scenario,
                        'taskname': eeg_scenario.label_id,
                        'simulacode': f"{eeg_scenario.label_id}_1"
                    }
                    session_id = self.db_manager.add_new_session(new_session_data)
                    if session_id is not None:
                        self.logger.info(
                            f"Created default 'EEG' session (ID: {session_id}) for new user {new_id}.")
                    else:
                        self.logger.error(
                            f"Failed to create default 'EEG' session for new user {new_id}.")
                else:
                    self.logger.warning("'EEG' scenario not found in DB. Cannot create default session.")
                self.populate_subjects_combo(select_id=new_id)
            else:
                print("Addition failed.")

    @Slot(int)
    def on_session_action_selected(self, index: int):
        """Handles session-related actions (edit, add)."""
        # ... (no changes here) ...
        if index == 0: return
        if index == 1:
            self._edit_session()
        elif index == 2:
            self._add_new_session()
        self.ui.comboBoxSes.setCurrentIndex(0)

    def on_anomaly_action_selected(self, index: int):
        """Handles the list of anomalies expected for a specific scenario (scenari_setup)"""
        # ... (no changes here) ...
        if index == 0: return
        if index == 1:
            self.select_anomalies()
        if index == 2:
            pass
        if index == 3:
            pass
        self.ui.comboAnomaly.setCurrentIndex(0)

    def select_anomalies(self):
        """list anomalies for a specific scenario : select, [edit and new anomaly only for advanced users)"""
        # ... (no changes here) ...
        self._manage_anomalies_for_current_session()

    def _edit_session(self):
        """Opens dialog to edit current session."""
        # ... (no changes here) ...
        if not self.current_session:
            print("No session selected to edit.")
            return
        scenarios_list = self.db_manager.get_all_scenarios()
        if not scenarios_list:
            print("WARNING: No scenarios found in database.")
        session_dict = self.current_session.__dict__.copy()
        self.logger.info(f"Editing session ID = {self.current_session.idsession}")
        self.edit_session_dialog = appendSession(
            recMode=1,
            recRow=session_dict,
            subject_id=self.current_subject.idcode,
            scenarios=scenarios_list,
            parent=self
        )
        self.edit_session_dialog.newSessionRec.connect(self._handle_update_session)
        self.edit_session_dialog.show()
        print(f"Opening edit window for session: {session_dict['idsession']}")

    def _add_new_session(self):
        """Opens dialog to add a new session."""
        # ... (no changes here) ...
        if not self.current_subject:
            print("Select a subject before adding a session.")
            return
        scenarios_list = self.db_manager.get_all_scenarios()
        if not scenarios_list:
            print("WARNING: No scenarios found in database.")
            return  # [CORRECTION] Better to stop if no scenarios

        self.logger.info(f"Adding new session for subject ID: {self.current_subject.idcode}")
        default_scenario = scenarios_list[0]
        default_taskname = default_scenario.label_id
        default_simulacode = f"{default_taskname}_1"
        empty_session = {
            'idcode': self.current_subject.idcode,
            'taskname': default_taskname,
            'simulacode': default_simulacode
        }
        self.add_session_dialog = appendSession(
            recMode=0,
            recRow=empty_session,
            subject_id=self.current_subject.idcode,
            scenarios=scenarios_list,
            parent=self
        )
        self.add_session_dialog.newSessionRec.connect(self._handle_append_session)
        self.add_session_dialog.show()
        print(f"Opening window to add new session for subject: {self.current_subject.idcode}")
        print(f"Default Scenario: {default_taskname}, Default Simulacode: {default_simulacode}")

    @Slot(object, bool)
    def _handle_update_session(self, updated_data: dict, success: bool):
        """Slot to receive updated session data."""
        # ... (no changes here) ...
        if success:
            if self.db_manager.update_session(updated_data):
                print("Session update successful. Reloading list.")
                self.populate_sessions_combo(self.current_subject.idcode)
            else:
                print("Session update failed.")

    @Slot(object, bool)
    def _handle_append_session(self, new_data: dict, success: bool):
        """Slot to receive new session data."""
        # ... (no changes here) ...
        if success:
            new_id = self.db_manager.add_new_session(new_data)
            if new_id is not None:
                print(f"New session added with ID: {new_id}. Reloading list.")
                self.populate_sessions_combo(self.current_subject.idcode)
            else:
                print("Session addition failed.")

    def context_help(self):
        """Shows help information for database management."""
        # ... (no changes here) ...
        pdf_path = 'help_docs/paper/Tut_Dbase.pdf'
        wb.open_new(pdf_path)

    def _manage_anomalies_for_current_session(self):
        """Opens dialog to manage anomalies related to the current session's scenario.
        """
        # ... (no changes here) ...
        if not self.current_session:
            print("Select a session first to manage its associated anomalies.")
            return
        subject_level_code = self.current_subject.level
        all_scenarios = self.db_manager.get_all_scenarios()
        selected_scenario = next((s for s in all_scenarios if s.label_id == self.current_session.taskname), None)
        if not selected_scenario:
            print(f"ERROR: Associated scenario ({self.current_session.taskname}) not found in database.")
            return
        scenario_dict = selected_scenario.__dict__.copy()
        anomalies_list = self.db_manager.get_anomalies_for_scenario(scenario_dict['scenario_id'])
        anomalies_dict_list = [a.__dict__ for a in anomalies_list]
        self.anomaly_manager_dialog = AnomalyDialog(
            scenario_data=scenario_dict,
            anomalies_data=anomalies_dict_list,
            level_code=subject_level_code,
            parent=self
        )
        self.anomaly_manager_dialog.anomaly_selected.connect(self._handle_anomaly_selection)
        self.anomaly_manager_dialog.newAnomalySaved.connect(self._handle_anomaly_update_notification)
        self.anomaly_manager_dialog.exec()

    @Slot(dict)
    def _handle_anomaly_selection(self, selected_anomaly_data: dict):
        """Receives the user-chosen anomaly and saves it to the session record."""
        # ... (no changes here) ...
        if not self.current_session:
            self.logger.warning("Logic error: Anomaly selected but no active session.")
            return
        new_simulacode = selected_anomaly_data.get('label_id')
        self.current_session.simulacode = new_simulacode
        session_data_for_update = self.current_session.__dict__.copy()

        # [MODIFICATION] We must convert datetime to string before passing it
        # to db_manager, because the session in memory might have a datetime object
        if isinstance(session_data_for_update['datetime'], datetime):
            session_data_for_update['datetime'] = session_data_for_update['datetime'].strftime('%Y-%m-%d %H:%M:%S')

        if self.db_manager.update_session(session_data_for_update):
            self.logger.info(
                f"Session {self.current_session.idsession} PERSISTENTLY updated with Anomaly: {new_simulacode}")
        else:
            self.logger.error(f"FAILED saving anomaly to DB for session {self.current_session.idsession}.")
        self._update_session_display()
        self.logger.info("You can now press 'StartSim' to start with the anomaly.")

    @Slot(bool)
    def _handle_anomaly_update_notification(self, success: bool):
        """Handles notification signal after adding/editing/deleting an anomaly."""
        # ... (no changes here) ...
        if success:
            print("Anomaly update completed. Proceed with saving JSON files if necessary.")

    # --- Funzione Helper per l'Header ---
    def _add_header(self, canvas, doc):
        """Disegna l'intestazione su ogni pagina del PDF."""
        canvas.saveState()

        # Impostazioni Font e Colore
        canvas.setFont('Helvetica-BoldOblique', 10)
        canvas.setFillColor(colors.darkblue)

        # Testo dell'intestazione
        header_text = "Neuroveal srl - 2025"

        # Posizionamento (in alto a destra o sinistra, qui facciamo destra)
        # A4 width è circa 595 punti. Margine destro 50.
        page_width, page_height = A4
        text_width = canvas.stringWidth(header_text, 'Helvetica-BoldOblique', 10)

        # Disegna a 50pt dal bordo destro, 30pt dal bordo superiore
        x_pos = page_width - 50 - text_width
        y_pos = page_height - 30

        canvas.drawString(x_pos, y_pos, header_text)

        # Linea divisoria opzionale sotto l'header
        canvas.setStrokeColor(colors.gray)
        canvas.line(50, y_pos - 5, page_width - 50, y_pos - 5)

        canvas.restoreState()

    def show_subject_progress(self):
        """
        Summarizes performance data of the selected subject from the session_decisions table
        and drafts a PDF report file.
        """
        if not self.current_subject:
            print("Select a subject to view progress.")
            return
        # ATTENTION: This logic assumes 'self.parent()' is the MainWindow
        # and has a 'learning_manager'. If you run this file alone, it will fail.
        main_window = self.parent()
        if not main_window or not hasattr(main_window, 'learning_manager'):
            print("ERROR: Cannot find 'learning_manager'. Run from main application.")
            return

        summary = main_window.learning_manager.get_performance_summary(self.current_subject.idcode)
        if not summary:
            print(
                f"No performance data found for {self.current_subject.firstname} {self.current_subject.lastname}")
            return

        subject_idcode = self.current_subject.idcode
        subject_name = f"{self.current_subject.firstname} {self.current_subject.lastname}"

        summary = main_window.learning_manager.get_performance_summary(subject_idcode)

        if not summary:
            print(f"No performance data found for {subject_name}. PDF file not generated.")
            return

        # --- PDF Generation ---

        # 1. File name definition: directory to save all .pdf reports
        report_dir = "reports_pdf"

        # [MODIFICATION] Use _get_resource_path to ensure
        # saving reports next to executable/script
        report_dir_path = self._get_resource_path(report_dir)
        os.makedirs(report_dir_path, exist_ok=True)  # Create folder if not exists

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(report_dir_path, f"Performance_Report_{subject_idcode}_{timestamp}.pdf")

        # 4. PDF Document creation
        doc = SimpleDocTemplate(filename, pagesize=A4,
                                rightMargin=50, leftMargin=50,
                                topMargin=50, bottomMargin=50)
        styles = getSampleStyleSheet()
        story = []

        # 5. Report Header
        title = Paragraph(f"<b>ION-Sim Performance Report</b>", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))

        subtitle = Paragraph(f"User: <b>{subject_name}</b> (ID: {subject_idcode})", styles['h2'])
        story.append(subtitle)
        story.append(Spacer(1, 6))

        info = Paragraph(f"Report Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles['Normal'])
        story.append(info)
        story.append(Spacer(1, 24))

        # 6. Data preparation for the table
        table_data = []

        # Table Headers
        table_data.append([
            "Anomaly", "Attempts", "Successes", "% Succ.",
            "Avg RT (s)", "Score (0-100)"
        ])

        # Data Population
        for item in summary:
            tentativi = item['tentativi']
            punteggio_totale = item['punteggio']
            punteggio_norm = 0.0
            if tentativi > 0:
                punteggio_norm = punteggio_totale / tentativi
            table_data.append([
                item['anomalia'],
                str(item['tentativi']),
                str(item['successi']),
                f"{item['percentuale_successo']:.1f}%",
                f"{item['tempo_medio']:.1f}",
                f"{punteggio_norm:.1f}"
            ])

        # 7. Table Creation and Styling
        # Calculate widths flexibly
        col_widths = [doc.width * 0.35, doc.width * 0.11, doc.width * 0.11, doc.width * 0.13, doc.width * 0.15,
                      doc.width * 0.15]

        table = Table(table_data, colWidths=col_widths)

        # Table Style Definition
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Header row background
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),  # Left align Anomaly column
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ])

        # Apply alternating colors for readability
        for i in range(1, len(table_data)):
            if i % 2 == 0:
                style.add('BACKGROUND', (0, i), (-1, i), colors.white)

        table.setStyle(style)
        story.append(table)

        # 8. Final Generation and Feedback
        try:
            # Passiamo la funzione _add_header agli argomenti onFirstPage e onLaterPages
            doc.build(
                story,
                onFirstPage=self._add_header,
                onLaterPages=self._add_header
            )
            print(f"PDF Report successfully generated in: {filename}")
            # Optional: open PDF file after generation
            webbrowser.open_new(filename)  # Use open_new for safety
        except Exception as e:
            print(f"ERROR during PDF creation: {e}")

        # 9. Summary print to console (useful for debug)
        print(f"\n--- Performance Summary of {self.current_subject.firstname} {self.current_subject.lastname}")
        for item in summary:
            # Calculation for console too
            tents = item['tentativi']
            score_avg = (item['punteggio'] / tents) if tents > 0 else 0.0

            print(f"\nTested Anomaly: {item['anomalia']}")
            print(f"  - Repetitions: {tents}, Successes: {item['successi']} ({item['percentuale_successo']:.1f}%)")
            print(f"  - Response Time (avg): {item['tempo_medio']:.1f}s, Norm Score: {score_avg:.1f}/100")
        print("\n")

    def closeEvent(self, event):
        """Ensures database connection closure when window closes."""
        # ... (no changes here) ...
        print("Closing application...")
        self.db_manager.close()
        event.accept()