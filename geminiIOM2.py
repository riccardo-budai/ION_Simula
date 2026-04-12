import os
import logging
import time
from datetime import datetime

# --- PYQT6 IMPORTS ---
from PySide6.QtCore import Qt, Signal, QObject, QThread, Slot
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QTextEdit, QMessageBox)

# --- GOOGLE GEMINI IMPORT ---
import google.generativeai as genai


# ========================================================================================
# 1. WORKER (Resta uguale)
# ========================================================================================
class GenerateResponseWorker(QObject):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, model_name, api_key, prompt, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.api_key = api_key
        self.prompt_text = prompt

    @Slot()
    def run(self):
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(self.prompt_text)
            self.finished.emit(response.text)
        except Exception as e:
            self.error.emit(str(e))


# ========================================================================================
# 2. FINESTRA DI INTERAZIONE (ChildWindow - La parte Grafica)
# ========================================================================================
class ChildWindow(QWidget):
    generationRequested = Signal(str, str)

    def __init__(self, mode="input", api_key=None, parent=None):
        # NOTA: Passiamo parent=None per renderla una finestra totalmente indipendente
        # visivamente, ma gestiamo la memoria manualmente se necessario.
        super().__init__(None)

        # Impostazioni finestra
        self.setWindowFlag(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.mode = mode
        self.api_key = api_key
        self.controller_ref = parent  # Manteniamo un riferimento al controller logico

        self.initUI()

        if self.mode == "input" and self.api_key:
            self.populate_models()

    def initUI(self):
        self.setWindowTitle("Interazione AI - Gemini")
        self.setGeometry(100, 100, 500, 400)

        # Stile
        self.setStyleSheet("""
            QWidget { background-color: #202020; font-family: Segoe UI; }
            QTextEdit { background-color: gray; color: blue; border: 1px solid #ccc; font-size: 12pt; padding: 5px; }
            QPushButton { background-color: #0078d7; color: white; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #005a9e; }
            QLabel { font-weight: bold; color: #333; }
            QComboBox { padding: 5px; }
        """)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Header
        header_layout = QHBoxLayout()
        self.lblInfo = QLabel("Prompt:" if self.mode == "input" else "Risposta Generata:")
        header_layout.addWidget(self.lblInfo)

        self.comboModel = QComboBox()
        if self.mode == "input":
            header_layout.addStretch()
            header_layout.addWidget(QLabel("Modello:"))
            self.comboModel.setMinimumWidth(200)
            header_layout.addWidget(self.comboModel)
        layout.addLayout(header_layout)

        # Body
        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)

        # Footer
        self.lblFilename = QLabel("")
        self.lblFilename.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.lblFilename)

        btn_layout = QHBoxLayout()
        if self.mode == "input":
            self.btnSaveDraft = QPushButton("Save Draft")
            self.btnGenerate = QPushButton("Save & Generate")
            self.btnExit = QPushButton("Close")

            self.btnSaveDraft.clicked.connect(self.save_draft)
            self.btnGenerate.clicked.connect(self.save_and_generate)
            self.btnExit.clicked.connect(self.close)

            btn_layout.addWidget(self.btnSaveDraft)
            btn_layout.addWidget(self.btnGenerate)
            btn_layout.addWidget(self.btnExit)
        else:
            self.btnClose = QPushButton("Close Gemini")
            self.btnClose.clicked.connect(self.close)
            btn_layout.addWidget(self.btnClose)
            self.text_edit.setReadOnly(True)

        layout.addLayout(btn_layout)

    def populate_models(self):
        self.comboModel.clear()
        self.comboModel.addItem("Caricamento...")
        safe_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
        self.comboModel.addItems(safe_models)
        self.comboModel.setCurrentIndex(1)  # Default sul primo safe (flash)

        try:
            genai.configure(api_key=self.api_key)
            found_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    clean_name = m.name.replace('models/', '')
                    found_models.append(clean_name)

            self.comboModel.clear()
            if found_models:
                found_models.sort(reverse=True)
                self.comboModel.addItems(found_models)
            else:
                self.comboModel.addItems(safe_models)
        except Exception:
            self.comboModel.clear()
            self.comboModel.addItems(safe_models)

    def save_draft(self):
        text = self.text_edit.toPlainText()
        if not text.strip(): return None
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = os.path.join("prompts", f"prompt_{timestamp}.txt")
        try:
            with open(filename, "w", encoding='utf-8') as f:
                f.write(text)
            self.lblFilename.setText(f"Salvato: {filename}")
            return filename
        except Exception:
            return None

    def save_and_generate(self):
        saved_file = self.save_draft()
        if saved_file:
            selected_model = self.comboModel.currentText()
            self.generationRequested.emit(saved_file, selected_model)
            self.close()
        else:
            QMessageBox.warning(self, "Attenzione", "Prompt vuoto.")


# ========================================================================================
# 3. CONTROLLER (LOGICA PURA - ORA COMPATIBILE CON MAIN)
# ========================================================================================
class gemini_controlIOM(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.logger = logging.getLogger(__name__)
        # Sostituisci con la tua chiave o usa una variabile d'ambiente
        self.API_KEY = 'AIzaSyCd0zvMqCrVv4LPP15F__5RcZOQfV2h1Tw'

        self._ensure_directories()

        # Variabili di stato
        self.generation_thread = None
        self.generation_worker = None
        self.current_window = None

        # --- METODI DI COMPATIBILITÀ (TRUCCHI PER MAIN.PY) ---

    def show(self):
        """Simula il metodo show() di un widget."""
        self.newPrompt()

    def isVisible(self):
        """
        Versione sicura: gestisce il caso in cui la finestra è stata
        distrutta dal C++ ma esiste ancora in Python.
        """
        if self.current_window is None:
            return False

        try:
            # Proviamo a chiedere se è visibile
            return self.current_window.isVisible()
        except RuntimeError:
            # Se otteniamo RuntimeError ("wrapped C/C++ object has been deleted")
            # significa che la finestra è stata chiusa.
            self.current_window = None
            return False

    # -----------------------------------------------------

    def _ensure_directories(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dir_prompts = os.path.join(base_dir, "prompts")
        self.dir_responses = os.path.join(base_dir, "responses")
        os.makedirs(self.dir_prompts, exist_ok=True)
        os.makedirs(self.dir_responses, exist_ok=True)

    def _on_window_destroyed(self):
        """Callback chiamata automaticamente quando la finestra viene chiusa."""
        self.current_window = None

    def newPrompt(self):
        # Chiudiamo eventuali finestre vecchie ancora vive
        if self.current_window is not None:
            try:
                self.current_window.close()
            except RuntimeError:
                pass  # Era già morta
            self.current_window = None

        # Creiamo la nuova finestra
        self.current_window = ChildWindow(mode="input", api_key=self.API_KEY, parent=self)

        # --- FIX CRITICO: Colleghiamo il segnale di distruzione ---
        # Quando la ChildWindow si chiude (WA_DeleteOnClose), avvisa il controller
        # di impostare self.current_window a None.
        self.current_window.destroyed.connect(self._on_window_destroyed)
        # ----------------------------------------------------------

        self.current_window.generationRequested.connect(self.handle_generation_request)
        self.current_window.show()

    def handle_generation_request(self, prompt_filename, model_name):
        self.logger.info(f"GEMINI: Avvio generazione. Modello: {model_name}")

        try:
            with open(prompt_filename, "r", encoding='utf-8') as f:
                prompt_text = f.read()
        except FileNotFoundError:
            return

        self.generation_thread = QThread()
        self.generation_worker = GenerateResponseWorker(model_name, self.API_KEY, prompt_text)
        self.generation_worker.moveToThread(self.generation_thread)

        self.generation_thread.started.connect(self.generation_worker.run)
        self.generation_worker.finished.connect(lambda resp: self.on_generation_success(resp, prompt_text, model_name))
        self.generation_worker.error.connect(self.on_generation_error)

        self.generation_worker.finished.connect(self.generation_thread.quit)
        self.generation_worker.deleteLater()
        self.generation_thread.finished.connect(self.generation_thread.deleteLater)

        self.generation_thread.start()

    def on_generation_success(self, response_text, original_prompt, model_used):
        self.logger.info("GEMINI: Generazione completata.")

        saved_path = self._save_full_response(original_prompt, response_text, model_used)

        # Chiudiamo l'input se ancora aperto e mostriamo l'output
        if self.current_window:
            self.current_window.close()

        self.current_window = ChildWindow(mode="output", parent=self)
        self.current_window.text_edit.setPlainText(response_text)
        if saved_path:
            filename = os.path.basename(saved_path)
            self.current_window.lblFilename.setText(f"Report: {filename}")
        self.current_window.show()

    def _save_full_response(self, prompt, response, model):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ts_file = datetime.now().strftime('%Y%m%d_%H%M%S')

        content = f"DATA: {timestamp}\nMODELLO: {model}\n\n--- PROMPT ---\n{prompt}\n\n--- RISPOSTA ---\n{response}"
        filepath = os.path.join(self.dir_responses, f"resp_{ts_file}.txt")

        try:
            with open(filepath, "w", encoding='utf-8') as f:
                f.write(content)
            return filepath
        except Exception as e:
            self.logger.error(str(e))
            return None

    def on_generation_error(self, error_msg):
        self.logger.error(f"GEMINI Error: {error_msg}")
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Errore AI")
        msg.setText(error_msg)
        msg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        msg.exec()