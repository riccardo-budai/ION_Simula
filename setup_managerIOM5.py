"""
    setup_managerIOM.py
    Manages Surgery Setup, Montage selection, Headstage Channel Mapping, and Stimulator Setup.
"""

import json
import os
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QGroupBox, QCheckBox, QPushButton,
                             QGridLayout, QFileDialog, QMessageBox, QLineEdit,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QWidget)
from PySide6.QtCore import Signal, Qt


class IOMSetupDialog(QDialog):
    # Segnale che emette la configurazione completa quando si clicca "Confirm"
    montage_confirmed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Titolo e Dimensioni
        self.setWindowTitle("IOM Surgery Setup & Hardware Manager")
        self.resize(980, 750)

        # Stile Scuro (Dark Theme)
        self.setStyleSheet("""
            QDialog { background-color: #333; color: white; }
            QGroupBox { border: 1px solid #555; margin-top: 10px; padding-top: 10px; font-weight: bold; }
            QGroupBox::title { color: #aaa; subcontrol-origin: margin; left: 10px; }
            QLabel { color: #ddd; }
            QComboBox { background-color: #444; color: white; padding: 5px; }
            QLineEdit { background-color: #222; color: #00ccff; padding: 5px; border: 1px solid #555; font-style: italic; }
            QLineEdit:focus { border: 1px solid #00ccff; background-color: #000; }
            QCheckBox { color: #aaa; }
            QCheckBox::indicator:checked { background-color: #00cc00; border: 1px solid #00ff00; }

            /* Tab Widget Styles */
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #444; color: #aaa; padding: 10px; min-width: 100px; }
            QTabBar::tab:selected { background: #666; color: white; border-bottom: 2px solid #00ccff; }

            /* Table Styles */
            QTableWidget { background-color: #222; gridline-color: #444; color: #ddd; alternate-background-color: #2a2a2a; }
            QHeaderView::section { background-color: #444; color: white; padding: 4px; border: 1px solid #555; }
        """)

        # --- CARICAMENTO PROCEDURE DA JSON ESTERNO ---
        self.surgical_procedures_data = self.load_procedures_json("sets/procedures_surg.json")

        # --- DEFINIZIONE MONTAGGI STANDARD (PRESETS) ---
        # Manteniamo i preset per la logica dei moduli, ma le procedure verranno dal JSON.
        # Nota: Dovrai assicurarti che le stringhe nel JSON matchino con queste chiavi se vuoi l'auto-selezione dei moduli.
        # Altrimenti, l'utente dovrà selezionare manualmente i moduli dopo aver scelto la procedura.
        self.montage_presets = {
            ("brain_surgery", "Standard Motor Mapping"): {  # Esempio di chiave generica
                "modules": ["VIRTUAL_PATIENT", "ANESTHESIA", "EEG", "ECOG", "MEP_CONTRA", "SEP_UL", "SEP_LL", "EMG_FREE"],
                "desc": "Standard Motor Mapping: Cortical & Peripheral monitoring."
            },
            # ... (Puoi mappare le procedure specifiche del JSON qui se vuoi automazione completa) ...
        }

        # Mappe interne
        self.channel_map = {}
        self.stim_map = {}

        self.init_ui()

    def load_procedures_json(self, filepath):
        """Carica il file JSON delle procedure chirurgiche."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get("surgical_procedures", {})
        except FileNotFoundError:
            QMessageBox.warning(self, "Warning", f"File '{filepath}' not found. Using empty procedures list.")
            return {}
        except json.JSONDecodeError:
            QMessageBox.critical(self, "Error", f"Error decoding '{filepath}'. Invalid JSON format.")
            return {}

    def init_ui(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # --- TAB WIDGET ---
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # TAB 1: General Setup
        self.tab_general = QWidget()
        self.setup_general_tab()
        self.tabs.addTab(self.tab_general, "1. Surgery & Modules")

        # TAB 2: Channel Mapping (Inputs)
        self.tab_channels = QWidget()
        self.setup_channels_tab()
        self.tabs.addTab(self.tab_channels, "2. Headstage Inputs")

        # TAB 3: Stimulator Mapping (Outputs)
        self.tab_stims = QWidget()
        self.setup_stims_tab()
        self.tabs.addTab(self.tab_stims, "3. Stimulator Outputs")

        # --- BOTTONI GLOBALI (SAVE/LOAD/CONFIRM) ---

        # File Operations
        file_box = QHBoxLayout()
        self.btn_load = QPushButton("📂 Load Setup (JSON)")
        self.btn_load.setStyleSheet("background-color: #555; padding: 5px;")
        self.btn_load.clicked.connect(self.load_setup_from_json)

        self.btn_save = QPushButton("💾 Save Setup (JSON)")
        self.btn_save.setStyleSheet("background-color: #555; padding: 5px;")
        self.btn_save.clicked.connect(self.save_setup_to_json)

        file_box.addWidget(self.btn_load)
        file_box.addWidget(self.btn_save)
        main_layout.addLayout(file_box)

        # Action Buttons
        btn_box = QHBoxLayout()
        self.btn_apply = QPushButton("🚀 CONFIRM SETUP & LAUNCH")
        self.btn_apply.setFixedHeight(45)
        self.btn_apply.setStyleSheet("""
            QPushButton { background-color: #006600; font-weight: bold; font-size: 14px; color: white; border-radius: 5px; }
            QPushButton:disabled { background-color: #222; color: #777; }
            QPushButton:hover { background-color: #008800; }
        """)
        self.btn_apply.clicked.connect(self.confirm_setup)
        self.btn_apply.setEnabled(False)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("padding: 10px;")
        btn_cancel.clicked.connect(self.reject)

        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(self.btn_apply)
        main_layout.addLayout(btn_box)

    # -------------------------------------------------------------------------
    # TAB 1: GENERAL SETUP UI
    # -------------------------------------------------------------------------
    def setup_general_tab(self):
        layout = QVBoxLayout()
        self.tab_general.setLayout(layout)

        # A. Selezione Chirurgia
        group_select = QGroupBox("Surgery Definition")
        grid_sel = QGridLayout()

        self.combo_type = QComboBox()
        self.combo_type.addItem("Select Type...")

        # Popola tipi dinamicamente dal JSON caricato
        if self.surgical_procedures_data:
            # Formatta le chiavi per renderle più leggibili (es. brain_surgery -> Brain Surgery)
            types = sorted(self.surgical_procedures_data.keys())
            self.combo_type.addItems(types)

        self.combo_type.currentTextChanged.connect(self.update_locations)

        self.combo_location = QComboBox()
        self.combo_location.setEnabled(False)
        # self.combo_location.currentTextChanged.connect(self.update_montage_preview) # Disabilitato per ora se non c'è mapping 1:1

        self.combo_laterality = QComboBox()
        self.combo_laterality.addItems(["Right", "Left", "Median", "Paramedian R", "Paramedian L"])
        self.combo_laterality.currentTextChanged.connect(self.regenerate_maps)

        grid_sel.addWidget(QLabel("Surgery Type:"), 0, 0)
        grid_sel.addWidget(self.combo_type, 0, 1)
        grid_sel.addWidget(QLabel("Surgery Procedure:"), 1, 0)
        grid_sel.addWidget(self.combo_location, 1, 1)
        grid_sel.addWidget(QLabel("Lesion Side:"), 2, 0)
        grid_sel.addWidget(self.combo_laterality, 2, 1)

        group_select.setLayout(grid_sel)
        layout.addWidget(group_select)

        # B. Moduli (Checkboxes)
        self.group_montage = QGroupBox("Active IOM Montage (Modules)")
        self.montage_layout = QGridLayout()
        self.module_checks = {}

        all_modules = [
            "ANESTHESIA", "EEG", "ECOG", "MEP_UpperLimb", "MEP_LowerLimb",
            "SEP_UpperLimb", "SEP_LowerLimb", "BAEP", "VEP", "EMG_FREE", "EMG_TRIGGERED",
            "EMG_CRANIAL", "DWAVE", "ANOMALY_INJECTION", "VIRTUAL_PATIENT"
        ]

        row, col = 0, 0
        for mod in all_modules:
            chk = QCheckBox(mod)
            self.module_checks[mod] = chk
            chk.clicked.connect(self.on_manual_change)
            chk.clicked.connect(self.regenerate_maps)

            self.montage_layout.addWidget(chk, row, col)
            col += 1
            if col > 3:
                col = 0
                row += 1

        self.group_montage.setLayout(self.montage_layout)
        layout.addWidget(self.group_montage)

        # C. Descrizione
        layout.addWidget(QLabel("Montage Description / Notes:"))
        self.edit_desc = QLineEdit()
        self.edit_desc.setPlaceholderText("Description auto-generated or custom...")
        layout.addWidget(self.edit_desc)

    # -------------------------------------------------------------------------
    # TAB 2: CHANNEL MAPPING UI
    # -------------------------------------------------------------------------
    def setup_channels_tab(self):
        layout = QVBoxLayout()
        self.tab_channels.setLayout(layout)

        layout.addWidget(QLabel("Headstage Channel Assignment (Recording Inputs)"))

        # Tabella Canali
        self.table_channels = QTableWidget(32, 3)
        self.table_channels.setHorizontalHeaderLabels(["Channel ID", "Signal/Muscle", "Side"])
        self.table_channels.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_channels.verticalHeader().setVisible(False)
        self.table_channels.setAlternatingRowColors(True)

        # Inizializza vuota
        for i in range(32):
            self.table_channels.setItem(i, 0, QTableWidgetItem(f"CH {i + 1}"))
            self.table_channels.setItem(i, 1, QTableWidgetItem("-"))
            self.table_channels.setItem(i, 2, QTableWidgetItem("-"))

        layout.addWidget(self.table_channels)

    # -------------------------------------------------------------------------
    # TAB 3: STIMULATOR MAPPING UI
    # -------------------------------------------------------------------------
    def setup_stims_tab(self):
        layout = QVBoxLayout()
        self.tab_stims.setLayout(layout)

        lbl = QLabel("Electrical & Sensory Stimulators Assignment (Outputs)")
        lbl.setStyleSheet("font-weight: bold; color: #ff9900;")
        layout.addWidget(lbl)

        self.table_stims = QTableWidget(10, 4)
        self.table_stims.setHorizontalHeaderLabels(["Type", "ID", "Site/Nerve/Side", "Notes"])
        self.table_stims.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table_stims.verticalHeader().setVisible(False)
        self.table_stims.setAlternatingRowColors(True)

        for i in range(10):
            self.table_stims.setItem(i, 0, QTableWidgetItem("-"))
            self.table_stims.setItem(i, 1, QTableWidgetItem(f"STIM {i + 1}"))
            self.table_stims.setItem(i, 2, QTableWidgetItem("-"))
            self.table_stims.setItem(i, 3, QTableWidgetItem("-"))

        layout.addWidget(self.table_stims)

        btn_refresh = QPushButton("Recalculate Hardware Map")
        btn_refresh.clicked.connect(self.regenerate_maps)
        layout.addWidget(btn_refresh)

    # -------------------------------------------------------------------------
    # LOGICA DI UI & PRESETS (AGGIORNATA CON JSON)
    # -------------------------------------------------------------------------
    def update_locations(self, surgery_type):
        """Popola le procedure in base al tipo di chirurgia selezionato dal JSON."""
        self.combo_location.blockSignals(True)
        self.combo_location.clear()
        self.combo_location.addItem("Select Procedure...")

        if surgery_type != "Select Type..." and surgery_type in self.surgical_procedures_data:
            procedures = self.surgical_procedures_data[surgery_type]

            # Gestione struttura nidificata (es. spine_surgery -> cervical -> procedure)
            if isinstance(procedures, dict):
                # Se è un dizionario (come spine_surgery), appiattiamo la lista o usiamo sottocategorie
                # Qui aggiungiamo tutto in un'unica lista formattata "Categoria: Procedura"
                for subcat, proc_list in procedures.items():
                    for proc in proc_list:
                        self.combo_location.addItem(f"{subcat.upper()}: {proc}")
            elif isinstance(procedures, list):
                # Se è una lista semplice (come brain_surgery)
                self.combo_location.addItems(procedures)

            self.combo_location.setEnabled(True)
        else:
            self.combo_location.setEnabled(False)

        self.combo_location.blockSignals(False)

    def update_montage_preview(self, location):
        """
        NOTA: Poiché ora carichiamo procedure dinamiche dal JSON, il mapping diretto
        con self.montage_presets potrebbe non funzionare se le stringhe non coincidono.

        Per ora, questa funzione è disabilitata nella connessione del segnale init_ui
        per permettere la selezione manuale dei moduli.

        In futuro, si può implementare una logica di "keyword matching" per indovinare il setup.
        Es. se la stringa contiene "acoustic neuroma" -> attiva BAEP e Facial Nerve.
        """
        pass  # Placeholder per logica futura

    def on_manual_change(self):
        self.btn_apply.setEnabled(True)
        if "Standard" in self.edit_desc.text() or not self.edit_desc.text():
            self.edit_desc.setText("Custom configuration (Modified)")

    def reset_checkboxes(self):
        for chk in self.module_checks.values():
            chk.setChecked(False)
            chk.setStyleSheet("color: #aaa;")

    # -------------------------------------------------------------------------
    # LOGICA MAPPING (CHANNELS + STIMULATORS)
    # -------------------------------------------------------------------------
    def regenerate_maps(self):
        self.regenerate_channel_map()
        self.regenerate_stimulator_map()

    def regenerate_channel_map(self):
        self.channel_map.clear()
        ch_idx = 1

        active_mods = [m for m, c in self.module_checks.items() if c.isChecked()]
        lat = self.combo_laterality.currentText()

        contra_side = "Left" if "Right" in lat else ("Right" if "Left" in lat else "Bi")
        ipsi_side = "Right" if "Right" in lat else ("Left" if "Left" in lat else "Bi")

        # 1. EEG
        if "EEG" in active_mods:
            for lbl in ["C3", "C4", "Cz", "Fz"]:
                if ch_idx > 32: break
                self.channel_map[f"CH{ch_idx}"] = {"name": f"EEG {lbl}", "side": "Scalp"}
                ch_idx += 1

        # 2. SEP
        if "SEP_UL" in active_mods:
            for name in ["Erb Point", "Cv7 (N13)", "C3'/C4' (N20)"]:
                if ch_idx > 32: break
                self.channel_map[f"CH{ch_idx}"] = {"name": f"SEP {name}", "side": "Bi"}
                ch_idx += 1

        if "SEP_LL" in active_mods:
            for name in ["Popliteal Fossa", "L1 (N22)", "Cz' (P37)"]:
                if ch_idx > 32: break
                self.channel_map[f"CH{ch_idx}"] = {"name": f"SEP {name}", "side": "Bi"}
                ch_idx += 1

        # 3. MUSCLES
        target_muscles = []
        muscles_ul = ["APB", "ADM", "Biceps", "Deltoid"]
        muscles_ll = ["Tibialis Ant.", "Abd. Hallucis", "Quadriceps"]
        muscles_cranial = ["Orb. Oculi", "Orb. Oris", "Mentalis", "Trapezius"]

        if "EMG_CRANIAL" in active_mods:
            target_muscles.extend([(m, ipsi_side) for m in muscles_cranial])

        if "MEP_FOUR_LIMB" in active_mods or "EMG_FREE" in active_mods:
            target_muscles.extend([(m, "Left") for m in muscles_ul + muscles_ll])
            target_muscles.extend([(m, "Right") for m in muscles_ul + muscles_ll])
        elif "MEP_CONTRA" in active_mods:
            target_muscles.extend([(m, contra_side) for m in muscles_ul + muscles_ll])
        elif "MEP_LL" in active_mods:
            target_muscles.extend([(m, "Left") for m in muscles_ll])
            target_muscles.extend([(m, "Right") for m in muscles_ll])

        target_muscles = list(dict.fromkeys(target_muscles))

        for m_name, m_side in target_muscles:
            if ch_idx > 31: break
            self.channel_map[f"CH{ch_idx}"] = {"name": f"{m_name} (+)", "side": m_side}
            ch_idx += 1
            self.channel_map[f"CH{ch_idx}"] = {"name": f"{m_name} (-)", "side": m_side}
            ch_idx += 1

        # 4. DWAVE
        if "DWAVE" in active_mods:
            if ch_idx <= 32:
                self.channel_map[f"CH{ch_idx}"] = {"name": "D-Wave", "side": "Spine"}
                ch_idx += 1

        self.refresh_channel_table()

    def regenerate_stimulator_map(self):
        self.stim_map.clear()
        stim_idx = 1

        active_mods = [m for m, c in self.module_checks.items() if c.isChecked()]

        if any(x in active_mods for x in ["MEP_CONTRA", "MEP_FOUR_LIMB", "MEP_LL"]):
            self.stim_map[f"STIM{stim_idx}"] = {
                "type": "Electrical (HV)",
                "site": "C3-C4 (Corkscrew)",
                "notes": "Transcranial (Multipulse)"
            }
            stim_idx += 1

        if "SEP_UL" in active_mods:
            self.stim_map[f"STIM{stim_idx}"] = {"type": "Electrical (CC)", "site": "Median Nerve (L)", "notes": "Wrist"}
            stim_idx += 1
            self.stim_map[f"STIM{stim_idx}"] = {"type": "Electrical (CC)", "site": "Median Nerve (R)", "notes": "Wrist"}
            stim_idx += 1

        if "SEP_LL" in active_mods:
            self.stim_map[f"STIM{stim_idx}"] = {"type": "Electrical (CC)", "site": "Tibial Post. (L)", "notes": "Ankle"}
            stim_idx += 1
            self.stim_map[f"STIM{stim_idx}"] = {"type": "Electrical (CC)", "site": "Tibial Post. (R)", "notes": "Ankle"}
            stim_idx += 1

        if "EMG_TRIGGERED" in active_mods or "EMG_CRANIAL" in active_mods:
            self.stim_map[f"STIM{stim_idx}"] = {
                "type": "Electrical (Probe)",
                "site": "Handheld Probe",
                "notes": "Monopolar/Bipolar (Surgeon)"
            }
            stim_idx += 1

        if "BAEP" in active_mods:
            self.stim_map[f"STIM{stim_idx}"] = {"type": "Acoustic (Click)", "site": "Ear Insert (L)",
                                                "notes": "Broadband Click"}
            stim_idx += 1
            self.stim_map[f"STIM{stim_idx}"] = {"type": "Acoustic (Click)", "site": "Ear Insert (R)",
                                                "notes": "Broadband Click"}
            stim_idx += 1

        if "VEP" in active_mods:
            self.stim_map[f"STIM{stim_idx}"] = {"type": "Visual (LED)", "site": "Goggles (Bi)",
                                                "notes": "Flash Stimulation"}
            stim_idx += 1

        if "ECOG" in active_mods:
            self.stim_map[f"STIM{stim_idx}"] = {"type": "Electrical (Cortical)", "site": "Strip/Grid",
                                                "notes": "Direct Cortical Stim"}
            stim_idx += 1

        self.refresh_stim_table()

    def refresh_channel_table(self):
        self.table_channels.blockSignals(True)
        self.table_channels.clearContents()

        for i in range(32):
            self.table_channels.setItem(i, 0, QTableWidgetItem(f"CH {i + 1}"))
            self.table_channels.setItem(i, 1, QTableWidgetItem("-"))
            self.table_channels.setItem(i, 2, QTableWidgetItem("-"))

        for k, v in self.channel_map.items():
            try:
                idx = int(k.replace("CH", "")) - 1
                if 0 <= idx < 32:
                    self.table_channels.setItem(idx, 1, QTableWidgetItem(v["name"]))
                    self.table_channels.setItem(idx, 2, QTableWidgetItem(v["side"]))
            except:
                pass

        self.table_channels.blockSignals(False)

    def refresh_stim_table(self):
        self.table_stims.blockSignals(True)
        self.table_stims.clearContents()

        for i in range(10):
            self.table_stims.setItem(i, 0, QTableWidgetItem("-"))
            self.table_stims.setItem(i, 1, QTableWidgetItem(f"STIM {i + 1}"))
            self.table_stims.setItem(i, 2, QTableWidgetItem("-"))
            self.table_stims.setItem(i, 3, QTableWidgetItem("-"))

        for k, v in self.stim_map.items():
            try:
                idx = int(k.replace("STIM", "")) - 1
                if 0 <= idx < 10:
                    self.table_stims.setItem(idx, 0, QTableWidgetItem(v["type"]))
                    self.table_stims.setItem(idx, 2, QTableWidgetItem(v["site"]))
                    self.table_stims.setItem(idx, 3, QTableWidgetItem(v["notes"]))
            except:
                pass

        self.table_stims.blockSignals(False)

    # -------------------------------------------------------------------------
    # SAVE / LOAD
    # -------------------------------------------------------------------------
    def save_setup_to_json(self):
        active_modules = [mod for mod, chk in self.module_checks.items() if chk.isChecked()]
        if not active_modules:
            QMessageBox.warning(self, "Warning", "Setup empty. Select modules first.")
            return

        setup_data = {
            "surgery_type": self.combo_type.currentText(),
            "location": self.combo_location.currentText(),
            "laterality": self.combo_laterality.currentText(),
            "description": self.edit_desc.text(),
            "active_modules": active_modules,
            "channel_mapping": self.channel_map,
            "stimulator_mapping": self.stim_map
        }

        filename, _ = QFileDialog.getSaveFileName(self, "Save Setup", "sets/custom_setup.json", "JSON (*.json)")
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(setup_data, f, indent=4)
                QMessageBox.information(self, "Saved", f"Setup saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def load_setup_from_json(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Setup", "sets/", "JSON (*.json)")
        if not filename: return

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.combo_type.blockSignals(True)
            self.combo_location.blockSignals(True)
            self.combo_laterality.blockSignals(True)

            try:
                # 1. General Info
                loaded_type = data.get("surgery_type", "")
                # Se il tipo non è in quelli standard del JSON, lo aggiungiamo
                if self.combo_type.findText(loaded_type) == -1: self.combo_type.addItem(loaded_type)
                self.combo_type.setCurrentText(loaded_type)

                # Aggiorniamo le location possibili ma selezioniamo quella del file
                # Se è una location custom (non nel JSON procedure), la aggiungiamo
                loaded_loc = data.get("location", "")
                # (Qui non puliamo la combo se non necessario, o ricarichiamo le opzioni valide per quel tipo)
                # Per semplicità, forziamo il testo
                self.combo_location.clear()
                self.combo_location.addItem(loaded_loc)
                self.combo_location.setCurrentText(loaded_loc)
                self.combo_location.setEnabled(True)

                self.combo_laterality.setCurrentText(data.get("laterality", "Right"))
                self.edit_desc.setText(data.get("description", ""))

                # 2. Modules
                self.reset_checkboxes()
                for mod in data.get("active_modules", []):
                    if mod in self.module_checks:
                        self.module_checks[mod].setChecked(True)
                        self.module_checks[mod].setStyleSheet("color: white; font-weight: bold;")

                # 3. Maps
                if "channel_mapping" in data:
                    self.channel_map = data["channel_mapping"]
                    self.refresh_channel_table()
                else:
                    self.regenerate_channel_map()

                if "stimulator_mapping" in data:
                    self.stim_map = data["stimulator_mapping"]
                    self.refresh_stim_table()
                else:
                    self.regenerate_stimulator_map()

                self.btn_apply.setEnabled(True)

            finally:
                self.combo_type.blockSignals(False)
                self.combo_location.blockSignals(False)
                self.combo_laterality.blockSignals(False)

            QMessageBox.information(self, "Loaded", "Setup loaded successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed: {str(e)}")

    def confirm_setup(self):
        active_modules = [mod for mod, chk in self.module_checks.items() if chk.isChecked()]
        if not active_modules:
            QMessageBox.warning(self, "Warning", "Select at least one module.")
            return

        config = {
            "surgery_type": self.combo_type.currentText(),
            "location": self.combo_location.currentText(),
            "laterality": self.combo_laterality.currentText(),
            "active_modules": active_modules,
            "channel_mapping": self.channel_map,
            "stimulator_mapping": self.stim_map
        }

        self.montage_confirmed.emit(config)
        self.accept()
