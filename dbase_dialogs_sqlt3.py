# file: dialogs.py

import json
import os
import sys

from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QComboBox, QFormLayout, QHBoxLayout, QTextEdit, \
    QListWidget, QListWidgetItem, QLineEdit, QSpinBox, QMessageBox
from PySide6.QtCore import Signal, QSize, Qt, QFile
from PySide6.QtUiTools import QUiLoader

from datetime import datetime


########################################################################################################################
class appendRecord(QDialog):
    """
    Finestra di dialogo per Aggiungere o Modificare un record Soggetto.
    recMode: 0 per Aggiungi (Append), 1 per Modifica (Edit)
    recRow: Dizionario o oggetto dati da visualizzare/modificare
    """
    newDataRec = Signal(dict, bool)  # Segnale: (dati_aggiornati, successo)

    def __init__(self, recMode, recRow, parent=None):
        super().__init__(parent)
        self.recMode = recMode
        # Usiamo una copia per evitare modifiche indesiderate se si preme "Annulla"
        self.recRow = recRow.copy()

        self.setWindowTitle(f"{'Modifica' if recMode == 1 else 'Aggiungi'} Soggetto")
        self.setFixedSize(QSize(350, 380))
        self.move(50 + 800 + 7 + 420 + 3, 50)

        # --- Layout Principale e Form Layout ---
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # --- Informazioni Statiche (Non modificabili) ---
        mode_text = "Modifica dati utente" if recMode == 1 else "Aggiungi Nuovo Utente"
        id_text = f"ID Utente: {self.recRow.get('idcode', 'N/A')}" if recMode == 1 else "ID Utente: Nuovo (Auto-assegnato)"

        # Data di inserimento (statica)
        if self.recMode == 0:
            # Modalità Aggiungi: usa la data di ora
            default_date_obj = datetime.now()
        else:
            # Modalità Modifica: la data arriva come STRINGA dal database
            datetime_str_from_db = self.recRow.get('datetime')
            if isinstance(datetime_str_from_db, str):
                try:
                    # Convertiamo la stringa del DB in un oggetto datetime
                    # Questo formato ('%Y-%m-%d %H:%M:%S') è come lo salviamo
                    default_date_obj = datetime.strptime(datetime_str_from_db, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # Se la stringa è corrotta o in un formato strano, usiamo 'ora'
                    default_date_obj = datetime.now()
            else:
                # Fallback se non è una stringa (magari è già un datetime o None)
                default_date_obj = self.recRow.get('datetime', datetime.now())

        # Ora siamo sicuri che default_date_obj è un oggetto datetime
        # e possiamo formattarlo
        datetime_str = default_date_obj.strftime('%d-%m-%Y %H:%M')
        self.lbl_datetime = QLabel(datetime_str)

        # Mostra 'fname' come statico (non modificabile)
        fname_text = self.recRow.get('fname', 'N/A')
        if recMode == 0:
            fname_text = "(Verrà generato al salvataggio)"
        self.lbl_fname = QLabel(fname_text)

        main_layout.addWidget(QLabel(f"<b>{mode_text}</b>"))
        main_layout.addWidget(QLabel(id_text))
        main_layout.addSpacing(10)

        # --- Creazione Widget Modificabili ---

        # 1. Nome e Cognome (QLineEdit)
        self.txt_firstname = QLineEdit(self.recRow.get('firstname', ''))
        self.txt_lastname = QLineEdit(self.recRow.get('lastname', ''))
        # [MODIFICA OPZIONALE] Forza l'input maiuscolo in tempo reale (cosmetico)
        self.txt_firstname.textChanged.connect(lambda t: self.txt_firstname.setText(t.upper()))
        self.txt_lastname.textChanged.connect(lambda t: self.txt_lastname.setText(t.upper()))

        # 2. Età (QSpinBox, come richiesto per intero positivo)
        self.spin_age = QSpinBox()
        self.spin_age.setRange(1, 120)  # Range per intero positivo
        self.spin_age.setValue(self.recRow.get('age', 18))

        # 3. [MODIFICA] Genere (QComboBox)
        self.combo_gender = QComboBox()
        gender_options = ["M", "F", "Other"]
        self.combo_gender.addItems(gender_options)
        current_gender = self.recRow.get('gender', 'M')  # Default 'M'
        if current_gender not in gender_options:
            current_gender = "Other"  # Se il dato vecchio non è valido
        self.combo_gender.setCurrentText(current_gender)

        # 4. Level (QComboBox)
        self.combo_level = QComboBox()
        self.combo_level.addItems(["ENTRY", "ADVANCED", "TUTOR"])
        current_level = self.recRow.get('level', 'ENTRY')
        self.combo_level.setCurrentText(current_level)

        # 5. DbaseClass (QComboBox)
        self.combo_dbaseclass = QComboBox()
        db_class_options = ["ION-Sim", "ION-Adv", "ION-Tut"]
        self.combo_dbaseclass.addItems(db_class_options)
        current_dbaseclass = self.recRow.get('dbaseclass', 'ION-Sim')
        self.combo_dbaseclass.setCurrentText(current_dbaseclass)

        # --- Aggiunta dei widget al Form Layout ---
        form_layout.addRow("Name:", self.txt_firstname)
        form_layout.addRow("Surname:", self.txt_lastname)
        form_layout.addRow("Age:", self.spin_age)
        form_layout.addRow("Gender:", self.combo_gender)  # [MODIFICA] Sostituito QLineEdit
        form_layout.addRow("Level:", self.combo_level)
        form_layout.addRow("Classe (Gruppo):", self.combo_dbaseclass)
        form_layout.addRow("Fname (ID file):", self.lbl_fname)
        form_layout.addRow("Insertion Date:", self.lbl_datetime)

        main_layout.addLayout(form_layout)
        main_layout.addStretch()  # Spingitore

        # --- Pulsanti Salva e Annulla ---
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Discard")
        btn_save = QPushButton("Save...")

        # Connessioni
        btn_cancel.clicked.connect(self.reject)
        btn_save.clicked.connect(self._save_data)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)

        main_layout.addLayout(btn_layout)

    def _save_data(self):
        """
        Raccoglie i dati dai widget, li valida e li emette.
        """

        # 1. Raccogli i dati dai widget
        # [MODIFICA] Applica .upper() per forzare il maiuscolo
        firstname = self.txt_firstname.text().strip().upper()
        lastname = self.txt_lastname.text().strip().upper()

        age = self.spin_age.value()

        # [MODIFICA] Leggi i valori dai QComboBox
        gender = self.combo_gender.currentText()
        level = self.combo_level.currentText()
        dbaseclass = self.combo_dbaseclass.currentText()

        # 2. Validazione semplice (Nome e Cognome obbligatori)
        if not firstname or not lastname:
            QMessageBox.warning(self, "Dati Mancanti", "I campi 'Nome' e 'Cognome' non possono essere vuoti.")
            return

        # 3. Prepara il dizionario 'data_to_send'
        data_to_send = self.recRow.copy()

        # 4. Aggiorna il dizionario con i nuovi valori
        data_to_send['firstname'] = firstname  # Ora è maiuscolo
        data_to_send['lastname'] = lastname  # Ora è maiuscolo
        data_to_send['age'] = age
        data_to_send['gender'] = gender  # Dal ComboBox
        data_to_send['level'] = level
        data_to_send['dbaseclass'] = dbaseclass

        if self.recMode == 0:  # Solo se Aggiungi
            data_to_send['datetime'] = datetime.now()
            # [MODIFICA] Genera fname usando i valori MAIUSCOLI convertiti in minuscolo
            data_to_send['fname'] = f"{lastname.lower()}_{firstname.lower()}"

        # 6. Emetti il segnale e chiudi
        self.newDataRec.emit(data_to_send, True)
        self.accept()
        self.close()

########################################################################################################################
class appendSession(QDialog):
    """
    Finestra di dialogo per Aggiungere o Modificare un record Sessione.
    """
    newSessionRec = Signal(dict, bool)  # Segnale per la sessione

    def __init__(self, recMode, recRow, subject_id, scenarios, parent=None):
        super().__init__(parent)
        self.recMode = recMode
        self.recRow = recRow
        self.subject_id = subject_id
        self.scenarios = scenarios  # <--- NUOVO: memorizza la lista di oggetti Scenari
        self.setWindowTitle(f"{'Modifica' if recMode == 1 else 'Aggiungi'} Sessione")

        self.setFixedSize(QSize(400, 250))
        self.move(50+800+7+420+3, 50)
        # self.setGeometry(50+800+7+420+5, 60, 400, 250)

        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # Etichette di base
        mode_text = "Modifica " if recMode == 1 else "Aggiungi Nuova"
        session_id_text = self.recRow.get('idsession', 'Nuova')
        datetime_text = self.recRow.get('datetime', 'Adesso')

        main_layout.addWidget(QLabel
            #(f"{mode_text} Sessione ID: {subject_id}/{self.recRow['idsession']} [{self.recRow['datetime']}]"))
            (f"{mode_text} Sessione ID: {subject_id}/{session_id_text} [{datetime_text}]"))

        # QComboBox per la selezione dello scenario
        self.combo_scenario = QComboBox()
        self._populate_scenario_combo()

        # Aggiungi il campo al form
        form_layout.addRow("Selected Scenario:", self.combo_scenario)

        # Etichetta di riepilogo per mostrare i dettagli (opzionale)
        self.lbl_details = QLabel("Dettagli Scenario: Nessuno selezionato")
        self.lbl_details.setWordWrap(True)
        form_layout.addRow(" ", self.lbl_details)

        initial_notes = self.recRow.get('sesnote', '')
        self.current_notes = QTextEdit(initial_notes)
        form_layout.addRow("notes...", self.current_notes)

        # Connetti il segnale per aggiornare i dettagli (e memorizzare la selezione)
        self.combo_scenario.currentIndexChanged.connect(self._update_scenario_details)
        #
        main_layout.addLayout(form_layout)
        if self.recMode == 1:
            self._preselect_scenario()
        if self.recMode == 0 or self.combo_scenario.currentIndex() == -1:
            self._update_scenario_details(0)
        # Pulsanti
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Save data Session")
        btn_notsave = QPushButton("Exit and Discard")
        btn_save.clicked.connect(self._simulate_save)
        btn_notsave.clicked.connect(self._discard)
        btn_layout.addWidget(btn_notsave)
        btn_layout.addWidget(btn_save)
        #
        main_layout.addLayout(btn_layout)


    def _preselect_scenario(self):
        """Pre-seleziona lo scenario basato sul taskname del record sessione."""
        # Se non ci sono scenari o non abbiamo un taskname, non possiamo fare nulla
        if not self.scenarios or 'taskname' not in self.recRow:
            return
        current_taskname = self.recRow['taskname']

        index_to_select = -1
        # Iteriamo sugli elementi della ComboBox per trovare la corrispondenza
        for i in range(self.combo_scenario.count()):
            scenario = self.combo_scenario.itemData(i)
            # Dobbiamo confrontare l'identificatore dello scenario (label_id)
            # con l'identificatore salvato nella sessione (taskname)
            if scenario and scenario.label_id == current_taskname:
                index_to_select = i
                break
        if index_to_select != -1:
            # Blocco temporaneamente i segnali per evitare chiamate multiple
            self.combo_scenario.blockSignals(True)
            self.combo_scenario.setCurrentIndex(index_to_select)
            self.combo_scenario.blockSignals(False)
            # Chiamiamo _update_scenario_details per aggiornare i dati e self.selected_scenario
            self._update_scenario_details(index_to_select)
        else:
            print(
                f"AVVISO: Scenario '{current_taskname}' della sessione non trovato nell'elenco degli scenari disponibili.")

    def _populate_scenario_combo(self):
        """Popola la combobox con i dati della lista Scenari."""
        self.combo_scenario.clear()
        if not self.scenarios:
            self.combo_scenario.addItem("Nessuno scenario disponibile.")
            self.combo_scenario.setEnabled(False)
            return
        for scenario in self.scenarios:
            display_text = f"ID: {scenario.nome_scenario}"
            self.combo_scenario.addItem(display_text, userData=scenario)

    def _update_scenario_details(self, index):
        """Aggiorna l'etichetta con i dettagli dello scenario selezionato."""
        if index < 0 or not self.scenarios:
            self.lbl_details.setText("Nessuno scenario selezionato.")
            self.selected_scenario = None
            return
        # Recupera l'oggetto Scenari memorizzato nel userData
        scenario = self.combo_scenario.itemData(index)
        self.selected_scenario = scenario  # Memorizza l'oggetto selezionato
        # Aggiorna l'etichetta di riepilogo
        if scenario:
            # 1. Utilizza l'operatore 'or ""' per garantire che sia una stringa
            #    se l'attributo è None.
            descrizione = scenario.descrizione or ""
            # trigger_tipo = scenario.trigger_tipo or "N/A"
            # details = (f"Tipo: {trigger_tipo}, Descrizione: {descrizione[:30]}...")
            details = (f"({descrizione})")
            self.lbl_details.setText(details)
        else:
            # Questo gestisce il caso in cui itemData(index) restituisce None
            self.lbl_details.setText("Errore: Dati scenario non validi.")
            self.selected_scenario = None

    def _simulate_save(self):
        # Dati di prova per la sessione (da modificare con i dati reali)
        if self.selected_scenario is None:
            # Qui potresti mostrare un QMessageBox di errore
            print("Errore: Seleziona uno scenario valido prima di salvare.")
            return
        updated_notes = self.current_notes.toPlainText()
        # Raccogli i dati dello scenario selezionato
        selected_sc = self.selected_scenario
        # Dati di base per la sessione
        data_to_send = {
            'idcode': self.subject_id,
            'sesnote': updated_notes,
            'fname': f"{selected_sc.label_id}_{self.subject_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            # Nome file generato
            'dbasename': 'ION-Sim',     # per default
            'datetime': datetime.now(),
            'session': selected_sc.nome_scenario,  # Corrisponde a session
            'taskname': selected_sc.label_id
        }

        # Mappa i campi dello scenario sui campi della sessione
        data_to_send['session'] = selected_sc.nome_scenario     # Corrisponde a session
        data_to_send['taskname'] = selected_sc.label_id         # Corrisponde a taskname
        data_to_send['simulacode'] = selected_sc.label_id       # Corrisponde a simulacode

        if self.recMode == 0:
            # MODALITÀ AGGIUNGI: Imposta il simulacode predefinito (taskname + _1)
            # Questo garantisce che la nuova sessione abbia un codice anomalia di base.
            data_to_send['simulacode'] = f"{selected_sc.label_id}_1"
        if self.recMode == 1:
            # In modalità modifica, mantieni l'idsession e l'idcode originali
            data_to_send['idsession'] = self.recRow.get('idsession')
            # In un'app reale, qui faresti anche il merge con i campi modificabili
            # e non derivati dallo scenario (es. sesnote)
            data_to_send['fname'] = self.recRow.get('fname')

        self.newSessionRec.emit(data_to_send, True)
        self.accept()
        self.close()

    def _discard(self):
        self.close()

########################################################################################################################
class AnomalyDialog(QDialog):
    """
    Finestra di dialogo per visualizzare e gestire le anomalie (scenari_anomaly)
    associate a uno specifico scenario (scenari_setup).
    """
    newAnomalySaved = Signal(bool)  # Segnale di salvataggio/modifica completata
    anomaly_selected = Signal(dict)

    def __init__(self, scenario_data: dict, anomalies_data: list, level_code: str, parent=None):
        super().__init__(parent)
        self.scenario = scenario_data
        self.anomalies = anomalies_data
        self.level_code = level_code

        # Stili e setup di base
        self.setWindowTitle(f"Gestione Anomalie per Scenario: {self.scenario.get('nome_scenario', 'N/A')}")
        self.setFixedSize(QSize(600, 450))
        self.move(50 + 800 + 7 + 420 + 3, 50)
        # self.setFixedSize(QSize(600, 450))
        # self.move(parent.pos().x() + parent.width() + 20, parent.pos().y())

        main_layout = QVBoxLayout(self)
        # 1. Intestazione Scenario
        header_label = QLabel(f"Scenario ID:{self.scenario.get('scenario_id')} | {self.scenario.get('nome_scenario')}")
        header_label.setStyleSheet("font-size: 14pt; color: Navy;")
        main_layout.addWidget(header_label)

        desc_label = QLabel(f"Descrizione: {self.scenario.get('descrizione', 'Nessuna descrizione.')}")
        desc_label.setWordWrap(True)
        main_layout.addWidget(desc_label)
        main_layout.addWidget(QLabel("-" * 50))

        # 2. Area Elenco Anomalie
        main_layout.addWidget(QLabel("Pre defined anomalies"))

        self.anomaly_list = QListWidget()
        self.anomaly_list.itemDoubleClicked.connect(self._edit_anomaly)  # Connettiamo il doppio click
        main_layout.addWidget(self.anomaly_list)

        self._populate_anomaly_list()

        # 3. Pulsanti Azioni
        btn_layout = QHBoxLayout()
        # definito il livello di user: autorizzazione a gestire le anomalie
        print(f"User level = {self.level_code}")
        if self.level_code == 'AUTO':
            self.btn_add = QPushButton("Anomaly selected automatically")
            self.btn_add.setEnabled(False)
            self._select_first_anomaly_auto()  # <--- Chiama il nuovo metodo AUTO
        elif self.level_code == 'ENTRY':
            self.btn_add = QPushButton("SELECT one anomaly")
            self.btn_add.clicked.connect(self._select_anomaly)
        elif self.level_code == 'ADVANCED':  # Usa elif per gestire i casi in modo esclusivo
            self.btn_add = QPushButton("MODIFY one anomaly (Double Click on list)")
            # Nota: colleghiamo la funzione al doppio click, il pulsante potrebbe aprire la prima modifica
            # o mostrare un'avvertenza se non è selezionato nulla. Per ora lasciamo la logica originale.
            self.btn_add.clicked.connect(self._edit_anomaly_wrapper)
        elif self.level_code == 'TUTOR':
            self.btn_add = QPushButton("ADD/modify anomalies")
            self.btn_add.clicked.connect(self._add_new_anomaly)
        else:
            # Caso di default o livello non riconosciuto
            self.btn_add = QPushButton("Limited access")
            # funzione disabilitata
            self.btn_add.setEnabled(False)

        self.btn_close = QPushButton("Exit")
        self.btn_close.clicked.connect(self.accept)
        #
        btn_layout.addWidget(self.btn_add)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        main_layout.addLayout(btn_layout)

    def _populate_anomaly_list(self):
        """Popola la QListWidget con le anomalie fornite."""
        self.anomaly_list.clear()
        if not self.anomalies:
            QListWidgetItem("Nessuna anomalia definita per questo scenario.").setForeground(Qt.GlobalColor.gray)
            return
        for anomaly in self.anomalies:
            # Assumiamo che anomaly sia un oggetto Anomaly o un dizionario con chiavi sensate
            item_text = f"[{anomaly.get('label_id', 'N/A')}] {anomaly.get('descrizione', 'Descrizione mancante')}"
            item = QListWidgetItem(item_text)
            # Puoi memorizzare l'intero oggetto/dict come user data
            item.setData(Qt.ItemDataRole.UserRole, anomaly)
            self.anomaly_list.addItem(item)

    def _edit_anomaly_wrapper(self):
        """Prepara e chiama _edit_anomaly quando attivato dal pulsante."""
        selected_items = self.anomaly_list.selectedItems()
        if selected_items:
            # Passa il primo elemento selezionato al metodo di modifica
            self._edit_anomaly(selected_items[0])
        else:
            print("Seleziona prima un'anomalia dalla lista per modificarla.")

    def _select_first_anomaly_auto(self):
        """Seleziona automaticamente la prima anomalia se l'utente è 'AUTO'."""
        if self.anomaly_list.count() > 0:
            # 1. Seleziona il primo elemento nella lista
            self.anomaly_list.setCurrentRow(0)
            # 2. Ottieni l'elemento QListWidgetItem e processalo
            first_item = self.anomaly_list.item(0)
            # 3. Lancia il processamento dell'anomalia selezionata
            self._process_selected_anomaly(first_item)
        else:
            print("AVVISO: Nessuna anomalia disponibile per la selezione AUTO.")

    def _select_anomaly(self):
        """Logica per gestire la selezione manuale da parte dell'utente ENTRY."""
        selected_items = self.anomaly_list.selectedItems()
        if not selected_items:
            print("AVVISO: Nessuna anomalia selezionata. Seleziona un elemento dalla lista.")
            return
        # Passa l'elemento selezionato alla funzione di processamento centralizzata
        self._process_selected_anomaly(selected_items[0])

        # Qui potresti emettere un segnale per comunicare alla finestra principale (dbaseMD)
        # che l'utente ha scelto questa anomalia per la sessione.
        # todo a seconda del livello user la selezione dell'anomalia può essere automatica o scelta manualmente
        # todo gestione nel file json corrispondente
        # Esempio: self.anomalySelected.emit(anomaly_data)

    def _process_selected_anomaly(self, item: QListWidgetItem):
        """Estrae i dati dall'elemento e processa la selezione."""
        if item is None:
            print("ERRORE: Elemento non valido passato al processore di anomalia.")
            return
        anomaly_data = item.data(Qt.ItemDataRole.UserRole)
        # Stampa i dettagli dell'anomalia selezionata
        print(f"Anomaly (ID {anomaly_data.get('anomaly_id')}) selected "
              f"[{self.level_code}]: {anomaly_data.get('descrizione')}"
              f" json[{anomaly_data.get('json_anomaly_id')}]")
        self.anomaly_selected.emit(anomaly_data)
        if self.level_code in ['ENTRY', 'AUTO']:
            self.accept()       # chiude dialogo

    def _add_new_anomaly(self):
        """Logica per aprire la finestra di aggiunta anomalia (sarà una nuova, piccola dialog)."""
        print("Apro la finestra per aggiungere una nuova anomalia.")
        # QUI aprirai un'altra QDialog per l'input dei campi: label_id, descrizione, json_anomaly_id
        # Passo l'ID dello scenario per legare l'anomalia
        scenario_id = self.scenario.get('scenario_id')
        # Esempio fittizio di apertura:
        # self.edit_anomaly_dialog = EditAnomalyDialog(mode=0, scenario_id=scenario_id, ...)
        # self.edit_anomaly_dialog.anomalySaved.connect(self._handle_anomaly_save)
        # self.edit_anomaly_dialog.exec()

        # Per ora, stamperemo solo un messaggio:
        print(f"Richiesta di aggiungere una nuova anomalia allo scenario ID: {scenario_id}")

    def _edit_anomaly(self, item: QListWidgetItem):
        """Logica per aprire la finestra di modifica anomalia al doppio click."""
        anomaly_data = item.data(Qt.ItemDataRole.UserRole)
        scenario_id = self.scenario.get('scenario_id')

        self.edit_dialog = EditAnomalyDialog(
            mode=1,                     # 1 per Modifica
            scenario_id=scenario_id,
            anomaly_data=anomaly_data,  # Passa i dati esistenti
            parent=self
        )
        self.edit_dialog.anomalySaved.connect(self._handle_anomaly_save)
        self.edit_dialog.exec()

    def _handle_anomaly_save(self, db_data: dict, json_content: str):
        """Gestisce il segnale emesso da EditAnomalyDialog."""
        print(f"Dati Anomalia ricevuti per salvataggio nel DB: {db_data}")
        print(f"Contenuto JSON ricevuto (salvato su disco): {json_content[:50]}...")

        # QUI andrebbe la LOGICA per chiamare i metodi del DatabaseManager
        # self.parent().db_manager.add_anomaly(db_data)  <-- Se anomaly_id è None
        # self.parent().db_manager.update_anomaly(db_data) <-- Se anomaly_id esiste

        # Esempio: assumendo che parent sia dbaseMD e abbia i metodi del db manager
        if db_data.get('anomaly_id') is None:
            # Nuovo record
            new_id = self.parent().db_manager.add_anomaly(db_data)
            if new_id is not None:
                print(f"Nuova anomalia aggiunta con ID: {new_id}")
        else:
            # Modifica record (necessita di update_anomaly)
            # Dobbiamo prima implementare update_anomaly nel database manager!
            print("Aggiornamento record Anomalia richiesto (metodo DB mancante).")

        # Dopo il salvataggio nel DB, ricarica la lista
        self._reload_anomaly_list()  # <--- Necessita di implementazione
        self.newAnomalySaved.emit(True)  # Notifica la finestra principale

    # [NUOVO METODO]
    def _reload_anomaly_list(self):
        """Ricarica i dati dell'anomalia dal DB e aggiorna la QListWidget."""
        scenario_id = self.scenario.get('scenario_id')

        # Accesso al gestore del database tramite il parent (dbaseMD)
        anomalies_list = self.parent().db_manager.get_anomalies_for_scenario(scenario_id)
        self.anomalies = [a.__dict__ for a in anomalies_list]
        self._populate_anomaly_list()
        print("Lista anomalie ricaricata.")

    # Quando la sub-dialog salva un'anomalia:
    # def _handle_anomaly_save(self, success: bool):
    #     if success:
    #         # Ricarica le anomalie dal database e aggiorna la lista
    #         self._reload_anomalies_from_db()
    #         self.newAnomalySaved.emit(True) # Emetti segnale se necessario

########################################################################################################################
'''
class EditAnomalyDialog(QDialog):
    """
    Finestra di dialogo per aggiungere o modificare un record di anomalia (scenari_anomaly)
    e gestire il suo file JSON associato.
    """
    anomalySaved = pyqtSignal(dict, str)  # Segnale: (dati_tabella_aggiornati, json_content)

    def __init__(self, mode: int, scenario_id: int, anomaly_data: dict = None, parent=None):
        super().__init__(parent)
        self.mode = mode  # 0: Aggiungi, 1: Modifica
        self.scenario_id = scenario_id
        self.anomaly_data = anomaly_data if anomaly_data else {}
        self.json_path_template = "res/anomalies_config/{json_filename}.json"  # Supponiamo un percorso standard

        # Variabili di stato per i dati:
        self.initial_json_content = {}
        self.current_json_content = {}

        self.setWindowTitle(f"{'Modifica' if mode == 1 else 'Aggiungi'} Anomalia")
        self.setFixedSize(QSize(500, 550))
        self.move(parent.pos().x() + parent.width() + 20, parent.pos().y())  # Spostalo in un posto visibile

        self._setup_ui()
        self._load_data_json()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # 1. Dati Tabella scenari_anomaly (Fissi)
        form_layout.addRow(QLabel("--- Dati Record Database ---"))
        self.txt_label_id = QLineEdit(self.anomaly_data.get('label_id', ''))
        form_layout.addRow("Label ID:", self.txt_label_id)

        self.txt_descrizione = QLineEdit(self.anomaly_data.get('descrizione', ''))
        form_layout.addRow("Descrizione Breve:", self.txt_descrizione)

        # Nome del file JSON (Chiave del sistema di iniezione)
        self.txt_json_id = QLineEdit(self.anomaly_data.get('json_anomaly_id', ''))
        form_layout.addRow("File JSON ID:", self.txt_json_id)

        main_layout.addLayout(form_layout)

        # 2. Area Configurazione JSON (Parametri Variabili)
        main_layout.addWidget(QLabel("\n--- Configurazione Parametri Anomalia (JSON) ---"))

        # A. Esempio di parametro comune (Trigger Type)
        self.combo_trigger = QComboBox()
        self.combo_trigger.addItems(['Timer', 'Randomized', 'External'])
        form_layout.addRow("Tipo di Iniezione:", self.combo_trigger)

        # B. Esempio di parametro variabile (Percentuale di Apparizione - SpinBox)
        self.spin_percentage = QSpinBox()
        self.spin_percentage.setRange(0, 100)
        self.spin_percentage.setSuffix(" %")
        form_layout.addRow("Percentuale Apparizione:", self.spin_percentage)

        # C. Area di visualizzazione JSON RAW (Testo non modificabile)
        self.txt_json_raw = QTextEdit()
        self.txt_json_raw.setReadOnly(True)
        main_layout.addWidget(QLabel("JSON Content (Visualizzazione):"))
        main_layout.addWidget(self.txt_json_raw)

        # 3. Pulsanti
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Salva Anomalia & JSON")
        btn_discard = QPushButton("Annulla")

        btn_save.clicked.connect(self._save_data)
        btn_discard.clicked.connect(self.reject)

        btn_layout.addWidget(btn_discard)
        btn_layout.addWidget(btn_save)
        main_layout.addLayout(btn_layout)

    def _load_data_json(self):
        """Carica i dati JSON esistenti o inizializza i campi per un nuovo record."""
        if self.mode == 1 and self.anomaly_data:
            # Modalità Modifica: Carica il JSON esistente
            json_filename = self.anomaly_data.get('json_anomaly_id')
            if json_filename:
                full_path = self.json_path_template.format(json_filename=json_filename)
                try:
                    with open(full_path, 'r') as f:
                        self.initial_json_content = json.load(f)
                        self.current_json_content = self.initial_json_content.copy()
                        # Aggiorna l'UI in base al contenuto JSON
                        self._update_ui_from_json()
                        self.txt_json_raw.setText(json.dumps(self.current_json_content, indent=2))
                        return
                except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                    print(f"AVVISO: Impossibile caricare il file JSON {full_path}: {e}")

        # Modalità Aggiungi o fallimento nel caricamento: Inizializza
        self.initial_json_content = {}
        self.current_json_content = {
            "trigger_type": self.combo_trigger.currentText(),
            "appearance_pct": self.spin_percentage.value(),
            "anom_config": {}  # Qui andranno i parametri specifici dello scenario
        }
        self.txt_json_raw.setText(json.dumps(self.current_json_content, indent=2))

        # Abilita i campi per l'aggiunta di un nuovo file
        self.txt_json_id.setEnabled(True)
        self.txt_label_id.setEnabled(True)

    def _update_ui_from_json(self):
        """Aggiorna i widget dell'UI con i valori dal JSON caricato."""

        # Aggiorna i widget comuni:
        trigger = self.current_json_content.get('trigger_type')
        if trigger:
            self.combo_trigger.setCurrentText(trigger)

        percentage = self.current_json_content.get('appearance_pct')
        if percentage is not None:
            self.spin_percentage.setValue(percentage)

        # NOTA: Per gestire i parametri variabili in 'anom_config', dovresti
        # dinamicamente aggiungere widget qui. Per ora, ci limitiamo a quelli fissi.

    def _prepare_json_for_save(self):
        """Aggiorna il JSON interno con i valori correnti dell'UI."""

        # Aggiorna i campi comuni dal form
        self.current_json_content['trigger_type'] = self.combo_trigger.currentText()
        self.current_json_content['appearance_pct'] = self.spin_percentage.value()

        # Qui potresti implementare la logica per aggiornare i parametri in 'anom_config'
        # basandosi sui widget dinamici (se li avessi aggiunti).

        # Aggiorna la visualizzazione RAW
        self.txt_json_raw.setText(json.dumps(self.current_json_content, indent=2))
        return json.dumps(self.current_json_content)

    def _save_data(self):
        """Salva i dati nel database e il JSON su disco."""

        # 1. Validazione base
        json_id = self.txt_json_id.text().strip()
        label_id = self.txt_label_id.text().strip()

        if not json_id or not label_id:
            print("ERRORE: I campi 'Label ID' e 'File JSON ID' non possono essere vuoti.")
            return

        # 2. Prepara il contenuto JSON
        json_content_str = self._prepare_json_for_save()

        # 3. Salva il file JSON su disco
        full_path = self.json_path_template.format(json_filename=json_id)
        try:
            # Crea la directory se non esiste
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(json_content_str)
            print(f"File JSON salvato con successo in: {full_path}")
        except OSError as e:
            print(f"ERRORE FATALE: Impossibile scrivere il file JSON su disco: {e}")
            return

        # 4. Prepara i dati per la tabella (database)
        db_data = {
            'anomaly_id': self.anomaly_data.get('anomaly_id', None),  # None se è un nuovo record
            'scenario_id': self.scenario_id,
            'label_id': label_id,
            'descrizione': self.txt_descrizione.text().strip(),
            'json_anomaly_id': json_id
            # 'data_creazione' è gestito dal DB
        }

        # 5. Emette il segnale per il dbaseManager (che farà l'INSERT/UPDATE)
        self.anomalySaved.emit(db_data, json_content_str)
        self.accept()
'''


class EditAnomalyDialog(QDialog):
    """
    Finestra di dialogo per aggiungere o modificare un record di anomalia (scenari_anomaly)
    e gestire il suo file JSON associato, utilizzando un file .ui.
    """
    anomalySaved = Signal(dict, str)  # Segnale: (dati_tabella_aggiornati, json_content)

    def __init__(self, mode: int, scenario_id: int, anomaly_data: dict = None, parent=None):
        super().__init__(parent)
        # 1. Caricamento del file UI
        ui_file_path = "res/anomalyIOMForm.ui"
        ui_file = QFile(ui_file_path)
        if not ui_file.open:
            print(f"Errore: Impossibile aprire il file {ui_file_path}")
            sys.exit(-1)
        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        ui_file.close()

        self.mode = mode  # 0: Aggiungi, 1: Modifica
        self.scenario_id = scenario_id
        self.anomaly_data = anomaly_data if anomaly_data else {}
        # NOTA: Percorso file JSON corretto (ripristinato il percorso standard)
        self.json_path_template = "{json_filename}"

        # Variabili di stato per i dati:
        self.initial_json_content = {}
        self.current_json_content = {}
        self.dynamic_widgets = {}  # Dizionario per i widget dinamici

        self.setWindowTitle(f"{'Modifica' if mode == 1 else 'Aggiungi'} Anomalia")

        # 2. Assegnazione e Setup UI (inclusa la preparazione dell'area dinamica)
        self._map_ui_widgets()

        # 3. Connessione dei Segnali
        self._connect_signals()

        self._load_data_json()  # Inizializza i dati e popola l'UI statica/dinamica

    def _map_ui_widgets(self):
        """
        Mappa i widget statici dal file .ui e prepara l'area per i widget dinamici.
        ATTENZIONE: groupBox_2 per i parametri ANOMALIA (anom_config)
        groupBox_5 per la visualizzazione JSON RAW.
        """
        # Mappatura dei widget statici (assunti da .ui):
        self.txt_label_id = self.txt_label_id
        self.txt_descrizione = self.txt_descrizione
        self.txt_json_id = self.txt_json_id
        self.combo_trigger = self.comboTrigger
        self.spin_percentage = self.spinPercentage
        # MANTENIAMO self.txt_json_raw come riferimento al widget QTextEdit in groupBox_5 nel .ui
        self.txt_json_raw = self.txt_json_raw
        self.btn_discard = self.pushButton
        self.btn_save = self.pushButton_3

        # 1. Area Dinamica ANOMALIA (groupBox_2: Scenario configuration)
        self.anomaly_params_layout = self.groupBox_2.layout()

        if self.anomaly_params_layout is None:
            # Crea il layout per i parametri di configurazione
            self.anomaly_params_layout = QVBoxLayout(self.groupBox_2)

        # Non è più necessario creare un layout per l'area JSON RAW se self.txt_json_raw
        # è un widget statico di grandi dimensioni. Se invece il tuo groupBox_5 deve contenere l'etichetta
        # e il QTextEdit, potresti volerlo impostare qui, ma assumiamo che self.txt_json_raw
        # sia già pronto nel .ui.

        # OPZIONALE: Se groupBox_5 deve mostrare solo il QTextEdit e l'etichetta:
        if self.groupBox_5.layout() is None:
            # Inizializza il layout di groupBox_5 se manca
            QVBoxLayout(self.groupBox_5)

    def _connect_signals(self):
        """Connette i segnali dei pulsanti e di altri widget alla logica."""
        self.btn_save.clicked.connect(self._save_data)
        self.btn_discard.clicked.connect(self.reject)

    def _load_data_json(self):
        """Carica i dati JSON esistenti o inizializza i campi per un nuovo record."""
        # Imposta i parametri statici iniziali
        self.combo_trigger.clear()
        self.combo_trigger.addItems(['Timer', 'Randomized', 'External'])
        self.spin_percentage.setRange(0, 100)
        self.spin_percentage.setSuffix(" %")

        # ----------------------------------------------------
        # 1. Caricamento Dati Esistenti (Modalità Modifica)
        # ----------------------------------------------------
        if self.mode == 1 and self.anomaly_data:
            json_filename = self.anomaly_data.get('json_anomaly_id')

            # Popola i campi statici del record
            self.txt_label_id.setText(self.anomaly_data.get('label_id', ''))
            self.txt_descrizione.setText(self.anomaly_data.get('descrizione', ''))
            self.txt_json_id.setText(self.anomaly_data.get('json_anomaly_id', ''))
            self.txt_json_id.setEnabled(False)
            self.txt_label_id.setEnabled(False)

            if json_filename:
                full_path = self.json_path_template.format(json_filename=json_filename)
                try:
                    with open(full_path, 'r') as f:
                        self.initial_json_content = json.load(f)
                        self.current_json_content = self.initial_json_content.copy()
                except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                    print(f"AVVISO: Impossibile caricare il file JSON {full_path}: {e}. Inizializzo con JSON vuoto.")

        # ----------------------------------------------------
        # 2. Inizializzazione (Aggiungi o Caricamento fallito)
        # ----------------------------------------------------
        if not self.current_json_content:
            # Se il caricamento è fallito o è Modalità Aggiungi (mode=0)
            self.current_json_content = {
                "trigger_type": self.combo_trigger.currentText() if self.combo_trigger.count() > 0 else 'Timer',
                "appearance_pct": self.spin_percentage.value(),
                "anom_config": {}
            }
            if self.mode == 0:
                # Inizializzazione di base per la configurazione dinamica dei NUOVI record
                self.current_json_content['anom_config'] = {
                    "base_value": 1.0,
                    "duration_s": 10,
                    "message_text": "New Anomaly Config"
                }

        # Abilita/Disabilita i campi per Aggiungi (se Modalità Aggiungi o fallimento caricamento)
        if self.mode == 0 or not self.anomaly_data:
            self.txt_json_id.setEnabled(True)
            self.txt_label_id.setEnabled(True)

        # ----------------------------------------------------
        # 3. Aggiornamento UI Finale
        # ----------------------------------------------------
        # Popola la visualizzazione JSON RAW
        self.txt_json_raw.setText(json.dumps(self.current_json_content, indent=2))

        # Popola i widget statici e crea i widget dinamici
        self._update_ui_from_json()

    def _create_dynamic_widgets_anomaly(self, anom_config: dict):
        """
        Pulisce e crea i widget di input dinamici in groupBox_2 (self.anomaly_params_layout)
        basati sul contenuto di anom_config.
        """
        # 1. Pulisci i widget esistenti nell'area di configurazione (groupBox_2)
        while self.anomaly_params_layout.count():
            item = self.anomaly_params_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                # Pulizia ricorsiva per FormLayout (necessario per rimuovere il layout precedente)
                inner_layout = item.layout()
                while inner_layout.count():
                    inner_item = inner_layout.takeAt(0)
                    if inner_item.widget() is not None:
                        inner_item.widget().deleteLater()
                # Dopo aver rimosso i widget interni, il layout interno è pulito e viene rimosso dal main layout

        self.dynamic_widgets.clear()  # Svuota il dizionario di riferimento

        # Se anom_config è vuoto, mostra un messaggio
        if not anom_config:
            self.anomaly_params_layout.addWidget(QLabel("Nessun parametro di configurazione dinamica."))
            self.anomaly_params_layout.addStretch(1)
            return

        # 2. Crea i nuovi widget nel FormLayout
        form_layout = QFormLayout()

        for key, value in anom_config.items():
            label_text = key.replace('_', ' ').title()

            if isinstance(value, (int, float)):
                widget = QLineEdit(str(value))
                widget.setPlaceholderText(f"Inserisci {label_text} (numerico)")
            elif isinstance(value, str):
                widget = QLineEdit(value)
                widget.setPlaceholderText(f"Inserisci {label_text}")
            else:
                widget = QLabel(str(value))

            form_layout.addRow(f"{label_text}:", widget)
            self.dynamic_widgets[key] = widget

        # Aggiungi il FormLayout al layout verticale di groupBox_2
        self.anomaly_params_layout.addLayout(form_layout)
        # Aggiungi un QSpacerItem per spingere i widget in alto
        self.anomaly_params_layout.addStretch(1)

    def _update_ui_from_json(self):
        """Aggiorna i widget dell'UI con i valori dal JSON caricato."""

        # Aggiorna i widget comuni statici:
        trigger = self.current_json_content.get('trigger_type')
        if trigger:
            self.combo_trigger.setCurrentText(trigger)

        percentage = self.current_json_content.get('appearance_pct')
        if percentage is not None:
            self.spin_percentage.setValue(percentage)

        # Gestione dei widget DINAMICI in groupBox_2:
        anom_config = self.current_json_content.get('anom_config', {})
        self._create_dynamic_widgets_anomaly(anom_config)  # Chiama la funzione di creazione aggiornata

    def _prepare_json_for_save(self):
        """Aggiorna il JSON interno con i valori correnti dell'UI, inclusi quelli dinamici."""

        # 1. Aggiorna i campi comuni statici
        self.current_json_content['trigger_type'] = self.combo_trigger.currentText()
        self.current_json_content['appearance_pct'] = self.spin_percentage.value()

        # 2. Aggiorna i parametri DINAMICI in 'anom_config' (logica non modificata)
        updated_anom_config = {}
        # ... (Logica di lettura dei valori da self.dynamic_widgets e conversione del tipo)
        for key, widget in self.dynamic_widgets.items():
            if isinstance(widget, QLineEdit):
                value_str = widget.text().strip()
                original_value = self.current_json_content.get('anom_config', {}).get(key)
                if isinstance(original_value, int):
                    try:
                        updated_anom_config[key] = int(value_str)
                    except ValueError:
                        updated_anom_config[key] = original_value
                elif isinstance(original_value, float):
                    try:
                        updated_anom_config[key] = float(value_str)
                    except ValueError:
                        updated_anom_config[key] = original_value
                else:
                    updated_anom_config[key] = value_str
            else:
                updated_anom_config[key] = self.current_json_content.get('anom_config', {}).get(key)

        self.current_json_content['anom_config'] = updated_anom_config

        # 3. Aggiorna la visualizzazione JSON RAW (groupBox_5 / txt_json_raw)
        json_content_str = json.dumps(self.current_json_content, indent=2)
        self.txt_json_raw.setText(json_content_str)
        return json_content_str

    def _save_data(self):
        """Salva i dati nel database e il JSON su disco."""
        # ... (Logica di Validazione e Salvataggio omessa per brevità, non modificata)
        json_id = self.txt_json_id.text().strip()
        label_id = self.txt_label_id.text().strip()
        descrizione = self.txt_descrizione.text().strip()

        if not json_id or not label_id:
            QMessageBox.warning(self, "Errore di Validazione",
                                "I campi 'Label ID' e 'File JSON ID' non possono essere vuoti.")
            return

        json_content_str = self._prepare_json_for_save()  # Chiama il metodo che aggiorna i dinamici e il JSON RAW

        # 3. Salva il file JSON su disco
        full_path = self.json_path_template.format(json_filename=json_id)
        # ... (Logica di salvataggio su disco)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(json_content_str)
            print(f"File JSON salvato con successo in: {full_path}")
        except OSError as e:
            QMessageBox.critical(self, "Errore di Salvataggio",
                                 f"ERRORE FATALE: Impossibile scrivere il file JSON su disco: {e}")
            return

        # 4. Prepara i dati per la tabella (database)
        db_data = {
            'anomaly_id': self.anomaly_data.get('anomaly_id', None),
            'scenario_id': self.scenario_id,
            'label_id': label_id,
            'descrizione': descrizione,
            'json_anomaly_id': json_id
        }

        # 5. Emette il segnale
        self.anomalySaved.emit(db_data, json_content_str)
        self.accept()