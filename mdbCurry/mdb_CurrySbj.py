import sys
import os
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
                             QComboBox, QPushButton, QTextEdit, QFileDialog,
                             QHBoxLayout, QMessageBox, QListWidget, QListWidgetItem,
                             QGroupBox, QSplitter)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QBrush

from mri_viewerIOM import CurryMRIViewer
from mesh_viewerIOM import CurryMeshViewer

# --- CONFIGURAZIONE CATEGORIE FILE ---

EXT_RAW = {
    '.dat', '.dap', '.cdt', '.cep', '.cnt', '.eeg', '.vhdr', '.vmrk'
}

EXT_RESULTS = {
    # Risultati numerici
    '.vcd', '.rs3', '.res', '.pom', '.cdr', '.map', '.avg', '.epi', '.spk',
    # Immagini
    # '.bmp', '.jpg', '.jpeg', '.png', '.tif',
    # --- NUOVO: MESH E BEM ---
    '.bo0', '.bo1', '.bo2',  # Boundary Output (BEM)
    '.bt1', '.bt2',  # BEM Tessellation
    '.bd1', '.bd2'  # BEM Description
}

# --- NUOVA LISTA IGNORE (Molto più ridotta) ---
EXT_IGNORE = {
    '.geo', '.sln',  # Geometry solutions (spesso binarie/complesse)
    '.log', '.bak', '.old',  # Log e backup
    '_attributes'
}

# --- DESCRIZIONI AGGIORNATE ---
FILE_DESCRIPTIONS = {
    '.dat': 'Dati Grezzi (Raw)',
    '.dap': 'Parametri Acquisizione',
    '.cdt': 'Dati Continui Curry',
    '.rs3': 'Ricostruzione Sorgenti',
    '.res': 'Risultati Analisi',
    '.pom': 'Modello Testa (Point)',
    '.avg': 'Media Segnali',
    '.jpg': 'Immagine/Screenshot',
    '.vcd': 'surface sep',

    # Mesh BEM
    '.s00': 'Mesh Pelle/Cranio (BEM)',
    '.s01': 'Mesh Corteccia (BEM)',
    '.s02': 'Mesh Corteccia (BEM)',
    '.s03': 'Mesh Corteccia (BEM)',
    '.s04': 'Mesh Corteccia (BEM)',
    '.s05': 'Mesh Corteccia (BEM)',
    '.bo0': 'Boundary Esterno (BEM)',
    '.bo1': 'Boundary Interno (BEM)',
    '.bt1': 'Triangolazione BEM',
    '.bd1': 'Descrittore BEM'
}
# Genera automaticamente le estensioni da .s00 a .s99
for i in range(100):
    ext = f".s{i:02d}"  # Crea la stringa .s00, .s01...
    EXT_RESULTS.add(ext) # La aggiunge alla lista dei risultati
    # Se non c'è già una descrizione specifica, ne mette una generica
    if ext not in FILE_DESCRIPTIONS:
        FILE_DESCRIPTIONS[ext] = "Mesh Superficie/BEM"

class CurryFileChecker(QWidget):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.base_data_folder = ""
        self.df = pd.DataFrame()

        self.setWindowTitle("🔍 Curry Data Explorer - Volume RM Manager")
        self.setGeometry(100, 100, 1300, 700)

        self.init_ui()
        self.load_data()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # --- TOP: CONTROLLI ---
        top_panel = QGroupBox("Impostazioni")
        top_layout = QVBoxLayout()

        folder_layout = QHBoxLayout()
        btn_folder = QPushButton("📂 Scegli Cartella Root")
        btn_folder.clicked.connect(self.select_root_folder)
        self.lbl_folder = QLabel("Nessuna cartella selezionata")
        self.lbl_folder.setStyleSheet("color: gray; font-style: italic;")
        folder_layout.addWidget(btn_folder)
        folder_layout.addWidget(self.lbl_folder)
        top_layout.addLayout(folder_layout)

        subj_layout = QHBoxLayout()
        lbl_subj = QLabel("👤 Soggetto:")
        lbl_subj.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.combo_subjects = QComboBox()
        self.combo_subjects.setEnabled(False)
        self.combo_subjects.currentIndexChanged.connect(self.check_files_for_subject)

        subj_layout.addWidget(lbl_subj)
        subj_layout.addWidget(self.combo_subjects, 1)
        top_layout.addLayout(subj_layout)

        top_panel.setLayout(top_layout)
        main_layout.addWidget(top_panel)

        # --- CENTER: TRE BOX LISTE ---
        lists_layout = QHBoxLayout()

        # BOX 1: Dati Grezzi
        self.grp_raw = QGroupBox("📡 Dati Grezzi")
        self.grp_raw.setStyleSheet("QGroupBox { font-weight: bold; color: #0078d7; }")
        raw_layout = QVBoxLayout()
        self.list_raw = QListWidget()
        self.list_raw.setAlternatingRowColors(True)
        raw_layout.addWidget(self.list_raw)
        self.grp_raw.setLayout(raw_layout)

        # BOX 2: Risultati
        self.grp_res = QGroupBox("📊 Risultati")
        self.grp_res.setStyleSheet("QGroupBox { font-weight: bold; color: #d83b01; }")
        res_layout = QVBoxLayout()
        self.list_results = QListWidget()
        self.list_results.setAlternatingRowColors(True)
        res_layout.addWidget(self.list_results)

        self.btn_view_mesh = QPushButton("🕸️ Visualizza Mesh/Sorgenti Selezionate")
        self.btn_view_mesh.setStyleSheet("background-color: #d83b01; color: white; padding: 5px;")
        self.btn_view_mesh.clicked.connect(self.launch_mesh_viewer)
        res_layout.addWidget(self.btn_view_mesh)

        self.grp_res.setLayout(res_layout)

        # BOX 3: VOLUMI RM (Metadata)
        self.grp_mri = QGroupBox("🧠 Volumi RM (Da _attributes)")
        self.grp_mri.setStyleSheet("QGroupBox { font-weight: bold; color: #881798; }")
        mri_layout = QVBoxLayout()
        self.list_mri = QListWidget()
        self.list_mri.setAlternatingRowColors(True)
        self.list_mri.itemClicked.connect(self.show_mri_details)
        mri_layout.addWidget(self.list_mri)

        # --- NUOVO PULSANTE VISUALIZZA ---
        self.btn_visualize = QPushButton("👁️ Visualizza Volume 3D")
        self.btn_visualize.setStyleSheet("background-color: #5C2D91; color: white; font-weight: bold; padding: 5px;")
        self.btn_visualize.setEnabled(False)  # Disabilitato all'inizio
        self.btn_visualize.clicked.connect(self.launch_3d_viewer)
        mri_layout.addWidget(self.btn_visualize)

        self.grp_mri.setLayout(mri_layout)

        lists_layout.addWidget(self.grp_raw)
        lists_layout.addWidget(self.grp_res)
        lists_layout.addWidget(self.grp_mri)

        main_layout.addLayout(lists_layout, 1)

        # --- BOTTOM: INFO ---
        self.lbl_status = QLabel("Pronto.")
        self.lbl_status.setStyleSheet("background-color: #333; color: white; padding: 5px;")
        main_layout.addWidget(self.lbl_status)

        # Area testo per vedere i dettagli dell'_attributes quando clicchi
        self.txt_details = QTextEdit()
        self.txt_details.setMaximumHeight(100)
        self.txt_details.setReadOnly(True)
        self.txt_details.setPlaceholderText("Clicca su un Volume RM per vedere i parametri di caricamento...")
        main_layout.addWidget(self.txt_details)

        self.setLayout(main_layout)

    def load_data(self):
        if not os.path.exists(self.csv_path):
            QMessageBox.critical(self, "Errore", f"CSV non trovato: {self.csv_path}")
            return

        try:
            self.df = pd.read_csv(self.csv_path)
            self.df['LastName'] = self.df['LastName'].fillna("").astype(str)
            self.df['FirstName'] = self.df['FirstName'].fillna("").astype(str)
            self.df['DisplayName'] = self.df['LastName'] + " " + self.df['FirstName']
            unique_subjects = sorted([s for s in self.df['DisplayName'].unique() if s.strip()])

            self.combo_subjects.clear()
            self.combo_subjects.addItems(unique_subjects)
            self.combo_subjects.setEnabled(True)
            self.lbl_status.setText(f"Database caricato: {len(unique_subjects)} soggetti.")

        except Exception as e:
            QMessageBox.critical(self, "Errore CSV", str(e))

    def select_root_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleziona Archivio Curry")
        if folder:
            self.base_data_folder = folder
            self.lbl_folder.setText(folder)
            self.lbl_folder.setStyleSheet("color: black; font-weight: bold;")
            if self.combo_subjects.count() > 0:
                self.check_files_for_subject()

    def parse_curry_attributes(self, file_path):
        """
        Legge il file _attributes riga per riga e crea un dizionario di parametri.
        """
        meta = {}
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue

                    # Il file sembra separato da spazi multipli o tabulazioni.
                    # Splittiamo al primo spazio bianco
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        key = parts[0]
                        value = parts[1]
                        meta[key] = value
        except Exception as e:
            print(f"Errore parsing {file_path}: {e}")
            return None
        return meta

    def show_mri_details(self, item):
        """
        Mostra i metadati salvati nell'item quando l'utente ci clicca sopra.
        """
        # Recuperiamo il dizionario salvato dentro l'item
        meta = item.data(Qt.ItemDataRole.UserRole)
        if meta:
            info = f"--- PARAMETRI DI CARICAMENTO ---\n"
            info += f"Paziente: {meta.get('patient_name', 'N/A')}\n"
            info += f"Dimensioni: {meta.get('xsize')} x {meta.get('ysize')} x {meta.get('zsize')} (Voxel)\n"
            info += f"Scaling: {meta.get('xscale')} / {meta.get('yscale')} / {meta.get('zscale')}\n"
            info += f"Totale Slice: {meta.get('image_number', '0')}\n"
            info += f"Header Size: {meta.get('header_size', '0')} bytes\n"
            info += f"Path: {meta.get('_full_path')}"
            self.txt_details.setText(info)
            self.btn_visualize.setEnabled(True)
            # Salviamo i metadati correnti in una variabile di classe per usarli dopo
            self.current_mri_meta = meta
        else:
            self.txt_details.setText("Nessun metadato disponibile.")
            self.btn_visualize.setEnabled(False)

    def launch_3d_viewer(self):
        if not hasattr(self, 'current_mri_meta') or not self.current_mri_meta:
            return

        # Feedback visivo
        self.lbl_status.setText("⏳ Caricamento Volume 3D in corso... attendere...")
        self.lbl_status.setStyleSheet("background-color: #e6e600; color: black;")
        QApplication.processEvents()

        try:
            # Istanzia la classe viewer
            viewer = CurryMRIViewer(self.current_mri_meta)

            # Carica i dati (questo può impiegare qualche secondo)
            success = viewer.load_data()

            if success:
                self.lbl_status.setText("✅ Visualizzatore avviato.")
                self.lbl_status.setStyleSheet("background-color: #107c10; color: white;")
                # Lancia la finestra Vedo (che è bloccante finché non la chiudi, o gira nel suo loop)
                viewer.show()
            else:
                QMessageBox.warning(self, "Errore", "Impossibile caricare i dati raw del volume.")
                self.lbl_status.setText("Errore caricamento.")

        except Exception as e:
            QMessageBox.critical(self, "Errore Critico", f"Errore durante la visualizzazione:\n{e}")
            import traceback
            print(traceback.format_exc())

    def launch_mesh_viewer(self):
        """
        Raccoglie i file selezionati nella lista Risultati/Mesh e li visualizza.
        """
        selected_items = self.list_results.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "Info",
                                    "Seleziona almeno un file (.vcd, .rs3, .pom, .cdr, .s00...) dalla lista Risultati.")
            return

        # Recuperiamo il percorso base corrente (l'ultimo usato per il check)
        # Nota: Dobbiamo ricostruirlo o salvarlo. Per semplicità, ricostruiamolo dal soggetto corrente.
        name = self.combo_subjects.currentText()
        if not name or not self.base_data_folder: return

        # (Logica per ritrovare la cartella target - copia identica a quella in check_files_for_subject)
        # ... Per evitare duplicazione codice, l'ideale sarebbe salvare self.current_target_path nella classe
        # Ma per ora ricalcoliamolo al volo:
        subset = self.df[self.df['DisplayName'] == name]
        row = subset.iloc[0]

        def clean(s):
            return str(s).upper().replace(",", "").replace(".", "").strip().replace(" ", "_")

        folder_name = "_".join([clean(row['LastName']), clean(row['FirstName'])])
        for suff in ["_M1", "_T", "_M2", "_TEST", "_PRE", "_POST"]:
            if folder_name.endswith(suff): folder_name = folder_name[:-len(suff)]; break
        folder_name = folder_name.strip("_")
        target_path = os.path.join(self.base_data_folder, folder_name)
        if not os.path.exists(target_path):  # Fallback spazio
            target_path = os.path.join(self.base_data_folder, folder_name.replace("_", " "))

        # --- AVVIO VIEWER ---
        self.lbl_status.setText("⏳ Elaborazione Mesh 3D...")
        QApplication.processEvents()

        viewer = CurryMeshViewer()
        loaded_count = 0

        for item in selected_items:
            # Il testo dell'item è "nomefile \n ↳ descrizione"
            # Prendiamo solo la prima riga
            full_text = item.text()
            rel_path = full_text.split('\n')[0].strip()

            # Pulizia eventuali caratteri extra se presenti (es. ★)
            rel_path = rel_path.replace("★ ", "").replace(" (Verificato)", "").replace(" (Extra)", "")

            full_path = os.path.join(target_path, rel_path)

            if viewer.load_file(full_path):
                loaded_count += 1

        if loaded_count > 0:
            self.lbl_status.setText(f"✅ Visualizzazione di {loaded_count} oggetti 3D.")
            viewer.show()
        else:
            QMessageBox.warning(self, "Errore",
                                "Impossibile caricare i file selezionati (formato non riconosciuto o file vuoti).")
            self.lbl_status.setText("Errore caricamento mesh.")

    def check_files_for_subject(self):
        try:
            name = self.combo_subjects.currentText()
            if not name or not self.base_data_folder: return

            self.list_raw.clear()
            self.list_results.clear()
            self.list_mri.clear()
            self.txt_details.clear()
            QApplication.processEvents()

            # --- TROVA CARTELLA PAZIENTE ---
            subset = self.df[self.df['DisplayName'] == name]
            if subset.empty: return
            row = subset.iloc[0]

            def clean(s):
                return str(s).upper().replace(",", "").replace(".", "").strip().replace(" ", "_")

            folder_name = "_".join([clean(row['LastName']), clean(row['FirstName'])])

            for suff in ["_M1", "_T", "_M2", "_TEST", "_PRE", "_POST"]:
                if folder_name.endswith(suff):
                    folder_name = folder_name[:-len(suff)]
                    break
            folder_name = folder_name.strip("_")

            target_path = os.path.join(self.base_data_folder, folder_name)

            if not os.path.exists(target_path):
                folder_space = folder_name.replace("_", " ")
                path_space = os.path.join(self.base_data_folder, folder_space)
                if os.path.exists(path_space):
                    target_path = path_space
                    folder_name = folder_space

            if not os.path.exists(target_path):
                self.lbl_status.setText(f"❌ Cartella non trovata: {folder_name}")
                self.lbl_status.setStyleSheet("background-color: #d83b01; color: white; padding: 5px;")
                return

            self.lbl_status.setText(f"📂 Cartella trovata: {folder_name}")
            self.lbl_status.setStyleSheet("background-color: #107c10; color: white; padding: 5px;")

            count_raw = 0
            count_res = 0
            count_volumes = 0

            # --- SCANSIONE FILE ---
            for root, dirs, files in os.walk(target_path):
                QApplication.processEvents()

                # --- 1. GESTIONE VOLUME RM (_attributes) ---
                if "_attributes" in files:
                    attr_path = os.path.join(root, "_attributes")
                    meta = self.parse_curry_attributes(attr_path)

                    if meta:
                        # Aggiungiamo il percorso completo ai metadati per uso futuro
                        meta['_full_path'] = root

                        # Creiamo una riga "RIASSUNTIVA" per il volume
                        rel_dir = os.path.relpath(root, target_path)
                        folder_display = "Root" if rel_dir == "." else rel_dir

                        dims = f"{meta.get('xsize', '?')}x{meta.get('ysize', '?')}x{meta.get('zsize', '?')}"
                        p_name = meta.get('patient_name', 'Unknown')

                        display_text = f"📦 Volume RM: {folder_display}\n   ↳ {dims} voxels | {p_name}"

                        item = QListWidgetItem(display_text)
                        item.setForeground(QBrush(QColor("#5C2D91")))  # Viola Scuro
                        item.setBackground(QBrush(QColor("#e6e6fa")))  # Lavanda
                        font = QFont()
                        font.setBold(True)
                        item.setFont(font)

                        # SALVIAMO IL DIZIONARIO DENTRO L'ITEM (Invisibile all'utente)
                        item.setData(Qt.ItemDataRole.UserRole, meta)

                        self.list_mri.addItem(item)
                        count_volumes += 1

                # --- 2. GESTIONE FILE NORMALI ---
                for filename in files:
                    if filename.startswith(".") or filename == "_attributes": continue

                    _, ext = os.path.splitext(filename)
                    ext = ext.lower()

                    if ext in EXT_IGNORE: continue

                    # Ignoriamo le slice numeriche (.001, .120) per non intasare la vista
                    # Verranno caricate tramite il volume _attributes
                    if ext[1:].isdigit() or ext in {'.dcm', '.ima'}:
                        continue

                    rel_dir = os.path.relpath(root, target_path)
                    display_text = filename if rel_dir == "." else f"{rel_dir}/{filename}"
                    desc = FILE_DESCRIPTIONS.get(ext, ext)

                    item_text = f"{display_text}\n   ↳ {desc}"
                    item = QListWidgetItem(item_text)

                    if ext in EXT_RAW:
                        item.setForeground(QBrush(QColor("#005a9e")))
                        self.list_raw.addItem(item)
                        count_raw += 1
                    elif ext in EXT_RESULTS:
                        item.setForeground(QBrush(QColor("#a4262c")))
                        self.list_results.addItem(item)
                        count_res += 1

            self.grp_raw.setTitle(f"📡 Dati Grezzi ({count_raw})")
            self.grp_res.setTitle(f"📊 Risultati ({count_res})")
            self.grp_mri.setTitle(f"🧠 Volumi RM ({count_volumes})")

        except Exception as e:
            self.lbl_status.setText(f"Errore: {str(e)}")
            self.lbl_status.setStyleSheet("background-color: red; color: white;")
            import traceback
            print(traceback.format_exc())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CurryFileChecker("../dataIOM/Elenco_File_Pazienti.csv")
    window.show()
    sys.exit(app.exec())