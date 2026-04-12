import sys
import os
import glob
import json
from multiprocessing import Process, freeze_support

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QComboBox, QListWidget, QListWidgetItem,
                             QPushButton, QLabel, QSplitter, QMessageBox, QFileDialog)
from PySide6.QtCore import Qt
import vedo

# --- CONFIGURAZIONE CATEGORIE E COLORI --------------------------------------------------------------------------------
CATEGORIES_MAP = {
    # 1. TRONCO E CERVELLETTO
    "Brainstem": {
        "keywords": ["Pons", "Midbrain", "Medulla", "stem", "Hypot", "mesencefalo"],
        "color": "salmon"
    },
    "Cerebellum": {
        "keywords": ["Cerebel"],
        "color": "gold"
    },

    # 2. LOBI CEREBRALI
    "Lobe: Frontal": {
        "keywords": ["frontal", "precentral", "orbitofrontal", "rectus", "straight", "prefrontal"],
        "color": "#4169E1"  # RoyalBlue
    },
    "Lobe: Parietal": {
        "keywords": ["parietal", "postcentral", "precuneus", "supramarginal", "angular"],
        "color": "#2E8B57"  # SeaGreen
    },
    "Lobe: Temporal": {
        "keywords": ["temporal", "fusiform", "parahippocampal", "uncus", "pole"],
        "color": "#FFA500"  # Orange
    },
    "Lobe: Occipital": {
        "keywords": ["occipital", "cuneus", "lingual", "calcarine"],
        "color": "#FF6347"  # Tomato
    },
    "Lobe: Limbic/Insula": {
        "keywords": ["cingulate", "cingulum", "insula", "fornix"],
        "color": "#FF69B4"  # HotPink
    },

    # 3. STRUTTURE PROFONDE
    "Deep Structures": {
        "keywords": ["thalamus", "putamen", "caudate", "hippocampus", "amygdala", "ventricle", "pallidus", "striatum"],
        "color": "lightblue"
    },

    # 4. SPINA DORSALE
    "Spine": {
        "keywords": ["cervical", "thoracic", "lumbar", "vertebra", "spine", "atlas", "axis", "sacrum"],
        "color": "#e3dac9"  # Bone
    },

    # 5. CORTECCIA GENERICA
    "Cortex (Generic)": {
        "keywords": ["cortex", "hemisphere", "gyrus", "lobule", "Left", "Right", "matter"],
        "color": "lightgray"
    }
}


# --- PROCESSO DI RENDERING --------------------------------------------------------------------------------------------
def launch_composite_viewer(items_data, window_title):
    try:
        actors = []
        print(f"Viewer avviato con {len(items_data)} oggetti misti.")

        for item in items_data:
            path = item['path']
            color = item['color']

            try:
                mesh = vedo.Mesh(path)
                mesh.color(color)
                mesh.lighting("glossy")
                actors.append(mesh)
            except Exception as e:
                print(f"Errore caricamento {path}: {e}")

        if not actors:
            return

        label = vedo.Text2D(f"Composizione: {len(actors)} elementi", pos="top-left")
        actors.append(label)

        vedo.show(actors,
                  title=window_title,
                  axes=1,
                  bg="bb",
                  viewup="z",
                  interactive=True).close()

    except Exception as e:
        print(f"Critical Viewer Error: {e}")


# --- GESTORE DATI ---
class AnatomyDataManager:
    def __init__(self, directory="3D_data"):
        self.directory = directory
        self.data = {}
        self._scan_files()

    def _scan_files(self):
        for cat in CATEGORIES_MAP.keys():
            self.data[cat] = []
        self.data["Other"] = []

        if not os.path.exists(self.directory):
            try:
                os.makedirs(self.directory)
            except:
                pass
            return

        files = glob.glob(os.path.join(self.directory, "*.obj"))

        for file_path in files:
            filename = os.path.basename(file_path)
            found = False
            for cat, props in CATEGORIES_MAP.items():
                if any(k.lower() in filename.lower() for k in props["keywords"]):
                    self.data[cat].append((filename, file_path))
                    found = True
                    break
            if not found:
                self.data["Other"].append((filename, file_path))

    def get_categories(self):
        return [cat for cat, files in self.data.items() if len(files) > 0]

    def get_files_in_category(self, category):
        return self.data.get(category, [])

    def get_color_for_category(self, category):
        return CATEGORIES_MAP.get(category, {}).get("color", "white")


# --- GUI PRINCIPALE (COMPOSER) ---
class AnatomyComposerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ION-Sim: Anatomy Composer (Ultimate)")
        self.resize(1100, 750)
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: white; }
            QLabel { font-weight: bold; color: #00ccff; margin-bottom: 5px; font-size: 14px; }
            QListWidget { background-color: #1e1e1e; border: 1px solid #555; color: #ddd; font-size: 13px; }
            QListWidget::item:selected { background-color: #005577; }
            QPushButton { padding: 8px; font-weight: bold; border-radius: 4px; background-color: #444; color: white; }
            QPushButton:hover { background-color: #555; }
            QComboBox { padding: 6px; background-color: #444; color: white; }
        """)

        self.manager = AnatomyDataManager()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ==========================
        # COLONNA SINISTRA (BROWSER)
        # ==========================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        left_layout.addWidget(QLabel("1. Sfoglia Categorie"))

        self.combo_cat = QComboBox()
        self.combo_cat.addItems(self.manager.get_categories())
        self.combo_cat.currentTextChanged.connect(self.on_category_change)
        left_layout.addWidget(self.combo_cat)

        self.list_source = QListWidget()
        self.list_source.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        left_layout.addWidget(self.list_source)

        # Bottoni Selezione Rapida
        hbox_sel = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(self.select_all_source)

        btn_lr = QPushButton("Toggle L/R")
        btn_lr.clicked.connect(self.toggle_left_right)

        hbox_sel.addWidget(btn_all)
        hbox_sel.addWidget(btn_lr)
        left_layout.addLayout(hbox_sel)

        # Bottone "AGGIUNGI"
        self.btn_add = QPushButton("AGGIUNGI ALLA SCENA >>>")
        self.btn_add.setStyleSheet("background-color: #006600; color: white; padding: 12px; margin-top: 10px;")
        self.btn_add.clicked.connect(self.add_to_scene)
        left_layout.addWidget(self.btn_add)

        # ==========================
        # COLONNA DESTRA (SCENA)
        # ==========================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        right_layout.addWidget(QLabel("2. Scena Composta"))

        self.list_scene = QListWidget()
        right_layout.addWidget(self.list_scene)

        # Bottoni Gestione Scena
        hbox_scene_btns = QHBoxLayout()
        btn_remove = QPushButton("Rimuovi Selezionati")
        btn_remove.setStyleSheet("background-color: #660000;")
        btn_remove.clicked.connect(self.remove_from_scene)

        btn_clear = QPushButton("Svuota Tutto")
        btn_clear.clicked.connect(self.list_scene.clear)

        hbox_scene_btns.addWidget(btn_remove)
        hbox_scene_btns.addWidget(btn_clear)
        right_layout.addLayout(hbox_scene_btns)

        # Bottone LANCIA RENDERING
        self.btn_render = QPushButton("🚀 APRI FINESTRA 3D COMPOSTA")
        self.btn_render.setStyleSheet(
            "background-color: #0055aa; color: white; padding: 15px; font-size: 15px; margin-top: 10px;")
        self.btn_render.clicked.connect(self.launch_composition)
        right_layout.addWidget(self.btn_render)

        # --- SEZIONE SYSTEM (Save, Load, Help, Exit) ---
        right_layout.addSpacing(20)
        right_layout.addWidget(QLabel("Sistema"))

        hbox_system = QHBoxLayout()

        self.btn_save = QPushButton("💾 Save")
        self.btn_save.clicked.connect(self.save_scene_to_json)

        # *** NUOVO BOTTONE LOAD ***
        self.btn_load = QPushButton("📂 Load")
        self.btn_load.clicked.connect(self.load_scene_from_json)

        self.btn_help = QPushButton("❓ Help")
        self.btn_help.clicked.connect(self.show_help)

        self.btn_exit = QPushButton("❌ Exit")
        self.btn_exit.setStyleSheet("background-color: #333; border: 1px solid #555;")
        self.btn_exit.clicked.connect(self.close)

        hbox_system.addWidget(self.btn_save)
        hbox_system.addWidget(self.btn_load)  # Aggiunto qui
        hbox_system.addWidget(self.btn_help)
        hbox_system.addWidget(self.btn_exit)
        right_layout.addLayout(hbox_system)

        # Layout Principale
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        # Init
        if self.combo_cat.count() > 0:
            self.on_category_change(self.combo_cat.currentText())

    # --- LOGICA ---
    def on_category_change(self, category):
        self.list_source.clear()
        files = self.manager.get_files_in_category(category)
        for filename, fullpath in files:
            item = QListWidgetItem(filename)
            item.setData(Qt.ItemDataRole.UserRole, fullpath)
            self.list_source.addItem(item)

    def select_all_source(self):
        for i in range(self.list_source.count()):
            self.list_source.item(i).setCheckState(Qt.CheckState.Checked)

    def toggle_left_right(self):
        left_items = []
        right_items = []
        for i in range(self.list_source.count()):
            item = self.list_source.item(i)
            name = item.text().lower()
            if "left" in name:
                left_items.append(item)
            elif "right" in name:
                right_items.append(item)

        if not left_items and not right_items: return

        l_checked = any(i.checkState() == Qt.CheckState.Checked for i in left_items)
        r_checked = any(i.checkState() == Qt.CheckState.Checked for i in right_items)

        next_l, next_r = Qt.CheckState.Unchecked, Qt.CheckState.Unchecked

        if not l_checked and not r_checked:
            next_l = Qt.CheckState.Checked
        elif l_checked and not r_checked:
            next_r = Qt.CheckState.Checked
        elif not l_checked and r_checked:
            next_l = next_r = Qt.CheckState.Checked

        for item in left_items: item.setCheckState(next_l)
        for item in right_items: item.setCheckState(next_r)

    def add_to_scene(self):
        current_cat = self.combo_cat.currentText()
        cat_color = self.manager.get_color_for_category(current_cat)
        for i in range(self.list_source.count()):
            src_item = self.list_source.item(i)
            if src_item.checkState() == Qt.CheckState.Checked:
                filename = src_item.text()
                fullpath = src_item.data(Qt.ItemDataRole.UserRole)
                scene_item = QListWidgetItem(f"[{current_cat}] {filename}")
                payload = {"path": fullpath, "color": cat_color, "name": filename, "category": current_cat}
                scene_item.setData(Qt.ItemDataRole.UserRole, payload)
                self.list_scene.addItem(scene_item)
                src_item.setCheckState(Qt.CheckState.Unchecked)

    def remove_from_scene(self):
        for item in self.list_scene.selectedItems():
            self.list_scene.takeItem(self.list_scene.row(item))

    def launch_composition(self):
        if self.list_scene.count() == 0: return
        composition_data = []
        for i in range(self.list_scene.count()):
            composition_data.append(self.list_scene.item(i).data(Qt.ItemDataRole.UserRole))
        p = Process(target=launch_composite_viewer, args=(composition_data, "ION-Sim: Scena Composta"))
        p.start()

    # --- FUNZIONI DI SALVATAGGIO E CARICAMENTO ---
    def save_scene_to_json(self):
        if self.list_scene.count() == 0:
            QMessageBox.warning(self, "Attenzione", "La scena è vuota.")
            return

        scene_export = []
        for i in range(self.list_scene.count()):
            payload = self.list_scene.item(i).data(Qt.ItemDataRole.UserRole)
            scene_export.append({
                "object_name": payload.get("name"),
                "category": payload.get("category"),
                "color_hex": payload.get("color"),
                "file_path": payload.get("path")
            })

        file_path, _ = QFileDialog.getSaveFileName(self, "Salva Scena JSON", "", "JSON Files (*.json)")

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(scene_export, f, indent=4)
                QMessageBox.information(self, "Successo", f"Salvato in:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Errore", str(e))

    def load_scene_from_json(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Carica Scena JSON", "", "JSON Files (*.json)")
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("Formato JSON non valido.")

            # Svuota la scena attuale o chiedi conferma
            self.list_scene.clear()

            loaded_count = 0
            for entry in data:
                path = entry['file_path']

                # --- SMART PATH CHECK ---
                # Se il file assoluto non esiste, prova a cercarlo nella cartella locale
                if not os.path.exists(path):
                    local_check = os.path.join("../Virtual_patient/3D_data", os.path.basename(path))
                    if os.path.exists(local_check):
                        path = local_check  # Aggiorna con il percorso corretto
                    else:
                        print(f"File mancante e non trovato localmente: {path}")
                        continue  # Salta questo file

                name = entry['object_name']
                cat = entry['category']
                color = entry['color_hex']

                # Ricostruisce l'elemento grafico
                scene_item = QListWidgetItem(f"[{cat}] {name}")
                payload = {"path": path, "color": color, "name": name, "category": cat}
                scene_item.setData(Qt.ItemDataRole.UserRole, payload)

                self.list_scene.addItem(scene_item)
                loaded_count += 1

            QMessageBox.information(self, "Caricamento", f"Caricati {loaded_count} oggetti.")

        except Exception as e:
            QMessageBox.critical(self, "Errore Load", f"Impossibile caricare:\n{str(e)}")

    def show_help(self):
        msg = """
        <b>GUIDA RAPIDA COMPOSER</b><br><br>
        1. <b>Scegli Categoria</b>: Usa il menu a sinistra.<br>
        2. <b>Seleziona</b>: Usa Checkbox o Toggle L/R.<br>
        3. <b>Aggiungi</b>: Premi 'AGGIUNGI ALLA SCENA'.<br>
        4. <b>Render</b>: Premi 'APRI FINESTRA 3D'.<br>
        5. <b>Save/Load</b>: Salva o ricarica composizioni JSON.
        """
        QMessageBox.information(self, "Help - ION Sim", msg)


if __name__ == "__main__":
    freeze_support()
    app = QApplication(sys.argv)
    window = AnatomyComposerGUI()
    window.show()
    sys.exit(app.exec())

