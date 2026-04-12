
import sys
import platform
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import QTimer, Qt

# Importiamo la classe per il widget VTK compatibile con Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import mne
# Aggiungiamo Text2D agli import
from vedo import Plotter, Mesh, settings, Text3D

# 1. SETUP
if platform.system() == "Darwin":
    settings.default_backend = "vtk"


class LogoWidget(QFrame):
    """
    Widget personalizzato che ospita la scena 3D.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Layout
        self.layout = QVBoxLayout()
        # Riduciamo i margini a 0 per sfruttare tutto lo spazio per il 3D
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

        # Timer animazione
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotate_step)

        self.is_ready = False

    def load_data_and_create_mesh(self):
        print("Inizio caricamento dati MNE...")

        try:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        except Exception as e:
            print(f"Errore download: {e}")
            return

        lh_path = fs_dir / 'surf' / 'lh.pial'
        rh_path = fs_dir / 'surf' / 'rh.pial'

        # Emisfero Sinistro
        p_lh, f_lh = mne.read_surface(lh_path)
        self.lh_pial = Mesh([p_lh, f_lh]).c("bisque").lighting("shiny")
        self.lh_pial.name = "lh_pial"

        # Emisfero Destro
        p_rh, f_rh = mne.read_surface(rh_path)
        self.rh_pial = Mesh([p_rh, f_rh]).c("bisque").lighting("plastic") # metallic, shiny, plastic
        self.rh_pial.name = "rh_pial"

        self.txt_mesh = Text3D("R&B-Lab 2025", pos=(10, 5, 70), s=15, c="aqua", alpha=1.0, justify='centered', depth=0.5)

        # Orientiamo il testo: Di base è steso a terra.
        # Lo ruotiamo di 90 gradi su X per "alzarlo" in piedi.
        self.txt_mesh.rotate_x(90)

        # Aggiungiamo TUTTO al plotter
        self.plt.add(self.lh_pial, self.rh_pial, self.txt_mesh)

        # Camera iniziale
        self.plt.show(zoom=1.5, viewup='z', bg='navy', interactive=False)

        self.is_ready = True
        print("Dati caricati e scena pronta.")

    def start_animation(self):
        if not self.is_ready:
            self.load_data_and_create_mesh()
        # Avvia timer (30ms)
        self.timer.start(50)

    def rotate_step(self):
        if self.is_ready:
            # Ruota attorno all'asse Z
            self.lh_pial.rotate(1.0, axis=(0, 0, 1))
            self.rh_pial.rotate(1.0, axis=(0, 0, 1))
            self.plt.render()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simula LOGO - PyQt6")
        self.resize(300, 200)  # Ho allargato leggermente la finestra

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Widget 3D
        self.logo_widget = LogoWidget()
        main_layout.addWidget(self.logo_widget, stretch=1)

        # Footer
        # info_label = QLabel("Animazione fluida in PyQt6")
        # info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # info_label.setStyleSheet("color: #555; font-size: 10px;")
        # main_layout.addWidget(info_label)

        # Avvio ritardato
        QTimer.singleShot(100, self.logo_widget.start_animation)

    def closeEvent(self, event):
        self.logo_widget.plt.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())