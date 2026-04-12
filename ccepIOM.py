"""
    CCEP cortico cortical evoked potential : intraoperatory responses obtained by stimulation of white matter
    and recording on cortex surface with grid electrodes
"""
import logging

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal, Qt
import vedo
from PyQt6.QtWidgets import QWidget, QApplication, QVBoxLayout
# from vedo.intertools import QVTKRenderWindowInteractor
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from nilearn.datasets import fetch_atlas_aal
import nibabel as nib


# --- FUNZIONE DI SIMULAZIONE SFAP (Modella Ricker) ---
def single_fiber_Ricker(t, amplitude, duration_tau, delay):
    """ Calcola un singolo Potenziale d'Azione (SFAP) (Modello Ricker). """
    t_shifted = t - delay
    return amplitude * (1 - (t_shifted ** 2 / duration_tau ** 2)) * np.exp(-t_shifted ** 2 / (2 * duration_tau ** 2))

# --- FUNZIONI DI SIMULAZIONE SFAP (Gaussiana Bipolare) ---
def single_fiber_ap(t, amplitude, duration_tau, delay):
    """ Calcola un singolo Potenziale d'Azione (SFAP) come somma di due Gaussiane. """
    t_shifted = t - delay
    # Parametri per la forma bipolare
    A_pos = amplitude * 1.5
    sigma_pos = duration_tau * 0.5
    A_neg = amplitude * 0.5
    sigma_neg = duration_tau * 1.0
    # Funzione Gaussiana: A * exp(-(t^2)/(2*sigma^2))
    phase_pos = A_pos * np.exp(-(t_shifted ** 2) / (2 * sigma_pos ** 2))
    phase_neg = A_neg * np.exp(-(t_shifted ** 2) / (2 * sigma_neg ** 2))
    return phase_pos - phase_neg


def extract_aal_region_mesh(atlas_data, region_name):
    """
    Estrae e converte in vedo.Mesh una singola regione da un oggetto Atlante AAL 3v2 di nilearn.
    """
    try:
        # 1. ⭐️ CORREZIONE PERCORSO NIFTI: L'atlante AAL 3v2 è un singolo file NIfTI.
        nifti_file_path = atlas_data.maps

        # 2. ⭐️ CORREZIONE ID REGIONE (Logica AAL3v2): ID = indice nella lista delle labels + 1
        # Il valore del voxel AAL è (indice nella lista labels) + 1
        region_index = atlas_data.labels.index(region_name)
        region_id = region_index + 1  # ID del voxel = indice + 1 (0 è background)

        # 3. Carica e prepara i dati
        img = nib.load(nifti_file_path)
        data = img.get_fdata()
        affine = img.affine

        # ... (Il resto della logica di estrazione resta invariata)
        mask = (data == region_id)

        # 4. Converte in vedo.Volume e poi in Mesh
        vedo_volume = vedo.Volume(mask.astype(np.uint8))
        region_mesh = vedo_volume.isosurface().c("red").smooth()

        # 5. Applica la trasformazione per posizionare correttamente il mesh nello spazio MNI
        region_mesh.apply_transform(affine)

        return region_mesh
    except ValueError:
        print(f"ERRORE: La regione '{region_name}' non è stata trovata nell'atlante AAL.")
        return None
    except Exception as e:
        print(f"ERRORE durante l'estrazione della regione {region_name}: {e}")
        return None

########################################################################################################################
class CCEPWorker(QtCore.QObject):
    """
    Esegue i calcoli intensivi (CCEP evoked, rumore, averaging) nel thread separato.
    """
    # Segnale: emette un dizionario {dist_cm: dati_mediati}, il contatore della media e il livello di rumore
    data_ready = pyqtSignal(dict, int, float)

    def __init__(self, simulator_params):
        super().__init__()

        # Inizializza gli attributi del worker dai parametri passati
        for key, value in simulator_params.items():
            setattr(self, key, value)

########################################################################################################################
class CcepClasController(QWidget):

    def __init__(self, json_anomaly_path=None, parent=None):
        super().__init__(parent)

        self.logger = logging.getLogger(__name__)
        self.json_anomaly_path = json_anomaly_path
        self.brain_mesh = None
        self.plotter = None

        # ⭐️ CARICA L'ATLANTE ALL'AVVIO
        # Visto il DeprecationWarning, useremo la versione AAL 3v2 se disponibile in futuro,
        # ma per ora usiamo la versione standard AAL SPM12.
        self.aal_atlas = fetch_atlas_aal(version='3v2')
        names = self.get_region_names()
        print("\n--- Nomi delle Regioni AAL ---")
        print(f"Totale Regioni trovate: {len(names)}")
        print(names)
        print("----------------------------\n")

        self.init_ui()

        # Carica la regione dal nuovo atlante
        # self.load_aal_region('Background')
        self.load_aal_region('Frontal_Inf_Oper_L')
        self.load_aal_region('Frontal_Inf_Oper_R')
        self.load_aal_region('Precentral_L')
        self.load_aal_region('Precentral_R')

    def init_ui(self):
        """ Configura il layout base e il Plotter Vedo."""
        self.setWindowTitle("ION-Sim CCEP Controller")
        self.resize(800, 600)

        # 1. Crea il layout principale (Verticale)
        main_layout = QVBoxLayout(self)
        self.qt_widget = QVTKRenderWindowInteractor(self)

        # 3. Collega il Plotter Vedo a questo Widget
        # Quando inizializzi vedo.Plotter, gli dici ESPLICITAMENTE quale RenderWindow usare.
        self.plotter = vedo.Plotter(
            bg='gray',
            axes=4,
            # Passa la Render Window del widget QT al Plotter Vedo
            qt_widget=self.qt_widget,
            interactive=False
        )

        user_style = vtk.vtkInteractorStyleUser()

        # self.plotter.interactor.SetInteractorStyle(user_style)
        # 2. Impedisce al QVTKRenderWindowInteractor di gestire gli eventi
        # self.qt_widget.SetInteractorStyle(user_style)
        # self.plotter.interactor.Disable()

        # 4. Aggiunge il Widget QT/VTK al layout di Qt
        main_layout.addWidget(self.qt_widget)
        # 5. Configurazione finale dell'interactor
        self.qt_widget.Initialize()
        self.qt_widget.Start()
        self.setLayout(main_layout)

    def get_region_names(self):
        """ Restituisce l'elenco dei nomi delle regioni nell'atlante AAL 3v2. """
        # ⭐️ L'attributo .labels contiene l'elenco dei nomi delle regioni.
        region_labels = self.aal_atlas.labels
        return region_labels

    def load_aal_region(self, region_name=None):
        """ Carica una regione specifica dall'atlante AAL e la visualizza. """

        # Chiama la funzione di estrazione, passando l'oggetto atlas scaricato
        region_mesh = extract_aal_region_mesh(self.aal_atlas, region_name)

        if region_mesh:
            print(f"Regione AAL '{region_name}' caricata con successo.")

            # Aggiunge la mesh della regione al plotter
            self.plotter.add(region_mesh)
            region_mesh.c('blue').alpha(0.7)

            # Aggiunge un'etichetta al centro
            self.plotter.add(vedo.Text3D(region_name, pos=region_mesh.center_of_mass(), s=5, c='green'))
            # Rende la scena
            self.plotter.render()
        else:
            print(f"Caricamento della regione AAL '{region_name}' fallito.")

    def loadVtk(self):
        # load file vtk/brain_v1b_labelled.vtk
        vtk_file_path = "vtk/brain_v1b_labelled.vtk"
        try:
            self.brain_mesh = vedo.load(vtk_file_path)
            if self.brain_mesh:
                # self.brain_mesh.cmap('Spectral').add_scalarbar().print()
                print(f"File VTK '{vtk_file_path}' caricato con successo come: {type(self.brain_mesh).__name__}")
                # Aggiunge il mesh del cervello al plotter
                self.plotter.add(self.brain_mesh.cmap(
                    'Spectral',
                    'AtlasLabels',
                    on='cells'
                ).add_scalarbar(title='Anatomy sites').alpha(0.5))

                # OPZIONALE: Configurazione base del mesh
                # self.brain_mesh.color('grey').alpha(0.2).wireframe(False)

                # Rende la scena (aggiorna la visualizzazione)
                self.plotter.render()

            else:
                print(f"ATTENZIONE: vedo.load() ha restituito None per '{vtk_file_path}'.")
        except FileNotFoundError:
            print(f"ERRORE: Impossibile trovare il file VTK a: {vtk_file_path}")
            self.brain_mesh = None
        except Exception as e:
            print(f"ERRORE durante il caricamento del file VTK: {e}")
            self.brain_mesh = None

if __name__ == '__main__':
    import sys
    app_qt = QApplication(sys.argv)
    main_window = CcepClasController()
    main_window.show()
    sys.exit(app_qt.exec())