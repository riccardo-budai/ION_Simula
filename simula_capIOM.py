import sys
import webbrowser as wb
import numpy as np
import pyqtgraph as pg
from pyqtgraph import BarGraphItem, InfiniteLine, LinearRegionItem
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSpinBox, QPushButton, QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt, QCoreApplication
# from PyQt6.QtGui import QFont, QCursor

# --- BIMODAL DISTRIBUTION PARAMETERS (Realistic) ---
BIMODAL_PARAMS = {
    # Peak 1: Fast Fibers (Group I)
    'mean1': 55.0,  # Mean Velocity (m/s)
    'std1': 5.0,  # Standard Deviation
    'range1': (40, 70),  # Truncation Range (min/max VC)

    # Peak 2: Intermediate/Slow Fibers (Group II)
    'mean2': 30.0,  # Mean Velocity (m/s)
    'std2': 8.0,  # Standard Deviation
    'range2': (15, 45),  # Truncation Range (min/max VC)
    'split_ratio': 0.70  # Percentage of fibers in Peak 1 (70% fast, 30% slow)
}

# --- PARAMETRO DEFAULT INIZIALE ---
INITIAL_DISTANCE_MM = 140
NUM_FIBERS = 18000


# --- FUNZIONI DI CALCOLO METRICHE ---

def calculate_cap_metrics(velocity_axis: np.ndarray, amplitude_data: np.ndarray) -> tuple:
    """
    Calcola la VC di picco e l'ampiezza massima per la curva CAP data.

    Ritorna (vc_peak, amplitude_peak).
    """
    if len(velocity_axis) == 0:
        return 0.0, 0.0

    # L'ampiezza di picco (Peak-to-Peak o massima assoluta)
    # Usiamo la massima assoluta per semplicità sul segnale filtrato
    amplitude_peak = np.max(np.abs(amplitude_data))

    # Trova l'indice del picco (dove l'ampiezza assoluta è massima)
    peak_index = np.argmax(np.abs(amplitude_data))
    vc_peak = velocity_axis[peak_index]

    return vc_peak, amplitude_peak


def generate_sfap(t, arrival_time):
    """Generates a Single Fiber Action Potential (SFAP)."""
    amplitude = 1.0
    width1 = 0.1
    width2 = 0.2
    relative_time = t - arrival_time
    positive_phase = -1.0 * amplitude * np.exp(-np.power(relative_time / width1, 2))
    negative_phase = 0.5 * amplitude * np.exp(-np.power((relative_time - 0.15) / width2, 2))
    return positive_phase + negative_phase


def generate_bimodal_velocities(n_fibers):
    """Generates a bimodal distribution of conduction velocities."""
    params = BIMODAL_PARAMS
    n_group1 = int(n_fibers * params['split_ratio'])
    n_group2 = n_fibers - n_group1

    # --- Group 1 (Fast) ---
    v1_all = np.random.normal(params['mean1'], params['std1'], n_group1 * 2)
    v1 = v1_all[(v1_all >= params['range1'][0]) & (v1_all <= params['range1'][1])][:n_group1]
    while v1.size < n_group1:
        v1 = np.append(v1, params['mean1'])
    v1 = v1[:n_group1]

    # --- Group 2 (Intermediate/Slow) ---
    v2_all = np.random.normal(params['mean2'], params['std2'], n_group2 * 2)
    v2 = v2_all[(v2_all >= params['range2'][0]) & (v2_all <= params['range2'][1])][:n_group2]
    while v2.size < n_group2:
        v2 = np.append(v2, params['mean2'])
    v2 = v2[:n_group2]

    all_velocities = np.concatenate([v1, v2])
    return all_velocities

# todo valutare ls modalità di modificare la distribuzione delle fibre in modalità deficit assonale
#  (blocco di conduzione) e demielinizzazione delle fibre (rallentameto e desincronizzazione assonale)

# --- PYQTGRAPH VISUALIZATION CLASS (Updated with ROI Selection) ---

class CAPSimulatorViewer(QWidget):
    def __init__(self, data, all_velocities, parent=None):
        super().__init__(parent)
        self.setWindowTitle("[ION-Sim] Bimodal CAP Simulation")
        self.setGeometry(100, 100, 1200, 600)

        self.all_velocities = all_velocities
        self.current_distance_mm = INITIAL_DISTANCE_MM

        self.data = data  # Dati iniziali del CAP
        self.full_cap_data = data  # Mantiene una copia per i calcoli ROI

        # Inizializzazione oggetti PyqtGraph
        self.cap_curve = None
        self.cap_curve_roi = None
        self.velocity_cursor = None
        self.cap_spot = None
        self.histogram_spot = None
        self.roi_selector = None

        self.roi_cursor = None  # Cursore specifico per il segnale ROI (rosso)
        self.roi_spot = None  # Marker sullo spot del segnale ROI
        self.hist_data = self._prepare_hist_data()

        # Etichette
        self.velocity_label = QLabel("VC selected = N/A")
        self.amplitude_label = QLabel("CAP amplitude = N/A")
        self.amplitude_roi_label = QLabel("CAP(ROI) amplitude = N/A")

        self.roi_label = QLabel("ROI VC Range: All")

        # NUOVA ETICHETTA per i risultati della ROI
        self.roi_metrics_label = QLabel("ROI Peak VC: N/A | Amp: N/A")

        self.roi_label.setStyleSheet("color: #E63946;")
        self.roi_metrics_label.setStyleSheet("color: #E63946; ")  # font-weight: bold; Evidenziamo il risultato

        self.current_pathology = "Normal"  # "Normal", "Axonal", "Demyelinating"

        self._setup_ui()
        self._initialize_plot_objects()
        self._plot_data()

    def _prepare_hist_data(self):
        """Prepara e memorizza i dati statici dell'istogramma (non cambiano con la distanza)."""
        x_pdf = np.linspace(5, 75, 1000)
        pdf1 = (1 / (BIMODAL_PARAMS['std1'] * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_pdf - BIMODAL_PARAMS['mean1']) / BIMODAL_PARAMS['std1']) ** 2)
        pdf2 = (1 / (BIMODAL_PARAMS['std2'] * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_pdf - BIMODAL_PARAMS['mean2']) / BIMODAL_PARAMS['std2']) ** 2)
        pdf_total = BIMODAL_PARAMS['split_ratio'] * pdf1 + (1 - BIMODAL_PARAMS['split_ratio']) * pdf2

        hist, bin_edges = np.histogram(self.all_velocities, bins=50, density=True)

        return {
            'hist': hist,
            'bin_edges': bin_edges,
            'x_pdf': x_pdf,
            'pdf_total': pdf_total,
            'all_velocities': self.all_velocities
        }

    def _initialize_plot_objects(self):
        """Crea tutti gli oggetti PlotItem una sola volta."""
        # 1. Inizializzazione CAP Plot
        velocity_axis_ms = self.data['velocity_axis_ms']
        min_vc = np.min(velocity_axis_ms)

        self.cap_curve = self.cap_plot_item.plot(
            [], [], pen=pg.mkPen(color='cyan', width=1), name='Full CAP'
        )
        self.cap_curve_roi = self.cap_plot_item.plot(
            [], [], pen=pg.mkPen(color='red', width=2), name='ROI Contribution'
        )

        self.velocity_cursor = InfiniteLine(
            movable=True,
            pen=pg.mkPen('yellow', width=2, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen('orange', width=3),
            bounds=[min_vc, 80]
        )
        self.velocity_cursor.setValue(np.mean(velocity_axis_ms))
        self.cap_plot_item.addItem(self.velocity_cursor)

        self.roi_cursor = InfiniteLine(
            movable=True,  # Lo impostiamo a False per farlo seguire il cursore principale
            pen=pg.mkPen('red', width=2, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen('red', width=3),
            bounds = [min_vc, 80]
        )
        self.roi_cursor.setValue(self.velocity_cursor.value() + 10)
        self.cap_plot_item.addItem(self.roi_cursor)

        self.cap_plot_item.addLine(y=0, pen=pg.mkPen('w', width=1, style=Qt.PenStyle.DashLine))
        self.cap_spot = pg.ScatterPlotItem(size=12, pen=pg.mkPen('r', width=1), brush=pg.mkBrush('white'))
        self.cap_plot_item.addItem(self.cap_spot)

        self.roi_spot = pg.ScatterPlotItem(size=12, pen=pg.mkPen('w', width=1), brush=pg.mkBrush('red'))
        self.cap_plot_item.addItem(self.roi_spot)

        # self.velocity_cursor.sigPositionChanged.connect(self._update_cursor_info)
        self.velocity_cursor.sigPositionChanged.connect(self._update_full_cap_info)
        self.roi_cursor.sigPositionChanged.connect(self._update_roi_cap_info)

        # 2. Inizializzazione Histogram Plot
        self.histogram_spot = pg.ScatterPlotItem(size=12, pen=pg.mkPen('w', width=1), brush=pg.mkBrush('yellow'))
        self.plot_hist.addItem(self.histogram_spot)

        # 3. Selettore ROI (LinearRegionItem) sul grafico dell'istogramma
        min_hist_vc = np.min(self.hist_data['bin_edges'])
        max_hist_vc = np.max(self.hist_data['bin_edges'])

        self.roi_selector = LinearRegionItem(
            values=[min_hist_vc + 10, max_hist_vc - 10],  # Posizione iniziale
            orientation='vertical',
            pen=pg.mkPen('red', width=2, style=Qt.PenStyle.DashLine),
            brush=pg.mkBrush(255, 0, 0, 30)  # Rosso trasparente
        )
        self.plot_hist.addItem(self.roi_selector)
        self.roi_selector.sigRegionChanged.connect(self._update_roi_cap)

    def _setup_ui(self):
        """Creates the base structure of the window and plot widgets."""
        main_layout = QVBoxLayout(self)

        title = QLabel("Compound Action Potential (CAP) Simulation with Bimodal Distribution of myelinated fibers")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16pt; margin-bottom: 10px;")
        main_layout.addWidget(title)

        # --- Controlli Distanza ---
        controls_layout = QHBoxLayout()
        frame_style = """
            QFrame {
                background-color: #3a506b; /* Blu/Grigio Scuro */
                border: 1px solid #1e1e1e;
                border-radius: 8px; 
                padding: 5px;
                margin: 2px;
            }
            QLabel {
                color: white; 
                padding-right: 5px;
            }
        """
        # 1. QFrame per la Distanza
        distance_frame = QFrame()
        distance_frame.setStyleSheet(frame_style)
        distance_h_layout = QHBoxLayout(distance_frame)
        distance_h_layout.setContentsMargins(5, 0, 5, 0)

        distance_label = QLabel("Conduction Distance (mm):")
        self.spinbox_distance = QSpinBox()
        self.spinbox_distance.setRange(50, 1000)
        self.spinbox_distance.setSingleStep(10)
        self.spinbox_distance.setValue(INITIAL_DISTANCE_MM)
        self.spinbox_distance.valueChanged.connect(self._handle_distance_change)

        distance_h_layout.addWidget(distance_label)
        distance_h_layout.addWidget(self.spinbox_distance)
        distance_frame.setMaximumWidth(300)

        controls_layout.addWidget(distance_frame)

        # 2. QFrame per l'Intervallo VC (ROI Range)
        roi_range_frame = QFrame()
        roi_range_frame.setStyleSheet(frame_style)
        roi_range_h_layout = QHBoxLayout(roi_range_frame)
        roi_range_h_layout.setContentsMargins(5, 0, 5, 0)

        # self.roi_label è già un membro della classe
        self.roi_label.setStyleSheet("color: #E63946;")  # Mantieni il colore del testo rosso

        roi_range_h_layout.addWidget(self.roi_label)
        roi_range_frame.setMaximumWidth(300)

        controls_layout.addWidget(roi_range_frame)

        # 3. QFrame per le Metriche (ROI Peak VC/Amp)
        roi_metrics_frame = QFrame()
        # Stile leggermente diverso per evidenziare i risultati
        roi_metrics_frame.setStyleSheet("""
                    QFrame {
                        background-color: #0077b6; /* Blu più vivo */
                        border: 1px solid #1e1e1e;
                        border-radius: 8px; 
                        padding: 5px;
                        margin: 2px;
                    }
                    QLabel {
                        color: white; 
                        font-weight: bold;
                    }
                """)
        roi_metrics_h_layout = QHBoxLayout(roi_metrics_frame)
        roi_metrics_h_layout.setContentsMargins(5, 0, 5, 0)

        # self.roi_metrics_label è già un membro della classe, usiamo lo stile del Frame
        self.roi_metrics_label.setStyleSheet("")

        roi_metrics_h_layout.addWidget(self.roi_metrics_label)
        roi_metrics_frame.setMaximumWidth(350)

        controls_layout.addWidget(roi_metrics_frame)

        # Spazio elastico per spingere il bottone a destra
        controls_layout.addStretch()

        # 4. Pulsante di AIUTO (Help Button)
        help_button = QPushButton("Help...", self)
        help_button.setMaximumWidth(150)
        # Scegliamo un colore complementare o neutro, come l'arancione, per distinguersi da blu/rosso
        help_button.setStyleSheet("""
                    QPushButton {
                        background-color: #D0A500; /* Arancione */
                        color: white;
                        font-weight: bold;
                        border-radius: 8px;
                        padding: 5px 15px;
                    }
                    QPushButton:hover {
                        background-color: #fa5252;
                    }
                """)
        # Collega il pulsante alla nuova funzione di aiuto
        help_button.clicked.connect(self._show_help)
        controls_layout.addWidget(help_button)

        # 4. Pulsante di Uscita (Exit Button)
        exit_button = QPushButton("Exit...", self)
        exit_button.setMaximumWidth(150)
        exit_button.setStyleSheet("""
                    QPushButton {
                        background-color: #4639E6; /* Rosso per l'uscita */
                        color: white;
                        font-weight: bold;
                        border-radius: 8px;
                        padding: 5px 15px;
                    }
                    QPushButton:hover {
                        background-color: #fa5252;
                    }
                """)
        # Collega il pulsante per chiudere l'applicazione
        exit_button.clicked.connect(self.close)

        controls_layout.addWidget(exit_button)
        main_layout.addLayout(controls_layout)
        # --------------------------------

        plots_container = QHBoxLayout()

        # Plot 1: CAP Waveform vs Velocity
        self.plot_cap = pg.PlotWidget()
        self.plot_cap.setBackground('#1e1e1e')
        self.plot_cap.setLabel('left', "Amplitude (µV)", color='#FFFFFF')
        self.plot_cap.setLabel('bottom', "Conduction Velocity (m/s)", color='#FFFFFF')
        self.plot_cap.getAxis('left').setTextPen('#FFFFFF')
        self.plot_cap.getAxis('bottom').setTextPen('#FFFFFF')
        self.plot_cap.getAxis('left').setPen('#FFFFFF')
        self.plot_cap.getAxis('bottom').setPen('#FFFFFF')

        self.plot_cap.showGrid(x=True, y=True, alpha=0.2)
        self.cap_plot_item = self.plot_cap.getPlotItem()
        plots_container.addWidget(self.plot_cap)

        # Plot 2: Velocity Distribution (Histogram)
        self.plot_hist = pg.PlotWidget()
        self.plot_hist.setBackground('#1e1e1e')
        self.plot_hist.setLabel('left', "Density", color='#FFFFFF')
        self.plot_hist.setLabel('bottom', "Conduction Velocity (m/s)", color='#FFFFFF')
        self.plot_hist.getAxis('left').setTextPen('#FFFFFF')
        self.plot_hist.getAxis('bottom').setTextPen('#FFFFFF')
        self.plot_hist.getAxis('left').setPen('#FFFFFF')
        self.plot_hist.getAxis('bottom').setPen('#FFFFFF')

        self.plot_hist.showGrid(x=True, y=True, alpha=0.2)
        self.hist_plot_item = self.plot_hist.getPlotItem()
        plots_container.addWidget(self.plot_hist)

        main_layout.addLayout(plots_container)
        '''
        pathology_frame_container = QHBoxLayout()  # Un layout container per allineare a destra se necessario

        pathology_frame = QFrame()
        # pathology_frame.setMaximumWidth(350)  # Lo allineiamo alla larghezza del grafico di destra

        pathology_frame.setStyleSheet("""
                QFrame {
                    background-color: #2a3c50; /* Grigio-Blu scuro */
                    border: 1px solid #505050;
                    border-radius: 5px;
                    padding: 5px;
                    margin: 2px;
                }
                QRadioButton {
                    color: white;
                }
            """)

        # pathology_v_layout = QVBoxLayout(pathology_frame)
        # pathology_v_layout.addWidget(QLabel("Simulazione Patologia Fibra Nervosa:"))

        radio_layout = QHBoxLayout(pathology_frame)

        # Gruppo per assicurare che solo un pulsante sia selezionato
        self.pathology_group = QButtonGroup(self)

        # Radio Button 1: Normale (Default)
        radio_normal = QRadioButton("Normal")
        radio_normal.setChecked(True)
        radio_normal.id = "Normal"

        # Radio Button 2: Deficit Assonale (Blocco di Conduzione)
        radio_axonal = QRadioButton("Axonal loss")
        radio_axonal.id = "Axonal"

        # Radio Button 3: Demielinizzazione
        radio_demyelinating = QRadioButton("Demyelinating")
        radio_demyelinating.id = "Demyelinating"

        self.pathology_group.addButton(radio_normal, 0)
        self.pathology_group.addButton(radio_axonal, 1)
        self.pathology_group.addButton(radio_demyelinating, 2)

        # Collega l'evento al gestore
        self.pathology_group.buttonClicked.connect(self._handle_pathology_change)

        radio_layout.addWidget(radio_normal)
        radio_layout.addWidget(radio_axonal)
        radio_layout.addWidget(radio_demyelinating)
        radio_layout.addStretch()  # Spinge i pulsanti a sinistra

        # pathology_v_layout.addLayout(radio_layout)

        # Aggiungi il frame a destra, sotto il plot_hist
        pathology_frame_container.addStretch(1)  # Spinge il frame a destra (sotto plot_hist)
        pathology_frame_container.addWidget(pathology_frame)

        main_layout.addLayout(pathology_frame_container)

        cursor_info_layout = QHBoxLayout()
        # Stile base per i frame dei cursori
        cursor_frame_style = """
                    QFrame {
                        background-color: #1a1a1a; /* Grigio scuro quasi nero */
                        border: 1px solid #505050;
                        border-radius: 5px; 
                        padding: 3px;
                        margin-right: 10px;
                    }
                """

        # --- 1. Frame per il Cursore Principale (CAP Totale - Giallo) ---
        main_cursor_frame = QFrame()
        main_cursor_frame.setStyleSheet(cursor_frame_style)
        main_cursor_h_layout = QHBoxLayout(main_cursor_frame)
        main_cursor_h_layout.setContentsMargins(5, 0, 5, 0)

        # Aggiunge le etichette al frame principale
        # self.velocity_label è già un membro della classe
        main_cursor_h_layout.addWidget(self.velocity_label)
        # self.amplitude_label è già un membro della classe
        main_cursor_h_layout.addWidget(self.amplitude_label)

        cursor_info_layout.addWidget(main_cursor_frame)

        # --- 2. Frame per il Cursore ROI (Segnale Filtrato - Rosso) ---
        roi_cursor_frame = QFrame()
        roi_cursor_frame.setStyleSheet(cursor_frame_style)
        roi_cursor_h_layout = QHBoxLayout(roi_cursor_frame)
        roi_cursor_h_layout.setContentsMargins(5, 0, 5, 0)

        # self.amplitude_roi_label contiene sia VC(ROI) che Amp(ROI)
        self.amplitude_roi_label.setStyleSheet("color: #FF6347;")  # Rosso/Arancio per l'evidenza

        roi_cursor_h_layout.addWidget(self.amplitude_roi_label)

        cursor_info_layout.addWidget(roi_cursor_frame)

        cursor_info_layout.addLayout(radio_layout)

        cursor_info_layout.addStretch()
        main_layout.addLayout(cursor_info_layout)
        '''
        # Layout principale per i controlli inferiori
        bottom_controls_layout = QHBoxLayout()

        # Stile base per i frame dei cursori/patologia
        info_frame_style = """
                    QFrame {
                        background-color: #1a1a1a; /* Grigio scuro quasi nero */
                        border: 1px solid #505050;
                        border-radius: 5px; 
                        padding: 3px;
                        margin-right: 10px;
                    }
                """

        # --- 1. Frame per il Cursore Principale (CAP Totale - Giallo) ---
        main_cursor_frame = QFrame()
        main_cursor_frame.setStyleSheet(info_frame_style)
        main_cursor_h_layout = QHBoxLayout(main_cursor_frame)
        main_cursor_h_layout.setContentsMargins(5, 0, 5, 0)

        main_cursor_h_layout.addWidget(self.velocity_label)
        main_cursor_h_layout.addWidget(self.amplitude_label)
        bottom_controls_layout.addWidget(main_cursor_frame)

        # --- 2. Frame per il Cursore ROI (Segnale Filtrato - Rosso) ---
        roi_cursor_frame = QFrame()
        roi_cursor_frame.setStyleSheet(info_frame_style)
        roi_cursor_h_layout = QHBoxLayout(roi_cursor_frame)
        roi_cursor_h_layout.setContentsMargins(5, 0, 5, 0)

        self.amplitude_roi_label.setStyleSheet("color: #FF6347;")
        roi_cursor_h_layout.addWidget(self.amplitude_roi_label)

        bottom_controls_layout.addWidget(roi_cursor_frame)

        # --- 3. Frame per la Patologia (Nuova posizione) ---
        pathology_frame = QFrame()
        # Non serve più pathology_frame_container; il frame viene aggiunto direttamente.

        pathology_frame.setStyleSheet("""
                        QFrame {
                            background-color: #2a3c50; /* Grigio-Blu scuro */
                            border: 1px solid #505050;
                            border-radius: 5px;
                            padding: 5px;
                            margin: 2px;
                        }
                        QRadioButton {
                            color: white;
                        }
                    """)

        radio_layout = QHBoxLayout(pathology_frame)

        # Gruppo per assicurare che solo un pulsante sia selezionato
        self.pathology_group = QButtonGroup(self)

        # Radio Button 1: Normale (Default)
        radio_normal = QRadioButton("Normal")
        radio_normal.setChecked(True)
        radio_normal.id = "Normal"

        # Radio Button 2: Deficit Assonale (Blocco di Conduzione)
        radio_axonal = QRadioButton("Axonal loss")
        radio_axonal.id = "Axonal"

        # Radio Button 3: Demielinizzazione
        radio_demyelinating = QRadioButton("Demyelinating")
        radio_demyelinating.id = "Demyelinating"

        self.pathology_group.addButton(radio_normal, 0)
        self.pathology_group.addButton(radio_axonal, 1)
        self.pathology_group.addButton(radio_demyelinating, 2)

        self.pathology_group.buttonClicked.connect(self._handle_pathology_change)

        radio_layout.addWidget(radio_normal)
        radio_layout.addWidget(radio_axonal)
        radio_layout.addWidget(radio_demyelinating)
        radio_layout.addStretch()  # Spinge i pulsanti a sinistra

        bottom_controls_layout.addWidget(pathology_frame)  # AGGIUNTA DEL FRAME PATOLOGIA

        bottom_controls_layout.addStretch()  # Spazio elastico finale
        # Aggiunge il layout contenente tutti i controlli inferiori al layout principale
        main_layout.addLayout(bottom_controls_layout)

    def _handle_pathology_change(self, button):
        """
        Gestisce la selezione di un nuovo stato patologico tramite i radio button.
        QUI dovrà essere aggiunta la logica per modificare BIMODAL_PARAMS.
        """
        pathology_id = button.id        # Ottiene l'ID che abbiamo assegnato al radio button
        self.current_pathology = pathology_id

        print(f"Modalità Patologia cambiata in: {self.current_pathology}")

        # LOGICA FUTURA:
        # 1. Modificare globalmente BIMODAL_PARAMS o creare un set di parametri per la patologia.
        # 2. Richiamare self.all_velocities = generate_bimodal_velocities(NUM_FIBERS) con i nuovi parametri.
        # 3. Aggiornare i grafici: self._update_cap_plot() e self._update_roi_cap().

        # Esempio di come la logica futura potrebbe iniziare:
        # if pathology_id == "Axonal":
        #     # Aumenta la deviazione standard per la demielinizzazione,
        #     # ma per Axonal devi ridurre NUM_FIBERS o l'ampiezza delle curve.
        #     pass

        # In questo momento, la simulazione torna semplicemente ai dati iniziali,
        # ma l'interfaccia è pronta.

    def _handle_distance_change(self, new_distance_mm: int):
        """
        Ricalcola la simulazione del CAP con la nuova distanza
        e aggiorna i plot.
        """
        self.current_distance_mm = new_distance_mm

        new_data = run_single_cap_simulation(
            NUM_FIBERS,
            new_distance_mm,
            self.all_velocities
        )
        self.data = new_data
        self.full_cap_data = new_data  # Aggiorna la copia dei dati completi

        self._update_cap_plot()
        self._update_roi_cap()  # Ricalcola l'ROI con i nuovi dati CAP

        # Aggiorna la posizione del cursore (o costringilo a fare un nuovo check)
        self.velocity_cursor.setValue(self.velocity_cursor.value())

    def _show_help(self):
        pdf_path = 'help_docs/paper/Tut_Cap.pdf'
        wb.open_new(pdf_path)

    def _update_roi_cap(self):
        """
        Ricalcola e aggiorna la curva del CAP in base all'intervallo di velocità selezionato (ROI).
        Aggiorna anche le metriche di picco della ROI.
        """
        if self.roi_selector is None:
            return

        # Ottieni i limiti di velocità selezionati dall'ROI
        vc_min, vc_max = self.roi_selector.getRegion()

        # 1. Richiama la simulazione filtrata
        data_roi = run_single_cap_simulation_filtered(
            NUM_FIBERS,
            self.current_distance_mm,
            self.all_velocities,
            vc_min,
            vc_max
        )

        # 2. Calcola le metriche di picco sulla curva ROI
        vc_peak_roi, amp_peak_roi = calculate_cap_metrics(
            data_roi['velocity_axis_ms'],
            data_roi['scaled_wave_valid']
        )

        # 3. Aggiorna le etichette
        self.roi_label.setText(f" ROI VC Range: {vc_min:.1f} to {vc_max:.1f} m/s")
        self.roi_metrics_label.setText(
            f"VC Peak: {vc_peak_roi:.1f} m/s | Amp Peak: {amp_peak_roi:.1f} µV"
        )

        # 4. Aggiorna la curva ROI
        self.cap_curve_roi.setData(
            data_roi['velocity_axis_ms'],
            data_roi['scaled_wave_valid']
        )

        # Forziamo il cursore a fare un nuovo check per aggiornare lo spot sull'istogramma
        self._update_full_cap_info()
        self._update_roi_cap_info()

    def _update_hist_spot(self, vc_value):
        """
        Aggiorna la posizione dello spot sull'istogramma basato sul cursore principale.
        """
        hist_data = self.hist_data
        hist = hist_data['hist']
        bin_edges = hist_data['bin_edges']

        # Aggiorna lo spot giallo (associato al cursore principale) sull'istogramma
        if hist.size > 0:
            bin_index = np.digitize(vc_value, bin_edges) - 1
            bin_index = max(0, min(bin_index, len(hist) - 1))
            bar_center = (bin_edges[bin_index] + bin_edges[bin_index + 1]) / 2
            bar_height = hist[bin_index]
            self.histogram_spot.setData([bar_center], [bar_height])
        else:
            self.histogram_spot.setData([], [])

    def _update_full_cap_info(self):
        """
        Aggiorna le etichette VC, Amplitudine CAP completa e lo spot sull'istogramma
        in base alla posizione del cursore giallo (principale).
        """
        vc_value = self.velocity_cursor.value()

        velocity_axis_full = self.data['velocity_axis_ms']
        amplitude_data_full = self.data['scaled_wave_valid']

        # --- CAP Plot Update - Full Signal (Giallo) ---
        idx_full = np.argmin(np.abs(velocity_axis_full - vc_value))
        amplitude_full = amplitude_data_full[idx_full]

        yellow_style = "color: #FFFF00;"

        # Update labels (VC e CAP totale)
        self.velocity_label.setStyleSheet(yellow_style)
        self.velocity_label.setText(f"<b>VC</b> selected = {vc_value:.1f} m/s")
        self.amplitude_label.setStyleSheet(yellow_style)
        self.amplitude_label.setText(f"  <b>CAP</b> amplitude = {amplitude_full:.1f} µV")

        # Update Full CAP spot (giallo/bianco)
        self.cap_spot.setData([velocity_axis_full[idx_full]], [amplitude_full])

        # Aggiorna lo spot sull'istogramma in base al cursore principale
        self._update_hist_spot(vc_value)

    def _update_roi_cap_info(self):
        """
        Aggiorna l'amplitudine CAP(ROI) e lo spot rosso in base alla posizione
        del cursore rosso (ROI).
        """
        vc_value = self.roi_cursor.value()

        # Dati del segnale ROI
        (velocity_axis_roi, amplitude_data_roi) = self.cap_curve_roi.getData()

        # --- CAP Plot Update - ROI Signal (Rosso) ---
        amplitude_roi = 0.0

        if velocity_axis_roi is not None and len(velocity_axis_roi) > 0:
            # Trova l'indice più vicino sulla curva ROI
            idx_roi = np.argmin(np.abs(velocity_axis_roi - vc_value))
            amplitude_roi = amplitude_data_roi[idx_roi]

            # Update ROI spot (rosso)
            self.roi_spot.setData([velocity_axis_roi[idx_roi]], [amplitude_roi])
        else:
            # Nascondi lo spot se non ci sono dati ROI validi
            self.roi_spot.setData([], [])

        # Aggiorna la label ROI Amplitude
        self.amplitude_roi_label.setText(
            f"  <b>VC(ROI)</b> = {vc_value:.1f} m/s amplitude = {amplitude_roi:.1f} µV")

    def _update_cap_plot(self):
        """Aggiorna solo la curva CAP e i limiti del cursore con i nuovi dati."""

        velocity_axis_ms = self.data['velocity_axis_ms']
        scaled_wave_valid = self.data['scaled_wave_valid']

        # 1. Aggiorna i dati della curva completa
        self.cap_curve.setData(velocity_axis_ms, scaled_wave_valid)

        # 2. Aggiorna il range X e i limiti del cursore
        min_vc = np.min(velocity_axis_ms)
        self.cap_plot_item.setXRange(80, min_vc)
        self.cap_plot_item.invertY(True)
        self.cap_plot_item.invertX(True)

        self.velocity_cursor.setBounds([min_vc, 80])
        self.roi_cursor.setBounds([min_vc, 80])

        self.cap_plot_item.enableAutoRange(axis='y', enable=False)  # eventualmente con checkbox ?
        self.cap_plot_item.enableAutoRange(axis='x', enable=False)

    def _plot_data(self):
        """Disegna l'istogramma statico e chiama l'aggiornamento per il CAP."""
        hist_data = self.hist_data

        # Chiamata all'aggiornamento dinamico del CAP (curva completa e cursore)
        self._update_cap_plot()
        self._update_roi_cap()  # Aggiorna la curva ROI

        # --- Plot 2: Bimodal Distribution (Istogramma Statico) ---

        # 1. Istogramma (BarGraphItem)
        bars = BarGraphItem(
            x=hist_data['bin_edges'][:-1],
            height=hist_data['hist'],
            width=hist_data['bin_edges'][1] - hist_data['bin_edges'][0],
            brush=pg.mkBrush(color=(90, 150, 255, 200))
        )
        self.plot_hist.addItem(bars)

        # 2. Curva Teorica (PDF)
        max_hist = np.max(hist_data['hist'])
        pdf_scaled = hist_data['pdf_total'] * (max_hist / np.max(hist_data['pdf_total']))

        self.plot_hist.plot(
            hist_data['x_pdf'],
            pdf_scaled,
            pen=pg.mkPen('red', width=3),
            name='Theoretical Distribution'
        )
        self.plot_hist.addLegend()

        # 3. Aggiornamento iniziale dello spot sull'istogramma
        self._update_full_cap_info()

    '''
    def _update_cursor_info(self):
        """
        Calculates and updates CAP amplitude and plots the corresponding spot
        on the histogram bar and the ROI spot.
        """
        vc_value = self.velocity_cursor.value()

        # 1. Aggiorna la posizione del cursore ROI per farlo seguire quello principale
        self.roi_cursor.setValue(vc_value)

        velocity_axis_full = self.data['velocity_axis_ms']
        amplitude_data_full = self.data['scaled_wave_valid']

        # Dati del segnale ROI (dal calcolo più recente)
        # Nota: dobbiamo ASSUMERE che l'ultimo run_single_cap_simulation_filtered abbia aggiornato self.cap_curve_roi
        # Otteniamo i dati della curva ROI direttamente da cap_curve_roi per consistenza:
        (velocity_axis_roi, amplitude_data_roi) = self.cap_curve_roi.getData()

        # --- 1. CAP Plot Update (Left) - Full Signal ---
        idx_full = np.argmin(np.abs(velocity_axis_full - vc_value))
        amplitude_full = amplitude_data_full[idx_full]

        # Update labels
        self.velocity_label.setText(f"<b>VC</b> selected = {vc_value:.1f} m/s")
        self.amplitude_label.setText(f"  <b>CAP</b> amplitude = {amplitude_full:.1f} µV")

        # Update Full CAP spot (giallo)
        self.cap_spot.setData([velocity_axis_full[idx_full]], [amplitude_full])

        amplitude_roi = 0.0
        # --- 2. CAP Plot Update (Left) - ROI Signal (Rosso) ---
        if velocity_axis_roi is not None and len(velocity_axis_roi) > 0:
            # Trova l'indice più vicino sulla curva ROI
            idx_roi = np.argmin(np.abs(velocity_axis_roi - vc_value))
            amplitude_roi = amplitude_data_roi[idx_roi]

            # Update ROI spot (rosso)
            self.roi_spot.setData([velocity_axis_roi[idx_roi]], [amplitude_roi])
        else:
            # Nascondi lo spot se non ci sono dati ROI validi
            self.roi_spot.setData([], [])

        self.amplitude_roi_label.setText(f"  <b>CAP(ROI)</b> amplitude = {amplitude_roi:.1f} µV")

        # --- 3. Histogram Plot Update (Right) ---
        hist_data = self.hist_data
        hist = hist_data['hist']
        bin_edges = hist_data['bin_edges']

        # ... (Logica di aggiornamento dell'istogramma invariata) ...
        if hist.size > 0:
            bin_index = np.digitize(vc_value, bin_edges) - 1
            bin_index = max(0, min(bin_index, len(hist) - 1))
            bar_center = (bin_edges[bin_index] + bin_edges[bin_index + 1]) / 2
            bar_height = hist[bin_index]
            self.histogram_spot.setData([bar_center], [bar_height])
        else:
            self.histogram_spot.setData([], [])
    '''

def run_single_cap_simulation(n_fibers: int, distance_mm: float, all_velocities: np.ndarray):
    """
    Runs the CAP calculation for a single distance.
    """
    distance_m = distance_mm / 1000.0

    # 1. CAP Calculation (Times)
    arrival_times_ms = (distance_m / all_velocities) * 1000

    # Calcolo dell'asse temporale
    min_time = np.min(arrival_times_ms) - 2.0
    max_time = np.max(arrival_times_ms) + 3.0
    time_step = 0.05
    time_points = np.arange(min_time, max_time, time_step)

    all_sfaps = generate_sfap(time_points[:, None], arrival_times_ms)
    cap_waveform = np.sum(all_sfaps, axis=1)
    scaled_wave = cap_waveform * (1000 / n_fibers)

    # 2. Data Preparation for PyQtGraph (Time -> Velocity)
    valid_indices = time_points > 0
    time_points_valid = time_points[valid_indices]
    scaled_wave_valid = scaled_wave[valid_indices]

    velocity_axis_ms = distance_m / (time_points_valid / 1000.0)

    # 3. Sort data by velocity (X-axis)
    sort_indices = np.argsort(velocity_axis_ms)
    velocity_axis_ms = velocity_axis_ms[sort_indices]
    scaled_wave_valid = scaled_wave_valid[sort_indices]

    return {
        'velocity_axis_ms': velocity_axis_ms,
        'scaled_wave_valid': scaled_wave_valid,
    }


def run_single_cap_simulation_filtered(n_fibers: int, distance_mm: float, all_velocities: np.ndarray, vc_min: float,
                                       vc_max: float):
    """
    Runs the CAP calculation for a single distance, filtering fibers based on VC range.
    """
    # 1. Filtra le fibre
    filtered_indices = (all_velocities >= vc_min) & (all_velocities <= vc_max)
    filtered_velocities = all_velocities[filtered_indices]

    # Se nessuna fibra è selezionata, restituisci dati vuoti
    if len(filtered_velocities) == 0:
        return {
            'velocity_axis_ms': np.array([vc_min, vc_max]),
            'scaled_wave_valid': np.array([0, 0]),
        }

    # 2. Ricalcola i parametri temporali (basati solo sulle fibre filtrate)
    distance_m = distance_mm / 1000.0

    arrival_times_ms = (distance_m / filtered_velocities) * 1000

    # L'asse temporale deve rimanere fisso o essere abbastanza ampio da contenere il segnale
    # Usiamo un asse temporale fisso per coerenza con l'asse delle ascisse (non strettamente necessario in questo plotting)
    # Ricalcoliamo i limiti temporali sulla base delle fibre filtrate
    min_time = np.min(arrival_times_ms) - 2.0 if len(filtered_velocities) > 0 else 0
    max_time = np.max(arrival_times_ms) + 3.0 if len(filtered_velocities) > 0 else 10
    time_step = 0.05
    time_points = np.arange(min_time, max_time, time_step)

    # 3. Calcola gli SFAP (l'ampiezza viene scalata sulla nuova popolazione)
    all_sfaps = generate_sfap(time_points[:, None], arrival_times_ms)
    cap_waveform = np.sum(all_sfaps, axis=1)

    # L'ampiezza è scalata sulla popolazione *totale* originale, non sulla popolazione filtrata,
    # per riflettere il contributo reale al segnale totale (n_fibers).
    scaled_wave = cap_waveform * (1000 / NUM_FIBERS)

    # 4. Data Preparation for PyQtGraph (Time -> Velocity)
    valid_indices = time_points > 0
    time_points_valid = time_points[valid_indices]
    scaled_wave_valid = scaled_wave[valid_indices]

    velocity_axis_ms = distance_m / (time_points_valid / 1000.0)

    # Sort data by velocity (X-axis)
    sort_indices = np.argsort(velocity_axis_ms)
    velocity_axis_ms = velocity_axis_ms[sort_indices]
    scaled_wave_valid = scaled_wave_valid[sort_indices]

    return {
        'velocity_axis_ms': velocity_axis_ms,
        'scaled_wave_valid': scaled_wave_valid,
    }


def run_simulation_and_plot():
    """
    Runs the simulation and prepares data for the initial view.
    """
    all_velocities_array = generate_bimodal_velocities(NUM_FIBERS)
    # Calcola il CAP per la distanza iniziale di default
    data_initial = run_single_cap_simulation(
        NUM_FIBERS,
        INITIAL_DISTANCE_MM,
        all_velocities_array
    )
    # Passiamo sia i dati iniziali che l'array di velocità alla classe Viewer
    return data_initial, all_velocities_array


# --- Script Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    initial_data, all_velocities_array = run_simulation_and_plot()
    viewer = CAPSimulatorViewer(initial_data, all_velocities_array)
    viewer.show()

    sys.exit(QCoreApplication.exec())
