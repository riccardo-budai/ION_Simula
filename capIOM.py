import json
import logging
import random
import sys
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PySide6.QtUiTools import QUiLoader
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
from PySide6.QtCore import Qt, Signal, Slot, QFile, QObject, QTimer, QThread, QMetaObject, Q_ARG
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout
import webbrowser as wb

from etc.anomaly_manager import AnomalyInjector

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

def butter_bandpass(lowcut, highcut, fs, order=5):
    """ Calcola i coefficienti del filtro Butterworth passa-banda. """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """ Applica un filtro passa-banda bidirezionale (zero-phase lag). """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # filtfilt applica il filtro in entrambe le direzioni per eliminare il ritardo di fase
    return filtfilt(b, a, data)

def apply_spline_smoothing(data, time, smoothing_factor=2):
    """ Applica uno smoothing usando l'interpolazione cubica su un sottoinsieme di punti. """
    n_points = len(data)
    # Seleziona solo 1/smoothing_factor dei punti per l'interpolazione
    sparse_indices = np.arange(0, n_points, smoothing_factor)
    if sparse_indices[-1] != n_points - 1:
        sparse_indices = np.append(sparse_indices, n_points - 1)
    cs = CubicSpline(time[sparse_indices], data[sparse_indices])
    return cs(time)

def apply_simple_smooth(data, window_len=15):
    """
    Applica una media mobile semplice (Moving Average) garantendo che
    la lunghezza dell'output sia UGUALE all'input (mode='same').

    Args:
        data (np.array): Segnale di input (dimensione N).
        window_len (int): Lunghezza della finestra di media (deve essere dispari).
    Returns:
        np.array: Segnale smussato (dimensione N).
    """
    if window_len % 2 == 0:
        # Assicuriamo che la finestra sia dispari per centrare correttamente con 'same'
        window_len += 1
    # Crea il kernel di media mobile (tutti 1)
    w = np.ones(window_len, 'd')
    # Esegue la convoluzione:
    # mode='same' assicura che l'output abbia la stessa lunghezza dell'input.
    smoothed_data = np.convolve(data, w / w.sum(), mode='same')
    return smoothed_data


# --- 2. CLASSE WORKER PER IL THREAD SEPARATO --------------------------------------------------------------------------
class CAPWorker(QObject):
    """
    Esegue i calcoli intensivi (CAP, rumore, averaging) nel thread separato.
    """
    # Segnale: emette un dizionario {dist_cm: dati_mediati}, il contatore della media e il livello di rumore
    data_ready = Signal(object, int, float)

    def __init__(self, simulator_params):
        super().__init__()

        # Inizializza gli attributi del worker dai parametri passati
        for key, value in simulator_params.items():
            setattr(self, key, value)

        self.injector: AnomalyInjector = simulator_params.get('ANOMALY_INJECTOR')

        # Variabili di stato del worker
        self.update_counter = 0
        self.cap_data_buffers = {dist: [] for dist in self.DISTANCES_CM}
        self.noise_level = 0.0
        self.FIFO_BLOCK_SIZE = 50

        self.current_filter_index = 0  # 0: Nessun Filtro
        self.FS = simulator_params['FS']  # Assumiamo che Fs sia passato

        # Setup del timer che girerà nel thread separato
        self.timer = QTimer()
        self.timer.timeout.connect(self._run_simulation_step)

    @Slot()
    def _init_timer(self):
        """ Slot chiamato per inizializzare il timer nel thread corretto. """
        self.timer = QTimer()
        self.timer.timeout.connect(self._run_simulation_step)

    @Slot(int)
    def start_simulation(self, interval_ms):
        """ Slot per avviare la simulazione. """
        self.timer.start(interval_ms)

    @Slot()
    def stop_simulation(self):
        """ Slot per fermare la simulazione. """
        self.timer.stop()

    @Slot(int)
    def update_filter_state(self, new_index):
        """ Slot per ricevere l'indice del nuovo filtro selezionato. """
        self.current_filter_index = new_index
        # Non è necessario resettare il buffer qui, si applica al prossimo step

    @Slot(int, int)
    def update_averaging_params(self, new_n_avg, new_fifo_block_size):
        """
        Slot per ricevere i nuovi parametri di averaging e FIFO dal thread GUI.
        """
        self.N_AVERAGES = new_n_avg
        self.FIFO_BLOCK_SIZE = new_fifo_block_size

        # Logica opzionale: se la nuova N_AVERAGES è più piccola, potresti voler
        # svuotare i buffer. Per ora, ci limitiamo ad aggiornare il valore.

        # todo self.logger.info(f"Worker: Aggiornato AVG={new_n_avg}, FIFO Block={new_fifo_block_size}")

    @Slot()
    def arm_anomaly_trigger(self):
        """
        Slot per armare l'iniettore di anomalie.
        Chiamato dal thread della GUI.
        """
        if self.injector and not self.injector.is_armed:
            # Calcola il tempo di simulazione *corrente* del worker
            # nel momento in cui questo slot viene eseguito.
            interval_s = self.timer.interval() / 1000.0
            current_sim_time = self.update_counter * interval_s

            # Chiama il metodo 'arm' dell'iniettore
            self.injector.arm_trigger(current_sim_time)

    @Slot()
    def disarm_anomaly_trigger(self):
        """
        Slot per resettare/disarmare l'iniettore di anomalie.
        Chiamato dal thread della GUI.
        """
        if self.injector:
            self.injector.reset()

    def _calculate_cap_at_distance(self, distance_m, noise_amplitude=0):
        """ Logica di calcolo del CAP per una distanza specifica. """
        cap_sum = np.zeros_like(self.time)

        for i in range(self.N_FIBERS):
            t_delay = (distance_m / self.fiber_vcs[i]) + self.MIN_DISPLAY_DELAY
            sfap_i = single_fiber_ap(self.time, self.fiber_amplitudes[i],
                                     self.fiber_taus[i], t_delay)
            cap_sum += sfap_i

        if self.injector:
            # Il tempo di simulazione per l'injector è il tempo 'corrente' che è l'ultimo
            # elemento del tuo array 'time' se stai simulando l'intero array ad ogni step.
            # Se invece stai simulando solo un punto per timestep, devi adattare il loop.

            # Poiché _calculate_cap_at_distance calcola l'intera forma d'onda CAP
            # (tutto l'array 'self.time') ad ogni chiamata, l'iniezione deve essere
            # applicata a tutti i punti dell'array:

            # Passiamo l'intero array 'time' (punti temporali) e l'array CAP calcolato
            # Passiamo il tempo di simulazione del *worker* (self.update_counter)

            # Nota: useremo current_time basato sul contatore, convertito in secondi,
            # come tempo di simulazione esterno per i trigger (Timer/Randomized).

            # L'intervallo è self.timer.interval() in ms, quindi self.interStim
            # Il tempo di simulazione è (update_counter * interStim / 1000)

            # Per ora, usiamo l'update_counter del worker come base del tempo di simulazione
            # per semplificare l'integrazione del trigger:

            # **ATTENZIONE**: Per accedere a self.update_counter, dobbiamo spostare la logica
            # di calcolo CAP dentro il loop _run_simulation_step dove il contatore è aggiornato.

            # --- SOLUZIONE ALTERNATIVA: Applicare l'iniezione nel _run_simulation_step ---
            # Visto che i dati di iniezione dipendono dal tempo esterno, è meglio
            # spostare l'iniezione in _run_simulation_step. Lasciamo _calculate_cap_at_distance
            # solo per il calcolo del CAP teorico + rumore gaussiano.

            # Lasciamo _calculate_cap_at_distance come era, senza l'iniezione di anomalia.
            # *Tuttavia, rimuoviamo l'aggiunta del rumore da qui per gestirla in _inject_data,*
            # *come specificato nella classe AnomalyInjector.*
            pass
        # Aggiungi rumore (uso noise_amplitude direttamente come deviazione standard)
        # if noise_amplitude > 0:
        #     noise = np.random.normal(0, noise_amplitude, self.time_points)
        #    cap_sum += noise

        return cap_sum

    def _run_simulation_step(self):
        """ Esegue un singolo step di calcolo e averaging (chiamato dal timer). """
        self.update_counter += 1
        interval_s = self.timer.interval() / 1000.0
        current_sim_time = self.update_counter * interval_s
        # Rumore costante
        self.noise_level = 1000

        current_avg_count = 0
        averaged_data = {}

        for dist_cm, dist_m in zip(self.DISTANCES_CM, self.DISTANCES_M):
            new_cap_data = self._calculate_cap_at_distance(dist_m)  #, noise_amplitude=self.noise_level)

            # 2. APPLICA L'ANOMALIA (se attiva o innescata)
            if self.injector:
                # L'array self.time contiene i punti temporali del CAP [0s a 0.04s]
                # current_sim_time è il tempo esterno del simulatore [0s, 0.25s, 0.50s, ...]
                new_cap_data = self.injector.apply_anomaly(
                    current_sim_time,
                    new_cap_data,
                    self.time  # L'array dei punti temporali del CAP
                )

            # --- Averaging e Filtri (logica non modificata) ---
            buffer = self.cap_data_buffers[dist_cm]
            buffer.append(new_cap_data)

            if len(buffer) > self.N_AVERAGES:
                # Se vuoi la tua logica di rimuovere N campioni ogni volta che supera N_AVERAGES:
                for _ in range(self.FIFO_BLOCK_SIZE):  # Usa il nuovo attributo
                    if buffer:
                        buffer.pop(0)

            current_avg_count = len(buffer)

            # Calcola la media (Averaging)
            if current_avg_count > 0:
                averaged_data[dist_cm] = np.mean(buffer, axis=0)
            else:
                averaged_data[dist_cm] = np.zeros_like(self.time)

            # --- APPLICAZIONE DEL FILTRO (NUOVO) ---
            if self.current_filter_index == 1:
                # 1: Smooth Means (Media mobile semplice)
                window_size = 3  # valore dispari
                averaged_data[dist_cm] = apply_simple_smooth(averaged_data[dist_cm], window_len=window_size)
            elif self.current_filter_index == 2:
                # 2: Spline Interpolation
                averaged_data[dist_cm] = apply_spline_smoothing(averaged_data[dist_cm], self.time, smoothing_factor=5)
            elif self.current_filter_index == 3:
                # 3: Band Pass 5 - 3000 Hz
                averaged_data[dist_cm] = apply_bandpass_filter(averaged_data[dist_cm], 5, 3000, self.FS)
            elif self.current_filter_index == 4:
                # 4: Band Pass 10 - 1500 Hz
                averaged_data[dist_cm] = apply_bandpass_filter(averaged_data[dist_cm], 10, 1500, self.FS)

        # Emette il segnale con i risultati al thread principale (GUI)
        self.data_ready.emit(averaged_data, current_avg_count, self.noise_level)

    @Slot(int)
    def update_fiber_structure(self, new_n_fibers):
        """
        Slot per ricevere il nuovo numero di fibre e rigenerare tutti gli array.
        Questo metodo deve essere chiamato dal thread GUI.
        """
        self.N_FIBERS = new_n_fibers
        # 2. Rigenera gli array che dipendono da N_FIBERS (CRUCIALE!)
        self.fiber_amplitudes = np.random.uniform(5, 15, self.N_FIBERS)
        self.fiber_taus = np.full(self.N_FIBERS, 0.0015)
        # Rigenera le Velocità di Conduzione basate sul nuovo N_FIBERS
        self.fiber_vcs = np.linspace(self.MIN_VC_MS, self.MAX_VC_MS, self.N_FIBERS)

        # 3. Svuota tutti i buffer di averaging (la simulazione è cambiata strutturalmente)
        for dist in self.DISTANCES_CM:
            self.cap_data_buffers[dist] = []

        # self.logger.info(f"Worker: Struttura fibre aggiornata a N={self.N_FIBERS}.


# --- 3. CLASSE SIMULATORE CAP MULTI-DISTANZA (GUI) --------------------------------------------------------------------
class MultiDistanceCAPSimulator(QWidget):
    simulation_closed = Signal(str)

    def __init__(self, json_anomaly_path=None, learning_manager=None, current_anomaly=None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setWindowFlag(Qt.WindowType.Window)
        self.json_anomaly_path = json_anomaly_path

        self.learning_manager = learning_manager
        self.current_anomaly = current_anomaly

        self.load_json_anomaly()

        self.injector = None
        if self.json_anomaly_path:
            self.injector = AnomalyInjector(self.json_anomaly_path)

        # --- Caricamento UI e inizializzazione ---
        # Assicurati che il file 'res/capIOMForm.ui' esista
        # self.ui = uic.loadUi('res/capIOMForm.ui', self)
        ui_file_path = "res/capIOMForm.ui"
        ui_file = QFile(ui_file_path)

        if not ui_file:
            print(f"Errore: Impossibile aprire il file {ui_file_path}")
            sys.exit(-1)
        loader = QUiLoader()
        #
        loader.registerCustomWidget(pg.GraphicsLayoutWidget)
        loader.registerCustomWidget(pg.PlotWidget)
        self.ui = loader.load(ui_file)
        ui_file.close()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Rimuove margini vuoti ai bordi
        layout.addWidget(self.ui)

        # Configurazione logo (lasciata come era nel tuo codice)
        percorso_logo = 'res/logos/logoCap5.png'
        pixmap = QPixmap(percorso_logo)
        scaled_pixmap = pixmap.scaled(self.ui.labelLogo.size(),
                                      Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        self.ui.labelLogo.setPixmap(scaled_pixmap)

        self.ui.setWindowTitle("ION-Sim CAP (Compound Action Potential)")
        # self.setGeometry(50, 415, 670, 480)     # extended = 800 x 480
        self.setFixedSize(670, 480)
        self.move(50, 415)

        self.VC_REF_DIST_CM = 20.0 - 7.0  # Differenza tra Wrist (7.0) ed Elbow (20.0)
        self.VC_REF_DIST_M = self.VC_REF_DIST_CM / 100.0
        self.cursor_1 = None
        self.cursor_2 = None

        # --- PARAMETRI PER IL WORKER ---
        self.DISTANCES_CM = [7.0, 20.0, 40.0, 55.0]
        self.DISTANCES_M = [d / 100.0 for d in self.DISTANCES_CM]
        self.MIN_VC_MS = 20.0
        self.MAX_VC_MS = 65.0
        self.N_AVERAGES = 100
        self.FIFO_BLOCK_SIZE = 50
        self.MIN_DISPLAY_DELAY = 0.005
        self.N_FIBERS = 100
        self.time_points = 500
        self.max_time = 0.035 + self.MIN_DISPLAY_DELAY
        self.time = np.linspace(0, self.max_time, self.time_points)
        self.fiber_amplitudes = np.random.uniform(5, 15, self.N_FIBERS)
        self.fiber_taus = np.full(self.N_FIBERS, 0.0015)
        self.fiber_vcs = np.linspace(self.MIN_VC_MS, self.MAX_VC_MS, self.N_FIBERS)

        # CALCOLO DEL TASSO DI CAMPIONAMENTO (CRUCIALE PER IL FILTRO)
        self.FS = (self.time_points - 1) / self.max_time

        self.plot_visibility = {dist: True for dist in self.DISTANCES_CM}

        # intervallo di aggiornamento equivalente all'intervallo di erogazione dello stimolo
        # in millisecondi : 4 stimoli al secondo = 250 ms
        # todo eventuale randomizzazione dell'intervallo
        self.interStim = 250

        # Raccogli i parametri da passare al Worker
        self.worker_params = {
            'DISTANCES_CM': self.DISTANCES_CM,
            'DISTANCES_M': self.DISTANCES_M,
            'MIN_DISPLAY_DELAY': self.MIN_DISPLAY_DELAY,
            'N_FIBERS': self.N_FIBERS,
            'FS': self.FS,
            'time': self.time,
            'N_AVERAGES': self.N_AVERAGES,
            'FIFO_BLOCK_SIZE': self.FIFO_BLOCK_SIZE,
            'MIN_VC_MS': self.MIN_VC_MS,
            'MAX_VC_MS': self.MAX_VC_MS,
            'fiber_amplitudes': self.fiber_amplitudes,
            'fiber_taus': self.fiber_taus,
            'fiber_vcs': self.fiber_vcs,
            'time_points': self.time_points,
            'ANOMALY_INJECTOR': self.injector
        }

        # --- SETUP QThread E WORKER (CORE DELLA MODIFICA) ---
        self.thread = QThread()
        self.worker = CAPWorker(self.worker_params)
        self.worker.moveToThread(self.thread)  # Muove il worker nel thread separato

        # Connessioni: Segnale del Worker -> Slot della GUI
        self.worker.data_ready.connect(self.update_gui_plot)

        # Avvia il thread (senza avviare ancora il timer interno)
        self.thread.start()
        QMetaObject.invokeMethod(self.worker, "_init_timer", Qt.ConnectionType.QueuedConnection)
        self.anomaly_active = False
        self.ui.pushBtAnomaly.setText('ION-Sim Anomaly -> inactive')
        self.ui.comboAnomaly.setEnabled(False)
        # --- GUI ---
        self._setup_gui()
        self.setup_choice_buttons()

        self.response_start_time = None
        '''
        if self.json_anomaly_path is not None:
            print(f"json = {self.json_anomaly_path}")
            self.load_json_anomaly()
        else:
            pass
        '''

    def _setup_gui(self):
        """ Configura l'interfaccia utente con PyQtGraph e le curve multiple. """
        # Connessioni
        self.ui.pushBtExit.clicked.connect(self.closeWin)
        self.ui.pushBtStart.clicked.connect(self.startStop)
        self.ui.pushBtHelp.clicked.connect(self.helpCap)
        self.ui.pushBtAnomaly.clicked.connect(self.handle_anomaly)

        self.ui.checkWrist.stateChanged.connect(lambda state: self.handle_rec_site_toggle(state, 7.0))
        self.ui.checkElbow.stateChanged.connect(lambda state: self.handle_rec_site_toggle(state, 20.0))
        self.ui.checkArmpit.stateChanged.connect(lambda state: self.handle_rec_site_toggle(state, 40.0))
        self.ui.checkErb.stateChanged.connect(lambda state: self.handle_rec_site_toggle(state, 55.0))

        # Configurazione plot
        self.ui.plt_cap.setBackground(background='#000000')
        self.ui.plt_cap.setAntialiasing(True)
        self.capPLT = self.ui.plt_cap.addPlot()

        # capPLT plot
        self.capPLT.setLabel('left', r"Potenziale (mV)")
        self.capPLT.setLabel('bottom', "Tempo (ms)")
        # self.capPLT.setTitle(f"CAP Averaging: 0 / {self.N_AVERAGES} Campioni")
        self.capPLT.showGrid(x=True, y=True)
        self.capPLT.enableAutoRange(axis='y', enable=False)
        self.capPLT.setYRange(-1000, 1000)
        self.capPLT.setXRange(min(self.time) * 1000, max(self.time) * 1000)

        self.cap_curves = {}
        # 4 colori per 4 distanze
        colors = [(255, 255, 0), (255, 0, 0), (0, 0, 255), (0, 150, 0)]

        for i, dist_cm in enumerate(self.DISTANCES_CM):
            curve = self.capPLT.plot(self.time, np.zeros_like(self.time),
                                     pen=pg.mkPen(color=colors[i], width=1),
                                     name=f"CAP a {dist_cm:.1f} cm")
            self.cap_curves[dist_cm] = curve
        self.capPLT.addLegend()

        # connect change of ISI in milliseconds
        self.ui.spinBoxIsi.valueChanged.connect(self.handle_stimIsi)
        self.ui.spinBoxAvg.valueChanged.connect(self.handle_avg)
        self.ui.spinBoxOvl.valueChanged.connect(self.handle_fifo_block)
        self.ui.spinBoxFibers.valueChanged.connect(self.handle_num_fibers)

        self.anomaly_active = False
        self.ui.pushBtAnomaly.setText('ION-Sim Anomaly -> inactive')

        # Cursore 1 (Colore Rosso, Tratteggiato)
        self.cursor_1 = pg.InfiniteLine(
            movable=True,
            pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine),
            label='T1: {value:.2f} ms',
            labelOpts={'position': 0.8, 'color': 'r'}
        )
        self.cursor_1.setValue(self.MIN_DISPLAY_DELAY * 1000 + 1)  # Posizione iniziale in ms
        self.capPLT.addItem(self.cursor_1)

        # Cursore 2 (Colore Blu, Tratteggiato)
        self.cursor_2 = pg.InfiniteLine(
            movable=True,
            pen=pg.mkPen('b', width=2, style=Qt.PenStyle.DashLine),
            label='T2: {value:.2f} ms',
            labelOpts={'position': 0.2, 'color': 'b'}
        )
        self.cursor_2.setValue(self.MIN_DISPLAY_DELAY * 1000 + 5)  # Posizione iniziale in ms
        self.capPLT.addItem(self.cursor_2)

        self.ui.comboFilter.addItem('Raw signal')
        self.ui.comboFilter.addItem('Smooth')
        self.ui.comboFilter.addItem('Spline')
        self.ui.comboFilter.addItem('5-3000 Hz')
        self.ui.comboFilter.addItem('10-1500 Hz')
        self.ui.comboFilter.currentIndexChanged.connect(self.handle_filter)
        self.ui.comboFilter.setCurrentIndex(0)

        self.cursor_1.sigPositionChanged.connect(self.update_vc_measurement)
        self.cursor_2.sigPositionChanged.connect(self.update_vc_measurement)
        self.update_vc_measurement()

    def setup_choice_buttons(self):
        """Prepara i pulsanti delle alternative leggendoli dal file JSON."""
        # TODO numero di pulsanti max = 4 compresa la risposta corretta
        print(f"ANOMALY SET = {self.anomaly_config}")
        combo = self.ui.comboAnswers
        confirm_button = self.ui.pushBtConfirm
        if not self.anomaly_config or 'learning_assessment' not in self.anomaly_config:
            combo.hide()
            confirm_button.hide()
            return
        combo.show()
        confirm_button.show()
        combo.setEnabled(True)
        confirm_button.setEnabled(True)

        assessment_data = self.anomaly_config['learning_assessment']
        correct_answer = assessment_data.get('correct_answer')
        distractors = assessment_data.get('distractors', [])
        if not correct_answer:
            self.logger.error("Dati di apprendimento incompleti nel JSON: 'correct_answer' mancante.")
            combo.hide()
            confirm_button.hide()
            return
        # Mettiamo tutte le scelte in una lista e mescoliamole
        combo.clear()
        choices = [correct_answer] + distractors
        random.shuffle(choices)
        combo.addItem("...")
        combo.addItems(choices)
        try:
            confirm_button.clicked.disconnect()
        except Exception:
            pass  # Nessuna connessione da rimuovere
        confirm_button.clicked.connect(self._handle_choice_confirmation)

    def _handle_choice_confirmation(self):
        """
        Questa funzione viene chiamata quando l'utente clicca su "Conferma Risposta".
        Valuta la scelta fatta nella ComboBox.
        """
        combo = self.ui.comboAnswers
        confirm_button = self.ui.pushBtConfirm

        # Recupera il testo dell'opzione attualmente selezionata
        chosen_action = combo.currentText()

        # 1. Controlla che l'utente abbia selezionato una risposta valida (non il placeholder)
        if combo.currentIndex() == 0:  # L'indice 0 è "Seleziona la tua risposta..."
            self.logger.warning("Nessuna risposta selezionata. Per favore, scegli un'opzione.")
            # Qui potresti mostrare un messaggio all'utente
            return

        if not self.learning_manager or not self.anomaly_config:
            return

        # Calcolo del tempo di risposta dal momento in cui l'anomalia è stata attivata
        tempo_risposta = None
        if self.response_start_time:
            tempo_risposta = (datetime.now() - self.response_start_time).total_seconds()
            self.logger.info(f"Tempo di risposta calcolato: {tempo_risposta:.2f} secondi.")

        # 2. Disabilita i controlli per evitare risposte multiple
        combo.setEnabled(False)
        confirm_button.setEnabled(False)
        # 3. Recupera la risposta corretta dal JSON caricato
        correct_action = self.anomaly_config['learning_assessment']['correct_answer']

        # 4. Il resto della logica di confronto e registrazione è IDENTICO a prima!
        if chosen_action == correct_action:
            is_correct = True
            punti = 100
            self.ui.pushBtRetray.setEnabled(False)
            self.logger.info(f"Risposta CORRETTA! Scelta: '{chosen_action}'")
            self.ui.textAnswers.append(f"{chosen_action} : CORRECT = points 100")
        else:
            is_correct = False
            punti = -20
            self.ui.pushBtRetray.setEnabled(True)
            self.logger.warning(f"Risposta SBAGLIATA. Scelta: '{chosen_action}', Corretta: '{correct_action}'")
            self.ui.textAnswers.append(f"{chosen_action} : ERROR = points -20")
        #
        self.learning_manager.record_decision(
            azione_presa=chosen_action,
            esito_corretto=is_correct,
            punti=punti,
            tempo_risposta_sec = tempo_risposta
        )

    def retray_answer(self):
        self.setup_choice_buttons()

    def load_json_anomaly(self):
        # json file
        self.anomaly_config = None
        if self.json_anomaly_path:
            try:
                with open(self.json_anomaly_path, 'r') as f:
                    self.anomaly_config = json.load(f)
                # print(f"{self.anomaly_config}")
                self.logger.info(f"EEG Simulator: Caricata configurazione anomalia da {self.json_anomaly_path}.")
            except json.JSONDecodeError as e:
                self.logger.error(f"Errore nella decodifica JSON per l'anomalia: {e}")
                self.anomaly_config = None  # Reimposta a None se c'è un errore
        #
        metadata = self.anomaly_config.get('metadata_db_mapping', {})
        nerve_trunk = self.anomaly_config.get('nerve_trunk', {})
        timing = self.anomaly_config.get('timing_control', {})
        injection = self.anomaly_config.get('injection_parameters', {})
        '''
        print("\n--- Parametri Anomalia Caricati ---")
        print(f"tronco nervoso testato: {nerve_trunk}")
        print(f"Codice Anomalia (label_id): {metadata.get('db_label_id', 'N/A')}")
        print(f"Schema Temporale: {timing.get('schema_temporale', 'N/A')}")
        print(f"Modalità Temporale: {timing.get('time_model', 'N/A')}")

        print(f"Trigger Type (tipo): {metadata.get('db_trigger_type', 'N/A')}")
        print(f"Percentuale Apparizione: {injection.get('appearance_pct', 'N/A')}%")
        print(f"signal affected: {injection.get('target_signal_affected', 'N/A')}")
        '''
        self.scenario_config = self.anomaly_config.get('scenario_specific_config', {})
        print("\n--- Parametri Specifici dello Scenario ---")
        if self.scenario_config:
            # Iteriamo su tutte le coppie chiave-valore nel dizionario specifico
            for key, value in self.scenario_config.items():
                print(f"  > {key}: {value}")
                # self.ui.comboScenario.addItem(f"{key}: {value}")
        else:
            print("Nessun parametro specifico per lo scenario trovato.")

    '''
    def handle_anomaly(self):
        if not self.anomaly_active:
            # **ATTIVAZIONE** della modalità Anomalia
            self.ui.pushBtAnomaly.setText('ION-Sim Anomaly: ACTIVE')
            self.setFixedSize(820, 480)  # La dimensione maggiore
            # self.ui.comboMetaData.setEnabled(True)
            # self.ui.comboScenario.setEnabled(True)
            self.ui.comboAnomaly.setEnabled(False)
            self.ui.pushBtAnomaly.setStyleSheet("background-color: darkred;")
            self.anomaly_active = True
            self.response_start_time = datetime.now()
            self.logger.info(f"Tempo di risposta INIZIATO: {self.response_start_time}")
        else:
            # **DISATTIVAZIONE** della modalità Anomalia (se era già attiva)
            self.ui.pushBtAnomaly.setText('ION-Sim Anomaly -> inactive')
            # self.setGeometry(50, 415, 670, 480)
            self.setFixedSize(670, 480)
            self.ui.comboAnomaly.setEnabled(False)
            # self.ui.comboMetaData.setEnabled(True)
            # self.ui.comboScenario.setEnabled(True)
            self.ui.pushBtAnomaly.setStyleSheet("background-color: darkgreen;")
            self.anomaly_active = False
    '''

    def handle_anomaly(self):
        if not self.anomaly_active:
            # **ATTIVAZIONE** della modalità Anomalia
            self.ui.pushBtAnomaly.setText('ION-Sim Anomaly: ACTIVE')
            self.setFixedSize(820, 480)  # La dimensione maggiore
            # self.ui.comboMetaData.setEnabled(True)
            # self.ui.comboScenario.setEnabled(True)
            self.ui.comboAnomaly.setEnabled(False)
            self.ui.pushBtAnomaly.setStyleSheet("background-color: darkred;")
            self.anomaly_active = True
            self.response_start_time = datetime.now()
            self.logger.info(f"Tempo di risposta INIZIATO: {self.response_start_time}")

            # --- CODICE MANCANTE (DA AGGIUNGERE) ---
            # Comunica al worker (in modo thread-safe) di 'armare' l'iniettore
            self.logger.info("Invio comando ARM al worker...")
            QMetaObject.invokeMethod(self.worker, "arm_anomaly_trigger", Qt.ConnectionType.QueuedConnection)

        else:
            # **DISATTIVAZIONE** della modalità Anomalia (se era già attiva)
            self.ui.pushBtAnomaly.setText('ION-Sim Anomaly -> inactive')
            # self.setGeometry(50, 415, 670, 480)
            self.setFixedSize(670, 480)
            self.ui.comboAnomaly.setEnabled(False)
            # self.ui.comboMetaData.setEnabled(True)
            # self.ui.comboScenario.setEnabled(True)
            self.ui.pushBtAnomaly.setStyleSheet("background-color: darkgreen;")
            self.anomaly_active = False

            # --- CODICE MANCANTE (DA AGGIUNGERE) ---
            # Comunica al worker (in modo thread-safe) di 'resettare' l'iniettore
            self.logger.info("Invio comando RESET al worker...")
            QMetaObject.invokeMethod(self.worker, "disarm_anomaly_trigger", Qt.ConnectionType.QueuedConnection)

    def handle_filter(self, new_index):
        # esegue il filtro selezionato sulla media corrente dei 4 segnali
        # Invia l'indice del filtro al Worker nel suo thread
        QMetaObject.invokeMethod(self.worker, "update_filter_state",
                                        Qt.ConnectionType.QueuedConnection, Q_ARG(int, new_index))
        # Nota: Non è necessario resettare l'averaging qui, a meno che tu non voglia
        # che l'utente ricominci la raccolta dati ad ogni cambio filtro.
        # self.logger.info(f"Filtro selezionato (indice): {new_index}")

    def show_report_window(self):
        """ Mostra la finestra dei report/log. """
        # self.report_window.show()
        # self.report_window.raise_()
        pass

    def helpCap(self):
        # todo eventually passed argument as contextual file pdf to open
        # self.setGeometry(50, 415, 800, 480)
        pdf_path = 'help_docs/docs_simula/colonneDorsali_lemniscoM.pdf'
        wb.open_new(pdf_path)

    def update_vc_measurement(self):
        """
        Calcola la differenza di tempo (delta T) tra i due cursori e l'equivalente VC.
        """
        if self.cursor_1 is None or self.cursor_2 is None:
            return
        # Cattura le posizioni attuali dei cursori (in ms, dato che l'asse X è in ms)
        t1_ms = self.cursor_1.value()
        t2_ms = self.cursor_2.value()

        # Calcola la differenza di tempo in secondi
        delta_t_ms = abs(t2_ms - t1_ms)
        delta_t_s = delta_t_ms / 1000.0  # Conversione da ms a s

        # Calcola la Velocità di Conduzione (VC = Delta D / Delta T)
        vc_ms = 0.0
        if delta_t_s != 0:
            vc_ms = self.VC_REF_DIST_M / delta_t_s

        # Aggiorna una label nella GUI (Devi avere una label, ad esempio self.ui.label_vc)
        # Nota: Ho aggiunto qui solo il log, dovrai connettere una label nell'UI.

        # Stampa i risultati nella console/log (o aggiorna la tua label)
        vc_text = f"VC misurata: {vc_ms:.1f} m/s"  # (Delta T: {delta_t_ms:.2f} ms)"
        self.ui.labelLatency.setText(vc_text)
        # self.capPLT.setTitle(self.capPLT.titleLabel.text + f" | Misura: {vc_ms:.2f} m/s")

        # Log utile per il debug
        # self.logger.info(f"Misura VC: {vc_ms:.2f} m/s")

    def handle_rec_site_toggle(self, state, distance_cm):
        """
        Gestisce il cambiamento di stato di un CheckBox e aggiorna la visibilità della curva.
        Args:
            state (int): Stato del CheckBox (0=Unchecked, 2=Checked).
            distance_cm (float): La distanza di registrazione associata a questo CheckBox.
        """
        # Se lo stato è Checked (2), la curva deve essere visibile (True)
        is_visible = (state == Qt.CheckState.Checked.value)
        # 1. Aggiorna lo stato nel dizionario di controllo
        self.plot_visibility[distance_cm] = is_visible
        # 2. Aggiorna immediatamente la visibilità della curva di PyQtGraph
        # Usiamo setVisible(bool) sul PlotDataItem
        if distance_cm in self.cap_curves:
            self.cap_curves[distance_cm].setVisible(is_visible)
            # print(f"Sito {distance_cm} cm visibile: {is_visible}")
    '''
    def handle_recPtStim(self, state):
        # visualizzazione delle sedi di registrazione
        if state == Qt.CheckState.Checked.value:
            print('point stim visualize')
            self.stimSite = False
        else:
            print('stimpoint NOT visualize')
            self.stimSite = True

    def handle_recPtWrist(self, state):
        # visualizzazione delle sedi di registrazione
        if state == Qt.CheckState.Checked.value:
            print('point Wrist visualize')

        else:
            print('Wrist point NOT visualize')

    def handle_recPtElbow(self, state):
        # visualizzazione delle sedi di registrazione
        if state == Qt.CheckState.Checked.value:
            print('point Elbow visualize')
        else:
            print('Elnow point NOT visualize')

    def handle_recPtErb(self, state):
        # visualizzazione delle sedi di registrazione
        if state == Qt.CheckState.Checked.value:
            print('point Erb visualize')
        else:
            print('Erb point NOT visualize')
    '''

    def handle_avg(self, new_value):
        # numero di medie nell'averaging
        self.N_AVERAGES = new_value
        QMetaObject.invokeMethod(self.worker, "update_averaging_params",
                                        Qt.ConnectionType.QueuedConnection,
                                        Q_ARG(int, self.N_AVERAGES), Q_ARG(int, self.FIFO_BLOCK_SIZE))

    def handle_fifo_block(self, new_value):
        """ Cattura il nuovo valore del blocco FIFO e lo invia al Worker. """
        self.FIFO_BLOCK_SIZE = new_value
        # Invia ENTRAMBI i parametri al Worker nel suo thread
        QMetaObject.invokeMethod(self.worker, "update_averaging_params",
                                        Qt.ConnectionType.QueuedConnection,
                                        Q_ARG(int, self.N_AVERAGES), Q_ARG(int, self.FIFO_BLOCK_SIZE))

    def handle_num_fibers(self, new_value):
        self.N_FIBERS = new_value * 1000
        self.ui.label_fiber.setText(f"fibers = {self.N_FIBERS}")
        QMetaObject.invokeMethod(self.worker, "update_fiber_structure",
                                        Qt.ConnectionType.QueuedConnection, Q_ARG(int, self.N_FIBERS))
        if self.ui.pushBtStart.text() == 'Stop':
            self.startStop()
            self.startStop()

    def handle_stimIsi(self, new_value):
        # corrispondente ISI in milliseconds
        new_value = round((1 / new_value) * 1000)
        self.ui.label_isi.setText(f"ISI = {new_value} ms")
        # update ISI value in ms
        self.interStim = new_value
        # gestione del timer per nuovo valore di ISI
        if self.ui.pushBtStart.text() == 'Stop':
            self.startStop()
        else:
            pass
        self.startStop()

    def closeWin(self):
        """ chiusura del timer e del thread"""
        QMetaObject.invokeMethod(self.worker, "stop_simulation", Qt.ConnectionType.QueuedConnection)
        # 2. Scollega il Worker dal Thread.
        self.worker.setParent(None)
        self.worker.thread = None
        # 3. Segnala al thread di uscire e attendi la terminazione.
        self.thread.quit()
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.close()

    def startStop(self):
        """ Attiva / disattiva il timer del Worker tramite invocazione. """
        # self.capPLT.enableAutoRange(axis='y', enable=True)

        if self.ui.pushBtStart.text() == 'Start':
            self.ui.pushBtStart.setText('Stop')
            # Avvia il timer del Worker con 100ms di intervallo (eseguito nel thread separato)
            QMetaObject.invokeMethod(self.worker, "start_simulation",
                                            Qt.ConnectionType.QueuedConnection, Q_ARG(int, self.interStim))
        else:
            self.ui.pushBtStart.setText('Start')
            # Ferma il timer del Worker
            QMetaObject.invokeMethod(self.worker, "stop_simulation", Qt.ConnectionType.QueuedConnection)

    @Slot(object, int, float)
    def update_gui_plot(self, averaged_data, current_avg_count, noise_level):
        """
        Slot per ricevere i dati dal Worker e aggiornare la UI (eseguito sul thread principale).
        """
        # Itera sui dati mediati ricevuti e aggiorna i grafici
        timems = self.time * 1000
        for dist_cm, data in averaged_data.items():
            if dist_cm in self.cap_curves:
                # Aggiornamento grafico: veloce e sicuro sul thread principale
                self.cap_curves[dist_cm].setData(timems, data)
                is_visible = self.plot_visibility.get(dist_cm, True)
                self.cap_curves[dist_cm].setVisible(is_visible)

        # Aggiorna il titolo del plot
        self.capPLT.setTitle(
            rf"CAP Averaging: {current_avg_count} / {self.N_AVERAGES} samples | Max noise: {noise_level:.1f} uV")

