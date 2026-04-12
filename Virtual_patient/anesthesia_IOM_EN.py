import sys
import math
import random
import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout,
                               QGroupBox, QGridLayout, QSlider, QPushButton,
                               QFrame, QApplication, QMessageBox, QComboBox, QSpinBox)
import pyqtgraph as pg


class AnesthesiaSimulator(QWidget):
    simulation_finished = Signal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("ION-Simula: Full TCI (Schnider/Minto) & Ventilator Workstation")
        self.resize(1000, 700)
        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: white; }
            QGroupBox { border: 1px solid #555; font-weight: bold; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: #aaa; }
            QPushButton { background-color: #006699; border: 1px solid #005577; border-radius: 5px; padding: 5px; font-weight: bold; color: white; }
            QPushButton:hover { background-color: #0077aa; }
            QPushButton:pressed { background-color: #004466; }
        """)

        # --- Variabili simulazione ---
        self.ecg_data = np.zeros(400)
        self.resp_data = np.zeros(200)
        self.ptr = 0
        self.resp_phase = 0.0
        self.ecg_phase = 0.0

        # Buffer per HRV
        self.rr_history = [1000.0 + random.uniform(-10, 10) for _ in range(50)]

        # Parametri Base
        self.base_hr = 60
        self.current_hr = 60
        self.base_sys = 120
        self.base_dia = 80
        self.body_temp = 36.8

        # Parametri Ventilatore e Ossigenazione
        self.ventilator_rate = 12  # Impostazione macchina
        self.current_spo2 = 99.0  # Saturazione iniziale

        # Sensibilità paziente e Carico Farmacologico
        self.patient_sensitivity = 1.0
        self.current_drug_load = 0.0

        self.init_ui()

        # Timer a 20ms (50 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(20)

    def init_ui(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # 1. Titolo
        title = QLabel("Anesthesia Workstation - Advanced TCI & Ventilation")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ccff;")
        title.setAlignment(Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(title)

        # 2. Area Monitoraggio
        monitor_layout = QHBoxLayout()

        # --- A. Colonna Sinistra: Timer + ECG + Resp ---
        ecg_container = QVBoxLayout()

        # Timer Anestesia
        self.lbl_timer = QLabel("Anesthesia Time: 00:00:00")
        self.lbl_timer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_timer.setStyleSheet("""
            font-family: Consolas, monospace; 
            font-size: 32px; 
            color: #00ffff;
            background-color: #000000;
            border: 4px solid #005555;
            border-radius: 8px;
            padding: 5px;
            margin-bottom: 5px;
        """)
        self.lbl_timer.setFixedHeight(70)
        ecg_container.addWidget(self.lbl_timer)

        # Grafico ECG
        self.ecg_plot = pg.PlotWidget(title="ECG Lead II")
        self.ecg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ecg_plot.setYRange(-2, 6)
        self.ecg_plot.getPlotItem().hideAxis('bottom')
        self.ecg_plot.setFixedHeight(220)
        self.ecg_curve = self.ecg_plot.plot(pen=pg.mkPen('#00ff00', width=2))
        ecg_container.addWidget(self.ecg_plot)

        # Grafico Ventilazione (Capnografia simulata)
        self.resp_plot = pg.PlotWidget(title="Ventilation / Airway Pressure")
        self.resp_plot.showGrid(x=False, y=True, alpha=0.2)
        self.resp_plot.getPlotItem().hideAxis('bottom')
        self.resp_plot.getPlotItem().hideAxis('left')
        self.resp_plot.setBackground('#1a1a1a')
        self.resp_plot.setFixedHeight(120)
        self.resp_curve = self.resp_plot.plot(pen=pg.mkPen('#ffff00', width=2))
        ecg_container.addWidget(self.resp_plot)

        monitor_layout.addLayout(ecg_container, stretch=6)

        # --- B. Colonna Destra: Vitali ---
        vitals_layout = QVBoxLayout()

        # RIGA 1: HR e NIBP
        row1_layout = QHBoxLayout()
        self.lbl_hr = self.create_digital_display("HR (bpm)", "60", "#00ff00")
        row1_layout.addWidget(self.lbl_hr)
        self.lbl_bp = self.create_digital_display("NIBP (mmHg)", "120/80", "#ff5500")
        row1_layout.addWidget(self.lbl_bp)
        vitals_layout.addLayout(row1_layout)

        # RIGA 2: SpO2, MAP, Temp
        row2_layout = QHBoxLayout()
        self.lbl_spo2 = self.create_digital_display("SpO2 (%)", "99", "#00ffff")
        row2_layout.addWidget(self.lbl_spo2)
        self.lbl_map = self.create_digital_display("MAP (mmHg)", "93", "#dddddd")
        row2_layout.addWidget(self.lbl_map)
        self.lbl_temp = self.create_digital_display("Temp (°C)", "36.8", "#ffcc00")
        row2_layout.addWidget(self.lbl_temp)
        vitals_layout.addLayout(row2_layout)

        # RIGA 3: Dati Ventilatore
        row3_layout = QHBoxLayout()
        self.lbl_resp_rate = self.create_digital_display("RR (resp.rate)", "12", "#ffff00")
        row3_layout.addWidget(self.lbl_resp_rate)
        self.lbl_etco2 = self.create_digital_display("EtCO2 (mmHg)", "35", "#ffff00")
        row3_layout.addWidget(self.lbl_etco2)
        vitals_layout.addLayout(row3_layout)

        # RIGA 4: HRV (Reintegrato R-R Cardiaco)
        hrv_group = QGroupBox("HRV Analysis")
        hrv_layout = QHBoxLayout()
        # Nuova label per l'intervallo R-R cardiaco
        self.lbl_rr = self.create_digital_display("R-R (ms)", "1000", "#ffffff")
        self.lbl_sdnn = self.create_digital_display("SDNN", "0", "#aaaaff")
        self.lbl_rmssd = self.create_digital_display("RMSSD", "0", "#aaaaff")

        hrv_layout.addWidget(self.lbl_rr)
        hrv_layout.addWidget(self.lbl_sdnn)
        hrv_layout.addWidget(self.lbl_rmssd)
        hrv_group.setLayout(hrv_layout)
        vitals_layout.addWidget(hrv_group)

        monitor_layout.addLayout(vitals_layout, stretch=4)
        main_layout.addLayout(monitor_layout)

        # --- SEZIONE CONTROLLI ---
        controls_main_layout = QHBoxLayout()

        # 1. Dati Paziente
        patient_group = QGroupBox("Patient Data (TCI Params)")
        pat_layout = QVBoxLayout()

        # Age & Sex
        row_demo = QHBoxLayout()
        row_demo.addWidget(QLabel("Age:"))
        self.spin_age = QSpinBox()
        self.spin_age.setRange(1, 100)
        self.spin_age.setValue(45)
        self.spin_age.valueChanged.connect(self.update_patient_metrics)
        row_demo.addWidget(self.spin_age)

        row_demo.addWidget(QLabel("Sex:"))
        self.combo_sex = QComboBox()
        self.combo_sex.addItems(["Male", "Female"])
        self.combo_sex.currentIndexChanged.connect(self.update_patient_metrics)
        row_demo.addWidget(self.combo_sex)
        pat_layout.addLayout(row_demo)

        # Height & Weight
        row_body = QHBoxLayout()
        row_body.addWidget(QLabel("Ht(cm):"))
        self.spin_height = QSpinBox()
        self.spin_height.setRange(50, 220)
        self.spin_height.setValue(175)
        self.spin_height.valueChanged.connect(self.update_patient_metrics)
        row_body.addWidget(self.spin_height)

        row_body.addWidget(QLabel("Wt(kg):"))
        self.spin_weight = QSpinBox()
        self.spin_weight.setRange(30, 150)
        self.spin_weight.setValue(75)
        self.spin_weight.valueChanged.connect(self.update_patient_metrics)
        row_body.addWidget(self.spin_weight)
        pat_layout.addLayout(row_body)

        # Computed Metrics (BMI, LBM)
        self.lbl_bmi_display = QLabel("BMI: --")
        self.lbl_bmi_display.setStyleSheet("color: #ff00ff; font-weight: bold;")
        pat_layout.addWidget(self.lbl_bmi_display)

        self.lbl_lbm_display = QLabel("LBM: -- kg")
        self.lbl_lbm_display.setStyleSheet("color: #00ccff; font-weight: bold;")
        pat_layout.addWidget(self.lbl_lbm_display)

        patient_group.setLayout(pat_layout)
        controls_main_layout.addWidget(patient_group, stretch=2)

        # 2. Ventilator Settings
        vent_group = QGroupBox("Mechanical Ventilator")
        vent_group.setStyleSheet("QGroupBox { border: 2px solid #ffff00; }")
        vent_layout = QGridLayout()

        vent_layout.addWidget(QLabel("Respiratory Rate (RR):"), 0, 0)
        self.spin_vent_rate = QSpinBox()
        self.spin_vent_rate.setRange(0, 40)
        self.spin_vent_rate.setValue(12)
        # self.spin_vent_rate.setSuffix(" RR")
        self.spin_vent_rate.valueChanged.connect(self.update_ventilator_settings)
        vent_layout.addWidget(self.spin_vent_rate, 0, 1)

        vent_layout.addWidget(QLabel("Tidal Volume (ml):"), 1, 0)
        self.spin_vent_vol = QSpinBox()
        self.spin_vent_vol.setRange(200, 800)
        self.spin_vent_vol.setValue(450)
        self.spin_vent_vol.setSingleStep(50)
        vent_layout.addWidget(self.spin_vent_vol, 1, 1)

        vent_group.setLayout(vent_layout)
        controls_main_layout.addWidget(vent_group, stretch=2)

        # 3. TCI Controls
        drugs_group = QGroupBox("TCI Effect Site Targets")
        drugs_layout = QGridLayout()

        # Sevoflurane
        drugs_layout.addWidget(QLabel("Sevoflurane (%):"), 0, 0)
        self.slider_sevo = QSlider(Qt.Orientation.Horizontal)
        self.slider_sevo.setRange(0, 80)
        self.slider_sevo.valueChanged.connect(self.update_drug_effects)
        drugs_layout.addWidget(self.slider_sevo, 0, 1)
        self.val_sevo = QLabel("0.0 %")
        drugs_layout.addWidget(self.val_sevo, 0, 2)

        # Propofol
        drugs_layout.addWidget(QLabel("Propofol (µg/ml):"), 1, 0)
        self.slider_prop = QSlider(Qt.Orientation.Horizontal)
        self.slider_prop.setRange(0, 80)
        self.slider_prop.valueChanged.connect(self.update_drug_effects)
        drugs_layout.addWidget(self.slider_prop, 1, 1)
        self.val_prop = QLabel("0.0")
        drugs_layout.addWidget(self.val_prop, 1, 2)

        # Remifentanil
        drugs_layout.addWidget(QLabel("Remifentanil (ng/ml):"), 2, 0)
        self.slider_remi = QSlider(Qt.Orientation.Horizontal)
        self.slider_remi.setRange(0, 150)
        self.slider_remi.valueChanged.connect(self.update_drug_effects)
        drugs_layout.addWidget(self.slider_remi, 2, 1)
        self.val_remi = QLabel("0.0")
        drugs_layout.addWidget(self.val_remi, 2, 2)

        # Ketamina
        drugs_layout.addWidget(QLabel("Ketamine (µg/ml):"), 3, 0)
        self.slider_ket = QSlider(Qt.Orientation.Horizontal)
        self.slider_ket.setRange(0, 50)
        self.slider_ket.valueChanged.connect(self.update_drug_effects)
        drugs_layout.addWidget(self.slider_ket, 3, 1)
        self.val_ket = QLabel("0.0")
        drugs_layout.addWidget(self.val_ket, 3, 2)

        # Button Auto-Set TCI
        self.btn_auto_tci = QPushButton("AUTO-SET: Schnider/Minto Targets")
        self.btn_auto_tci.setFixedHeight(30)
        self.btn_auto_tci.clicked.connect(self.apply_auto_tci_targets)
        drugs_layout.addWidget(self.btn_auto_tci, 4, 0, 1, 3)

        drugs_group.setLayout(drugs_layout)
        controls_main_layout.addWidget(drugs_group, stretch=3)

        main_layout.addLayout(controls_main_layout)

        # Footer
        btn_layout = QHBoxLayout()
        self.btn_exit = QPushButton("Exit Simulation")
        self.btn_exit.setFixedSize(150, 40)
        self.btn_exit.setStyleSheet("background-color: #cc0000; color: white; font-weight: bold;")
        self.btn_exit.clicked.connect(self.close)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_exit)
        main_layout.addLayout(btn_layout)

        self.update_patient_metrics()

    def create_digital_display(self, title, start_val, color):
        frame = QFrame()
        frame.setStyleSheet("background-color: #151515; border: 1px solid #333; border-radius: 4px; margin: 1px;")
        frame.setFixedHeight(50)
        l = QHBoxLayout()
        l.setContentsMargins(5, 0, 5, 0)
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("font-size: 11px; color: #aaa;")
        lbl_v = QLabel(start_val)
        lbl_v.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {color};")
        lbl_v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        l.addWidget(lbl_t)
        l.addWidget(lbl_v)
        frame.setLayout(l)
        return frame

    def get_lbm(self):
        """Calcolo Massa Magra (Lean Body Mass) - Formula James"""
        weight = self.spin_weight.value()
        height = self.spin_height.value()
        sex = self.combo_sex.currentText()
        if height == 0: return 0

        # Conversione in cm per formula ma la logica base usa kg e cm
        if sex == "Male":
            lbm = (1.1 * weight) - 128 * ((weight / height) ** 2)
        else:
            lbm = (1.07 * weight) - 148 * ((weight / height) ** 2)

        if lbm > weight: lbm = weight
        if lbm < 30: lbm = 30  # Limite fisiologico
        return lbm

    def update_patient_metrics(self):
        # Update BMI / LBM Displays
        weight = self.spin_weight.value()
        height = self.spin_height.value() / 100.0  # metri
        age = self.spin_age.value()

        if height > 0:
            bmi = weight / (height * height)
            self.lbl_bmi_display.setText(f"BMI: {bmi:.1f}")

        lbm = self.get_lbm()
        self.lbl_lbm_display.setText(f"LBM: {int(lbm)} kg")

        # Base vitals based on age
        if age > 65:
            self.base_hr = 60 + random.randint(-2, 2)
            self.base_sys = 135
            self.base_dia = 80
        else:
            self.base_hr = 70
            self.base_sys = 120
            self.base_dia = 75

        self.update_drug_effects()

    def update_ventilator_settings(self):
        self.ventilator_rate = self.spin_vent_rate.value()

        self.lbl_resp_rate.layout().itemAt(1).widget().setText(str(self.ventilator_rate))
        if self.ventilator_rate == 0:
            self.lbl_resp_rate.layout().itemAt(1).widget().setStyleSheet(
                "font-size: 20px; font-weight: bold; color: #ff0000;")
            self.lbl_resp_rate.layout().itemAt(1).widget().setText("OFF")
        else:
            self.lbl_resp_rate.layout().itemAt(1).widget().setStyleSheet(
                "font-size: 20px; font-weight: bold; color: #ffff00;")

    def apply_auto_tci_targets(self):
        """Calcola target ottimali in base a Schnider/Minto"""
        age = self.spin_age.value()
        lbm = self.get_lbm()
        weight = self.spin_weight.value()

        # Propofol (Schnider base logic approx)
        base_prop = 4.0
        if age > 30:
            reduction_p = (age - 30) * 0.005 * base_prop
            base_prop -= reduction_p
        # Aggiustamento per obesità (LBM vs Weight)
        if lbm < (weight * 0.75):
            base_prop *= 0.9

        # Remifentanil (Minto base logic approx)
        base_remi = 5.0
        if age > 30:
            reduction_r = (age - 30) * 0.015 * base_remi
            base_remi -= reduction_r

        if base_prop < 1.5: base_prop = 1.5
        if base_remi < 1.0: base_remi = 1.0

        # Reset Gas e Ketamina
        self.slider_sevo.setValue(0)
        self.slider_ket.setValue(0)

        # Set TCI Targets
        self.slider_prop.setValue(int(base_prop * 10))
        self.slider_remi.setValue(int(base_remi * 10))

        QMessageBox.information(self, "Auto-Set TCI",
                                f"Schnider/Minto Targets Applied:\n\n"
                                f"Propofol Ce: {base_prop:.1f} µg/ml\n"
                                f"Remifentanil Ce: {base_remi:.1f} ng/ml\n\n"
                                f"Adjusted for Age ({age}) and LBM ({int(lbm)}kg).")

    def update_drug_effects(self):
        sevo = self.slider_sevo.value() / 10.0
        prop = self.slider_prop.value() / 10.0
        remi = self.slider_remi.value() / 10.0
        ket = self.slider_ket.value() / 10.0

        self.val_sevo.setText(f"{sevo:.1f} %")
        self.val_prop.setText(f"{prop:.1f}")
        self.val_remi.setText(f"{remi:.1f}")
        self.val_ket.setText(f"{ket:.1f}")

        # Calcolo carico depressivo combinato
        # Sevo ha un forte impatto ipnotico/vasodilatatorio
        depressive = (sevo * 2.0) + (prop * 3.0) + (remi * 2.5)
        stimulant = (ket * 3.0)

        self.current_drug_load = depressive - stimulant

        # Effetto su HR e BP
        target_hr = self.base_hr - (self.current_drug_load * 0.5)
        if target_hr < 35: target_hr = 35
        if target_hr > 180: target_hr = 180
        self.current_hr = target_hr

    def update_simulation(self):
        self.ptr += 1

        # Timer
        total_seconds = int(self.ptr * 0.020)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        self.lbl_timer.setText(f"Anesthesia Time: {hours:02}:{minutes:02}:{seconds:02}")

        # --- SPO2 Logic ---
        if self.ventilator_rate < 8:
            drop_rate = 0.05
            if self.ventilator_rate == 0: drop_rate = 0.1
            self.current_spo2 -= drop_rate
        else:
            if self.current_spo2 < 99.0:
                self.current_spo2 += 0.2
        if self.current_spo2 < 50: self.current_spo2 = 50
        if self.current_spo2 > 100: self.current_spo2 = 100

        # --- ECG Drawing ---
        safe_hr = max(30, self.current_hr)
        ticks_per_beat = (60.0 / safe_hr) / 0.020

        self.ecg_phase += 1
        if self.ecg_phase >= ticks_per_beat:
            self.ecg_phase = 0

        local_step = int(self.ecg_phase)
        ecg_sig = random.gauss(0, 0.05)

        if 4 <= local_step <= 8:
            ecg_sig += 0.2 * math.sin(math.pi * (local_step - 4) / 4.0)
        elif local_step == 12:
            ecg_sig -= 0.5
        elif local_step == 13:
            ecg_sig += 4.0
        elif local_step == 14:
            ecg_sig -= 1.0
        elif 20 <= local_step <= 32:
            ecg_sig += 0.5 * math.sin(math.pi * (local_step - 20) / 12.0)

        self.ecg_data[:-1] = self.ecg_data[1:]
        self.ecg_data[-1] = ecg_sig
        self.ecg_curve.setData(self.ecg_data)

        # --- Ventilation Drawing ---
        resp_sig = 0.0
        if self.ventilator_rate > 0:
            vent_freq_hz = self.ventilator_rate / 60.0
            phase_increment = 2 * math.pi * vent_freq_hz * 0.020
            self.resp_phase += phase_increment

            cycle_pos = (self.resp_phase / (2 * math.pi)) % 1.0
            if cycle_pos < 0.33:
                resp_sig = math.sin(cycle_pos * 3 * math.pi) * 10
            else:
                decay_pos = (cycle_pos - 0.33) * 1.5
                resp_sig = 10 * math.exp(-decay_pos * 5)
        else:
            resp_sig = random.gauss(0, 0.05)

        self.resp_data[:-1] = self.resp_data[1:]
        self.resp_data[-1] = resp_sig
        self.resp_curve.setData(self.resp_data)

        # --- Aggiornamento Dati Numerici e HRV (ogni 200ms) ---
        if self.ptr % 10 == 0:
            # Calcolo HRV basato su R-R
            base_rr = 60000 / safe_hr

            rsa_oscillation = 0
            if self.ventilator_rate > 0:
                # L'RSA è sincronizzata con la fase del ventilatore
                rsa_amp = 30.0 / (1.0 + self.current_drug_load * 0.5)  # I farmaci riducono HRV
                rsa_oscillation = math.sin(self.resp_phase) * rsa_amp

            current_rr = base_rr + rsa_oscillation + random.uniform(-5, 5)  # Rumore naturale

            self.rr_history.append(current_rr)
            self.rr_history.pop(0)

            # Calcolo metriche
            sdnn = np.std(self.rr_history)
            diffs = np.diff(self.rr_history)
            rmssd = np.sqrt(np.mean(diffs ** 2))

            # Update Labels HRV
            self.lbl_rr.layout().itemAt(1).widget().setText(f"{int(current_rr)}")
            self.lbl_sdnn.layout().itemAt(1).widget().setText(f"{sdnn:.1f}")
            self.lbl_rmssd.layout().itemAt(1).widget().setText(f"{rmssd:.1f}")

            # Vitals
            hr_noise = random.uniform(-1, 1)
            real_hr = self.current_hr + hr_noise

            bp_drop = self.current_drug_load * 2.0
            curr_sys = self.base_sys - bp_drop + random.uniform(-2, 2)
            curr_dia = self.base_dia - (bp_drop * 0.7) + random.uniform(-1, 1)
            val_map = curr_dia + (curr_sys - curr_dia) / 3.0

            self.lbl_hr.layout().itemAt(1).widget().setText(str(int(real_hr)))
            self.lbl_bp.layout().itemAt(1).widget().setText(f"{int(curr_sys)}/{int(curr_dia)}")
            self.lbl_map.layout().itemAt(1).widget().setText(f"{int(val_map)}")

            spo2_disp = int(self.current_spo2)
            self.lbl_spo2.layout().itemAt(1).widget().setText(f"{spo2_disp}")
            if spo2_disp < 90:
                self.lbl_spo2.layout().itemAt(1).widget().setStyleSheet(
                    "font-size: 20px; font-weight: bold; color: #ff0000;")
            else:
                self.lbl_spo2.layout().itemAt(1).widget().setStyleSheet(
                    "font-size: 20px; font-weight: bold; color: #00ffff;")

    def closeEvent(self, event):
        self.simulation_finished.emit()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pg.setConfigOption('background', '#1e1e1e')
    window = AnesthesiaSimulator()
    window.show()
    sys.exit(app.exec())
