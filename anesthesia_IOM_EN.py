import sys
import math
import random
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout,
                             QGroupBox, QGridLayout, QSlider, QPushButton,
                             QFrame, QApplication, QMessageBox, QComboBox, QSpinBox)
import pyqtgraph as pg


class AnesthesiaSimulator(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ION-Simula: TCI Anesthesia (Prop/Remi/Ket/Sevo)")
        self.resize(950, 900)  # Finestra leggermente più larga
        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: white; }
            QGroupBox { border: 1px solid #555; font-weight: bold; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: #aaa; }
            QPushButton { background-color: #444; border: 1px solid #666; border-radius: 5px; padding: 5px; font-weight: bold; }
            QPushButton:hover { background-color: #555; }
            QPushButton:pressed { background-color: #333; }
        """)

        # --- Variabili simulazione ---
        self.ecg_data = np.zeros(400)
        self.rsa_data = np.zeros(100)
        self.ptr = 0
        self.resp_phase = 0.0
        self.last_filtered_rsa = 60.0

        # Parametri Base
        self.base_hr = 60
        self.current_hr = 60
        self.base_sys = 120
        self.base_dia = 80
        self.body_temp = 37.0  # Temperatura iniziale

        # Sensibilità paziente
        self.patient_sensitivity = 1.0
        self.current_drug_load = 0.0  # Carico depressivo netto

        # Buffer HRV
        self.rr_history = [1000.0 + random.uniform(-10, 10) for _ in range(30)]

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(40)  # ~25 FPS

    def init_ui(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # 1. Titolo
        title = QLabel("Anesthesia Monitor - Advanced TCI Mode")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ccff;")
        title.setAlignment(Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(title)

        # 2. Area Monitoraggio
        monitor_layout = QHBoxLayout()

        # --- A. Colonna Sinistra: Timer + ECG ---
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
        self.ecg_plot.setFixedHeight(300)
        self.ecg_curve = self.ecg_plot.plot(pen=pg.mkPen('#00ff00', width=2))

        ecg_container.addWidget(self.ecg_plot)
        monitor_layout.addLayout(ecg_container, stretch=5)

        # --- B. Colonna Destra: Vitali ---
        vitals_layout = QVBoxLayout()

        # RIGA 1: HR e NIBP
        row1_layout = QHBoxLayout()
        self.lbl_hr = self.create_digital_display("HR (bpm)", "60", "#00ff00")
        row1_layout.addWidget(self.lbl_hr)
        self.lbl_bp = self.create_digital_display("NIBP (mmHg)", "120/80", "#ff5500")
        row1_layout.addWidget(self.lbl_bp)
        vitals_layout.addLayout(row1_layout)

        # RIGA 2: SpO2, MAP, Temp (Nuovi!)
        row2_layout = QHBoxLayout()

        self.lbl_spo2 = self.create_digital_display("SpO2 (%)", "98", "#00ffff")
        row2_layout.addWidget(self.lbl_spo2)

        # MAP (Mean Arterial Pressure) - Bianco/Grigio chiaro
        self.lbl_map = self.create_digital_display("MAP (mmHg)", "93", "#dddddd")
        row2_layout.addWidget(self.lbl_map)

        # Temp - Giallo/Arancio
        self.lbl_temp = self.create_digital_display("Temp (°C)", "37.0", "#ffcc00")
        row2_layout.addWidget(self.lbl_temp)

        vitals_layout.addLayout(row2_layout)

        # RIGA 3: Grafico Respiro
        self.rsa_plot = pg.PlotWidget(title="Resp. Cycle (RSA)")
        self.rsa_plot.showGrid(x=False, y=True, alpha=0.2)
        self.rsa_plot.getPlotItem().hideAxis('bottom')
        self.rsa_plot.getPlotItem().hideAxis('left')
        self.rsa_plot.setBackground('#1a1a1a')
        self.rsa_plot.setFixedHeight(100)
        self.rsa_curve = self.rsa_plot.plot(pen=pg.mkPen('#ffff00', width=2))
        vitals_layout.addWidget(self.rsa_plot)

        # RIGA 4: Valore Resp
        self.lbl_resp = self.create_digital_display("Resp (rpm)", "12", "#ffff00")
        vitals_layout.addWidget(self.lbl_resp)

        monitor_layout.addLayout(vitals_layout, stretch=4)
        main_layout.addLayout(monitor_layout)

        # 3. Analisi HRV
        hrv_group = QGroupBox("Hemodynamics & HRV")
        hrv_layout = QHBoxLayout()
        self.lbl_rr = self.create_digital_display("R-R (ms)", "1000", "#ffffff")
        self.lbl_sdnn = self.create_digital_display("SDNN", "0", "#aaaaff")
        self.lbl_rmssd = self.create_digital_display("RMSSD", "0", "#aaaaff")
        hrv_layout.addWidget(self.lbl_rr)
        hrv_layout.addWidget(self.lbl_sdnn)
        hrv_layout.addWidget(self.lbl_rmssd)
        hrv_group.setLayout(hrv_layout)
        main_layout.addWidget(hrv_group)

        # --- DATI PAZIENTE ---
        patient_group = QGroupBox("Patient Data (TCI Demographics)")
        pat_layout = QHBoxLayout()

        pat_layout.addWidget(QLabel("Age:"))
        self.spin_age = QSpinBox()
        self.spin_age.setRange(1, 100)
        self.spin_age.setValue(45)
        self.spin_age.valueChanged.connect(self.update_patient_metrics)
        pat_layout.addWidget(self.spin_age)

        pat_layout.addSpacing(10)
        pat_layout.addWidget(QLabel("Sex:"))
        self.combo_sex = QComboBox()
        self.combo_sex.addItems(["Male", "Female"])
        self.combo_sex.currentIndexChanged.connect(self.update_patient_metrics)
        pat_layout.addWidget(self.combo_sex)

        pat_layout.addSpacing(10)
        pat_layout.addWidget(QLabel("Ht:"))
        self.spin_height = QSpinBox()
        self.spin_height.setRange(50, 220)
        self.spin_height.setValue(175)
        self.spin_height.valueChanged.connect(self.update_patient_metrics)
        pat_layout.addWidget(self.spin_height)

        pat_layout.addSpacing(10)
        pat_layout.addWidget(QLabel("Wt:"))
        self.spin_weight = QSpinBox()
        self.spin_weight.setRange(30, 150)
        self.spin_weight.setValue(75)
        self.spin_weight.valueChanged.connect(self.update_patient_metrics)
        pat_layout.addWidget(self.spin_weight)

        pat_layout.addSpacing(20)
        info_layout = QVBoxLayout()
        self.lbl_bmi_display = QLabel("BMI: --")
        self.lbl_bmi_display.setStyleSheet("color: #ff00ff; font-weight: bold;")
        self.lbl_lbm_display = QLabel("LBM: -- kg")
        self.lbl_lbm_display.setStyleSheet("color: #00ccff; font-weight: bold;")

        info_layout.addWidget(self.lbl_bmi_display)
        info_layout.addWidget(self.lbl_lbm_display)
        pat_layout.addLayout(info_layout)
        pat_layout.addStretch()
        patient_group.setLayout(pat_layout)
        main_layout.addWidget(patient_group)

        # --- CONTROLLI TCI ---
        controls_group = QGroupBox("TCI Infusion Control (Effect Site)")
        controls_layout = QGridLayout()

        # 1. Sevoflurane
        lbl_sevo = QLabel("Sevoflurane (%)")
        self.slider_sevo = QSlider(Qt.Orientation.Horizontal)
        self.slider_sevo.setRange(0, 80)
        self.slider_sevo.valueChanged.connect(self.update_drug_effects)
        self.val_sevo = QLabel("0.0 %")
        self.val_sevo.setStyleSheet("color: #ff9900; font-weight: bold; font-size: 14px;")
        controls_layout.addWidget(lbl_sevo, 0, 0)
        controls_layout.addWidget(self.slider_sevo, 0, 1)
        controls_layout.addWidget(self.val_sevo, 0, 2)

        # 2. Propofol
        lbl_prop = QLabel("Propofol TCI (µg/ml)")
        self.slider_prop = QSlider(Qt.Orientation.Horizontal)
        self.slider_prop.setRange(0, 80)
        self.slider_prop.valueChanged.connect(self.update_drug_effects)
        self.val_prop = QLabel("0.0 µg/ml")
        self.val_prop.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 14px;")
        controls_layout.addWidget(lbl_prop, 1, 0)
        controls_layout.addWidget(self.slider_prop, 1, 1)
        controls_layout.addWidget(self.val_prop, 1, 2)

        # 3. Remifentanil
        lbl_remi = QLabel("Remifentanil TCI (ng/ml)")
        self.slider_remi = QSlider(Qt.Orientation.Horizontal)
        self.slider_remi.setRange(0, 150)
        self.slider_remi.valueChanged.connect(self.update_drug_effects)
        self.val_remi = QLabel("0.0 ng/ml")
        self.val_remi.setStyleSheet("color: #ff00ff; font-weight: bold; font-size: 14px;")
        controls_layout.addWidget(lbl_remi, 2, 0)
        controls_layout.addWidget(self.slider_remi, 2, 1)
        controls_layout.addWidget(self.val_remi, 2, 2)

        # 4. Ketamina (Nuovo!)
        lbl_ket = QLabel("Ketamine TCI (µg/ml)")
        self.slider_ket = QSlider(Qt.Orientation.Horizontal)
        self.slider_ket.setRange(0, 50)  # 0.0 - 5.0 ug/ml
        self.slider_ket.valueChanged.connect(self.update_drug_effects)
        self.val_ket = QLabel("0.0 µg/ml")
        self.val_ket.setStyleSheet("color: #ff55ff; font-weight: bold; font-size: 14px;")
        controls_layout.addWidget(lbl_ket, 3, 0)
        controls_layout.addWidget(self.slider_ket, 3, 1)
        controls_layout.addWidget(self.val_ket, 3, 2)

        # Buttons
        self.btn_auto_tci = QPushButton("AUTO-SET: Schnider/Minto Targets")
        self.btn_auto_tci.setFixedHeight(40)
        self.btn_auto_tci.setStyleSheet("background-color: #006699; color: white; font-size: 13px; font-weight: bold;")
        self.btn_auto_tci.clicked.connect(self.apply_auto_tci_targets)
        controls_layout.addWidget(self.btn_auto_tci, 4, 0, 1, 3)

        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)

        # Footer
        btn_layout = QHBoxLayout()
        self.btn_help = QPushButton("Help")
        self.btn_help.clicked.connect(self.show_help)
        self.style_button(self.btn_help)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.clicked.connect(self.close)
        self.style_button(self.btn_exit, color="#cc0000")

        btn_layout.addWidget(self.btn_help)
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
        lbl_v.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
        lbl_v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        l.addWidget(lbl_t)
        l.addWidget(lbl_v)
        frame.setLayout(l)
        return frame

    def style_button(self, btn, color="#555555"):
        btn.setFixedSize(100, 30)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {color}; color: white; border-radius: 4px; font-weight: bold; }}")

    def get_lbm(self):
        weight = self.spin_weight.value()
        height = self.spin_height.value()
        sex = self.combo_sex.currentText()
        if height == 0: return 0

        if sex == "Male":
            lbm = (1.1 * weight) - 128 * ((weight / height) ** 2)
        else:
            lbm = (1.07 * weight) - 148 * ((weight / height) ** 2)

        if lbm > weight: lbm = weight
        if lbm < 30: lbm = 30
        return lbm

    def update_patient_metrics(self):
        weight = self.spin_weight.value()
        height = self.spin_height.value() / 100.0
        age = self.spin_age.value()

        bmi = weight / (height * height)
        self.lbl_bmi_display.setText(f"BMI: {bmi:.1f}")

        lbm = self.get_lbm()
        self.lbl_lbm_display.setText(f"LBM: {int(lbm)} kg")

        if age > 65:
            self.base_hr = 60 + random.randint(-2, 2)
            self.base_sys = 135
            self.base_dia = 80
        elif age < 12:
            self.base_hr = 95 + random.randint(-5, 5)
            self.base_sys = 95
            self.base_dia = 55
        else:
            self.base_hr = 70
            self.base_sys = 120
            self.base_dia = 75

        self.update_drug_effects()

    def apply_auto_tci_targets(self):
        age = self.spin_age.value()
        lbm = self.get_lbm()
        weight = self.spin_weight.value()

        # Propofol
        base_prop = 4.0
        if age > 30:
            reduction_p = (age - 30) * 0.005 * base_prop
            base_prop -= reduction_p
        if lbm < (weight * 0.75):
            base_prop *= 0.9

        # Remifentanil
        base_remi = 5.0
        if age > 30:
            reduction_r = (age - 30) * 0.015 * base_remi
            base_remi -= reduction_r

        if base_prop < 1.5: base_prop = 1.5
        if base_remi < 1.0: base_remi = 1.0

        self.slider_prop.setValue(int(base_prop * 10))
        self.slider_remi.setValue(int(base_remi * 10))
        self.slider_sevo.setValue(0)
        self.slider_ket.setValue(0)  # Reset Ketamine on auto-set

        QMessageBox.information(self, "Auto-Set Applied",
                                f"Schnider/Minto Adjusted Targets:\n\n"
                                f"Propofol Ce: {base_prop:.1f} µg/ml\n"
                                f"Remifentanil Ce: {base_remi:.1f} ng/ml\n\n"
                                f"Adjusted for Age ({age}) and LBM ({int(lbm)}kg).")

    def update_drug_effects(self):
        sevo = self.slider_sevo.value() / 10.0
        prop = self.slider_prop.value() / 10.0
        remi = self.slider_remi.value() / 10.0
        ket = self.slider_ket.value() / 10.0  # Ketamine

        self.val_sevo.setText(f"{sevo:.1f} %")
        self.val_prop.setText(f"{prop:.1f} µg/ml")
        self.val_remi.setText(f"{remi:.1f} ng/ml")
        self.val_ket.setText(f"{ket:.1f} µg/ml")

        age = self.spin_age.value()
        sensitivity = 1.0 + max(0, (age - 40) * 0.015)

        # Calcolo carico depressivo (Prop/Remi/Sevo)
        depressive_load = (sevo * 2.5) + (prop * 3.0) + (remi * 2.0)

        # Effetto KETAMINA (Simpaticomimetico)
        # La ketamina aumenta HR e BP, contrastando i depressivi.
        # Fattore stimolante arbitrario per la simulazione
        stimulant_load = (ket * 2.5)

        # Carico netto (Se positivo = depressione, Se negativo = stimolazione)
        net_load = (depressive_load - stimulant_load) * sensitivity
        self.current_drug_load = net_load

        # Remifentanil ha effetto bradicardizzante specifico extra
        remi_effect = (remi * 3.0) * sensitivity

        # Target HR Calculation
        target_hr = self.base_hr - (self.current_drug_load * 0.5) - remi_effect

        if target_hr < 30: target_hr = 30
        if target_hr > 180: target_hr = 180

        self.current_hr = target_hr

    def update_simulation(self):
        self.ptr += 1

        # Timer logic
        total_seconds = int(self.ptr * 0.040)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        self.lbl_timer.setText(f"Anesthesia Time: {hours:02}:{minutes:02}:{seconds:02}")

        # Temp Simulation (Leggera ipotermia progressiva)
        # Scende lentamente ogni ciclo, si stabilizza se bassa
        if self.ptr % 100 == 0:  # Ogni 4 secondi circa
            if self.body_temp > 35.5:
                # Scende più veloce se c'è molto vasodilatatore (Prop/Sevo)
                cooling_factor = 0.005 + (max(0, self.current_drug_load) * 0.001)
                self.body_temp -= cooling_factor

            # Aggiorna label Temp
            self.lbl_temp.layout().itemAt(1).widget().setText(f"{self.body_temp:.1f}")

        # ECG Drawing
        cycle_len = 50
        step = self.ptr % cycle_len
        signal = random.gauss(0, 0.05)
        if 3 <= step <= 9:
            signal += 0.3 * math.sin(math.pi * (step - 3) / 6.0)
        elif step == 14:
            signal -= 0.5
        elif step == 15:
            signal += 1.5
        elif step == 16:
            signal += 5.0
        elif step == 17:
            signal += 1.5
        elif step == 18:
            signal -= 1.0
        elif 24 <= step <= 38:
            signal += 0.6 * math.sin(math.pi * (step - 24) / 14.0)

        self.ecg_data[:-1] = self.ecg_data[1:]
        self.ecg_data[-1] = signal
        self.ecg_curve.setData(self.ecg_data)

        # Vital Updates
        if self.ptr % 10 == 0:
            hr_noise = random.uniform(-1, 1)
            real_hr = self.current_hr + hr_noise

            # BP Logic (Net load affects BP)
            # Ketamine raises BP, Propofol lowers it.
            # current_drug_load is already (Depressive - Stimulant)
            bp_drop = self.current_drug_load * 2.0

            curr_sys = self.base_sys - bp_drop + random.uniform(-2, 2)
            curr_dia = self.base_dia - (bp_drop * 0.7) + random.uniform(-1, 1)

            if curr_sys < 40: curr_sys = 40
            if curr_dia < 20: curr_dia = 20

            # Calcolo MAP
            val_map = curr_dia + (curr_sys - curr_dia) / 3.0

            self.lbl_hr.layout().itemAt(1).widget().setText(str(int(real_hr)))
            self.lbl_bp.layout().itemAt(1).widget().setText(f"{int(curr_sys)}/{int(curr_dia)}")
            self.lbl_map.layout().itemAt(1).widget().setText(f"{int(val_map)}")

            # RSA
            base_rr = 60000 / max(30, real_hr)
            self.resp_phase += 0.3
            remi_val = self.slider_remi.value() / 10.0
            rsa_amp = 40 - (self.current_drug_load * 0.8) - (remi_val * 15)
            if rsa_amp < 1: rsa_amp = 1
            rsa_val = math.sin(self.resp_phase) * rsa_amp
            rr_raw = base_rr + rsa_val + random.gauss(0, 5)

            self.rr_history.append(rr_raw)
            self.rr_history.pop(0)

            val_sdnn = np.std(self.rr_history)
            diffs = np.diff(self.rr_history)
            val_rmssd = np.sqrt(np.mean(diffs ** 2))

            self.lbl_rr.layout().itemAt(1).widget().setText(f"{int(rr_raw)}")
            self.lbl_sdnn.layout().itemAt(1).widget().setText(f"{val_sdnn:.1f}")
            self.lbl_rmssd.layout().itemAt(1).widget().setText(f"{val_rmssd:.1f}")

            inst_bpm = 60000 / rr_raw
            self.last_filtered_rsa = (self.last_filtered_rsa * 0.85) + (inst_bpm * 0.15)
            self.rsa_data[:-1] = self.rsa_data[1:]
            self.rsa_data[-1] = self.last_filtered_rsa
            self.rsa_curve.setData(self.rsa_data)

    def show_help(self):
        msg = """
        <b>TCI Simulator (Schnider, Minto & Ketamine)</b><br><br>

        <b>Ketamine Effect:</b><br>
        Ketamine is sympathomimetic. Increasing Ketamine will raise HR and BP,
        counteracting the depressive effects of Propofol/Sevo.<br><br>

        <b>New Metrics:</b><br>
        - <b>MAP:</b> Mean Arterial Pressure (approx Dia + 1/3 Pulse Pressure).<br>
        - <b>Temp:</b> Simulates progressive hypothermia during anesthesia.
        """
        QMessageBox.information(self, "TCI Sim Help", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pg.setConfigOption('background', '#1e1e1e')
    window = AnesthesiaSimulator()
    window.show()
    sys.exit(app.exec())