import sys
import random

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QFrame, QApplication, QMessageBox
import pyqtgraph as pg



class HeadstageSimulator(QWidget):
    def __init__(self):
        super().__init__()

        # Configurazione della finestra principale
        self.setWindowTitle("ION-Simula: Headstage Unit")
        self.resize(640, 480)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        # Dizionario per tenere traccia dei nostri "LED"
        self.leds = {}

        # Setup dell'interfaccia
        self.init_ui()

    def init_ui(self):
        # Layout principale verticale
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # --- Titolo ---
        title_label = QLabel("Simulazione Testina 32 Canali + GND/REF")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title_label)

        # --- Area Canali (Griglia) ---
        # Usiamo LayoutWidget di pyqtgraph per gestire la griglia
        grid_container = pg.LayoutWidget()
        main_layout.addWidget(grid_container)

        # Creazione LED per i 32 canali (4 righe x 8 colonne)
        rows, cols = 4, 8
        for i in range(32):
            ch_num = i + 1
            # Creiamo il LED e l'etichetta
            led_widget = self.create_led_widget(f"CH {ch_num}")
            self.leds[f"CH{ch_num}"] = led_widget['indicator']

            # Calcolo posizione griglia
            r = i // cols
            c = i % cols
            grid_container.layout.addWidget(led_widget['frame'], r, c)

        # --- Area GND e REF ---
        # Li mettiamo in un layout orizzontale separato sotto la griglia
        ref_layout = QHBoxLayout()
        ref_layout.addStretch()

        # Ground
        gnd_widget = self.create_led_widget("GND")
        self.leds["GND"] = gnd_widget['indicator']
        ref_layout.addWidget(gnd_widget['frame'])

        # Spaziatore
        ref_layout.addSpacing(20)

        # Reference
        ref_widget = self.create_led_widget("REF")
        self.leds["REF"] = ref_widget['indicator']
        ref_layout.addWidget(ref_widget['frame'])

        ref_layout.addStretch()
        main_layout.addLayout(ref_layout)

        # --- Area Bottoni ---
        btn_layout = QHBoxLayout()

        # Bottone Help
        self.btn_help = QPushButton("Help")
        self.btn_help.clicked.connect(self.show_help)
        self.style_button(self.btn_help)
        btn_layout.addWidget(self.btn_help)

        # Bottone Test Impedenza
        self.btn_test = QPushButton("Test Impedenza")
        self.btn_test.clicked.connect(self.run_impedance_test)
        self.style_button(self.btn_test, color="#007acc")  # Blu
        btn_layout.addWidget(self.btn_test)

        # Bottone Exit
        self.btn_exit = QPushButton("Exit")
        self.btn_exit.clicked.connect(self.close)
        self.style_button(self.btn_exit, color="#cc0000")  # Rosso
        btn_layout.addWidget(self.btn_exit)

        main_layout.addLayout(btn_layout)

    def create_led_widget(self, label_text):
        """
        Crea un piccolo widget contenente un 'LED' (QLabel rotonda) e un testo.
        """
        frame = QFrame()
        layout = QVBoxLayout()
        frame.setLayout(layout)

        # Il LED è una label con bordi arrotondati
        led = QLabel()
        led.setFixedSize(20, 20)
        # Stile di base (Grigio = spento)
        self.set_led_color(led, "grey")

        lbl = QLabel(label_text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("font-size: 10px; color: #aaaaaa;")

        layout.addWidget(led, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl, 0, Qt.AlignmentFlag.AlignCenter)

        return {'frame': frame, 'indicator': led}

    def set_led_color(self, led_widget, color):
        """ Cambia il colore del LED simulato usando i fogli di stile CSS """
        color_map = {
            "grey": "#444444",
            "green": "#00ff00",  # Buona impedenza
            "red": "#ff0000"  # Cattiva impedenza
        }
        hex_color = color_map.get(color, "#444444")

        # Border-radius 10px rende un quadrato 20x20 un cerchio perfetto
        led_widget.setStyleSheet(f"""
            background-color: {hex_color}; 
            border-radius: 10px; 
            border: 1px solid #666;
        """)

    def style_button(self, btn, color="#555555"):
        btn.setFixedSize(120, 40)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: white;
                color: black;
            }}
        """)

    def run_impedance_test(self):
        """ Simula la misura di impedenza cambiando i colori dei LED """
        print("Avvio test impedenza...")

        # Simula GND e REF (solitamente devono essere stabili)
        self.set_led_color(self.leds["GND"], "green")
        self.set_led_color(self.leds["REF"], "green")

        # Simula i 32 canali
        for i in range(1, 33):
            key = f"CH{i}"
            # 80% probabilità che sia buono (verde), 20% che sia cattivo (rosso)
            status = "green" if random.random() > 0.2 else "red"
            self.set_led_color(self.leds[key], status)

    def show_help(self):
        QMessageBox.information(
            self,
            "Help - ION-Simula",
            "Simulatore Headstage Elettrodi.\n\n"
            "- Clicca 'Test Impedenza' per simulare una verifica dei contatti.\n"
            "- Verde: Impedenza Ottimale (< 50 kOhm)\n"
            "- Rosso: Impedenza Alta / Canale Aperto\n"
            "- Grigio: Canale Inattivo"
        )


# Blocco principale per l'esecuzione
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Abilita il tema scuro di default per PyQtGraph/Qt
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')

    window = HeadstageSimulator()
    window.show()

    sys.exit(app.exec())