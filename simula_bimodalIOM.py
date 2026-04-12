import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QPushButton, QGridLayout, QFrame)
from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtGui import QFont, QCursor


class BimodalVisualizer(QWidget):
    """
    An application that visualizes a bimodal distribution of nerve conduction
    velocities using PyQtGraph, similar to a web concept.
    """

    def __init__(self):
        super().__init__()
        # Traduzione: 'Visualizzatore Distribuzione Bimodale - pyqtgraph'
        self.setWindowTitle('Bimodal Distribution Visualizer - R&B-Lab')
        self.setFixedSize(1000, 700)
        self.move(100, 100)
        self.setStyleSheet("background-color: #f8fafc;")

        # --- Data and Palettes ---
        self.x_data, self.y_data = self._generate_bimodal_data()
        self._setup_palettes()

        # --- User Interface ---
        self._init_ui()

        # --- Initial State ---
        self.update_palette('scientific_blue')
        self.update_velocity_indicator(40)

    def _generate_bimodal_data(self):
        """Generates data for the bimodal curve."""

        # ... (implementation remains the same) ...
        def gaussian(x, mu, sigma, a):
            return a * np.exp(-np.power(x - mu, 2) / (2 * np.power(sigma, 2)))

        x = np.arange(10, 70.5, 0.5)
        peak1 = gaussian(x, 28, 6, 0.7)
        peak2 = gaussian(x, 55, 4.5, 1)
        y = peak1 + peak2
        return x, y

    def _setup_palettes(self):
        """Defines the color palettes."""
        # ... (implementation remains the same) ...
        self.palettes = {
            'scientific_blue': {
                'line': (59, 130, 246),
                'fill': (96, 165, 250, 50),
                'point': (37, 99, 235),
                'point_highlight': (255, 255, 255),
                'value_text_style': 'color: #3b82f6;'
            },
            'bio_green': {
                'line': (16, 185, 129),
                'fill': (52, 211, 153, 50),
                'point': (5, 150, 105),
                'point_highlight': (255, 255, 255),
                'value_text_style': 'color: #10b981;'
            },
            'tech_gray': {
                'line': (75, 85, 99),
                'fill': (156, 163, 175, 50),
                'point': (31, 41, 55),
                'point_highlight': (255, 255, 255),
                'value_text_style': 'color: #4b5563;'
            }
        }

    def _init_ui(self):
        """Creates and organizes the user interface widgets."""
        # --- Main Layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- Title ---
        # Traduzione: 'Concept Logo Interattivo'
        title = QLabel('Interactive Bimodal Distribution')
        title.setFont(QFont('Inter', 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #1e293b;")

        # Traduzione: 'Distribuzione Bimodale della Velocità di Conduzione Nervosa'
        subtitle = QLabel('Distribution of Nerve Conduction Velocity')
        subtitle.setFont(QFont('Inter', 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #475569; margin-bottom: 20px;")

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        # --- Main Container (Plot + Controls) ---
        main_container = QFrame()
        main_container.setStyleSheet("background-color: white; border-radius: 12px;")
        main_container_layout = QHBoxLayout(main_container)
        main_container_layout.setContentsMargins(20, 20, 20, 20)

        # --- Plot ---
        self.plot_widget = pg.PlotWidget(background='light gray')
        self.plot_widget.showAxis('left', False)
        self.plot_widget.showAxis('bottom', True)
        self.plot_widget.setMenuEnabled(False)
        # Traduzione: Ticks sull'asse X (tempo)
        self.plot_widget.getAxis('bottom').setTicks([[(10, '10 m/s'), (40, '40 m/s'), (70, '70 m/s')]])
        self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color=(100, 116, 139)))

        self.curve = self.plot_widget.plot(self.x_data, self.y_data)
        self.scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen(width=2))
        self.plot_widget.addItem(self.scatter)

        main_container_layout.addWidget(self.plot_widget, 2)

        # --- Controls Panel ---
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(20)

        # Traduzione: 'Unione di Scienza e Design'
        desc_title = QLabel("Union of Science and Design")
        desc_title.setFont(QFont('Inter', 16, QFont.Weight.Bold))
        desc_title.setStyleSheet("color: #1e293b;")

        # Traduzione: Descrizione
        description = QLabel(
            "This concept merges the graphical representation of a biological "
            "phenomenon with its physical structure: the myelinated nerve fiber."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #475569;")

        # Traduzione: 'Esplora la Velocità di Conduzione:'
        slider_label = QLabel("Explore Conduction Velocity:")
        slider_label.setFont(QFont('Inter', 10, QFont.Weight.Medium))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(10, 70)
        self.slider.setValue(40)
        self.slider.valueChanged.connect(self.update_velocity_indicator)

        # Info Box
        info_box = QFrame()
        info_box.setStyleSheet("background-color: #f8fafc; border-radius: 8px; padding: 10px;")
        info_box_layout = QVBoxLayout(info_box)
        # Testo iniziale, aggiornato in update_velocity_indicator
        self.velocity_label = QLabel("Selected Velocity: 40 m/s")
        self.velocity_label.setFont(QFont('Inter', 10, QFont.Weight.Medium))
        self.fiber_type_label = QLabel("...")
        self.fiber_type_label.setWordWrap(True)
        info_box_layout.addWidget(self.velocity_label)
        info_box_layout.addWidget(self.fiber_type_label)

        controls_layout.addWidget(desc_title)
        controls_layout.addWidget(description)
        controls_layout.addStretch()
        controls_layout.addWidget(slider_label)
        controls_layout.addWidget(self.slider)
        controls_layout.addWidget(info_box)
        controls_layout.addStretch()

        main_container_layout.addLayout(controls_layout, 1)
        main_layout.addWidget(main_container)

        # --- Palette Section ---
        # Traduzione: 'Palette Colori e Stile'
        palette_label = QLabel("Color Palette and Style")
        palette_label.setFont(QFont('Inter', 14, QFont.Weight.Bold))
        palette_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        palette_label.setStyleSheet("color: #1e293b; margin-top: 20px;")

        palette_buttons_layout = QHBoxLayout()
        palette_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Traduzione: 'Scientifico', 'Biologico', 'Tecnologico'
        btn_blue = QPushButton("Scientific")
        btn_green = QPushButton("Biological")
        btn_gray = QPushButton("Technical")

        self.palette_buttons = {'scientific_blue': btn_blue, 'bio_green': btn_green, 'tech_gray': btn_gray}

        for name, btn in self.palette_buttons.items():
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    color: #334155;
                    border: 1px solid #cbd5e1;
                    padding: 8px 16px;
                    border-radius: 16px;
                    font-size: 14px;
                }
                QPushButton:hover { background-color: #f1f5f9; }
            """)
            btn.clicked.connect(lambda _, n=name: self.update_palette(n))
            palette_buttons_layout.addWidget(btn)

        main_layout.addWidget(palette_label)
        main_layout.addLayout(palette_buttons_layout)

    def update_velocity_indicator(self, value):
        """Updates the point on the graph and the text labels."""
        velocity = float(value)
        # Traduzione: 'Velocità Selezionata'
        self.velocity_label.setText(f"<b>Selected Velocity:</b> {velocity:.1f} m/s")

        # Traduzione: Descrizioni dei tipi di fibra
        if 10 <= velocity < 40:
            info = "Corresponds to the slower fiber population (peak ~28 m/s)."
        elif 40 <= velocity < 50:
            info = "Lies in the interval between the two fiber populations."
        else:
            info = "Corresponds to the faster fiber population (peak ~55 m/s)."

        self.fiber_type_label.setText(info)

        # Finds the closest point on the graph
        idx = (np.abs(self.x_data - velocity)).argmin()
        self.scatter.setData([self.x_data[idx]], [self.y_data[idx]])

    def update_palette(self, name):
        """Updates the plot colors based on the selected palette."""
        # ... (implementation remains the same) ...
        palette = self.palettes[name]
        self.current_palette_name = name

        # Update button styles
        for btn_name, btn in self.palette_buttons.items():
            if btn_name == name:
                btn.setStyleSheet(btn.styleSheet() + "border: 2px solid #3b82f6;")
            else:
                btn.setStyleSheet(btn.styleSheet().replace("border: 2px solid #3b82f6;", "border: 1px solid #cbd5e1;"))

        # Update plot colors
        pen = pg.mkPen(color=palette['line'], width=3)
        brush = pg.mkBrush(color=palette['fill'])
        self.curve.setPen(pen)
        self.curve.setBrush(brush)
        self.curve.setFillLevel(0)

        scatter_pen = pg.mkPen(color=palette['point_highlight'], width=2)
        scatter_brush = pg.mkBrush(color=palette['point'])
        self.scatter.setPen(scatter_pen)
        self.scatter.setBrush(scatter_brush)

        # Update label text color
        self.velocity_label.setStyleSheet(palette['value_text_style'] + "font-weight: bold;")
        self.update_velocity_indicator(self.slider.value())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont("Inter")
    app.setFont(font)

    main_win = BimodalVisualizer()
    main_win.show()
    sys.exit(QCoreApplication.exec())
