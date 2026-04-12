import sys
import os
import datetime
import numpy as np
os.environ["QT_API"] = "pyside6"
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import Qt, Slot, Signal

# 1. Define the Class inheriting from PlotWidget
class SignalTrackerWidget(pg.PlotWidget):
    """
    A custom PlotWidget with a cursor (InfiniteLine)
    that can be moved with the mouse or arrow keys.

    A CurvePoint (arrow + text) follows the cursor along the curve,
    displaying the x,y values.
    """

    marker_added = Signal(dict)

    # 2. The Constructor (accepts your parameters)
    def __init__(self, x_data, y_data, label_x="X-Axis", label_y="Y-Axis",
                 pen='c',
                 parent=None):

        super().__init__(parent=parent)

        self.x_data = x_data
        self.y_data = y_data

        # --- Basic Plot Settings ---
        self.plotItem = self.getPlotItem()
        self.plotItem.setLabel('bottom', label_x)
        self.plotItem.setLabel('left', label_y)
        self.plotItem.showGrid(x=True, y=True, alpha=0.1)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        # --- Creation of Graphic Objects ---

        # 1. The main curve ('pen' color)
        self.curve = self.plotItem.plot(self.x_data, self.y_data, pen=pen)

        # 2. Movable elements
        self.curvePoint = pg.CurvePoint(self.curve)
        self.plotItem.addItem(self.curvePoint)

        self.text_label = pg.TextItem(anchor=(0.5, -1.0))
        self.text_label.setParentItem(self.curvePoint)

        self.arrow = pg.ArrowItem(angle=0)
        self.arrow.setParentItem(self.curvePoint)

        # 3. The vertical line/cursor
        self.linecur = pg.InfiniteLine(angle=90,
                                       movable=True,
                                       pen=pg.mkPen('y', width=3),
                                       hoverPen=pg.mkPen('r', width=3))
        self.plotItem.addItem(self.linecur)

        # --- Connections (Signals & Slots) ---
        self.linecur.sigPositionChanged.connect(self._update_from_line_pos)

        # --- Initial Setup ---
        if len(self.x_data) > 0:
            self.linecur.setPos(self.x_data[0])

    # 3. Class Methods

    def _update_from_line_pos(self):
        """
        This method is the "engine".
        It is called every time 'linecur' moves.
        It reads the line's pos. and updates the CurvePoint.
        """

        # 1. Get the current X position of the line
        current_x_pos = self.linecur.value()

        if len(self.x_data) == 0:
            return

        # 2. Find the nearest index in the 'x' array
        #    (We use 'self.x_data' and 'self.y_data')
        index = np.argmin(np.abs(self.x_data - current_x_pos))

        # 3. Update the text
        self.text_label.setText('[%0.1f   %0.1f]' % (self.x_data[index], self.y_data[index]))

        # 4. Update the arrow/text position (CurvePoint)
        pos_percent = float(index) / (len(self.x_data) - 1)
        self.curvePoint.setPos(pos_percent)

        # 5. "Snap" the line to the exact X data position
        #    We block signals to avoid an infinite loop
        self.linecur.blockSignals(True)
        self.linecur.setPos(self.x_data[index])
        self.linecur.blockSignals(False)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if len(self.x_data) == 0 or (self.x_data[-1] == self.x_data[0]):
            return

        # Calculate a "smart" step (e.g., 1% of total duration)
        step = (self.x_data[-1] - self.x_data[0]) / 100.0
        current_pos = self.linecur.value()

        if event.key() == Qt.Key.Key_Right:
            self.linecur.setPos(current_pos + step)

        elif event.key() == Qt.Key.Key_Left:
            self.linecur.setPos(current_pos - step)

        elif event.key() == Qt.Key.Key_A:
            current_x_pos = current_pos
            # 2. Find the "snapped" index and X/Y values
            index = np.argmin(np.abs(self.x_data - current_x_pos))
            snapped_x = self.x_data[index]
            snapped_y = self.y_data[index]

            # 3. Get the timestamp
            ts = datetime.datetime.now().isoformat()
            # 4. Create the dictionary
            marker_data = {
                "timestamp": ts,
                "x_value": snapped_x,
                "y_value": snapped_y
            }
            # 5. Emit the signal with the dictionary!
            self.marker_added.emit(marker_data)
            # (Optional visual feedback in the console)
            # print(f"Marker added: {marker_data}")

    def set_amplitude_scale(self, amplitude_uV):
        """
        Manually sets the Y-Range of the plot, centered
        around zero, with a defined amplitude.

        Args:
            amplitude_uV (float): The value to use for the range
                                  (e.g., 500 -> Y-Range from -500 to +500)
        """
        # Disable auto-range to keep the scale fixed
        self.plotItem.enableAutoRange(axis='y', enable=False)

        self.plotItem.invertY(True)

        # Set the Y-Range
        self.plotItem.setYRange(-amplitude_uV, amplitude_uV)

    def update_data(self, x_data, y_data):
        """
        Public method to update the curve data from outside.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.curve.setData(self.x_data, self.y_data)

        # When data changes, we also update the text
        # to reflect the current cursor position
        self._update_from_line_pos()

    @Slot(float)
    def set_auto_v_line(self, x_pos):
        """
        Sposta programmaticamente la linea verticale (marker).
        Usato per impostare l'onset automatico dal simulatore principale.
        """

        # [CORREZIONE] Il nome corretto della variabile è self.linecur

        # Se l'utente sta trascinando attivamente la linea,
        # non aggiornarla automaticamente (l'utente ha priorità).
        if self.linecur.moving:
            return

        # Spostando 'self.linecur', il segnale 'sigPositionChanged'
        # viene emesso automaticamente.
        # Questo segnale è già connesso a '_update_from_line_pos',
        # che aggiornerà il testo e il 'curvePoint' per noi.
        # Non serve altro!
        self.linecur.setPos(x_pos)

# This block runs ONLY if you execute this .py file directly
if __name__ == '__main__':
    # 1. Create the application
    # (We use the standard QApplication for a more robust example)
    app = QApplication(sys.argv)

    # 2. Prepare sample data
    x = np.linspace(-20, 20, 2000)  # More points for smoother movement
    y = np.sin(x) / x

    # 3. Create a main window
    # (In your project, this could be your existing GUI)
    main_window = QMainWindow()
    main_window.setWindowTitle("SignalTrackerWidget Class Example")

    # 4. Create an INSTANCE of your new class
    tracker_plot = SignalTrackerWidget(
        x_data=x,
        y_data=y,
        label_x="Time (s)",
        label_y="Amplitude (V)"
    )

    # 5. Insert the widget into the main window
    main_window.setCentralWidget(tracker_plot)

    # 6. Show the window and start the app
    main_window.setGeometry(100, 100, 800, 600)
    main_window.show()

    sys.exit(app.exec())
