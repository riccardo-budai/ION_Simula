import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

# È buona pratica creare l'app all'inizio
app = pg.mkQApp()

x = np.linspace(-20, 20, 1000)
y = np.sin(x) / x
plot = pg.plot()
plot.setYRange(-1, 2)
plot.setWindowTitle('pyqtgraph example: text')
plot.setZValue(0)
curve = plot.plot(x, y)

## Testo statico (invariato)
text = pg.TextItem(
    html='<div style="text-align: center"><span style="color: #FFF;">This is the</span><br><span style="color: #FF0; font-size: 16pt;">PEAK</span></div>',
    anchor=(-0.3, 0.5), angle=45, border='w', fill=(0, 0, 255, 100))
text.setZValue(1000)
plot.addItem(text)
text.setPos(0, y.max())

## Freccia statica
arrow = pg.ArrowItem(pos=(0, y.max()), angle=-45)
plot.addItem(arrow)

## Elementi animati (testo e freccia)
curvePoint = pg.CurvePoint(curve)
plot.addItem(curvePoint)
text2 = pg.TextItem("test", anchor=(0.5, -1.0))
text2.setParentItem(curvePoint)
arrow2 = pg.ArrowItem(angle=45)
arrow2.setParentItem(curvePoint)

# Ho reso la linea più spessa e gialla
linecur = pg.InfiniteLine(angle=90,
                          movable=True,
                          pen=pg.mkPen('y', width=3),
                          hoverPen=pg.mkPen('r', width=3))
plot.addItem(linecur)

# Questa funzione è il nuovo "motore".
# Legge la posizione X della linea e aggiorna tutto il resto.
def aggiorna_in_base_alla_linea():
    # 1. Ottieni la posizione X attuale della linea
    posizione_x = linecur.value()

    # 2. Trova l'indice (index) più vicino nell'array 'x'
    #    Questo "snappa" la posizione ai tuoi dati
    index = np.argmin(np.abs(x - posizione_x))

    # 3. Aggiorna il testo
    text2.setText('[%0.1f, %0.1f]' % (x[index], y[index]))

    # 4. Aggiorna la posizione della freccia/testo (curvePoint)
    #    (Usando la posizione percentuale, come prima)
    percentuale_pos = float(index) / (len(x) - 1)
    curvePoint.setPos(percentuale_pos)

    # 5. (Opzionale) Riadatta la linea alla posizione "snappata"
    #    Questo evita che la linea si trovi "tra" due punti dati
    #    Disabilitiamo i segnali per evitare un loop infinito!
    linecur.blockSignals(True)
    linecur.setPos(x[index])
    linecur.blockSignals(False)


# 1. Collegare il Mouse:
#    Ogni volta che la linea viene mossa (trascinata),
#    chiama la nostra nuova funzione.
linecur.sigPositionChanged.connect(aggiorna_in_base_alla_linea)


# 2. Modificare la gestione Tasti:
#    Ora i tasti non chiamano più 'update', ma
#    spostano *direttamente* la linea.
def gestisci_tasti(event):
    # Questo step definisce di quanto si sposta la linea
    # con una singola pressione della freccia
    step = 0.1
    posizione_attuale = linecur.value()

    if event.key() == QtCore.Qt.Key.Key_Right:
        # Sposta la linea -> questo attiverà
        # 'sigPositionChanged' e aggiornerà tutto.
        linecur.setPos(posizione_attuale + step)

    elif event.key() == QtCore.Qt.Key.Key_Left:
        # Sposta la linea -> questo attiverà
        # 'sigPositionChanged' e aggiornerà tutto.
        linecur.setPos(posizione_attuale - step)

# Collega la funzione dei tasti (invariato)
plot.keyPressEvent = gestisci_tasti
plot.setFocus()

# Rimuoviamo tutto ciò che riguarda il timer
# index = 0  (non più necessario come globale)
# def update(): ... (sostituita)
# timer = QtCore.QTimer() ... (rimosso)
# timer.start(10) (rimosso)

# aggiorna la posizione iniziale
aggiorna_in_base_alla_linea()

if __name__ == '__main__':
    pg.exec()