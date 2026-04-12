import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

x = np.linspace(-20, 20, 1000)
y = np.sin(x) / x
plot = pg.plot()  ## create an empty plot widget
plot.setYRange(-1, 2)
plot.setWindowTitle('pyqtgraph example: text')
plot.setZValue(0)
curve = plot.plot(x, y)  ## add a single curve

## Create text object, use HTML tags to specify color/size
text = pg.TextItem(
    html='<div style="text-align: center"><span style="color: #FFF;">This is the</span><br><span style="color: #FF0; font-size: 16pt;">PEAK</span></div>',
    anchor=(-0.3, 0.5), angle=45, border='w', fill=(0, 0, 255, 100))
text.setZValue(1000)
plot.addItem(text)
text.setPos(0, y.max())

## Draw an arrowhead next to the text box
arrow = pg.ArrowItem(pos=(0, y.max()), angle=-45)
plot.addItem(arrow)

## Set up an animated arrow and text that track the curve
curvePoint = pg.CurvePoint(curve)
plot.addItem(curvePoint)
text2 = pg.TextItem("test", anchor=(0.5, -1.0))
text2.setParentItem(curvePoint)
arrow2 = pg.ArrowItem(angle=45)
# arrow2.setParentItem(curvePoint)

linecur = pg.InfiniteLine(angle=90, movable=True)
plot.addItem(linecur)

## update position every 10ms
index = 0


def update():
    global curvePoint, index
    index = (index + 1) % len(x)
    nuova_posizione_x = x[index]
    curvePoint.setPos(float(index) / (len(x) - 1))
    linecur.setPos(nuova_posizione_x)
    text2.setText('[%0.1f, %0.1f]' % (x[index], y[index]))


timer = QtCore.QTimer()
timer.timeout.connect(update)
# timer.start(10)

update()

###############################

### ------------------------------------ ###
### NUOVO: GESTIONE PRESSIONE TASTI      ###
### ------------------------------------ ###

# 1. Definiamo la funzione che gestirà l'evento
def gestisci_tasti(event):
    # 'event' contiene le informazioni su quale tasto è stato premuto

    # Definiamo di quanto si sposta la linea a ogni pressione
    # Sentiti libero di cambiare questo valore!
    step = 0.1

    # Prendiamo la posizione X attuale della linea
    posizione_attuale = linecur.value()

    # 2. Controlliamo quale tasto è stato premuto
    if event.key() == QtCore.Qt.Key.Key_Right:
        # Se è la freccia destra, sposta la linea a destra
        # linecur.setPos(posizione_attuale + step)
        update()

    elif event.key() == QtCore.Qt.Key.Key_Left:
        # Se è la freccia sinistra, sposta la linea a sinistra
        # linecur.setPos(posizione_attuale - step)
        update()

    # Nota: se premiamo altri tasti, questa funzione
    # semplicemente non fa nulla.


# 3. Colleghiamo la nostra funzione al widget del grafico
plot.keyPressEvent = gestisci_tasti

# 4. FONDAMENTALE: Diamo il "focus" al grafico.
# Solo il widget che ha il "focus" (cioè quello attivo)
# può "sentire" la pressione dei tasti.
# Potrebbe essere necessario cliccare prima sul grafico
# per fargli ottenere il focus.
plot.setFocus()


if __name__ == '__main__':
    pg.exec()