import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
import time

app = QtWidgets.QApplication([])

win = pg.plot(title="Real-time line plot")
win.showGrid(x=True, y=True)
curve = win.plot(pen='y')

data = np.zeros(1000)
ptr = 0

def update():
    global data, ptr
    data[:-1] = data[1:]
    data[-1] = np.sin(ptr * 0.1)
    curve.setData(data)
    ptr += 1

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

# Non-blocking update loop (like plt.pause)
for i in range(1000):
    print("Loop iteration:", i)
    time.sleep(0.01)
    QtWidgets.QApplication.processEvents()
