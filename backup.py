import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout
import matplotlib
from matplotlib.figure import Figure 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
matplotlib.use("QT5Agg")
import os
import numpy as np
from PyQt5.QtCore import QThread 
from time import sleep
Form = uic.loadUiType(os.path.join(os.getcwd(),'gui.ui'))[0]
from PyQt5 import QtCore




class IntroWindow (Form, QMainWindow):
    def __init__(self):
        Form.__init__(self)
        QMainWindow.__init__(self)
        self.setupUi(self)
        
        self.fig = Figure()
        
        self.ax = self.fig.add_axes([0.1, 0.1, .8 , .8], frameon = False)
        self.canvas = FigureCanvas(self.fig)
        self.x = np.linspace(0,2*np.pi, 1000)
        self.line1, = self.ax.plot(self.x, np.sin(self.x), 'r')

        l = QVBoxLayout(self.widget)
        l.addWidget(self.canvas)
        
        self.plotThread = None
        
        self.startButton.clicked.connect(self.start)
        
        
    def start(self):
        num = np.random.random()
        self.plotThread = PlotThread(num)
        self.plotThread.update_plot.connect(self.update)
        self.plotThread.start()
    
    def update (self , x , y ):
        self.line1.set_data(x,y)
        self.fig.canvas.draw()
        
class PlotThread (QThread):
    update_plot = QtCore.pyqtSignal(np.ndarray,np.ndarray)
    def __init__(self, n):
        QThread.__init__(self)
        self.n = n
    def run (self):
        x = np.linspace(0,np.pi*2,1000)
        for i in range (10):
            y = np.sin((i+1)*x)*np.exp(-self.n*x)
            self.update_plot.emit(x,y)
            sleep(0.2)
       

app = QApplication(sys.argv)
w = IntroWindow()
w.show()
sys.exit(app.exec())

