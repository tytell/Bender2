from __future__ import print_function, unicode_literals
import sys
import os
import logging
import random
from PyQt4 import QtGui, QtCore
import numpy as np

from matplotlib.backends import qt_compat

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from setupDialog import SetupDialog
from settings import SETTINGS_FILE

assert qt_compat.QT_API == qt_compat.QT_API_PYQT

class BenderDAQ(object):
    def __init__(self, input, output, stim, geometry, outfilename):
        self.input = input
        self.output = output
        self.stim = stim
        self.geometry = geometry
        self.outfilename = outfilename

    def make_stimulus(self):
        if self.stim['type'] == 'sine':
            self.make_sine_stimulus()
        elif self.stim['type'] == 'frequencySweep':
            self.make_freqsweep_stimulus()
        else:
            assert False

    def make_sine_stimulus(self):
        dur = self.stim['waitPre'] + self.stim['cycles']/self.stim['frequency'] + \
            self.stim['waitPost']

        t = np.arange(0.0,dur, 1/self.output['frequency']) - self.stim['waitPre']
        
        pos = self.stim['amplitude'] * np.sin(2*np.pi*self.stim['frequency']*t)
    def setup_channels(self):
        pass

class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        # self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class BenderWindow(QtGui.QWidget):
    def __init__(self):
        super(BenderWindow, self).__init__()

        doneButton = QtGui.QPushButton("Done")
        doneButton.clicked.connect(self.close)

        self.forceAxes = MplCanvas(self)
        self.torqueAxes = MplCanvas(self)
        grid = QtGui.QGridLayout()
        grid.addWidget(self.forceAxes, 0,0)
        grid.addWidget(self.torqueAxes, 0,1)

        self.forceAxes.axes.plot([0, 1, 2, 3],[1, 2, 5, 1],'ro-')
        self.forceAxes.axes.set_ylabel('Force (N)')
        self.torqueAxes.axes.plot([0, 1, 2, 3],[8, 1, 2, 1],'gs-')
        self.torqueAxes.axes.set_ylabel('Torque (N m)')

        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(doneButton)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(grid)
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.readSettings()

    def readSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        settings.beginGroup("BenderWindow")
        self.resize(settings.value("size", QtCore.QSize(800, 600)).toSize())
        self.move(settings.value("position", QtCore.QSize(200, 200)).toPoint())
        settings.endGroup()

    def writeSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        logging.debug('Writing settings!')

        settings.beginGroup("BenderWindow")
        settings.setValue("size", self.size())
        settings.setValue("position", self.pos())
        settings.endGroup()

    def start(self):
        setup = SetupDialog()
        if not setup.exec_():
            return False

        self.show()
        return True

    def closeEvent(self, event):
        self.writeSettings()
        event.accept()


def main():
    logging.basicConfig(level=logging.DEBUG)
    app = QtGui.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    settings = QtCore.QSettings('bender.ini', QtCore.QSettings.IniFormat)

    bw = BenderWindow()
    if bw.start():
        return app.exec_()
    else:
        return False

if __name__ == '__main__':
    sys.exit(main())

