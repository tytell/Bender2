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


class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class BenderWindow(QtGui.QWidget):
    def __init__(self):
        super(BenderWindow, self).__init__()

        doneButton = QtGui.QPushButton("Done")
        doneButton.clicked.connect(self.close)

        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(doneButton)

        vbox = QtGui.QVBoxLayout()
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

        logging.debug('setup!')
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

