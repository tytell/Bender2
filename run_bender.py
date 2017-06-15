from __future__ import print_function, unicode_literals
import sys
import logging
from PyQt5 import QtGui, QtCore

from settings import EXPERIMENT_TYPE

if EXPERIMENT_TYPE == 'whole body':
    from bender_wholebody import BenderWindow_WholeBody
    BenderWindow = BenderWindow_WholeBody
elif EXPERIMENT_TYPE == 'isolated muscle':
    from bender_ergometer import BenderWindow_Ergometer
    BenderWindow = BenderWindow_Ergometer
else:
    raise ValueError('Unknown experiment type {}'.format(EXPERIMENT_TYPE))

def main():
    logging.basicConfig(level=logging.DEBUG)
    app = QtGui.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    bw2 = BenderWindow()
    bw2.show()

    return app.exec_()
