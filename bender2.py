from __future__ import print_function, unicode_literals
import sys
import os
import logging
from PyQt4 import QtGui, QtCore

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

from bender2_ui import Ui_Bender2Window

sineParameterDefs = [
    {'name': 'Amplitude', 'type': 'float', 'value': 15.0, 'step': 1.0, 'suffix': 'deg'},
    {'name': 'Frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
    {'name': 'Cycles', 'type': 'int', 'value': 10}
]

freqSweepParameterDefs = [
    {'name': 'Amplitude', 'type': 'float', 'value': 15.0, 'step': 1.0, 'suffix': 'deg'},
    {'name': 'Start Frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
    {'name': 'End Frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
    {'name': 'Duration', 'type': 'float', 'value': 300.0, 'suffix': 'sec'}
]

parameterDefinitions = [
    {'name': 'DAQ', 'type': 'group', 'children': [
        {'name': 'Input', 'type': 'group', 'children': [
            {'name': 'Input Frequency', 'type': 'float', 'value': 1000.0, 'step': 500.0, 'siPrefix': True,
             'suffix': 'Hz'},
            {'name': 'xForce', 'type': 'str', 'value': 'Dev1/ai0'},
            {'name': 'yForce', 'type': 'str', 'value': 'Dev1/ai1'},
            {'name': 'zForce', 'type': 'str', 'value': 'Dev1/ai2'},
            {'name': 'xTorque', 'type': 'str', 'value': 'Dev1/ai3'},
            {'name': 'yTorque', 'type': 'str', 'value': 'Dev1/ai4'},
            {'name': 'zTorque', 'type': 'str', 'value': 'Dev1/ai5'},
            {'name': 'Encoder', 'type': 'str', 'value': 'Dev1/ctr0'},
            {'name': 'Counts per revolution', 'type': 'int', 'value': 10000, 'limits': (1, 100000)}
        ]}
    ]},
    {'name': 'Stimulus', 'type': 'group', 'children': [
        {'name': 'Type', 'type': 'list', 'values': ['Sine', 'Frequency Sweep'], 'value': 'Sine'},
        {'name': 'Parameters', 'type': 'group', 'children': sineParameterDefs}
    ]},
    {'name': 'Save', 'type': 'action'},
    {'name': 'Load', 'type': 'action'}
]

class Bender2Window(QtGui.QWidget):
    def __init__(self):
        super(Bender2Window, self).__init__()

        self.initUI()
        self.params = Parameter.create(name='params', type='group', children=parameterDefinitions)
        self.ui.parameterTree.setParameters(self.params, showTop=False)

        for child in self.params.children():
            print(child)

        stimtype = self.params.child('Stimulus', 'Type')
        stimtype.sigValueChanging.connect(self.changeStimType)

    def initUI(self):
        ui = Ui_Bender2Window()
        ui.setupUi(self)
        self.ui = ui

    def changeStimType(self, param, value):
        print('{} = {}'.format(param, value))

def main():
    logging.basicConfig(level=logging.DEBUG)
    app = QtGui.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    # settings = QtCore.QSettings('bender.ini', QtCore.QSettings.IniFormat)

    bw2 = Bender2Window()
    bw2.show()

    return app.exec_()

if __name__ == '__main__':
    sys.exit(main())


