from __future__ import print_function, unicode_literals
import sys
import os
import string
import logging
from PyQt4 import QtGui, QtCore
import xml.etree.ElementTree as ET
import pickle
import numpy as np
from scipy import integrate, interpolate

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import pyqtgraph as pg

from bender_ui import Ui_BenderWindow

from benderdaq import BenderDAQ

try:
    import PyDAQmx as daq
except ImportError:
    pass

from settings import SETTINGS_FILE

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


stimParameterDefs = {
    'Sine': [
        {'name': 'Amplitude', 'type': 'float', 'value': 15.0, 'step': 1.0, 'suffix': 'deg'},
        {'name': 'Frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'Cycles', 'type': 'int', 'value': 10},
        {'name': 'Activation', 'type': 'group', 'children': [
            {'name': 'On', 'type': 'bool', 'value': True},
            {'name': 'Start cycle', 'type': 'int', 'value': 3},
            {'name': 'Phase', 'type': 'float', 'value': 0.0, 'step': 10.0, 'suffix': '%'},
            {'name': 'Duty', 'type': 'float', 'value': 30.0, 'step': 10.0, 'suffix': '%'},
            {'name': 'Left voltage', 'type': 'float', 'value': 2.0, 'step': 1.0, 'suffix': 'V'},
            {'name': 'Left voltage scale', 'type': 'float', 'value': 1.0, 'step': 1.0, 'suffix': 'V/V'},
            {'name': 'Right voltage', 'type': 'float', 'value': 2.0, 'step': 1.0, 'suffix': 'V'},
            {'name': 'Right voltage scale', 'type': 'float', 'value': 0.4, 'step': 1.0, 'suffix': 'V/V'},
            {'name': 'Pulse rate', 'type': 'float', 'value': 75.0, 'step': 5.0, 'suffix': 'Hz'},
        ]}
    ],
    'Frequency Sweep': [
        {'name': 'Start frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'End frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'Frequency change', 'type': 'list', 'values': ['Exponential','Linear'], 'value': 'Exponential'},
        {'name': 'Duration', 'type': 'float', 'value': 300.0, 'suffix': 'sec'},
        {'name': 'Amplitude', 'type': 'float', 'value': 15.0, 'step': 1.0, 'suffix': 'deg'},
        {'name': 'Frequency exponent', 'type': 'float', 'value': 0.0, 'limits': (-1, 0)}
    ]
}

parameterDefinitions = [
    {'name': 'DAQ', 'type': 'group', 'children': [
        {'name': 'Input', 'type': 'group', 'children': [
            {'name': 'Sampling frequency', 'type': 'float', 'value': 1000.0, 'step': 500.0, 'siPrefix': True,
             'suffix': 'Hz'},
            {'name': 'xForce', 'type': 'str', 'value': 'Dev1/ai0'},
            {'name': 'yForce', 'type': 'str', 'value': 'Dev1/ai1'},
            {'name': 'zForce', 'type': 'str', 'value': 'Dev1/ai2'},
            {'name': 'xTorque', 'type': 'str', 'value': 'Dev1/ai3'},
            {'name': 'yTorque', 'type': 'str', 'value': 'Dev1/ai4'},
            {'name': 'zTorque', 'type': 'str', 'value': 'Dev1/ai5'},
            {'name': 'Load calibration', 'type': 'action'},
            {'name': 'Encoder', 'type': 'str', 'value': 'Dev1/ctr0'},
            {'name': 'Counts per revolution', 'type': 'int', 'value': 10000, 'limits': (1, 100000)}
        ]},
        {'name': 'Output', 'type': 'group', 'children': [
            {'name': 'Sampling frequency', 'type': 'float', 'value': 10000.0, 'step': 1000.0, 'siPrefix': True,
             'suffix': 'Hz', 'readonly': True},
            {'name': 'Left stimulus', 'type': 'str', 'value': 'Dev1/ao0'},
            {'name': 'Right stimulus', 'type': 'str', 'value': 'Dev1/ao1'},
            {'name': 'Digital port', 'type': 'str', 'value': 'Dev1/port0'}
        ]},
        {'name': 'Update rate', 'type': 'float', 'value': 10.0, 'suffix': 'Hz'}
    ]},
    {'name': 'Motor parameters', 'type': 'group', 'children': [
        {'name': 'Maximum speed', 'type': 'float', 'value': 400.0, 'step': 50.0, 'suffix': 'RPM'},
        {'name': 'Minimum pulse frequency', 'type': 'float', 'value': 1000.0, 'step': 100.0, 'siPrefix': True,
         'suffix': 'Hz'},
        {'name': 'Maximum pulse frequency', 'type': 'float', 'value': 5000.0, 'step': 100.0, 'siPrefix': True,
         'suffix': 'Hz'},
    ]},
    {'name': 'Geometry', 'type': 'group', 'children': [
        {'name': 'doutvert', 'tip': 'Vertical distance from transducer to center of pressure', 'type': 'float',
         'value': 0.011, 'step': 0.001, 'siPrefix': True, 'suffix': 'm'},
        {'name': 'douthoriz', 'tip': 'Horizontal distance from transducer to center of pressure', 'type': 'float',
         'value': 0.0, 'step': 0.001, 'siPrefix': True, 'suffix': 'm'},
        {'name': 'din', 'tip': 'Horizontal distance from center of pressure to center of rotation', 'type': 'float',
         'value': 0.035, 'step': 0.001, 'siPrefix': True, 'suffix': 'm'},
        {'name': 'dclamp', 'tip': 'Horizontal distance between the edges of the clamps', 'type': 'float',
         'value': 0.030, 'step': 0.001, 'siPrefix': True, 'suffix': 'm'},
        {'name': 'Cross-section', 'type': 'group', 'children': [
            {'name': 'width', 'type': 'float', 'value': 0.021, 'step': 0.001, 'siPrefix': True, 'suffix': 'm'},
            {'name': 'height', 'type': 'float', 'value': 0.021, 'step': 0.001, 'siPrefix': True, 'suffix': 'm'},
        ]}
    ]},
    {'name': 'Stimulus', 'type': 'group', 'children': [
        {'name': 'Type', 'type': 'list', 'values': ['Sine', 'Frequency Sweep'], 'value': 'Sine'},
        {'name': 'Parameters', 'type': 'group', 'children': stimParameterDefs['Sine']},
        {'name': 'Wait before', 'type': 'float', 'value': 1.0, 'suffix': 's'},
        {'name': 'Wait after', 'type': 'float', 'value': 1.0, 'suffix': 's'},
    ]}
]


class BenderWindow(QtGui.QMainWindow):
    def __init__(self):
        super(BenderWindow, self).__init__()

        self.initUI()

        self.params = Parameter.create(name='params', type='group', children=parameterDefinitions)
        self.ui.parameterTreeWidget.setParameters(self.params, showTop=False)

        stimtype = self.params.child('Stimulus', 'Type')
        self.curStimType = stimtype.value()
        self.connectParameterSlots()

        self.stimParamState = dict()

        self.ui.browseOutputPathButton.clicked.connect(self.browseOutputPath)
        self.ui.fileNamePatternEdit.editingFinished.connect(self.updateFileName)
        self.ui.curFileNumberBox.valueChanged.connect(self.updateFileName)
        self.ui.restartNumberingButton.clicked.connect(self.restartNumbering)

        self.ui.saveParametersButton.clicked.connect(self.saveParams)
        self.ui.loadParametersButton.clicked.connect(self.loadParams)

        self.ui.plot1Widget.setLabel('left', "Angle", units='deg')
        self.ui.plot1Widget.setLabel('bottom', "Time", units='sec')

        self.bender = BenderDAQ()
        self.ui.goButton.clicked.connect(self.startAcquisition)

        self.readSettings()

    def initUI(self):
        ui = Ui_BenderWindow()
        ui.setupUi(self)
        self.ui = ui

    def connectParameterSlots(self):
        self.params.child('Stimulus', 'Type').sigValueChanged.connect(self.changeStimType)
        self.params.child('Stimulus').sigTreeStateChanged.connect(self.generateStimulus)
        self.params.child('Motor parameters', 'Maximum pulse frequency').sigValueChanged.connect(self.updateOutputFrequency)

    def disconnectParameterSlots(self):
        try:
            self.params.child('Stimulus', 'Type').sigValueChanged.disconnect(self.changeStimType)
            self.params.child('Stimulus').sigTreeStateChanged.disconnect(self.generateStimulus)
            self.params.child('Motor parameters', 'Maximum pulse frequency').sigValueChanged.disconnect(
                self.updateOutputFrequency)
        except TypeError:
            pass

    def startAcquisition(self):
        self.bender.start()

        self.ui.plot2Widget.plot(x=self.bender.t[0:len(self.bender.encoder_in_data)],
                                 y=self.bender.encoder_in_data, clear=True, pen='r')
        self.ui.plot2Widget.setXLink(self.ui.plot1Widget)

    def browseOutputPath(self):
        outputPath = QtGui.QFileDialog.getExistingDirectory(self, "Choose output directory")
        if outputPath:
            self.ui.outputPathEdit.setText(outputPath)

    def changeStimType(self, param, value):
        stimParamGroup = self.params.child('Stimulus', 'Parameters')
        self.stimParamState[self.curStimType] = stimParamGroup.saveState()
        try:
            self.disconnectParameterSlots()

            if value in self.stimParamState:
                stimParamGroup.restoreState(self.stimParamState[value], blockSignals=True)
            else:
                stimParamGroup.clearChildren()
                stimParamGroup.addChildren(stimParameterDefs[value])
        finally:
            self.connectParameterSlots()
        self.curStimType = value
        self.generateStimulus()

    def generateStimulus(self):
        self.bender.make_stimulus(self.params)

        if self.bender.t is not None:
            self.ui.plot1Widget.plot(x=self.bender.t, y=self.bender.pos, clear=True)
            self.ui.plot1Widget.plot(x=self.bender.t, y=self.bender.vel, pen='r')
            Lbrush = pg.mkBrush(pg.hsvColor(0.0, sat=0.4, alpha=0.3))
            Rbrush = pg.mkBrush(pg.hsvColor(0.5, sat=0.4, alpha=0.3))
            for onoff in self.bender.Lonoff:
                self.ui.plot1Widget.addItem(pg.LinearRegionItem(onoff, movable=False, brush=Lbrush))

            for onoff in self.bender.Ronoff:
                self.ui.plot1Widget.addItem(pg.LinearRegionItem(onoff, movable=False, brush=Rbrush))

            self.ui.plot2Widget.plot(x=self.bender.tout, y=self.bender.motorpulses, clear=True, pen='r')
            self.ui.plot2Widget.plot(x=self.bender.tout, y=self.bender.motordirection, pen='b')

            self.ui.plot2Widget.setXLink(self.ui.plot1Widget)

            self.updateFileName()

    def updateFileName(self):
        pattern = self.ui.fileNamePatternEdit.text()
        filename = self.getFileName(pattern)
        self.ui.fileNameLabel.setText(filename)

    def updateOutputFrequency(self):
        self.params["DAQ", "Output", "Sampling frequency"] = self.params["Motor parameters", "Maximum pulse frequency"] * 2

    def restartNumbering(self):
        self.ui.curFileNumberBox.setValue(1)

    def getFileName(self, filepattern):
        stim = self.params.child('Stimulus', 'Parameters')

        class SafeDict(dict):
            def __missing__(self, key):
                return '{' + key + '}'

        logging.debug('Stimulus/Type = {}'.format(self.params['Stimulus', 'Type']))
        stimtype = str(self.params['Stimulus', 'Type'])
        if stimtype == 'Sine':
            data = SafeDict({'tp': 'sin',
                             'f': stim['Frequency'],
                             'a': stim['Amplitude'],
                             'ph': stim['Activation', 'Phase'],
                             'lv': stim['Activation', 'Left voltage'],
                             'rv': stim['Activation', 'Right voltage'],
                             'num': self.ui.curFileNumberBox.value()})

            if not stim['Activation', 'On']:
                data['lv'] = 0
                data['rv'] = 0
        elif stimtype == 'Frequency Sweep':
            data = SafeDict({'tp': 'freqsweep',
                             'a': stim['Amplitude'],
                             'f0': stim['Start frequency'],
                             'f1': stim['End frequency']})
        else:
            assert False

        # default formats for the different types
        fmt = {'f': '.2f',
               'a': '.0f',
               'ph': '02.0f',
               'lv': '.0f',
               'rv': '.0f',
               'f0': '.1f',
               'f1': '.1f',
               'num': '03d'}

        filepattern = str(filepattern)
        for key1, fmt1 in fmt.iteritems():
            if key1 in data:
                filepattern = filepattern.replace('{'+key1+'}', '{'+key1+':'+fmt1+'}')

        filename = string.Formatter().vformat(filepattern, (), data)

        filename = filename.replace('.', '_')

        return filename

    def saveParams(self):
        paramFile = QtGui.QFileDialog.getSaveFileName(self, "Choose parameter file", filter="INI files (.ini)")

        settings = QtCore.QSettings(paramFile, QtCore.QSettings.IniFormat)
        settings.beginGroup("ParameterTree")
        self.writeParameters(settings, self.params)
        settings.endGroup()

    def loadParams(self):
        paramFile = QtGui.QFileDialog.getOpenFileName(self, "Choose parameter file", filter="INI files (*.ini)")

        settings = QtCore.QSettings(paramFile, QtCore.QSettings.IniFormat)
        settings.beginGroup("ParameterTree")
        self.readParameters(settings, self.params)
        settings.endGroup()

    def readSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        settings.beginGroup("BenderWindow")
        self.resize(settings.value("size", QtCore.QSize(800, 600)).toSize())
        self.move(settings.value("position", QtCore.QSize(200, 200)).toPoint())

        self.ui.plotSplitter.restoreState(settings.value("plotSplitter").toByteArray())
        self.ui.verticalSplitter.restoreState(settings.value("verticalSplitter").toByteArray())

        settings.endGroup()

        settings.beginGroup("File")
        self.ui.outputPathEdit.setText(settings.value("OutputPath").toString())
        self.ui.fileNamePatternEdit.setText(settings.value("FileNamePattern").toString())
        v, ok = settings.value("CurrentFileNumber").toInt()
        if ok:
            self.ui.curFileNumberBox.setValue(v)
        settings.endGroup()

        settings.beginGroup("ParameterTree")

        try:
            self.disconnectParameterSlots()

            if settings.contains("Stimulus/Type"):
                stimtype = str(settings.value("Stimulus/Type").toString())
                if stimtype in stimParameterDefs:
                    stimParamGroup = self.params.child('Stimulus', 'Parameters')
                    stimParamGroup.clearChildren()
                    stimParamGroup.addChildren(stimParameterDefs[stimtype])
                else:
                    assert False
                self.curStimType = stimtype

            self.readParameters(settings, self.params)
        finally:
            self.connectParameterSlots()

        settings.endGroup()

        self.updateOutputFrequency()
        self.generateStimulus()
        self.updateFileName()

    def writeSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        logging.debug('Writing settings!')

        settings.beginGroup("BenderWindow")
        settings.setValue("size", self.size())
        settings.setValue("position", self.pos())
        settings.setValue("verticalSplitter", self.ui.verticalSplitter.saveState())
        settings.setValue("plotSplitter", self.ui.plotSplitter.saveState())
        settings.endGroup()

        settings.beginGroup("File")
        settings.setValue("OutputPath", self.ui.outputPathEdit.text())
        settings.setValue("FileNamePattern", self.ui.fileNamePatternEdit.text())
        settings.setValue("CurrentFileNumber", self.ui.curFileNumberBox.value())
        settings.endGroup()

        settings.beginGroup("ParameterTree")
        self.writeParameters(settings, self.params)
        settings.endGroup()

    def writeParameters(self, settings, params):
        for ch in params:
            if ch.hasChildren():
                settings.beginGroup(ch.name())
                settings.setValue("Expanded", ch.opts['expanded'])
                self.writeParameters(settings, ch)
                settings.endGroup()
            elif ch.type() in ['float', 'int', 'list', 'str']:
                settings.setValue(ch.name(), ch.value())

    def readParameters(self, settings, params):
        for ch in params:
            if ch.hasChildren():
                settings.beginGroup(ch.name())
                expanded = settings.value("Expanded").toBool()
                ch.setOpts(expanded=expanded)

                self.readParameters(settings, ch)
                settings.endGroup()
            else:
                if ch.type() == 'float':
                    v, ok = settings.value(ch.name()).toDouble()
                    if ok:
                        ch.setValue(v)
                elif ch.type() == 'int':
                    v, ok = settings.value(ch.name()).toInt()
                    if ok:
                        ch.setValue(v)
                elif ch.type() in ['str', 'list']:
                    if settings.contains(ch.name()):
                        ch.setValue(str(settings.value(ch.name()).toString()))

    def closeEvent(self, event):
        self.writeSettings()
        event.accept()

def main():
    logging.basicConfig(level=logging.DEBUG)
    app = QtGui.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    # settings = QtCore.QSettings('bender.ini', QtCore.QSettings.IniFormat)

    bw2 = BenderWindow()
    bw2.show()

    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())


