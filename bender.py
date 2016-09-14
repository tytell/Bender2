from __future__ import print_function, unicode_literals
import sys
import os
import string
import logging
from PyQt4 import QtGui, QtCore
import xml.etree.ElementTree as ElementTree
import pickle
import numpy as np
from scipy import signal, integrate, interpolate
import h5py
from copy import copy

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import pyqtgraph as pg

from bender_ui import Ui_BenderWindow

from benderdaq import BenderDAQ
from benderfile import BenderFile

from settings import SETTINGS_FILE, MOTOR_TYPE

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

velocityDriverParams = [
    {'name': 'Maximum speed', 'type': 'float', 'value': 400.0, 'step': 50.0, 'suffix': 'RPM'},
    {'name': 'Minimum pulse frequency', 'type': 'float', 'value': 1000.0, 'step': 100.0, 'siPrefix': True,
     'suffix': 'Hz'},
    {'name': 'Maximum pulse frequency', 'type': 'float', 'value': 5000.0, 'step': 100.0, 'siPrefix': True,
     'suffix': 'Hz'},
]

stepperParams = [
    {'name': 'Steps per revolution', 'type': 'float', 'value': 6400}
]

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
            {'name': 'Get calibration...', 'type': 'action'},
            {'name': 'Calibration file', 'type': 'str', 'readonly': True},
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
    {'name': 'Motor parameters', 'type': 'group', 'children':
        stepperParams if MOTOR_TYPE == 'stepper' else velocityDriverParams},
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
    plotNames = {'X torque': 3,
                 'Y force': 1,
                 'X force': 0,
                 'Y torque': 4,
                 'Z force': 2,
                 'Z torque': 5}

    def __init__(self):
        super(BenderWindow, self).__init__()

        self.initUI()

        self.params = Parameter.create(name='params', type='group', children=parameterDefinitions)
        self.ui.parameterTreeWidget.setParameters(self.params, showTop=False)

        if MOTOR_TYPE == 'stepper':
            self.params.child('DAQ', 'Output', 'Sampling frequency').setWritable()
            self.params['DAQ', 'Output', 'Sampling frequency'] = 100000

        stimtype = self.params.child('Stimulus', 'Type')
        self.curStimType = stimtype.value()
        self.connectParameterSlots()

        self.stimParamState = dict()

        self.calibration = None
        self.filter = None

        self.ui.browseOutputPathButton.clicked.connect(self.browseOutputPath)
        self.ui.fileNamePatternEdit.editingFinished.connect(self.updateFileName)
        self.ui.nextFileNumberBox.valueChanged.connect(self.updateFileName)
        self.ui.restartNumberingButton.clicked.connect(self.restartNumbering)

        self.ui.saveParametersButton.clicked.connect(self.saveParams)
        self.ui.loadParametersButton.clicked.connect(self.loadParams)

        self.ui.plot1Widget.setLabel('left', "Angle", units='deg')
        self.ui.plot1Widget.setLabel('bottom', "Time", units='sec')

        self.bender = BenderDAQ()
        self.bender.sigUpdate.connect(self.updateAcquisitionPlot)
        self.bender.sigDoneAcquiring.connect(self.endAcquisition)

        self.ui.goButton.clicked.connect(self.startAcquisition)

        self.workLabels = None
        self.activationPlot2 = None

        self.readSettings()

    def initUI(self):
        ui = Ui_BenderWindow()
        ui.setupUi(self)
        self.ui = ui

    def connectParameterSlots(self):
        self.params.child('Stimulus', 'Type').sigValueChanged.connect(self.changeStimType)
        self.params.child('Stimulus').sigTreeStateChanged.connect(self.generateStimulus)
        self.params.child('DAQ', 'Update rate').sigValueChanged.connect(self.generateStimulus)
        if MOTOR_TYPE == 'velocity':
            self.params.child('Motor parameters', 'Maximum pulse frequency').sigValueChanged.connect(self.updateOutputFrequency)
        self.params.child('Motor parameters').sigTreeStateChanged.connect(self.generateStimulus)
        self.params.child('DAQ', 'Input', 'Get calibration...').sigActivated.connect(self.getCalibration)

    def disconnectParameterSlots(self):
        try:
            self.params.child('Stimulus', 'Type').sigValueChanged.disconnect(self.changeStimType)
            self.params.child('Stimulus').sigTreeStateChanged.disconnect(self.generateStimulus)
            self.params.child('DAQ', 'Update rate').sigValueChanged.disconnect(self.generateStimulus)
            if MOTOR_TYPE == 'velocity':
                self.params.child('Motor parameters', 'Maximum pulse frequency').sigValueChanged.disconnect(
                    self.updateOutputFrequency)
            self.params.child('Motor parameters').sigTreeStateChanged.disconnect(self.generateStimulus)
            self.params.child('DAQ', 'Input', 'Get calibration...').sigActivated.disconnect(self.getCalibration)
        except TypeError:
            logging.warning('Problem disconnecting parameter slots')
            pass

    def startAcquisition(self):
        if self.calibration is None or self.calibration.size == 0:
            ret = QtGui.QMessageBox.warning(self, "You need to have a calibration!", buttons=QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel,
                                            defaultButton=QtGui.QMessageBox.Ok)
            if ret == QtGui.QMessageBox.Cancel:
                return
            self.getCalibration()

        self.ui.goButton.setText('Abort')
        self.ui.goButton.clicked.disconnect(self.startAcquisition)
        self.ui.goButton.clicked.connect(self.bender.abort)

        self.ui.overlayCheck.setChecked(False)

        pattern = self.ui.fileNamePatternEdit.text()
        filename = self.getFileName(pattern)
        self.ui.fileNameLabel.setText(filename)
        self.curFileName = filename

        self.encoderPlot = self.ui.plot1Widget.plot(pen='k')

        self.plot2 = self.ui.plot2Widget.plot(pen='k', clear=True)
        self.overlayPlot = self.ui.plot2Widget.plot(pen='r', clear=False)

        self.ui.plot2Widget.setLabel('left', self.ui.plotYBox.currentText(), units='unscaled')
        self.ui.plot2Widget.setLabel('bottom', "Time", units='sec')

        yname = str(self.ui.plotYBox.currentText())
        if yname in self.plotNames:
            self.plotYNum = self.plotNames[yname]
        elif 'X torque' in yname:
            self.plotYNum = self.plotNames['X torque']
        elif 'Y force' in yname:
            self.plotYNum = self.plotNames['Y force']
        else:
            self.plotYNum = 0

        self.bender.start()

        # self.ui.plot2Widget.plot(x=self.bender.t[0:len(self.bender.encoder_in_data)],
        #                          y=self.bender.encoder_in_data, clear=True, pen='r')
        # self.ui.plot2Widget.setXLink(self.ui.plot1Widget)

    def updateAcquisitionPlot(self, t, aidata, encdata):
        logging.debug('updatePlot')
        t = t.flatten()
        encdata = encdata.flatten()
        aidata = aidata.reshape((len(t), -1))

        self.encoderPlot.setData(x=t, y=encdata)
        self.plot2.setData(x=t, y=aidata[:, self.plotYNum])

        logging.debug('updateAcquisitionPlot end')

    def endAcquisition(self):
        self.ui.goButton.setText('Go')
        self.ui.goButton.clicked.disconnect(self.bender.abort)
        self.ui.goButton.clicked.connect(self.startAcquisition)

        self.data0 = np.dot(self.bender.analog_in_data, self.calibration)
        self.data = self.filterData()

        self.updatePlot()

        filepath = str(self.ui.outputPathEdit.text())
        filename, ext = os.path.splitext(self.curFileName)
        with BenderFile(os.path.join(filepath, filename + '.h5'), allowoverwrite=True) as benderFile:
            benderFile.setupFile(self.bender, self.params)
            benderFile.saveRawData(self.bender.analog_in_data, self.bender.encoder_in_data, self.params)
            benderFile.saveCalibratedData(self.data, self.calibration, self.params)

        self.ui.nextFileNumberBox.setValue(self.ui.nextFileNumberBox.value() + 1)

        self.ui.plotXBox.currentIndexChanged.connect(self.changePlotX)
        self.ui.plotYBox.currentIndexChanged.connect(self.changePlotY)

        self.ui.filterCheck.stateChanged.connect(self.filterChecked)
        self.ui.filterCutoffBox.valueChanged.connect(self.filterCutoffChanged)

        self.ui.overlayCheck.stateChanged.connect(self.overlayChecked)
        self.ui.overlayFromBox.currentIndexChanged.connect(self.overlayFromChanged)
        self.ui.overlayColorBox.currentIndexChanged.connect(self.overlayColorChanged)

    def filterChecked(self, state):
        self.data = self.filterData(buildfilter=True)
        self.updatePlot()

    def filterCutoffChanged(self, cutoff):
        self.data = self.filterData(buildfilter=True)
        self.updatePlot()

    def filterData(self, buildfilter=False, data0=None):
        if self.ui.filterCheck.isChecked():
            if self.filter is None or buildfilter:
                cutoff = self.ui.filterCutoffBox.value()
                sampfreq = self.params['DAQ', 'Input', 'Sampling frequency']

                b, a = signal.butter(5, cutoff / (sampfreq / 2))
                self.filter = (b, a)

            if data0 is None:
                data0 = self.data0

            data = []
            if data0.ndim == 1:
                data = signal.filtfilt(self.filter[0], self.filter[1], data0)
            else:
                for data1 in data0.T:
                    data1 = signal.filtfilt(self.filter[0], self.filter[1], data1)
                    data.append(data1)

                data = np.array(data).T
        else:
            data = copy(self.data0)
        return data

    def updatePlot(self):
        self.changePlot(xname=self.ui.plotXBox.currentText(), yname=self.ui.plotYBox.currentText(),
                        colorbyname=self.ui.colorByBox.currentText())

    def getX(self, xname):
        if xname == 'Time (sec)':
            x = self.bender.t
            xunit = 'sec'
            self.ui.plot2Widget.setXLink(self.ui.plot1Widget)
        elif xname == 'Time (cycles)':
            x = self.bender.tnorm
            xunit = 'cycles'
        elif xname == 'Phase':
            x = np.mod(self.bender.phase, 1)
            xunit = ''
        elif xname == 'Angle':
            x = self.bender.encoder_in_data
            xunit = 'deg'
        else:
            assert False

        return x, xunit

    def getY(self, yname):
        if yname == 'Body torque from X torque':
            y, yunit = self.getBodyTorque('X torque')
        elif yname == 'Body torque from Y force':
            y, yunit = self.getBodyTorque('Y force')
        elif str(yname) in self.plotNames:
            y = self.data[:, self.plotNames[str(yname)]]
            if 'force' in yname:
                yunit = 'N'
            elif 'torque' in yname:
                yunit = 'N m'
            else:
                yunit = ''
                logging.debug('Unrecognized y variable unit: %s', yname)
        else:
            assert False

        return y, yunit

    def changePlot(self, xname, yname, colorbyname):
        self.ui.plot2Widget.setXLink(None)

        x, xunit = self.getX(xname)
        y, yunit = self.getY(yname)

        tnorm = self.bender.tnorm

        self.ui.plot2Widget.clear()

        maxcyc = np.ceil(np.max(tnorm))
        if xname != 'Time (sec)':
            for cyc in range(-1, int(maxcyc)):
                iscycle = np.logical_and(tnorm >= cyc, tnorm <= cyc + 1)
                if any(iscycle):
                    self.ui.plot2Widget.plot(pen='k', clear=False, x=x[iscycle], y=y[iscycle])
        else:
            self.ui.plot2Widget.plot(pen='k', clear=False, x=x, y=y)

        self.ui.plot2Widget.setLabel('left', yname, units=yunit)
        self.ui.plot2Widget.setLabel('bottom', xname, units=xunit)

        if xname == 'Time (sec)':
            Lbrush = pg.mkBrush(pg.hsvColor(0.0, sat=0.4, alpha=0.3))
            Rbrush = pg.mkBrush(pg.hsvColor(0.5, sat=0.4, alpha=0.3))

            for onoff in self.bender.Lonoff:
                act1 = pg.LinearRegionItem(onoff, movable=False, brush=Lbrush)
                self.ui.plot2Widget.addItem(act1)
            for onoff in self.bender.Ronoff:
                act1 = pg.LinearRegionItem(onoff, movable=False, brush=Rbrush)
                self.ui.plot2Widget.addItem(act1)
        else:
            Lpen = pg.mkPen(color=pg.hsvColor(0.0, sat=0.4), width=4)
            Rpen = pg.mkPen(pg.hsvColor(0.5, sat=0.4), width=4)

            t = self.bender.t
            for onoff in self.bender.Lonoff:
                ison = np.logical_and(t >= onoff[0], t < onoff[1])
                self.ui.plot2Widget.plot(pen=Lpen, clear=False, x=x[ison], y=y[ison])

            for onoff in self.bender.Ronoff:
                ison = np.logical_and(t >= onoff[0], t < onoff[1])
                self.ui.plot2Widget.plot(pen=Rpen, clear=False, x=x[ison], y=y[ison])

        ymed = np.nanmedian(y)
        logging.debug('ymed={}'.format(ymed))

        self.getWork(yctr=ymed)

        self.ui.plot2Widget.autoRange()
        logging.debug('changePlot end')

    def changePlotX(self, xind):
        xname = self.ui.plotXBox.itemText(xind)
        yname = self.ui.plotYBox.currentText()
        colorbyname = self.ui.colorByBox.currentText()
        self.changePlot(xname, yname, colorbyname)

    def changePlotY(self, yind):
        yname = self.ui.plotYBox.itemText(yind)
        xname = self.ui.plotXBox.currentText()
        colorbyname = self.ui.colorByBox.currentText()
        self.changePlot(xname, yname, colorbyname)

    def overlayChecked(self, state):
        self.updateOverlay()

    def overlayFromChanged(self, fromName):
        self.updateOverlay()

    def overlayColorChanged(self, color):
        self.overlayPlot.setPen(color)

    def loadOtherData(self, otherFile):
        try:
            with h5py.File(otherFile, 'r') as f:
                yname = str(self.ui.plotYOverlayBox.currentText())
                dsetName = BenderFile.datasetNames[yname]

                g = f['Calibrated']
                y = np.array(g[dsetName])

                y = self.filterData(data0=y)

                if 'force' in yname:
                    yunit = 'N'
                elif 'torque' in yname:
                    yunit = 'N m'
                else:
                    yunit = ''
                    logging.debug('Unrecognized y variable unit: %s', yname)

                xname = self.ui.plotXBox.currentText()

                g = f['NominalStimulus']
                if xname == 'Time (sec)':
                    x = g['t']
                    xunit = 'sec'
                elif xname == 'Time (cycles)':
                    x = g['tnorm']
                    xunit = 'cycles'
                elif xname == 'Phase':
                    x = g['Phase']
                    xunit = ''
                elif xname == 'Angle':
                    g = f['RawInput']
                    x = g['Encoder']
                    xunit = 'deg'
                else:
                    assert False

                x = np.array(x)
        except IOError:
            x, y = None, None
            xunit, yunit = '', ''

        return x, xunit, y, yunit

    def updateOverlay(self):
        if self.ui.overlayCheck.isChecked():
            xname = self.ui.plotXBox.currentText()
            yname = self.ui.plotYOverlayBox.currentText()

            if self.ui.overlayFromBox.currentText() == "Current file":
                x, xunit = self.getX(xname)

                if yname == self.ui.plotYBox.currentText():
                    return

                y, yunit = self.getY(yname)
            else:
                if self.ui.overlayFromBox.currentText() == "Other file...":
                    otherFile = QtGui.QFileDialog.getOpenFileName(self, "Choose other data file",
                                                                  filter="*.h5")
                    if not otherFile:
                        self.ui.overlayCheck.setChecked(False)
                        return
                else:
                    otherFile = self.ui.overlayFromBox.currentText()

                ind = self.ui.overlayFromBox.findText(otherFile)
                if ind == -1:
                    self.ui.overlayFromBox.addItem(otherFile)
                else:
                    self.ui.overlayFromBox.setCurrentIndex(ind)

                x, xunit, y, yunit = self.loadOtherData(str(otherFile))

            self.overlayPlot.setData(x=x, y=y)
            if yname != self.ui.plotYBox.currentText():
                self.ui.plot2Widget.setLabel('right', yname, units=yunit)
            self.ui.plot2Widget.setLabel('bottom', xname, units=xunit)
            self.ui.plot2Widget.autoRange()
        else:
            self.overlayPlot.setData(x=[], y=[])
            self.ui.plot2Widget.hideAxis('right')
            self.ui.plot2Widget.autoRange()

    def getBodyTorque(self, yname):
        y, yunit = self.getY(yname)
        if 'Body torque' in yname:
            return y, yunit
        elif yname == 'X torque':
            y = copy(y)
            y *= -self.params['Geometry', 'din'] / self.params['Geometry','doutvert']
            yunit = 'N m'

            tnorm = self.bender.tnorm
            y0 = np.mean(y[tnorm < 0])
            y -= y0
        elif yname == 'Y force':
            y = copy(y)
            y *= self.params['Geometry', 'din']
            yunit = 'N m'

            tnorm = self.bender.tnorm
            y0 = np.mean(y[tnorm < 0])
            y -= y0
        else:
            logging.warning("Can't calculate body torque from %s", yname)
        return y, yunit

    def getWork(self, yctr=None):
        tnorm = self.bender.tnorm

        maxcyc = np.max(tnorm)
        if np.ceil(maxcyc) - maxcyc < 0.01:
            maxcyc = np.ceil(maxcyc)
        else:
            maxcyc = np.floor(maxcyc)

        angle = self.bender.encoder_in_data

        yname = self.ui.plotYBox.currentText()
        y, yunit = self.getBodyTorque(yname)

        xname = self.ui.plotXBox.currentText()
        x, xunit = self.getX(xname)

        if yctr is None:
            vr = self.ui.plot2Widget.viewRange()
            yctr = (vr[1][1] + vr[1][0])/2
        logging.debug('yctr = {}'.format(yctr))

        work = []
        xmean = []
        for c in range(0, int(maxcyc) - 1):
            iscycle = np.logical_and(tnorm >= c, tnorm <= c + 1)
            if any(iscycle):
                w1 = integrate.trapz(y[iscycle], x=angle[iscycle])
                work.append(w1)

                xmean1 = np.mean(x[iscycle])
                xmean.append(xmean1)

                text = pg.TextItem('{:.4f}'.format(w1), color='b')
                self.ui.plot2Widget.addItem(text)
                text.setPos(xmean1, yctr)

    def browseOutputPath(self):
        outputPath = QtGui.QFileDialog.getExistingDirectory(self, "Choose output directory")
        if outputPath:
            self.ui.outputPathEdit.setText(outputPath)

    def getCalibration(self):
        calibrationFile = self.params['DAQ', 'Input', 'Calibration file']
        if not calibrationFile:
            calibrationFile = QtCore.QString()
        calibrationFile = QtGui.QFileDialog.getOpenFileName(self, "Choose calibration file", directory=calibrationFile,
                                                            filter="*.cal")
        if calibrationFile:
            self.params['DAQ', 'Input', 'Calibration file'] = calibrationFile

            self.loadCalibration()

    def loadCalibration(self):
        calibrationFile = self.params['DAQ', 'Input', 'Calibration file']
        if not calibrationFile:
            return
        if not os.path.exists(calibrationFile):
            raise IOError("Calibration file %s not found", calibrationFile)

        try:
            tree = ElementTree.parse(calibrationFile)
            cal = tree.getroot().find('Calibration')
            if cal is None:
                raise IOError('Not a calibration XML file')

            mat = []
            for ax in cal.findall('UserAxis'):
                txt = ax.get('values')
                row = [float(v) for v in txt.split()]
                mat.append(row)

        except IOError:
            logging.warning('Bad calibration file')
            return

        self.calibration = np.array(mat).T

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

    def generateStimulus(self, showwarning=True):
        try:
            self.bender.make_stimulus(self.params)
        except ValueError as err:
            if showwarning:
                QtGui.QMessageBox.warning(self, 'Warning', err.strerror)
            else:
                raise
            return

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
        self.ui.nextFileNameLabel.setText(filename)

    def updateOutputFrequency(self):
        if MOTOR_TYPE == 'velocity':
            self.params["DAQ", "Output", "Sampling frequency"] = self.params["Motor parameters", "Maximum pulse frequency"] * 2

    def restartNumbering(self):
        self.ui.nextFileNumberBox.setValue(1)

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
                             'num': self.ui.nextFileNumberBox.value()})

            if not stim['Activation', 'On']:
                data[' lv'] = 0
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
        v, ok = settings.value("NextFileNumber").toInt()
        if ok:
            self.ui.nextFileNumberBox.setValue(v)
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

        try:
            self.updateOutputFrequency()
            self.generateStimulus(showwarning=False)
            self.updateFileName()

            self.loadCalibration()
        except ValueError:
            # skip over problems with the settings
            pass

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
        settings.setValue("NextFileNumber", self.ui.nextFileNumberBox.value())
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


