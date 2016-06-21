from __future__ import print_function, unicode_literals
import sys
import os
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

try:
    import PyDAQmx as daq
except ImportError:
    pass

from settings import SETTINGS_FILE

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class BenderDAQ(object):
    def __init__(self, parameters):
        self.params = parameters

        self.t = None
        self.pos = None
        self.vel = None
        self.digital_out = None

    def make_stimulus(self):
        logging.debug('Stimulus.type = {}'.format(self.params['Stimulus', 'Type']))
        if self.params['Stimulus', 'Type'] == 'Sine':
            self.make_sine_stimulus()
        elif self.params['Stimulus', 'Type'] == 'Frequency Sweep':
            self.make_freqsweep_stimulus()
        else:
            assert False

    def make_sine_stimulus(self):
        logging.debug('Stimulus.wait before = {}'.format(self.params['Stimulus', 'Wait before']))
        stim = self.params.child('Stimulus', 'Parameters')
        dur = self.params['Stimulus', 'Wait before'] + stim['Cycles'] / stim['Frequency'] + \
              self.params['Stimulus', 'Wait after']
        self.duration = dur

        tout = np.arange(0.0, dur, 1 / self.params['DAQ', 'Output', 'Sampling frequency']) - \
               self.params['Stimulus', 'Wait before']

        pos = stim['Amplitude'] * np.sin(2 * np.pi * stim['Frequency'] * tout)
        pos[tout < 0] = 0
        pos[tout > stim['Cycles'] / stim['Frequency']] = 0

        vel = 2.0 * np.pi * stim['Frequency'] * stim['Amplitude'] * \
              np.sin(2 * np.pi * stim['Frequency'] * tout)
        vel[tout < 0] = 0
        vel[tout > stim['Cycles'] / stim['Frequency']] = 0

        phase = tout * stim['Frequency']
        phase[tout < 0] = -1
        phase[tout > stim['Cycles'] / stim['Frequency']] = -1

        # self.digital_out_data = self.make_motor_pulses(tout, vel)

        # make activation
        actburstdur = stim['Activation','Duty']/100.0 / stim['Frequency']
        actburstdur = np.floor(actburstdur * stim['Activation','Pulse rate'] * 2) / (stim['Activation','Pulse rate'] * 2)
        actburstduty = actburstdur * stim['Frequency']

        actpulsephase = tout[np.logical_and(tout > 0, tout < actburstdur)] * stim['Activation','Pulse rate']
        burst = (np.mod(actpulsephase, 1) < 0.5).astype(np.float)

        bendphase = phase - 0.25
        bendphase[phase == -1] = -1

        Lactcmd = np.zeros_like(tout)
        Ractcmd = np.zeros_like(tout)
        Lonoff = []
        Ronoff = []
        for c in range(int(stim['Activation','Start cycle']), int(stim['Cycles'])):
            tstart = (c - 0.25 + stim['Activation','Phase']) / stim['Frequency']
            tend = tstart + actburstdur
            Lonoff.append([tstart, tend])
            Ronoff.append(np.array([tstart, tend]) + 0.5/stim['Frequency'])

            np.place(Lactcmd, np.logical_and(bendphase >= c + stim['Activation','Phase'],
                                             bendphase < c + stim['Activation','Phase'] + actburstduty),
                     burst)

            np.place(Ractcmd, np.logical_and(bendphase >= c + 0.5 + stim['Activation','Phase'],
                                             bendphase < c + 0.5 + stim['Activation','Phase'] + actburstduty),
                     burst)

        Lactcmd = Lactcmd * stim['Activation','Left voltage'] / stim['Activation','Left voltage scale']
        Ractcmd = Ractcmd * stim['Activation','Right voltage'] / stim['Activation','Right voltage scale']

        self.analog_out_data = np.row_stack((Lactcmd, Ractcmd))

        self.t = np.arange(0.0, dur, 1 / self.params['DAQ', 'Input', 'Sampling frequency']) - \
                 self.params['Stimulus', 'Wait before']
        self.pos = interpolate.interp1d(tout, pos, assume_sorted=True)(self.t)
        self.vel = interpolate.interp1d(tout, vel, assume_sorted=True)(self.t)

        self.Lact = interpolate.interp1d(tout, Lactcmd, assume_sorted=True)(self.t)
        self.Ract = interpolate.interp1d(tout, Ractcmd, assume_sorted=True)(self.t)
        self.Lonoff = np.array(Lonoff)
        self.Ronoff = np.array(Ronoff)

    def make_motor_pulses(self, t, vel):
        velfrac = vel / (self.output.motorMaxSpeed / 60 * 360)
        if np.any(np.abs(velfrac) > 1):
            raise ValueError('Motion is too fast!')

        motorpulserate = np.abs(velfrac) * (self.output.motorMaxFreq - self.output.motorMinFreq) \
                         + self.output.motorMinFreq
        motorpulsephase = integrate.cumtrapz(motorpulserate, x=t, initial=self.output.motorMinFreq)

        motorpulses = (np.mod(motorpulsephase, 1) <= 0.5).astype(np.uint8)
        motordirection = (velfrac <= 0).astype(np.uint8)
        motorenable = np.ones_like(motordirection)
        motorenable[-5:] = 0

        dig = np.packbits(np.column_stack((np.zeros((len(motorpulses), 5), dtype=np.uint8),
                                           motorenable,
                                           motorpulses,
                                           motordirection)))
        return dig

    def setup_channels(self):
        # analog input
        self.analog_in = daq.Task()

        self.analog_in.CreateAIVoltageChan(self.input.xForce, 'Fx', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_in.CreateAIVoltageChan(self.input.yForce, 'Fy', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_in.CreateAIVoltageChan(self.input.zForce, 'Fz', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_in.CreateAIVoltageChan(self.input.xTorque, 'Tx', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_in.CreateAIVoltageChan(self.input.yTorque, 'Ty', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_in.CreateAIVoltageChan(self.input.zTorque, 'Tz', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)

        nsamps = int(self.duration * self.input.frequency)
        self.analog_in.CfgSampClkTiming("", self.input.frequency, daq.DAQmx_Val_Rising,
                                        daq.DAQmx_Val_FiniteSamps, nsamps)

        # encoder input
        self.encoder_in = daq.Task()

        self.encoder_in.CreateCIAngEncoderChan(self.input.encoder, 'encoder',
                                               daq.DAQmx_Val_X4, False,
                                               0, daq.DAQmx_Val_AHighBHigh,
                                               daq.DAQmx_Val_Degrees,
                                               self.input.encoderCountsPerRev, 0, None)
        self.encoder_in.CfgSampClkTiming("ai/SampleClock", self.input.frequency, daq.DAQmx_Val_Rising,
                                         daq.DAQmx_Val_FiniteSamps, nsamps)

        # analog output (stimulus)
        self.analog_out = daq.Task()
        aobyteswritten = daq.int32()

        self.analog_out.CreateAOVoltageChan(self.output.leftStim, 'Lstim',
                                            -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_out.CreateAOVoltageChan(self.output.rightStim, 'Rstim',
                                            -10, 10, daq.DAQmx_Val_Volts, None)

        # set the output sample frequency and number of samples to acquire
        self.analog_out.CfgSampClkTiming("", self.output.frequency, daq.DAQmx_Val_Rising,
                                         daq.DAQmx_Val_FiniteSamps, self.analog_out_data.shape[1])
        # make sure the output starts at the same time as the input
        self.analog_out.CfgDigEdgeStartTrig("ai/StartTrigger", daq.DAQmx_Val_Rising)

        # write the output data
        self.analog_out.WriteAnalogF64(self.analog_out_data.shape[1], False, 10,
                                       daq.DAQmx_Val_GroupByChannel,
                                       self.analog_out_data, daq.byref(aobyteswritten), None)

        if aobyteswritten.value != self.analog_out_data.shape[1]:
            raise IOError('Problem with writing output data')

        # digital output (motor)
        self.digital_out = daq.Task()
        dobyteswritten = daq.int32()

        self.digital_out.CreateDOChan(self.output.digitalPort, '', daq.DAQmx_Val_ChanForAllLines)
        # use the analog output clock for digital output
        self.digital_out.CfgSampClkTiming("ao/SampleClock", self.output.frequency,
                                          daq.DAQmx_Val_Rising,
                                          daq.DAQmx_Val_FiniteSamps,
                                          len(self.digital_out_data))

        # write the digital data
        self.digital_out.WriteDigitalU8(len(self.digital_out_data), False, 10,
                                        daq.DAQmx_Val_GroupByChannel,
                                        self.digital_out_data, daq.byref(dobyteswritten), None)

        if dobyteswritten.value != len(self.digital_out_data):
            raise IOError('Problem with writing digital data')

    def start(self):
        # start the digital and analog output tasks.  They won't
        # do anything until the analog input starts
        self.digital_out.StartTask()
        self.analog_out.StartTask()
        self.encoder_in.StartTask()

        nsamps = int(self.duration * self.input.frequency)

        self.analog_in_data = np.zeros((6, nsamps), dtype=np.float64)
        aibytesread = daq.int32()

        self.encoder_in_data = np.zeros((nsamps,), dtype=np.float64)
        encbytesread = daq.int32()

        # read the input data
        self.analog_in.ReadAnalogF64(self.analog_in_data.shape[1], self.duration + 10.0,
                                     daq.DAQmx_Val_GroupByChannel,
                                     self.analog_in_data, self.analog_in_data.size, daq.byref(aibytesread), None)
        self.encoder_in.ReadCounterF64(len(self.encoder_in_data), self.duration + 10.0, self.encoder_in_data,
                                       self.encoder_in_data.size, daq.byref(encbytesread), None)

        self.analog_in_data = self.analog_in_data.T

        del self.analog_in
        del self.encoder_in
        del self.analog_out
        del self.digital_out


stimParameterDefs = {
    'Sine': [
        {'name': 'Amplitude', 'type': 'float', 'value': 15.0, 'step': 1.0, 'suffix': 'deg'},
        {'name': 'Frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'Cycles', 'type': 'int', 'value': 10},
        {'name': 'Activation', 'type': 'group', 'children': [
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
        {'name': 'Start Frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'End Frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
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
            {'name': 'Sampling frequency', 'type': 'float', 'value': 100000.0, 'step': 1000.0, 'siPrefix': True,
             'suffix': 'Hz'},
            {'name': 'Left stimulus', 'type': 'str', 'value': 'Dev1/ao0'},
            {'name': 'Right stimulus', 'type': 'str', 'value': 'Dev1/ao1'},
            {'name': 'Digital port', 'type': 'str', 'value': 'Dev1/port0'}
        ]},
        {'name': 'Motor parameters', 'type': 'group', 'children': [
            {'name': 'Maximum speed', 'type': 'float', 'value': 400.0, 'step': 50.0, 'suffix': 'RPM'},
            {'name': 'Minumum pulse frequency', 'type': 'float', 'value': 1000.0, 'step': 100.0, 'siPrefix': True,
             'suffix': 'Hz'},
            {'name': 'Maximum pulse frequency', 'type': 'float', 'value': 5000.0, 'step': 100.0, 'siPrefix': True,
             'suffix': 'Hz'},
        ]}
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

        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        self.initUI()

        self.params = Parameter.create(name='params', type='group', children=parameterDefinitions)
        self.ui.parameterTreeWidget.setParameters(self.params, showTop=False)

        stimtype = self.params.child('Stimulus', 'Type')
        self.curStimType = stimtype.value()
        stimtype.sigValueChanged.connect(self.changeStimType)

        self.params.child('Stimulus').sigTreeStateChanged.connect(self.generateStimulus)

        self.stimParamState = dict()

        self.ui.browseOutputPathButton.clicked.connect(self.browseOutputPath)

        self.ui.saveParametersButton.clicked.connect(self.saveParams)
        self.ui.loadParametersButton.clicked.connect(self.loadParams)

        self.ui.plot1Widget.setLabel('left', "Angle", units='deg')
        self.ui.plot1Widget.setLabel('bottom', "Time", units='sec')

        self.readSettings()

    def initUI(self):
        ui = Ui_BenderWindow()
        ui.setupUi(self)
        self.ui = ui

    def browseOutputPath(self):
        outputPath = QtGui.QFileDialog.getExistingDirectory(self, "Choose output directory")
        if outputPath:
            self.ui.outputPathEdit.setText(outputPath)

    def changeStimType(self, param, value):
        stimParamGroup = self.params.child('Stimulus', 'Parameters')
        self.stimParamState[self.curStimType] = stimParamGroup.saveState()
        if value in self.stimParamState:
            stimParamGroup.restoreState(self.stimParamState[value])
        else:
            stimParamGroup.clearChildren()
            stimParamGroup.addChildren(stimParameterDefs[value])
        self.curStimType = value

    def generateStimulus(self):
        self.bender = BenderDAQ(self.params)
        self.bender.make_stimulus()

        self.ui.plot1Widget.plot(x=self.bender.t, y=self.bender.pos, clear=True)

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

        settings.beginGroup("ParameterTree")
        self.readParameters(settings, self.params)
        settings.endGroup()

    def writeSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        logging.debug('Writing settings!')

        settings.beginGroup("BenderWindow")
        settings.setValue("size", self.size())
        settings.setValue("position", self.pos())
        settings.setValue("verticalSplitter", self.ui.verticalSplitter.saveState())
        settings.setValue("plotSplitter", self.ui.plotSplitter.saveState())
        settings.endGroup()

        settings.beginGroup("ParameterTree")
        self.writeParameters(settings, self.params)
        settings.endGroup()

    def writeParameters(self, settings, params):
        for ch in params:
            if ch.hasChildren():
                settings.beginGroup(ch.name())
                self.writeParameters(settings, ch)
                settings.endGroup()
            elif ch.type() in ['float', 'int', 'list', 'str']:
                settings.setValue(ch.name(), ch.value())

    def readParameters(self, settings, params):
        for ch in params:
            if ch.hasChildren():
                settings.beginGroup(ch.name())
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


