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
from bender import BenderWindow
from wholebody_params import parameterDefinitions, stimParameterDefs, stepperParams, velocityDriverParams, encoderParams, pwmParams

try:
    import PyDAQmx as daq
except ImportError:
    pass

from settings import MOTOR_TYPE, COUNTER_TYPE

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


# to update the UI -
# run Designer.exe to modify,
# then python C:\Anaconda2\Lib\site-packages\PyQt4\uic\pyuic.py bender.ui -o bender_ui.py

class BenderWindow_WholeBody(BenderWindow):
    plotNames = {'X torque': 3,
                 'Y force': 1,
                 'X force': 0,
                 'Y torque': 4,
                 'Z force': 2,
                 'Z torque': 5}

    def __init__(self):
        self.bender = BenderDAQ_WholeBody()
        self.benderFileClass = BenderFile_WholeBody
        self.stimParameterDefs = stimParameterDefs

        super(BenderWindow_WholeBody, self).__init__()

        self.bender.sigUpdate.connect(self.updateAcquisitionPlot)
        self.bender.sigDoneAcquiring.connect(self.endAcquisition)

        self.ui.plot1Widget.setLabel('left', "Angle", units='deg')
        self.ui.plot1Widget.setLabel('bottom', "Time", units='sec')
        self.ui.plot1Widget.setToolTip('Left = positive')


    def setup_parameters(self):
        self.params = Parameter.create(name='params', type='group', children=parameterDefinitions)
        self.ui.parameterTreeWidget.setParameters(self.params, showTop=False)

        if MOTOR_TYPE == 'stepper':
            self.params.child('Motor parameters').addChildren(stepperParams)
            self.params.child('DAQ', 'Output', 'Sampling frequency').setWritable()
            self.params['DAQ', 'Output', 'Sampling frequency'] = 100000
        elif MOTOR_TYPE == 'velocity':
            self.params.child('Motor parameters').addChildren(velocityDriverParams)
        elif MOTOR_TYPE == 'none':
            self.params.child('Motor parameters').remove()
            self.params.child('Stimulus', 'Type').setValues(['None'])
            self.params.child('Stimulus', 'Type').setValue('None')
        else:
            raise ValueError('Unknown MOTOR_TYPE {}'.format(MOTOR_TYPE))

        if COUNTER_TYPE == 'encoder':
            self.params.child('DAQ', 'Input').addChildren(encoderParams)
        elif COUNTER_TYPE == 'pwm':
            self.params.child('DAQ', 'Input').addChildren(pwmParams)
        else:
            raise ValueError('Unknown COUNTER_TYPE {}'.format(COUNTER_TYPE))

    def connectParameterSlots(self):
        self.params.child('Stimulus', 'Type').sigValueChanged.connect(self.changeStimType)
        self.params.child('Stimulus').sigTreeStateChanged.connect(self.generateStimulus)
        self.params.child('DAQ', 'Update rate').sigValueChanged.connect(self.generateStimulus)
        if MOTOR_TYPE == 'velocity':
            self.params.child('Motor parameters', 'Maximum pulse frequency').sigValueChanged.connect(
                self.updateOutputFrequency)
        elif MOTOR_TYPE == 'stepper':
            self.params.child('DAQ', 'Output', 'Sampling frequency').sigValueChanged.connect(self.generateStimulus)

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
            elif MOTOR_TYPE == 'stepper':
                self.params.child('DAQ', 'Output', 'Sampling frequency').sigValueChanged.disconnect(
                    self.generateStimulus)

            self.params.child('Motor parameters').sigTreeStateChanged.disconnect(self.generateStimulus)
            self.params.child('DAQ', 'Input', 'Get calibration...').sigActivated.disconnect(self.getCalibration)
        except TypeError:
            logging.warning('Problem disconnecting parameter slots')
            pass

    def startAcquisition(self):
        if self.calibration is None or self.calibration.size == 0:
            ret = QtGui.QMessageBox.warning(self, "You need to have a calibration!",
                                            buttons=QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel,
                                            defaultButton=QtGui.QMessageBox.Ok)
            if ret == QtGui.QMessageBox.Cancel:
                return
            self.getCalibration()

        super(BenderWindow_WholeBody, self).startAcquisition()

    def set_plot2(self):
        self.plot2 = self.ui.plot2Widget.plot(pen='k', clear=True)
        self.overlayPlot = self.ui.plot2Widget.plot(pen='r', clear=False)

        self.ui.plot2Widget.setLabel('left', self.ui.plotYBox.currentText(), units='unscaled')
        self.ui.plot2Widget.setLabel('bottom', "Time", units='sec')

        self.ui.plotYBox.addItems(['X torque', 'Y force', 'Body torque from X torque', 'Body torque from Y force',
                                   'X force', 'Z force', 'Z torque', 'Channel 4', ' Channel 5'])
        self.ui.plotXBox.addItems(['Time (sec)', 'Time (cycles)', 'Phase', 'Angle'])

        yname = str(self.ui.plotYBox.currentText())
        if yname in self.plotNames:
            self.plotYNum = self.plotNames[yname]
        elif 'X torque' in yname:
            self.plotYNum = self.plotNames['X torque']
        elif 'Y force' in yname:
            self.plotYNum = self.plotNames['Y force']
        else:
            self.plotYNum = 0

    def endAcquisition(self):
        self.data0 = np.dot(self.bender.analog_in_data[:, :6], self.calibration)
        self.data = self.filterData()

        super(BenderWindow_WholeBody, self).endAcquisition()

    def show_stim(self, x, y, xname, plotwidget):
        if xname == 'Time (sec)':
            Lbrush = pg.mkBrush(pg.hsvColor(0.0, sat=0.4, alpha=0.3))
            Rbrush = pg.mkBrush(pg.hsvColor(0.5, sat=0.4, alpha=0.3))

            for onoff in self.bender.Lonoff:
                leftplot = pg.LinearRegionItem(onoff, movable=False, brush=Lbrush)
                plotwidget.addItem(leftplot)
            for onoff in self.bender.Ronoff:
                rightplot = pg.LinearRegionItem(onoff, movable=False, brush=Rbrush)
                plotwidget.addItem(rightplot)
        else:
            Lpen = pg.mkPen(color=pg.hsvColor(0.0, sat=0.4), width=4)
            Rpen = pg.mkPen(pg.hsvColor(0.5, sat=0.4), width=4)

            t = self.bender.t
            for onoff in self.bender.Lonoff:
                ison = np.logical_and(t >= onoff[0], t < onoff[1])
                plotwidget.plot(pen=Lpen, clear=False, x=x[ison], y=y[ison])

            for onoff in self.bender.Ronoff:
                ison = np.logical_and(t >= onoff[0], t < onoff[1])
                plotwidget.plot(pen=Rpen, clear=False, x=x[ison], y=y[ison])

    def getBodyTorque(self, yname):
        y, yunit = self.getY(yname)
        if 'Body torque' in yname:
            return y, yunit
        elif yname == 'Z torque':
            y = copy(y)
            y *= (self.params['Geometry', 'dclamp'] / 2 + self.params['Geometry', 'douthoriz']) / self.params[
                'Geometry', 'doutvert']
            yunit = 'N m'

            tnorm = self.bender.tnorm
            y0 = np.mean(y[tnorm < 0])
            y -= y0
        elif yname == 'Y force':
            y = copy(y)
            y *= -(self.params['Geometry', 'dclamp'] / 2 + self.params['Geometry', 'douthoriz'])
            yunit = 'N m'

            tnorm = self.bender.tnorm
            y0 = np.mean(y[tnorm < 0])
            y -= y0
        else:
            logging.warning("Can't calculate body torque from %s", yname)
        return y, yunit

    def getWork(self, yctr=None):
        angle = self.bender.angle_in_data

        yname = self.ui.plotYBox.currentText()
        y, yunit = self.getBodyTorque(yname)

        xname = self.ui.plotXBox.currentText()
        x, xunit = self.getX(xname)

        self._calcWork(x, angle, y, yctr=yctr)

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

    def updateOutputFrequency(self):
        if MOTOR_TYPE == 'velocity':
            self.params["DAQ", "Output", "Sampling frequency"] = self.params[
                                                                     "Motor parameters", "Maximum pulse frequency"] * 2


class BenderDAQ_WholeBody(BenderDAQ):
    def __init__(self):
        super(BenderDAQ_WholeBody, self).__init__()

        if MOTOR_TYPE == 'stepper':
            self.make_motor_pulses = self.make_motor_stepper_pulses
        elif MOTOR_TYPE == 'velocity':
            self.make_motor_pulses = self.make_motor_velocity_pulses
        else:
            raise Exception("Unknown motor type %s", MOTOR_TYPE)

    def make_motor_stepper_pulses(self, t, pos, vel, tout):
        poshi = interpolate.interp1d(t, pos, kind='linear', assume_sorted=True, bounds_error=False,
                                     fill_value=0.0)(tout)
        velhi = interpolate.interp1d(t, vel, kind='linear', assume_sorted=True, bounds_error=False,
                                     fill_value=0.0)(tout)

        if self.params['Motor parameters', 'Sign convention'] == 'Left is negative':
            poshi = -poshi
            velhi = -velhi

        outsampfreq = self.params['DAQ', 'Output', 'Sampling frequency']
        stepsperrev = self.params['Motor parameters', 'Steps per revolution']
        if outsampfreq == 0 or stepsperrev == 0:
            raise ValueError('Problems with parameters')

        stepsize = 360.0 / stepsperrev
        maxspeed = stepsize * outsampfreq / 2

        if np.any(np.abs(vel) > maxspeed):
            raise ValueError('Motion is too fast!')

        stepnum = np.floor(poshi / stepsize)
        dstep = np.diff(stepnum)
        motorstep = np.concatenate((np.array([0], dtype='uint8'), (dstep != 0).astype('uint8')))
        motordirection = (velhi <= 0).astype('uint8')

        motorenable = np.ones_like(motordirection, dtype='uint8')
        motorenable[-5:] = 0

        self.motorpulses = motorstep
        self.motordirection = motordirection

        dig = np.packbits(np.column_stack((np.zeros((len(motorstep), 5), dtype=np.uint8),
                                           motorenable,
                                           motorstep,
                                           motordirection)))
        return dig

    def make_motor_velocity_pulses(self, t, pos, vel, tout):

        velhi = interpolate.interp1d(t, vel, kind='linear', assume_sorted=True, bounds_error=False,
                                     fill_value=0.0)(tout)

        if self.params['Motor parameters', 'Sign convention'] == 'Left is negative':
            velhi = -velhi

        motorParams = self.params.child('Motor parameters')

        velfrac = velhi / (motorParams['Maximum speed'] / 60 * 360)
        if np.any(np.abs(velfrac) > 1):
            raise ValueError('Motion is too fast!')

        motorpulserate = np.abs(velfrac) * (
        motorParams['Maximum pulse frequency'] - motorParams['Minimum pulse frequency']) \
                         + motorParams['Minimum pulse frequency']
        motorpulsephase = integrate.cumtrapz(motorpulserate, x=tout, initial=0)

        motorpulses = (np.mod(motorpulsephase, 1) <= 0.5).astype(np.uint8)
        motordirection = (velfrac <= 0).astype(np.uint8)
        motorenable = np.ones_like(motordirection)
        motorenable[-5:] = 0

        self.motorpulses = motorpulses
        self.motordirection = motordirection

        dig = np.packbits(np.column_stack((np.zeros((len(motorpulses), 5), dtype=np.uint8),
                                           motorenable,
                                           motorpulses,
                                           motordirection)))
        return dig

    def make_sine_stimulus(self):
        super(BenderDAQ_WholeBody, self).make_sine_stimulus()

        stim = self.params.child('Stimulus', 'Parameters')
        dur = self.params['Stimulus', 'Wait before'] + stim['Cycles'] / stim['Frequency'] + \
              self.params['Stimulus', 'Wait after']

        t = self.t

        # make activation
        actburstdur = stim['Activation', 'Duty'] / 100.0 / stim['Frequency']
        actburstdur = np.floor(actburstdur * stim['Activation', 'Pulse rate'] * 2) / (
        stim['Activation', 'Pulse rate'] * 2)
        actburstduty = actburstdur * stim['Frequency']

        actpulsephase = t[np.logical_and(t > 0, t < actburstdur)] * stim['Activation', 'Pulse rate']
        burst = (np.mod(actpulsephase, 1) < 0.5).astype(np.float)

        bendphase = self.tnorm - 0.25
        bendphase[self.tnorm == -1] = -1

        Lactcmd = np.zeros_like(t)
        Ractcmd = np.zeros_like(t)
        Lonoff = []
        Ronoff = []

        if stim['Activation', 'On']:
            actphase = stim['Activation', 'Phase'] / 100.0
            for c in range(int(stim['Activation', 'Start cycle']), int(stim['Cycles'])):
                k = np.argmax(bendphase >= c + actphase)
                tstart = t[k]
                # tstart = (c - 0.25 + actphase) / stim['Frequency']
                tend = tstart + actburstdur

                if stim['Activation', 'Left voltage'] != 0:
                    if any(bendphase >= c + actphase):
                        Lonoff.append([tstart, tend])
                    np.place(Lactcmd, np.logical_and(bendphase >= c + actphase,
                                                     bendphase < c + actphase + actburstduty),
                             burst)
                if stim['Activation', 'Right voltage'] != 0:
                    if any(bendphase >= c + actphase + 0.5):
                        Ronoff.append(np.array([tstart, tend]) + 0.5 / stim['Frequency'])

                    np.place(Ractcmd, np.logical_and(bendphase >= c + 0.5 + actphase,
                                                     bendphase < c + 0.5 + actphase + actburstduty),
                             burst)

            Lactcmd = Lactcmd * stim['Activation', 'Left voltage'] / stim['Activation', 'Left voltage scale']
            Ractcmd = Ractcmd * stim['Activation', 'Right voltage'] / stim['Activation', 'Right voltage scale']

        self.Lact = Lactcmd
        self.Ract = Ractcmd
        self.Lonoff = np.array(Lonoff)
        self.Ronoff = np.array(Ronoff)

        Lacthi = interpolate.interp1d(t, Lactcmd, kind='linear', assume_sorted=True, bounds_error=False,
                                      fill_value=0.0)(self.tout)
        Racthi = interpolate.interp1d(t, Ractcmd, kind='linear', assume_sorted=True, bounds_error=False,
                                      fill_value=0.0)(self.tout)

        self.analog_out_data = np.row_stack((Lacthi, Racthi))

    def make_ramp_stimulus(self):
        super(BenderDAQ_WholeBody, self).make_ramp_stimulus()

        stim = self.params.child('Stimulus', 'Parameters')
        t = self.t

        holddur = stim['Hold duration']
        amp = stim['Amplitude']
        rate = stim['Rate']
        rampdur = amp / rate

        # make activation
        actburstdur = stim['Activation', 'Duration']
        actburstdur = np.ceil(actburstdur * stim['Activation', 'Pulse rate'] * 2) / (
        stim['Activation', 'Pulse rate'] * 2)

        actpulsephase = t[np.logical_and(t > 0, t < actburstdur)] * stim['Activation', 'Pulse rate']
        burst = (np.mod(actpulsephase, 1) < 0.5).astype(np.float)

        Lactcmd = np.zeros_like(t)
        Ractcmd = np.zeros_like(t)
        Lonoff = []
        Ronoff = []

        stimdelay = stim['Activation', 'Delay']

        if stim['Activation', 'During'] == 'Hold':
            isact = np.logical_and(t > rampdur + stimdelay, t <= rampdur + stimdelay + actburstdur)
            tstart = rampdur + stimdelay
            tend = rampdur + stimdelay + actburstdur
        elif stim['Activation', 'During'] == 'Ramp':
            isact = np.logical_and(t > stimdelay, t <= actburstdur + stimdelay)
            tstart = stimdelay
            tend = actburstdur + stimdelay

        if stim['Activation', 'Stim side'] == 'Left':
            onoff = Lonoff
            actcmd = Lactcmd
        else:
            onoff = Ronoff
            actcmd = Ractcmd

        if stim['Activation', 'Stim voltage'] != 0:
            onoff.append([tstart, tend])
            np.place(actcmd, isact, burst)

        Lactcmd = Lactcmd * stim['Activation', 'Stim voltage'] / stim['Activation', 'Left voltage scale']
        Ractcmd = Ractcmd * stim['Activation', 'Stim voltage'] / stim['Activation', 'Right voltage scale']

        tout = self.tout

        Lacthi = interpolate.interp1d(t, Lactcmd, kind='linear', assume_sorted=True, bounds_error=False,
                                      fill_value=0.0)(tout)
        Racthi = interpolate.interp1d(t, Ractcmd, kind='linear', assume_sorted=True, bounds_error=False,
                                      fill_value=0.0)(tout)

        self.analog_out_data = np.row_stack((Lacthi, Racthi))

        self.Lact = Lactcmd
        self.Ract = Ractcmd
        self.Lonoff = np.array(Lonoff)
        self.Ronoff = np.array(Ronoff)

    def make_motor_signal(self, t, pos, vel):
        self.digital_out_data = self.make_motor_pulses(t, pos, vel, self.tout)

    def setup_input_channels(self):
        # analog input
        assert (self.duration is not None)

        self.analog_in = daq.Task()

        inputParams = self.params.child('DAQ', 'Input')

        self.analog_in.CreateAIVoltageChan(inputParams['xForce'], 'Fx', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_in.CreateAIVoltageChan(inputParams['yForce'], 'Fy', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_in.CreateAIVoltageChan(inputParams['zForce'], 'Fz', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_in.CreateAIVoltageChan(inputParams['xTorque'], 'Tx', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_in.CreateAIVoltageChan(inputParams['yTorque'], 'Ty', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_in.CreateAIVoltageChan(inputParams['zTorque'], 'Tz', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)

        self.analog_in.CreateAIVoltageChan(inputParams['Left stim'], 'Lstim', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_in.CreateAIVoltageChan(inputParams['Right stim'], 'Rstim', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)

        self.ninsamps = int(1.0 / self.params['DAQ', 'Update rate'] * inputParams['Sampling frequency'])
        self.inputbufferlen = 2 * self.ninsamps
        self.analog_in.CfgSampClkTiming("", inputParams['Sampling frequency'], daq.DAQmx_Val_Rising,
                                        daq.DAQmx_Val_ContSamps, self.inputbufferlen)

        # angle input
        self.angle_in = daq.Task()

        self.angle_in.CreateCIAngEncoderChan(inputParams['Encoder'], 'encoder',
                                             daq.DAQmx_Val_X4, False,
                                             0, daq.DAQmx_Val_AHighBHigh,
                                             daq.DAQmx_Val_Degrees,
                                             inputParams['Counts per revolution'], 0, None)
        self.angle_in.CfgSampClkTiming("ai/SampleClock", inputParams['Sampling frequency'], daq.DAQmx_Val_Rising,
                                       daq.DAQmx_Val_ContSamps, self.inputbufferlen)

    def get_analog_output_names(self):
        outputParams = self.params.child('DAQ', 'Output')

        return [outputParams['Left stimulus'], outputParams['Right stimulus']], ['Lstim', 'Rstim']


class BenderFile_WholeBody(BenderFile):
    def __init__(self, *args, **kwargs):
        super(BenderFile_WholeBody, self).__init__(*args, **kwargs)

    def setupFile(self, bender, params):
        super(BenderFile_WholeBody, self).setupFile(bender, params)

        # add activation parameters
        gout = self.h5file.require_group('Output')

        dset = gout.create_dataset('Lact', data=bender.Lact)
        dset.attrs['HardwareChannel'] = params['DAQ', 'Output', 'Left stimulus']
        try:
            dset.attrs['VoltScale'] = params['Stimulus', 'Parameters', 'Activation', 'Left voltage scale']
        except KeyError:
            pass

        dset = gout.create_dataset('Ract', data=bender.Ract)
        dset.attrs['HardwareChannel'] = params['DAQ', 'Output', 'Right stimulus']
        try:
            dset.attrs['VoltScale'] = params['Stimulus', 'Parameters', 'Activation', 'Right voltage scale']
        except KeyError:
            pass

        dset = gout.create_dataset('DigitalOut', data=bender.digital_out_data)
        dset.attrs['HardwareChannel'] = params['DAQ', 'Output', 'Digital port']

        stim = params.child('Stimulus', 'Parameters')
        if params['Stimulus', 'Type'] == 'Sine':
            gout.attrs['ActivationOn'] = stim['Activation', 'On']
            gout.attrs['ActivationDuty'] = stim['Activation', 'Duty']
            gout.attrs['ActivationStartPhase'] = stim['Activation', 'Phase']
            gout.attrs['ActivationStartCycle'] = stim['Activation', 'Start cycle']
            gout.attrs['ActivationPulseFreq'] = stim['Activation', 'Pulse rate']
            gout.attrs['Left voltage'] = stim['Activation', 'Left voltage']
            gout.attrs['Right voltage'] = stim['Activation', 'Right voltage']
        elif params['Stimulus', 'Type'] == 'Ramp':
            gout.attrs['ActivationDuring'] = stim['Activation', 'During']
            gout.attrs['ActivationDuration'] = stim['Activation', 'Duration']
            gout.attrs['ActivationDelay'] = stim['Activation', 'Delay']
            gout.attrs['ActivationPulseFreq'] = stim['Activation', 'Pulse rate']
            gout.attrs['StimVoltage'] = stim['Activation', 'Stim voltage']
            gout.attrs['StimSide'] = stim['Activation', 'Stim side']

    def saveRawData(self, aidata, encdata, params):
        gin = self.h5file.require_group('RawInput')

        for i, aichan in enumerate(['xForce', 'yForce', 'zForce', 'xTorque', 'yTorque', 'zTorque', 'Left stim', 'Right stim']):
            dset = gin.create_dataset(aichan, data=aidata[:, i])
            dset.attrs['HardwareChannel'] = params['DAQ', 'Input', aichan]

        dset = gin.create_dataset('Encoder', data=encdata)
        dset.attrs['HardwareChannel'] = params['DAQ', 'Input', 'Encoder']
        dset.attrs['CountsPerRev'] = params['DAQ', 'Input', 'Counts per revolution']

    def saveCalibratedData(self, data, calibration, params):
        gin = self.h5file.require_group('Calibrated')

        for i, aichan in enumerate(['xForce', 'yForce', 'zForce', 'xTorque', 'yTorque', 'zTorque']):
            dset = gin.create_dataset(aichan, data=data[:, i])

        gin.create_dataset('CalibrationMatrix', data=calibration)



