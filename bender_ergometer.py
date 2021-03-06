from __future__ import print_function, unicode_literals
import logging
import numpy as np
from scipy import signal, integrate, interpolate
from copy import copy
from itertools import dropwhile

from PyQt4 import QtCore, QtGui

from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import pyqtgraph as pg

from benderdaq import BenderDAQ
from benderfile import BenderFile
from bender import BenderWindow
from ergometer_params import parameterDefinitions, stimParameterDefs, perturbationDefs

try:
    import PyDAQmx as daq
except ImportError:
    pass

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


# to update the UI -
# run Designer.exe to modify,
# then python C:\Anaconda2\Lib\site-packages\PyQt4\uic\pyuic.py bender.ui -o bender_ui.py

class BenderWindow_Ergometer(BenderWindow):
    plotNames = {'Force': 1,
                 'Length': 0,
                 'Stim': 2,
                 'Stim voltage': 2,
                 'Stim current': 3}

    def __init__(self):
        self.bender = BenderDAQ_Ergometer()
        self.stimParameterDefs = stimParameterDefs
        self.perturbationDefs = perturbationDefs
        self.benderFileClass = BenderFile_Ergometer

        super(BenderWindow_Ergometer, self).__init__()

        self.bender.sigUpdate.connect(self.updateAcquisitionPlot)
        self.bender.sigDoneAcquiring.connect(self.endAcquisition)

        self.ui.plot1Widget.setLabel('left', "Length", units='mm')
        self.ui.plot1Widget.setLabel('bottom', "Time", units='sec')
        self.ui.plot1Widget.setToolTip('Longer = positive')

    def initUI(self):
        super(BenderWindow_Ergometer, self).initUI()
        self.ui.plotYBox.addItems(['Force', 'Length', 'Stim voltage', 'Stim current'])
        self.ui.plotXBox.addItems(['Time (sec)', 'Time (cycles)', 'Phase', 'Length'])

    def setup_parameters(self):
        self.params = Parameter.create(name='params', type='group', children=parameterDefinitions)
        self.ui.parameterTreeWidget.setParameters(self.params, showTop=False)

        self.curPertType = self.params['Stimulus', 'Perturbations', 'Type']
        self.perturbationState = dict()

    def connectParameterSlots(self):
        self.params.child('Stimulus', 'Type').sigValueChanged.connect(self.changeStimType)
        self.params.child('Stimulus', 'Perturbations', 'Type').sigValueChanged.connect(self.changePerturbationType)
        self.params.child('Stimulus', 'Parameters').sigTreeStateChanged.connect(self.generateStimulus)
        self.params.child('Stimulus', 'Perturbations', 'Parameters').sigTreeStateChanged.connect(self.generateStimulus)
        self.params.child('Stimulus', 'Wait before').sigValueChanged.connect(self.generateStimulus)
        self.params.child('Stimulus', 'Wait after').sigValueChanged.connect(self.generateStimulus)
        self.params.child('DAQ', 'Update rate').sigValueChanged.connect(self.generateStimulus)
        self.params.child('DAQ', 'Output', 'Sampling frequency').sigValueChanged.connect(self.generateStimulus)
        self.params.child('DAQ', 'Input', 'Sampling frequency').sigValueChanged.connect(self.generateStimulus)

        self.params.child('Motor parameters').sigTreeStateChanged.connect(self.generateStimulus)

        try:
            self.params.child('Stimulus', 'Parameters', 'Activation', 'Type').sigValueChanged\
                .connect(self.changeActivationType)
        except Exception as e:
            logging.debug('Trouble with activation type: {}'.format(e))

        try:
            self.params.child('Stimulus', 'Perturbations', 'Parameters', 'Load frequencies...').sigActivated\
                .connect(self.loadPerturbationFreqs)
            self.params.child('Stimulus', 'Perturbations', 'Parameters', 'Randomize phases...').sigActivated\
                .connect(self.randPerturbationPhases)
        except Exception:
            pass

    def disconnectParameterSlots(self):
        try:
            self.params.child('Stimulus', 'Type').sigValueChanged.disconnect(self.changeStimType)
            self.params.child('Stimulus', 'Perturbations', 'Type').sigValueChanged.disconnect(self.changePerturbationType)
            self.params.child('Stimulus', 'Parameters').sigTreeStateChanged.disconnect(self.generateStimulus)
            self.params.child('Stimulus', 'Perturbations', 'Parameters').sigTreeStateChanged.disconnect(self.generateStimulus)
            self.params.child('Stimulus', 'Wait before').sigValueChanged.disconnect(self.generateStimulus)
            self.params.child('Stimulus', 'Wait after').sigValueChanged.disconnect(self.generateStimulus)
            self.params.child('DAQ', 'Update rate').sigValueChanged.disconnect(self.generateStimulus)
            self.params.child('DAQ', 'Output', 'Sampling frequency').sigValueChanged.disconnect(self.generateStimulus)
            self.params.child('DAQ', 'Input', 'Sampling frequency').sigValueChanged.disconnect(self.generateStimulus)

            self.params.child('Motor parameters').sigTreeStateChanged.disconnect(self.generateStimulus)

            try:
                self.params.child('Stimulus', 'Parameters', 'Activation', 'Type').sigValueChanged \
                    .disconnect(self.changeActivationType)
            except Exception:
                pass

            try:
                self.params.child('Stimulus', 'Perturbations', 'Load frequencies...').sigActivated.disconnect(
                    self.loadPerturbationFreqs)
                self.params.child('Stimulus', 'Perturbations', 'Randomize phases...').sigActivated.disconnect(
                    self.randPerturbationPhases)
            except Exception:
                pass

        except TypeError:
            logging.warning('Problem disconnecting parameter slots')
            pass

    def changeStimType(self, param, value):
        super(BenderWindow_Ergometer, self).changeStimType(param, value)

        try:
            p = self.params.child('Stimulus', 'Parameters', 'Activation', 'Type')
        except Exception:
            return

        self.changeActivationType(p, p.value())

    def changeActivationType(self, param, value):
        actgp = self.params.child('Stimulus', 'Parameters', 'Activation')
        if value == 'Generate train':
            isreadonly = False
        else:
            isreadonly = True

        iter = dropwhile(lambda x: x != param, actgp)
        iter.next()
        for p in iter:
            p.setReadonly(isreadonly)

    def changePerturbationType(self, param, value):
        pertGroup = self.params.child('Stimulus', 'Perturbations', 'Parameters')
        self.perturbationState[self.curPertType] = pertGroup.saveState()
        try:
            self.disconnectParameterSlots()

            if value in self.perturbationState:
                pertGroup.restoreState(self.perturbationState[value], blockSignals=True)
            else:
                pertGroup.clearChildren()
                pertGroup.addChildren(perturbationDefs[value])
        finally:
            self.connectParameterSlots()
        self.curPertType = value
        self.generateStimulus()

    def loadPerturbationFreqs(self):
        fn = QtGui.QFileDialog.getOpenFileName(self, 'Choose perturbation frequency file...')
        if len(fn) is 0:
            return

        freqs = []
        with open(fn, "r") as f:
            for ln in f:
                try:
                    ln = ln.strip()
                    val = float(ln)         # check that it converts properly to a float
                    freqs.append(ln)
                except ValueError:
                    pass                    # ignore lines that don't convert to floats

        freqstr = ' '.join(freqs)
        self.params['Stimulus', 'Perturbations', 'Parameters', 'Frequencies'] = freqstr
        self.randPerturbationPhases()

    def randPerturbationPhases(self):
        freqs = self.params['Stimulus', 'Perturbations', 'Parameters', 'Frequencies'].split()

        phases = np.random.rand(len(freqs))
        phasestr = ' '.join(['{:.3f}'.format(p) for p in phases])
        self.params['Stimulus', 'Perturbations', 'Parameters', 'Phases'] = phasestr

    def loadCalibration(self):
        pass

    def endAcquisition(self):
        self.data0 = self.bender.analog_in_data
        self.data0[:, self.plotNames['Length']] *= self.params['Motor parameters', 'Length scale']
        self.data0[:, self.plotNames['Length']] += self.params['Motor parameters', 'Initial length'] - \
                                                   self.data0[0, self.plotNames['Length']]
        self.data0[:, self.plotNames['Force']] *= self.params['DAQ', 'Input', 'Force scale']
        self.data0[:, self.plotNames['Stim voltage']] *= self.params['DAQ', 'Input', 'Stim voltage scale']
        self.data0[:, self.plotNames['Stim current']] *= self.params['DAQ', 'Input', 'Stim current scale']

        self.data = self.filterData()

        self.length_in_data = self.data[:, self.plotNames['Length']]

        super(BenderWindow_Ergometer, self).endAcquisition()

    def set_plot2(self):
        self.plot2 = self.ui.plot2Widget.plot(pen='k', clear=True)
        self.overlayPlot = self.ui.plot2Widget.plot(pen='r', clear=False)

        self.ui.plot2Widget.setLabel('left', self.ui.plotYBox.currentText(), units='unscaled')
        self.ui.plot2Widget.setLabel('bottom', "Time", units='sec')

        yname = str(self.ui.plotYBox.currentText())
        if yname in self.plotNames:
            self.plotYNum = self.plotNames[yname]
        else:
            self.plotYNum = 0

    def show_stim(self, x,y, xname, plotwidget):
        if xname == 'Time (sec)':
            br = pg.mkBrush(pg.hsvColor(0.0, sat=0.4, alpha=0.3))

            for onoff in self.bender.actonoff:
                actplot = pg.LinearRegionItem(onoff, movable=False, brush=br)
                plotwidget.addItem(actplot)
        else:
            pen = pg.mkPen(color=pg.hsvColor(0.0, sat=0.4), width=4)

            t = self.bender.t
            for onoff in self.bender.actonoff:
                ison = np.logical_and(t >= onoff[0], t < onoff[1])
                plotwidget.plot(pen=pen, clear=False, x=x[ison], y=y[ison])

    def getWork(self, yctr=None):
        len = self.data[:, self.plotNames['Length']]

        yname = self.ui.plotYBox.currentText()
        y, yunit = self.getY(yname)

        xname = self.ui.plotXBox.currentText()
        x, xunit = self.getX(xname)

        self._calcWork(x, -len, y, yctr=yctr)


class BenderDAQ_Ergometer(BenderDAQ):
    def __init__(self):
        super(BenderDAQ_Ergometer, self).__init__()

        self.ninchannels = 4

    def make_zero_stimulus(self):
        super(BenderDAQ_Ergometer, self).make_zero_stimulus()

        if self.t is None:
            return

        self.act = None
        self.actonoff = np.array([])

        self.analog_out_data = None

    def make_sine_stimulus(self):
        super(BenderDAQ_Ergometer, self).make_sine_stimulus()

        if self.t is None:
            return

        stim = self.params.child('Stimulus', 'Parameters')
        dur = self.params['Stimulus', 'Wait before'] + stim['Cycles'] / stim['Frequency'] + \
              self.params['Stimulus', 'Wait after']

        t = self.t
        dt = t[1] - t[0]

        bendphase = self.tnorm - 0.25
        bendphase[self.tnorm == -1] = -1

        actburstdur = stim['Activation','Duty']/100.0 / stim['Frequency']
        actburstdur = np.floor(actburstdur * stim['Activation','Pulse rate'] * 2) / (stim['Activation','Pulse rate'] * 2)
        actburstduty = actburstdur * stim['Frequency']

        actcmd = np.zeros_like(t)
        actonoff = []

        # make activation
        if stim['Activation', 'On']:
            if stim['Activation', 'Type'] == 'Sync pulse':
                pulsedur = 0.01 / dt

                actphase = stim['Activation', 'Phase'] / 100.0
                for c in range(int(stim['Activation', 'Start cycle']), int(stim['Cycles'])):
                    k = np.searchsorted(bendphase, c+actphase, side='left')
                    if k < len(bendphase):
                        tstart = t[k]
                        tend = tstart + actburstdur

                        actcmd[k:k+pulsedur] = 5
                        actonoff.append([tstart, tend])

            elif stim['Activation', 'Type'] == 'Generate train':
                npulses = int(np.floor(actburstdur * stim['Activation', 'Pulse rate']))
                tburst = t[np.logical_and(t > 0, t < actburstdur)]
                burst = np.zeros_like(tburst)
                for i in range(npulses):
                    tpulsestart = i/stim['Activation', 'Pulse rate']
                    ispulse = np.logical_and(tburst >= tpulsestart, tburst < tpulsestart + stim['Activation', 'Pulse duration'])
                    burst[ispulse] = 5.0

                actphase = stim['Activation', 'Phase'] / 100.0
                for c in range(int(stim['Activation', 'Start cycle']), int(stim['Cycles'])):
                    k = np.searchsorted(bendphase, c+actphase, side='left')
                    tstart = t[k]
                    # tstart = (c - 0.25 + actphase) / stim['Frequency']
                    tend = tstart + actburstdur

                    if k < len(bendphase):
                        actonoff.append([tstart, tend])
                        np.place(actcmd, np.logical_and(bendphase >= c + actphase,
                                                         bendphase < c + actphase + actburstduty),
                                 burst)

        if any(np.abs(self.lengthcmd) > 10):
            QtGui.QMessageBox.warning(None, 'Warning', 'Movement is too large. Clipping')
            self.lengthcmd[self.lengthcmd > 10.0] = 10.0
            self.lengthcmd[self.lengthcmd < -10.0] = 10.0

        self.act = actcmd
        self.actonoff = np.array(actonoff)

        acthi = interpolate.interp1d(t, actcmd, kind='linear', assume_sorted=True, bounds_error=False,
                                      fill_value=0.0)(self.tout)

        self.analog_out_data = np.row_stack((self.lengthcmd, acthi))

    def make_ramp_stimulus(self):
        super(BenderDAQ_Ergometer, self).make_ramp_stimulus()

        if self.t is None:
            return

        stim = self.params.child('Stimulus', 'Parameters')
        t = self.t
        dt = t[1] - t[0]

        holddur = stim['Hold duration']
        amp = stim['Amplitude']
        rate = stim['Rate']
        rampdur = amp / rate

        # make activation
        actburstdur = stim['Activation','Duration']
        actburstdur = np.ceil(actburstdur * stim['Activation','Pulse rate'] * 2) / (stim['Activation','Pulse rate'] * 2)

        npulses = int(np.floor(actburstdur * stim['Activation', 'Pulse rate']))
        burst = np.zeros_like(t[np.logical_and(t > 0, t < actburstdur)])
        for i in range(npulses):
            tpulsestart = i / stim['Activation', 'Pulse rate']
            ispulse = np.logical_and(t >= tpulsestart, t < tpulsestart + stim['Activation', 'Pulse duration'])
            burst[ispulse] = 5.0

        actcmd = np.zeros_like(t)
        actonoff = []

        stimdelay = stim['Activation', 'Delay']

        if stim['Activation', 'During'] == 'Hold':
            isact = np.logical_and(t > rampdur+stimdelay, t <= rampdur+stimdelay+actburstdur)
            tstart = rampdur+stimdelay
            tend = rampdur+stimdelay+actburstdur
        elif stim['Activation', 'During'] == 'Ramp':
            isact = np.logical_and(t > stimdelay, t <= actburstdur+stimdelay)
            tstart = stimdelay
            tend = actburstdur+stimdelay

        if stim['Activation', 'Type'] == 'Sync pulse':
            pulsedur = int(0.01 / dt)
            k = np.searchsorted(t, tstart)
            actcmd[k:k+pulsedur] = 5
        else:
            np.place(actcmd, isact, burst)
            actcmd = actcmd * stim['Activation', 'Voltage'] / stim['Activation', 'Voltage scale']
        actonoff.append([tstart, tend])

        tout = self.tout

        acthi = interpolate.interp1d(t, actcmd, kind='linear', assume_sorted=True, bounds_error=False,
                                      fill_value=0.0)(tout)

        if any(np.abs(self.lengthcmd) > 10):
            QtGui.QMessageBox.warning(None, 'Warning', 'Movement is too large. Clipping')
            self.lengthcmd[self.lengthcmd > 10.0] = 10.0
            self.lengthcmd[self.lengthcmd < -10.0] = 10.0

        self.analog_out_data = np.row_stack((self.lengthcmd, acthi))

        self.act = actcmd
        self.actonoff = np.array(actonoff)

    def make_motor_signal(self, t, pos, vel):
        self.lengthcmd = interpolate.interp1d(t, pos, kind='linear', assume_sorted=True, bounds_error=False,
                                              fill_value=0.0)(self.tout)
        self.lengthcmd = self.lengthcmd / self.params['Motor parameters', 'Length scale']

    def setup_input_channels(self):
        # analog input
        assert(self.duration is not None)

        self.analog_in = daq.Task()

        inputParams = self.params.child('DAQ', 'Input')

        self.analog_in.CreateAIVoltageChan(inputParams['Length'], 'length', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_in.CreateAIVoltageChan(inputParams['Force'], 'force', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_in.CreateAIVoltageChan(inputParams['Stim voltage'], 'stimvolts', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_in.CreateAIVoltageChan(inputParams['Stim current'], 'stimcurrent', daq.DAQmx_Val_Cfg_Default,
                                           -10, 10, daq.DAQmx_Val_Volts, None)

        self.ninsamps = int(1.0/self.params['DAQ', 'Update rate'] * inputParams['Sampling frequency'])
        self.inputbufferlen = 2*self.ninsamps
        self.analog_in.CfgSampClkTiming("", inputParams['Sampling frequency'], daq.DAQmx_Val_Rising,
                                        daq.DAQmx_Val_ContSamps, self.inputbufferlen)

        # angle input
        self.angle_in = None

    def get_analog_output_names(self):
        outputParams = self.params.child('DAQ', 'Output')

        return [outputParams['Length'], outputParams['Stimulus']], ['length', 'stim']


class BenderFile_Ergometer(BenderFile):
    def __init__(self, *args, **kwargs):
        super(BenderFile_Ergometer, self).__init__(*args, **kwargs)

    def setupFile(self, bender, params):
        super(BenderFile_Ergometer, self).setupFile(bender, params)

        # add activation parameters
        try:
            if bender.act is not None:
                gout = self.h5file.require_group('Output')

                dset = gout.create_dataset('act', data=bender.act)
                dset.attrs['HardwareChannel'] = params['DAQ', 'Output', 'Stimulus']

                dset = gout.create_dataset('length', data=bender.lengthcmd)
                dset.attrs['HardwareChannel'] = params['DAQ', 'Output', 'Length']
                dset.attrs['LengthScale'] = params['Motor parameters', 'Length scale']

                stim = params.child('Stimulus', 'Parameters')
                if params['Stimulus', 'Type'] == 'Sine':
                    gout.attrs['ActivationOn'] = stim['Activation', 'On']
                    gout.attrs['ActivationDuty'] = stim['Activation', 'Duty']
                    gout.attrs['ActivationStartPhase'] = stim['Activation', 'Phase']
                    gout.attrs['ActivationStartCycle'] = stim['Activation', 'Start cycle']
                    gout.attrs['ActivationPulseFreq'] = stim['Activation', 'Pulse rate']
                    gout.attrs['ActivationPulseDur'] = stim['Activation', 'Pulse duration']
                elif params['Stimulus', 'Type'] == 'Ramp':
                    gout.attrs['ActivationDuring'] = stim['Activation', 'During']
                    gout.attrs['ActivationDuration'] = stim['Activation', 'Duration']
                    gout.attrs['ActivationDelay'] = stim['Activation', 'Delay']
                    gout.attrs['ActivationPulseFreq'] = stim['Activation', 'Pulse rate']
                    gout.attrs['ActivationPulseDur'] = stim['Activation', 'Pulse duration']
        except Exception as err:
            self.errors.append('Problem saving activation')

    def saveRawData(self, aidata, angdata, params):
        gin = self.h5file.require_group('RawInput')

        for i, aichan in enumerate(['Length', 'Force', 'Stim voltage', 'Stim current']):
            dset = gin.create_dataset(aichan, data=aidata[:, i])
            dset.attrs['HardwareChannel'] = params['DAQ', 'Input', aichan]

    def saveCalibratedData(self, data, calibration, params):
        pass

