from __future__ import print_function, unicode_literals
import sys
import os
import string
import logging
from PyQt4 import QtGui, QtCore
from itertools import cycle
import xml.etree.ElementTree as ElementTree
import pickle
import numpy as np
from scipy import signal, integrate, interpolate
import h5py
from copy import copy
import datetime

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import pyqtgraph as pg

from bender_ui import Ui_BenderWindow

from benderdaq import BenderDAQ
from benderfile import BenderFile

from settings import SETTINGS_FILE, MOTOR_TYPE, TIME_DEBUG

from double_params import parameterDefinitions, stimParameterDefs, perturbationDefs, ChannelGroup

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# python C:\Anaconda2\Lib\site-packages\PyQt4\uic\pyuic.py bender.ui -o bender_ui.py


class RasterGroup(pg.VTickGroup):
    def __init__(self, xvals=None, yrange=None, pen=None):
        pg.VTickGroup.__init__(self, xvals, yrange, pen)

    def setData(self, x, y):
        pg.VTickGroup.setXVals(self, vals=x)
        # ignore y


class BenderWindow(QtGui.QMainWindow):
    plotNames = {'X torque': 3,
                 'Y force': 1,
                 'X force': 0,
                 'Y torque': 4,
                 'Z force': 2,
                 'Z torque': 5}
    penOrder = ['k', 'b', 'g', 'r']
    markerOrder = ['o', 's', 't', 'd', '+']

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

        self.curPertType = self.params['Stimulus', 'Perturbations', 'Type']

        self.stimParamState = dict()

        self.filter = None

        self.ui.browseOutputPathButton.clicked.connect(self.browseOutputPath)
        self.ui.fileNamePatternEdit.editingFinished.connect(self.updateFileName)
        self.ui.nextFileNumberBox.valueChanged.connect(self.updateFileName)
        self.ui.restartNumberingButton.clicked.connect(self.restartNumbering)

        self.ui.saveParametersButton.clicked.connect(self.saveParams)
        self.ui.loadParametersButton.clicked.connect(self.loadParams)

        self.ui.plot1Widget.setLabel('left', "Angle", units='deg')
        self.ui.plot1Widget.setLabel('bottom', "Time", units='sec')

        self.plotwidgets = []
        self.plots = []
        self.spikeplots = []
        self.thresholdLines = []
        self.spikeThreshold = None
        self.nchannels = 0
        self.ischanneloverlay = True

        self.bender = BenderDAQ()
        self.bender.sigUpdate.connect(self.updateAcquisitionPlot)
        self.bender.sigDoneAcquiring.connect(self.endAcquisition)

        self.ui.goButton.clicked.connect(self.startAcquisition)

        self.workLabels = None
        self.activationPlot2 = None

        self.t = np.array([])
        self.data = np.array([])
        self.spikeind = None
        self.spikex = None
        self.spikeamp = None

        self.readSettings()

        self.ui.plotTypeCombo.currentIndexChanged.connect(self.changePlotType)
        self.ui.spikeTypeCombo.currentIndexChanged.connect(self.changeSpikeType)
        self.ui.channelOverlayCombo.currentIndexChanged.connect(self.changeChannelOverlay)

    def initUI(self):
        ui = Ui_BenderWindow()
        ui.setupUi(self)
        self.ui = ui

    def connectParameterSlots(self):
        self.params.child('Stimulus', 'Type').sigValueChanged.connect(self.changeStimType)
        self.params.child('Stimulus', 'Perturbations', 'Type').sigValueChanged.connect(self.changePerturbationType)
        try:
            self.params.child('Stimulus', 'Parameters', 'Type').sigValueChanged.connect(self.changeSineType)
        except Exception:
            pass

        self.params.child('Stimulus', 'Parameters').sigTreeStateChanged.connect(self.generateStimulus)
        self.params.child('Stimulus', 'Perturbations', 'Parameters').sigTreeStateChanged.connect(self.generateStimulus)
        self.params.child('Stimulus', 'Wait before').sigValueChanged.connect(self.generateStimulus)
        self.params.child('Stimulus', 'Wait after').sigValueChanged.connect(self.generateStimulus)

        self.params.child('DAQ', 'Update rate').sigValueChanged.connect(self.generateStimulus)
        if MOTOR_TYPE == 'velocity':
            self.params.child('Motor parameters', 'Maximum pulse frequency').sigValueChanged.connect(self.updateOutputFrequency)
        elif MOTOR_TYPE == 'stepper':
            self.params.child('DAQ', 'Output', 'Sampling frequency').sigValueChanged.connect(self.generateStimulus)

        self.params.child('Motor parameters').sigTreeStateChanged.connect(self.generateStimulus)
        self.params.child('DAQ', 'Input', 'Channels').sigTreeStateChanged.connect(self.channelsChanged)

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
            try:
                self.params.child('Stimulus', 'Parameters', 'Type').sigValueChanged.disconnect(self.changeSineType)
            except Exception:
                pass
            self.params.child('Stimulus', 'Parameters').sigTreeStateChanged.disconnect(self.generateStimulus)

            try:
                self.params.child('Stimulus', 'Perturbations', 'Parameters').sigTreeStateChanged.disconnect(self.generateStimulus)
            except Exception:
                pass

            self.params.child('Stimulus', 'Wait before').sigValueChanged.disconnect(self.generateStimulus)
            self.params.child('Stimulus', 'Wait after').sigValueChanged.disconnect(self.generateStimulus)


            self.params.child('DAQ', 'Update rate').sigValueChanged.disconnect(self.generateStimulus)
            if MOTOR_TYPE == 'velocity':
                self.params.child('Motor parameters', 'Maximum pulse frequency').sigValueChanged.disconnect(
                    self.updateOutputFrequency)
            elif MOTOR_TYPE == 'stepper':
                self.params.child('DAQ', 'Output', 'Sampling frequency').sigValueChanged.disconnect(self.generateStimulus)

            self.params.child('Motor parameters').sigTreeStateChanged.disconnect(self.generateStimulus)
            self.params.child('DAQ', 'Input', 'Channels').sigTreeStateChanged.disconnect(self.channelsChanged)

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

    def make_spike_plots(self, start=0):
        assert(len(self.spikeplots) == start)
        if self.ui.spikeTypeCombo.currentIndex() == 0 or \
                        self.ui.spikeTypeCombo.currentIndex() == 1:
            br = pg.mkBrush('w')

            for i, (pw, pen, marker) in enumerate(zip(self.plotwidgets, cycle(self.penOrder), cycle(self.markerOrder))):
                if i < start:
                    continue
                self.spikeplots.append(pw.plot(pen=None, symbolPen=pen, symbolBrush=br, symbol=marker))

        elif self.ui.spikeTypeCombo.currentIndex() == 2:
            # raster plot
            if self.ischanneloverlay:
                yr = np.linspace(0.9, 1, self.nchannels+1)
                yr = yr[:-1]
                yh = (yr[1] - yr[0])*0.9
            else:
                yr = 0.95*np.ones((self.nchannels,))
                yh = 0.05

            for i, (pw, yr1, pen, marker) in enumerate(zip(self.plotwidgets, yr, cycle(self.penOrder),
                                                           cycle(self.markerOrder))):
                if i < start:
                    continue
                r = RasterGroup(yrange=[yr1, yr1+yh], pen=pen)
                pw.addItem(r)
                self.spikeplots.append(r)

    def initializeChannels(self):
        channels = self.params.child('DAQ', 'Input', 'Channels').children()
        nchannels = len(channels)
        self.nchannels = nchannels

        # make the plot widgets
        if self.ui.channelOverlayCombo.currentIndex() == 0:
            plotwidget = self.ui.plot2Layout.addPlot()
            self.plotwidgets = [plotwidget] * nchannels
            self.ischanneloverlay = True
        elif self.ui.channelOverlayCombo.currentIndex() == 1:
            plotwidget = self.ui.plot2Layout.addPlot()
            self.plotwidgets = [plotwidget]
            for _ in range(nchannels-1):
                self.ui.plot2Layout.nextRow()
                pw = self.ui.plot2Layout.addPlot()
                self.plotwidgets.append(pw)
            self.ischanneloverlay = False

        # make the plots
        assert(len(self.plots) == 0)
        for pw, pen in zip(self.plotwidgets, cycle(self.penOrder)):
            self.plots.append(pw.plot(pen=pen))

        # make the spike plots
        self.make_spike_plots()

        # make the threshold lines
        self.spikeThreshold = np.tile(np.array([[-0.3, 0.3]]), (nchannels, 1))

        br = pg.mkBrush(pg.hsvColor(0.5, sat=0.4, alpha=0.3))

        self.thresholdLines = []
        threshline = pg.LinearRegionItem(self.spikeThreshold[0],
                                         orientation=pg.LinearRegionItem.Horizontal,
                                         movable=True, brush=br)
        self.plotwidgets[0].addItem(threshline)
        self.thresholdLines.append(threshline)
        threshline.sigRegionChangeFinished.connect(self.thresholdChanged)

        if not self.ischanneloverlay:
            for thresh, pw in zip(self.spikeThreshold[1:], self.plotwidgets[1:]):
                threshline = pg.LinearRegionItem(thresh,
                                                 orientation=pg.LinearRegionItem.Horizontal,
                                                 movable=True, brush=br)
                pw.addItem(threshline)
                self.thresholdLines.append(threshline)
                threshline.sigRegionChangeFinished.connect(self.thresholdChanged)

        # make empty data arrays
        self.t = np.array([])
        self.data = np.zeros((0, self.nchannels))
        self.spikeind = [[] for _ in range(self.nchannels)]
        self.spikeamp = [np.array([]) for _ in range(self.nchannels)]

    def channelsChanged(self):
        channels = self.params.child('DAQ', 'Input', 'Channels').children()
        newnchannels = len(channels)

        if newnchannels > self.nchannels:
            curchannels = self.nchannels
            if self.spikeThreshold is None:
                addthresh = 0.1
            else:
                addthresh = np.mean(self.spikeThreshold, axis=0)

            addthresh = np.tile(addthresh, (newnchannels-curchannels, 1))

            self.spikeThreshold = np.vstack((self.spikeThreshold, addthresh))

            if self.ui.channelOverlayCombo.currentIndex() == 0:
                self.plotwidgets = np.tile([self.plotwidgets[0]], (newnchannels,))
            elif self.ui.channelOverlayCombo.currentIndex() == 1:
                br = pg.mkBrush(pg.hsvColor(0.5, sat=0.4, alpha=0.3))

                for thresh in self.spikeThreshold[newnchannels:]:
                    self.ui.plot2Layout.nextRow()
                    pw = self.ui.plot2Layout.addPlot()
                    self.plotwidgets.append(pw)

                    threshline = pg.LinearRegionItem(thresh,
                                                     orientation=pg.LinearRegionItem.Horizontal,
                                                     movable=True, brush=br)

                    pw.addItem(threshline)
                    self.thresholdLines.append(threshline)
                    threshline.sigRegionChangeFinished.connect(self.thresholdChanged)

            # make the plots
            br = pg.mkBrush('w')
            for i, (pw, pen, marker) in enumerate(zip(self.plotwidgets, cycle(self.penOrder),
                                                           cycle(self.markerOrder))):
                if i < self.nchannels:
                    continue
                self.plots.append(pw.plot(pen=pen))

            self.make_spike_plots(start=self.nchannels)
            self.nchannels = newnchannels

        elif newnchannels < self.nchannels:
            for p, sp, ln, pw in zip(self.plots[newnchannels:], self.spikeplots[newnchannels:], \
                                     self.thresholdLines[newnchannels:], self.plotwidgets[newnchannels:]):
                pw.removeItem(p)
                pw.removeItem(sp)
                pw.removeItem(ln)
                if pw != self.plotwidgets[0]:
                    self.ui.plot2Layout.removeItem(pw)
            self.plots = self.plots[:newnchannels]
            self.spikeplots = self.spikeplots[:newnchannels]
            self.thresholdLines = self.thresholdLines[:newnchannels]
            self.plotwidgets = self.plotwidgets[:newnchannels]

            self.nchannels = newnchannels

        # clear the data arrays
        self.t = np.array([])
        self.data = np.zeros((0, self.nchannels))
        self.spikeind = [[] for _ in range(self.nchannels)]
        self.spikeamp = [np.array([]) for _ in range(self.nchannels)]

    def changeChannelOverlay(self, index):
        if self.ui.channelOverlayCombo.currentIndex() == 0:
            for plotwidget, p, sp in zip(self.plotwidgets[1:], self.plots[1:], self.spikeplots[1:]):
                plotwidget.removeItem(p)
                plotwidget.removeItem(sp)
                self.plotwidgets[0].addItem(p)
                self.plotwidgets[0].addItem(sp)
                self.ui.plot2Layout.removeItem(plotwidget)

            self.plotwidgets = np.tile([self.plotwidgets[0]], (self.nchannels,))
            self.plotwidgets[0].setLabel('left', 'Voltage', units='V')

            self.ischanneloverlay = True

        elif self.ui.channelOverlayCombo.currentIndex() == 1:
            self.plotwidgets = [self.plotwidgets[0]]
            br = pg.mkBrush(pg.hsvColor(0.5, sat=0.4, alpha=0.3))
            for thresh, p, sp in zip(self.spikeThreshold[1:], self.plots[1:], self.spikeplots[1:]):
                self.ui.plot2Layout.nextRow()
                pw = self.ui.plot2Layout.addPlot()
                self.plotwidgets.append(pw)

                self.plotwidgets[0].removeItem(p)
                self.plotwidgets[0].removeItem(sp)

                pw.addItem(p)
                pw.addItem(sp)

                threshline = pg.LinearRegionItem(thresh,
                                                 orientation=pg.LinearRegionItem.Horizontal,
                                                 movable=True, brush=br)

                pw.addItem(threshline)
                self.thresholdLines.append(threshline)
                threshline.sigRegionChangeFinished.connect(self.thresholdChanged)

            for pw, chan in zip(self.plotwidgets, self.params.child('DAQ', 'Input', 'Channels').children()):
                pw.setLabel('left', chan.value(), units='V')

            for pw in self.plotwidgets[:-1]:
                pw.hideAxis('bottom')

            self.ischanneloverlay = False

    def changePlotType(self, index):
        # TODO: Add plot vs caudal phase
        if index == 2 or index == 3:
            self.ui.spikeTypeCombo.setCurrentIndex(1)

        self.make_plot(self.t, self.tnorm, self.freq, self.data, self.encdata)

    def changeSpikeType(self, index):
        if index == 0:
            for sp in self.spikeplots:
                sp.setData(x=[], y=[])
        else:
            for pw, sp in zip(self.plotwidgets, self.spikeplots):
                pw.removeItem(sp)

            self.spikeplots = []
            self.make_spike_plots()

            self.find_spikes(self.data)

        if len(self.t) > 0 and len(self.data) > 0:
            self.make_plot(self.t, self.tnorm, self.freq, self.data, self.encdata)

    def thresholdChanged(self, threshline):
        rgn = threshline.getRegion()
        logging.debug("threshold changed: {}".format(rgn))

        if len(self.thresholdLines) > 1:
            i = self.thresholdLines.index(threshline)
            self.spikeThreshold[i] = rgn
        else:
            self.spikeThreshold[:, :] = rgn

        if len(self.data) > 0:
            self.find_spikes(self.data)
            self.make_plot(self.t, self.tnorm, self.freq, self.data, self.encdata)

    def find_spikes(self, data, append=False, offset=0):
        if data is None or data.size == 0:
            return

        if append:
            spikeind = self.spikeind
            spikeamp = self.spikeamp
        else:
            spikeind = [[] for _ in range(self.nchannels)]
            spikeamp = [np.array([]) for _ in range(self.nchannels)]

        for i, (chan, thresh) in enumerate(zip(np.rollaxis(data, 1), self.spikeThreshold)):
            spikeindhi = signal.argrelmax(chan, order=2)
            spikeindhi = spikeindhi[0]
            ishi = chan[spikeindhi] > thresh[1]
            spikeindhi = spikeindhi[ishi]

            spikeindlo = signal.argrelmin(chan, order=2)
            spikeindlo = spikeindlo[0]
            islo = chan[spikeindlo] < thresh[0]
            spikeindlo = spikeindlo[islo]

            spikeind1 = np.sort(np.concatenate((spikeindhi, spikeindlo)), kind='mergesort')

            spikeind[i].extend((spikeind1+offset).astype(np.int).tolist())
            spikeamp[i] = np.append(spikeamp[i], chan[spikeind1])

        self.spikeind = spikeind
        self.spikeamp = spikeamp

    def make_phase(self, t):
        phase = t * self.params['Stimulus', 'Parameters', 'Frequency']

        cycles = np.floor(phase)
        phase -= cycles

        return phase, cycles

    def make_plot(self, t, tnorm, freq, aidata, encdata):
        if self.ui.plotTypeCombo.currentText() == 'Raw data vs. time':
            x = t
            xs = self.bender.t
            phase = []
            showraw = True
        elif self.ui.plotTypeCombo.currentText() == 'Raw data vs. phase':
            x = tnorm
            xs = self.bender.tnorm
            showraw = True
        elif self.ui.plotTypeCombo.currentText() == 'Phase raster':
            x = tnorm
            xs = self.bender.tnorm
            showraw = False
        elif self.ui.plotTypeCombo.currentText() == 'Phase vs. frequency':
            if freq is None:
                self.ui.plotTypeCombo.setCurrentIndex(0)
                return
            else:
                x = freq
                xs = self.bender.f
                showraw = False

        self.stim_plot[0].setData(x=xs, y=self.bender.pos[0])
        self.stim_plot[1].setData(x=xs, y=self.bender.pos[1])

        self.encoderPlot1.setData(x=x, y=encdata[:, 0])
        self.encoderPlot2.setData(x=x, y=encdata[:, 1])

        for p, pw, chan in zip(self.plots, self.plotwidgets, np.rollaxis(aidata, 1)):
            if showraw:
                p.setData(x=x, y=chan)
            else:
                p.setData(x=[], y=[])

        if self.ui.spikeTypeCombo.currentIndex() > 0:
            for p, pw, sind, sa in zip(self.spikeplots, self.plotwidgets, self.spikeind, self.spikeamp):
                sx = x[sind]
                if self.ui.plotTypeCombo.currentText() == 'Raw data vs. time' or \
                    self.ui.plotTypeCombo.currentText() == 'Raw data vs. phase':
                    p.setData(x=sx, y=sa)
                elif self.ui.plotTypeCombo.currentText() == 'Phase raster':
                    ph = np.mod(sx, 1)
                    p.setData(x=sx, y=ph)
                elif self.ui.plotTypeCombo.currentText() == 'Phase vs. frequency':
                    ph = tnorm[sind]
                    ph = np.mod(ph, 1)
                    p.setData(x=sx, y=ph)

    def startAcquisition(self):
        self.ui.goButton.setText('Abort')
        self.ui.goButton.clicked.disconnect(self.startAcquisition)
        self.ui.goButton.clicked.connect(self.bender.abort)

        pattern = self.ui.fileNamePatternEdit.text()
        filename = self.getFileName(pattern)
        self.ui.fileNameLabel.setText(filename)
        self.curFileName = filename

        try:
            self.ui.plot1Widget.removeItem(self.encoderPlot1)
            self.ui.plot1Widget.removeItem(self.encoderPlot2)
        except AttributeError as e:
            pass

        self.encoderPlot1 = self.ui.plot1Widget.plot(pen='k', name='Encoder 1')
        self.encoderPlot2 = self.ui.plot1Widget.plot(pen='g', name='Encoder 2')

        self.plotwidgets[-1].setLabel('bottom', "Time", units='sec')

        for pw in self.plotwidgets:
            pw.setXLink(self.ui.plot1Widget)

        # reset data
        self.t = np.array([])
        self.tnorm = np.array([])
        self.data = np.zeros((0, self.nchannels))
        self.spikeind = [[] for _ in range(self.nchannels)]
        self.spikeamp = [np.array([]) for _ in range(self.nchannels)]

        self.start_acq_time = datetime.datetime.now()
        self.last_acq_time = datetime.datetime.now()
        self.n_acq = 0

        self.bender.start()

    def updateAcquisitionPlot(self, aidata, encdata):
        if self.ui.spikeTypeCombo.currentIndex() > 0:
            self.find_spikes(aidata[-1, :, :], append=True, offset=(aidata.shape[0]-1)*aidata.shape[1])

        aidata = aidata.reshape((-1, self.nchannels))
        encdata = encdata.reshape((-1, 2))

        t = self.bender.t[:aidata.shape[0]]
        tnorm = self.bender.tnorm[:aidata.shape[0]]
        try:
            freq = self.bender.f[:aidata.shape[0]]
        except Exception:
            freq = None

        self.encoderPlot1.setData(x=t.reshape((-1,)), y=encdata[:, 0])
        self.encoderPlot2.setData(x=t.reshape((-1,)), y=encdata[:, 1])

        self.t = t
        self.tnorm = tnorm
        self.freq = freq
        self.data = aidata
        self.encdata = encdata

        self.make_plot(t, tnorm, freq, aidata, encdata)

        logging.debug('updateAcquisitionPlot end')

    def endAcquisition(self):
        self.ui.goButton.setText('Go')
        self.ui.goButton.clicked.disconnect(self.bender.abort)
        self.ui.goButton.clicked.connect(self.startAcquisition)

        self.t = self.bender.t
        self.tnorm = self.bender.tnorm
        try:
            self.freq = self.bender.f
        except Exception:
            self.freq = None
        self.data = self.bender.analog_in_data
        self.encdata = self.bender.encoder_in_data

        self.make_plot(self.bender.t, self.tnorm, self.freq, self.data, self.encdata)

        filepath = str(self.ui.outputPathEdit.text())
        filename, ext = os.path.splitext(self.curFileName)
        with BenderFile(os.path.join(filepath, filename + '.h5'), allowoverwrite=True) as benderFile:
            benderFile.setupFile(self.bender, self.params)
            benderFile.saveRawData(self.bender.analog_in_data, self.bender.encoder_in_data, self.params)

        self.ui.nextFileNumberBox.setValue(self.ui.nextFileNumberBox.value() + 1)

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

    def changeSineType(self, param, value):
        stim = self.params.child('Stimulus', 'Parameters')

        self.disconnectParameterSlots()
        try:
            if value in ['Rostral only', 'Caudal only', 'Same amplitude', 'Same frequency']:
                try:
                    stim.child('Rostral frequency').setName('Frequency')
                except Exception:
                    pass
                stim.child('Caudal frequency').setReadonly(True)
                stim.child('Phase offset').setReadonly(False)
            elif value == 'Different frequency':
                try:
                    stim.child('Frequency').setName('Rostral frequency')
                except Exception:
                    pass
                stim.child('Caudal frequency').setReadonly(False)
                stim.child('Phase offset').setReadonly(True)

            if value == 'Rostral only':
                stim.child('Rostral amplitude').setReadonly(False)
                stim.child('Caudal amplitude').setReadonly(True)
            elif value == 'Caudal only':
                stim.child('Rostral amplitude').setReadonly(True)
                stim.child('Caudal amplitude').setReadonly(False)
            else:
                stim.child('Rostral amplitude').setReadonly(False)
                stim.child('Caudal amplitude').setReadonly(False)

            if value == 'Same amplitude':
                amp = max((stim['Rostral amplitude'], stim['Caudal amplitude']))
                stim['Rostral amplitude'] = amp
                stim['Caudal amplitude'] = amp
        finally:
            self.connectParameterSlots()
            self.generateStimulus()

    def changePerturbationType(self, param, value):
        # TODO: Debug perturbations
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

    def generateStimulus(self, showwarning=True):
        try:
            self.bender.make_stimulus(self.params)
        except ValueError as err:
            if showwarning:
                QtGui.QMessageBox.warning(self, 'Warning', str(err))
            else:
                raise
            return

        if self.bender.t is not None:
            self.ui.plot1Widget.clear()
            self.stim_plot = [self.ui.plot1Widget.plot(x=self.bender.t, y=p1, pen=col1)
                              for p1, col1 in zip(self.bender.pos, ['b','r'])]
            self.t = self.bender.t
            self.tnorm = self.bender.tnorm
            self.freq = None
            self.data = np.array([])
            self.encdata = np.array([])

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

        try:
            logging.debug('Stimulus/Type = {}'.format(self.params['Stimulus', 'Type']))
            stimtype = str(self.params['Stimulus', 'Type'])
            if stimtype == 'Sine':
                if stim['Type'] == 'Rostral only':
                    rostamp = stim['Rostral amplitude']
                    caudamp = 0.0
                elif stim['Type'] == 'Caudal only':
                    rostamp = 0.0
                    caudamp = stim['Caudal amplitude']
                else:
                    rostamp = stim['Rostral amplitude']
                    caudamp = stim['Caudal amplitude']

                if stim['Type'] == 'Different frequency':
                    rostfreq = stim['Rostral frequency']
                    caudfreq = stim['Caudal frequency']
                    freq = min((rostfreq, caudfreq))
                    phaseoff = 0.0
                else:
                    rostfreq = stim['Frequency']
                    caudfreq = stim['Frequency']
                    freq = rostfreq
                    phaseoff = stim['Phase offset']

                data = SafeDict({'tp': 'sin',
                                 'f': freq,
                                 'rf': rostfreq,
                                 'cf': caudfreq,
                                 'a': caudamp,
                                 'ca': caudamp,
                                 'ra': rostamp,
                                 'phoff': phaseoff,
                                 'num': self.ui.nextFileNumberBox.value()})

            elif stimtype == 'Frequency Sweep':
                data = SafeDict({'tp': 'freqsweep',
                                 'a': stim['Caudal amplitude'],
                                 'ca': stim['Caudal amplitude'],
                                 'ra': stim['Rostral amplitude'],
                                 'f0': stim['Start frequency'],
                                 'f1': stim['End frequency'],
                                 'phoff': stim['Phase offset'],
                                 'num': self.ui.nextFileNumberBox.value(),
                                 'f': stim['Start frequency']})

            elif stimtype == 'None':
                data = SafeDict({'tp': 'none',
                                 'num': self.ui.nextFileNumberBox.value()})

            else:
                assert False
        except Exception:
            data = SafeDict({'tp': 'none',
                                 'num': self.ui.nextFileNumberBox.value()})

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

        v, ok = settings.value("channelOverlay").toInt()
        if ok:
            self.ui.channelOverlayCombo.setCurrentIndex(v)

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
            self.readChannels(settings)
        finally:
            self.connectParameterSlots()

        settings.endGroup()

        try:
            self.updateOutputFrequency()
            if self.params['Stimulus', 'Type'] == 'Sine':
                self.changeSineType(self.params.child('Stimulus', 'Parameters', 'Type'),
                                    self.params['Stimulus', 'Parameters', 'Type'])
            self.generateStimulus(showwarning=False)
            self.initializeChannels()
            self.updateFileName()
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
        settings.setValue("channelOverlay", self.ui.channelOverlayCombo.currentIndex())
        settings.endGroup()

        settings.beginGroup("File")
        settings.setValue("OutputPath", self.ui.outputPathEdit.text())
        settings.setValue("FileNamePattern", self.ui.fileNamePatternEdit.text())
        settings.setValue("NextFileNumber", self.ui.nextFileNumberBox.value())
        settings.endGroup()

        settings.beginGroup("ParameterTree")
        self.writeParameters(settings, self.params)
        settings.setValue("DAQ/Input/Channels/nchannels", self.nchannels)

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

    def readChannels(self, settings):
        settings.beginGroup('DAQ/Input/Channels')
        nchan, ok = settings.value('nchannels').toInt()
        if not ok:
            nchan = 1

        chansettings = settings.childKeys()

        channelgroup = self.params.child('DAQ', 'Input', 'Channels')
        i = 0
        for hwchan in chansettings:
            if str(hwchan) == 'Expanded':
                continue
            channelgroup.addChild(
                dict(name=str(hwchan), type='str', value=str(settings.value(hwchan).toString()),
                     removable=True, renamable=True))
            i += 1
            if i == nchan:
                break
        settings.endGroup()

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


