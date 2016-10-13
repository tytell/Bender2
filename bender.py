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

from settings import SETTINGS_FILE, MOTOR_TYPE

TIME_DEBUG = True

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class ChannelGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add channel"
        pTypes.GroupParameter.__init__(self, **opts)

    def addNew(self):
        newname = "Channel {}".format(len(self.childs)+1)
        hwchan = "ai{}".format(len(self.childs))

        self.addChild(
            dict(name=hwchan, type='str', value=newname, removable=True, renamable=True))

stimParameterDefs = {
    'Sine': [
        {'name': 'Caudal amplitude', 'type': 'float', 'value': 15.0, 'step': 1.0, 'suffix': 'deg'},
        {'name': 'Rostral amplitude', 'type': 'float', 'value': 15.0, 'step':1.0, 'suffix': 'deg'},
        {'name': 'Distance between', 'type': 'int', 'value': 20, 'step': 1, 'suffix': 'seg'},
        {'name': 'Base phase offset', 'type': 'float', 'value': 0.0, 'readonly': True},
        {'name': 'Additional phase offset', 'type': 'float', 'value': 0.0},
        {'name': 'Frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'Cycles', 'type': 'int', 'value': 10},
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
        {'name': 'Device name', 'type': 'str', 'value': 'Dev1'},
        {'name': 'Input', 'type': 'group', 'children': [
            {'name': 'Sampling frequency', 'type': 'float', 'value': 1000.0, 'step': 500.0, 'siPrefix': True,
             'suffix': 'Hz'},
            ChannelGroup(name="Channels", children=[]),
            {'name': 'Encoder 1', 'type': 'str', 'value': 'ctr0'},
            {'name': 'Encoder 2', 'type': 'str', 'value': 'ctr2'},
            {'name': 'Counts per revolution', 'type': 'int', 'value': 10000, 'limits': (1, 100000)}
        ]},
        {'name': 'Output', 'type': 'group', 'children': [
            {'name': 'Sampling frequency', 'type': 'float', 'value': 10000.0, 'step': 1000.0, 'siPrefix': True,
             'suffix': 'Hz', 'readonly': True},
            {'name': 'Digital port', 'type': 'str', 'value': 'port0'}
        ]},
        {'name': 'Update rate', 'type': 'float', 'value': 10.0, 'suffix': 'Hz'}
    ]},
    {'name': 'Motor parameters', 'type': 'group', 'children':
        stepperParams if MOTOR_TYPE == 'stepper' else velocityDriverParams},
    {'name': 'Stimulus', 'type': 'group', 'children': [
        {'name': 'Type', 'type': 'list', 'values': ['Sine', 'Frequency Sweep'], 'value': 'Sine'},
        {'name': 'Parameters', 'type': 'group', 'children': stimParameterDefs['Sine']},
        {'name': 'Ramp duration', 'type': 'float', 'value': 0.5, 'suffix': 's'},
        {'name': 'Wait before', 'type': 'float', 'value': 1.0, 'suffix': 's'},
        {'name': 'Wait after', 'type': 'float', 'value': 1.0, 'suffix': 's'},
    ]}
]


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
        self.params.child('Stimulus').sigTreeStateChanged.connect(self.generateStimulus)
        try:
            self.params.child('Stimulus', 'Parameters', 'Distance between').sigValueChanged.connect(self.updatePhaseOffset)
        except Exception as exc:
            if 'has no child named' in str(exc):
                pass
            else:
                raise

        self.params.child('DAQ', 'Update rate').sigValueChanged.connect(self.generateStimulus)
        if MOTOR_TYPE == 'velocity':
            self.params.child('Motor parameters', 'Maximum pulse frequency').sigValueChanged.connect(self.updateOutputFrequency)
        elif MOTOR_TYPE == 'stepper':
            self.params.child('DAQ', 'Output', 'Sampling frequency').sigValueChanged.connect(self.generateStimulus)

        self.params.child('Motor parameters').sigTreeStateChanged.connect(self.generateStimulus)
        self.params.child('DAQ', 'Input', 'Channels').sigTreeStateChanged.connect(self.channelsChanged)

    def disconnectParameterSlots(self):
        try:
            self.params.child('Stimulus', 'Type').sigValueChanged.disconnect(self.changeStimType)
            self.params.child('Stimulus').sigTreeStateChanged.disconnect(self.generateStimulus)
            try:
                self.params.child('Stimulus', 'Parameters', 'Distance between').\
                    sigValueChanged.disconnect(self.updatePhaseOffset)
            except Exception as exc:
                if 'has no child named' in str(exc):
                    pass
                else:
                    raise

            self.params.child('DAQ', 'Update rate').sigValueChanged.disconnect(self.generateStimulus)
            if MOTOR_TYPE == 'velocity':
                self.params.child('Motor parameters', 'Maximum pulse frequency').sigValueChanged.disconnect(
                    self.updateOutputFrequency)
            elif MOTOR_TYPE == 'stepper':
                self.params.child('DAQ', 'Output', 'Sampling frequency').sigValueChanged.disconnect(self.generateStimulus)

            self.params.child('Motor parameters').sigTreeStateChanged.disconnect(self.generateStimulus)
            self.params.child('DAQ', 'Input', 'Channels').sigTreeStateChanged.disconnect(self.channelsChanged)
        except TypeError:
            logging.warning('Problem disconnecting parameter slots')
            pass

    def updatePhaseOffset(self):
        try:
            dist = self.params['Stimulus', 'Parameters', 'Distance between']
            self.params['Stimulus', 'Parameters', 'Base phase offset'] = dist * 0.01
        except Exception as exc:
            if 'has no child named' in str(exc):
                pass
            else:
                raise

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
            for p, sp, ln, pw in self.plots[newnchannels:], self.spikeplots[newnchannels:], \
                                 self.thresholdLines[newnchannels:], self.plotwidgets[newnchannels:]:
                pw.removeItem(p)
                pw.removeItem(sp)
                pw.removeItem(ln)
                if pw != self.plotwidgets[0]:
                    self.ui.plot2Layout.removeItem(pw)

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
        self.make_plot(self.bender.t, self.data)

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
            self.make_plot(self.t, self.data)

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
            self.make_plot(self.t, self.data)

    def find_spikes(self, data, append=False, offset=0):
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

    def make_plot(self, t, data):
        showraster = False
        if self.ui.plotTypeCombo.currentIndex() == 0:
            x = t
            phase = []
        elif self.ui.plotTypeCombo.currentIndex() == 1:
            phase, cycles = self.make_phase(t)
            x = phase + cycles
        elif self.ui.plotTypeCombo.currentIndex() == 2:
            phase, cycles = self.make_phase(t)
            x = phase + cycles
            showraster = True

        for p, pw, chan in zip(self.plots, self.plotwidgets, np.rollaxis(data, 1)):
            if not showraster:
                p.setData(x=x, y=chan)
            else:
                p.setData(x=[], y=[])

        if self.ui.spikeTypeCombo.currentIndex() > 0:
            for p, pw, sind, sa in zip(self.spikeplots, self.plotwidgets, self.spikeind, self.spikeamp):
                sx = x[sind]
                if not showraster:
                    p.setData(x=sx, y=sa)
                else:
                    cyc = np.floor(sx)
                    ph = phase - cyc
                    p.setData(x=ph, y=cyc)

    def startAcquisition(self):
        self.ui.goButton.setText('Abort')
        self.ui.goButton.clicked.disconnect(self.startAcquisition)
        self.ui.goButton.clicked.connect(self.bender.abort)

        pattern = self.ui.fileNamePatternEdit.text()
        filename = self.getFileName(pattern)
        self.ui.fileNameLabel.setText(filename)
        self.curFileName = filename

        self.encoderPlot1 = self.ui.plot1Widget.plot(pen='k', name='Encoder 1')
        self.encoderPlot2 = self.ui.plot1Widget.plot(pen='g', name='Encoder 2')

        self.plotwidgets[-1].setLabel('bottom', "Time", units='sec')

        for pw in self.plotwidgets:
            pw.setXLink(self.ui.plot1Widget)

        # reset data
        self.t = np.array([])
        self.data = np.zeros((0, self.nchannels))
        self.spikeind = [[] for _ in range(self.nchannels)]
        self.spikeamp = [np.array([]) for _ in range(self.nchannels)]

        self.start_acq_time = datetime.datetime.now()
        self.last_acq_time = datetime.datetime.now()
        self.n_acq = 0

        self.bender.start()

    def updateAcquisitionPlot(self, t, aidata, encdata):
        if TIME_DEBUG:
            starttime = datetime.datetime.now()
            self.n_acq += 1
            avgdt = (starttime - self.start_acq_time) / self.n_acq
            curdt = starttime - self.last_acq_time
            logging.debug('updatePlot: avg dt={}, current dt={}'.format(avgdt.total_seconds(), curdt.total_seconds()))
            self.last_acq_time = starttime

        encdata = encdata.reshape((-1, 2))
        self.encoderPlot1.setData(x=t.reshape((-1,)), y=encdata[:, 0])
        self.encoderPlot2.setData(x=t.reshape((-1,)), y=encdata[:, 1])

        if self.ui.spikeTypeCombo.currentIndex() > 0:
            self.find_spikes(aidata[-1, :, :], append=True, offset=(aidata.shape[0]-1)*aidata.shape[1])

        self.make_plot(t.reshape((-1,)), aidata.reshape((-1, self.nchannels)))

        if TIME_DEBUG:
            endtime = datetime.datetime.now()
            logging.debug('updatePlot: duration={}'.format((endtime-starttime).total_seconds()))

        logging.debug('updateAcquisitionPlot end')

    def endAcquisition(self):
        self.ui.goButton.setText('Go')
        self.ui.goButton.clicked.disconnect(self.bender.abort)
        self.ui.goButton.clicked.connect(self.startAcquisition)

        self.t = self.bender.t
        self.data = self.bender.analog_in_data

        self.make_plot(self.bender.t, self.data)

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
            self.ui.plot1Widget.plot(x=self.bender.t, y=self.bender.pos1, clear=True)
            self.ui.plot1Widget.plot(x=self.bender.t, y=self.bender.pos2, pen='r')

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
                             'a': stim['Caudal amplitude'],
                             'ca': stim['Caudal amplitude'],
                             'ra': stim['Rostral amplitude'],
                             'phoff': stim['Additional phase offset'],
                             'num': self.ui.nextFileNumberBox.value()})

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
            self.updatePhaseOffset()
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
        chansettings = settings.childKeys()

        channelgroup = self.params.child('DAQ', 'Input', 'Channels')
        for i in range(chansettings.count()):
            hwchan = str(chansettings[i])
            if hwchan == 'Expanded':
                continue
            channelgroup.addChild(
                dict(name=hwchan, type='str', value=str(settings.value(hwchan).toString()),
                     removable=True, renamable=True))
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


