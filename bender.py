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

from settings import SETTINGS_FILE, EXPERIMENT_TYPE

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')



fileNameTips = {
    'None': '''{tp}: None''',
    'Sine': '''{tp} 'sin',
{f}: Frequency,
{a}: Amplitude,
{ph}: Phase,
{lv}: Left voltage,
{rv}: Right voltage,
{num}: Trial number''',
    'Frequency Sweep': '''{tp}: 'freqsweep',
{a}: Amplitude,
{f0}: Start frequency,
{f1}: End frequency,
{num}: Trial number''',
    'Ramp': '''{tp}: 'ramp',
{a}: stim['Amplitude'],
{r}: Rate,
{v}: Stim voltage,
{s}: Stim side,
{num}: Trial number'''
}

# to update the UI -
# run Designer.exe to modify,
# then python C:\Anaconda2\Lib\site-packages\PyQt4\uic\pyuic.py bender.ui -o bender_ui.py

class BenderWindow(QtGui.QMainWindow):
    def __init__(self):
        super(BenderWindow, self).__init__()

        self.initUI()

        self.setup_parameters()

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

        self.ui.goButton.clicked.connect(self.startAcquisition)

        self.workLabels = None
        self.activationPlot2 = None

        self.readSettings()
        self.set_plot2()

    def initUI(self):
        ui = Ui_BenderWindow()
        ui.setupUi(self)
        self.ui = ui

    def setup_parameters(self):
        # don't call this one directly. Override it in a subclass
        assert False

    def startAcquisition(self):
        self.ui.goButton.setText('Abort')
        self.ui.goButton.clicked.disconnect(self.startAcquisition)
        self.ui.goButton.clicked.connect(self.bender.abort)

        self.ui.overlayCheck.setChecked(False)

        pattern = self.ui.fileNamePatternEdit.text()
        filename = self.getFileName(pattern)
        self.ui.fileNameLabel.setText(filename)
        self.curFileName = filename

        self.anglePlot = self.ui.plot1Widget.plot(pen='k')
        self.set_plot2()

        self.bender.start()

        # self.ui.plot2Widget.plot(x=self.bender.t[0:len(self.bender.encoder_in_data)],
        #                          y=self.bender.encoder_in_data, clear=True, pen='r')
        # self.ui.plot2Widget.setXLink(self.ui.plot1Widget)

    def set_plot2(self):
        assert False

    def updateAcquisitionPlot(self, t, aidata, angdata):
        logging.debug('updatePlot')
        t = t.flatten()
        aidata = aidata.reshape((len(t), -1))

        if angdata.size != 0:
            angdata = angdata.flatten()
            self.anglePlot.setData(x=t, y=angdata)
        self.plot2.setData(x=t, y=aidata[:, self.plotYNum])

        logging.debug('updateAcquisitionPlot end')

    def endAcquisition(self):
        self.ui.goButton.setText('Go')
        self.ui.goButton.clicked.disconnect(self.bender.abort)
        self.ui.goButton.clicked.connect(self.startAcquisition)

        self.updatePlot()

        filepath = str(self.ui.outputPathEdit.text())
        filename, ext = os.path.splitext(self.curFileName)
        with self.benderFileClass(os.path.join(filepath, filename + '.h5'), allowoverwrite=True) as benderFile:
            benderFile.setupFile(self.bender, self.params)
            benderFile.saveRawData(self.bender.analog_in_data, self.bender.angle_in_data, self.params)
            if self.calibration is not None:
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
        elif xname == 'Length':
            x = self.length_in_data
            xunit = 'mm'
        else:
            assert False

        return x, xunit

    def getY(self, yname):
        yn = str(yname)
        if yname == 'Body torque from X torque':
            y, yunit = self.getBodyTorque('X torque')
        elif yname == 'Body torque from Y force':
            y, yunit = self.getBodyTorque('Y force')
        elif yname == 'Channel 4':
            y = self.bender.analog_in_data[:, 4]
            yunit = 'V'
        elif yname == 'Channel 5':
            y = self.bender.analog_in_data[:, 5]
            yunit = 'V'
        elif yn in self.plotNames:
            y = self.data[:, self.plotNames[yn]]
            if 'force' in yn.lower():
                yunit = 'N'
            elif 'torque' in yn.lower():
                yunit = 'N m'
            elif 'length' in yn.lower():
                yunit = 'mm'
            elif 'stim' in yn.lower():
                yunit = 'V'
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

        self.show_stim(x,y, xname, self.ui.plot2Widget)

        ymed = np.nanmedian(y)
        logging.debug('ymed={}'.format(ymed))

        self.getWork(yctr=ymed)

        self.ui.plot2Widget.autoRange()
        logging.debug('changePlot end')

    def show_stim(self, x,y, xname, plotwidget):
        assert False

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

    def getWork(self, yctr=None):
        pass

    def _calcWork(self, x, angle, y, yctr=None):
        tnorm = self.bender.tnorm

        maxcyc = np.max(tnorm)
        if np.ceil(maxcyc) - maxcyc < 0.01:
            maxcyc = np.ceil(maxcyc)
        else:
            maxcyc = np.floor(maxcyc)

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

    def changeStimType(self, param, value):
        stimParamGroup = self.params.child('Stimulus', 'Parameters')
        self.stimParamState[self.curStimType] = stimParamGroup.saveState()
        try:
            self.disconnectParameterSlots()

            if value in self.stimParamState:
                stimParamGroup.restoreState(self.stimParamState[value], blockSignals=True)
            else:
                stimParamGroup.clearChildren()
                stimParamGroup.addChildren(self.stimParameterDefs[value])
        finally:
            self.connectParameterSlots()
        self.curStimType = value
        self.generateStimulus()

        self.ui.fileNamePatternEdit.setToolTip(fileNameTips[self.curStimType])

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
            self.ui.plot1Widget.plot(x=self.bender.t, y=self.bender.pos, clear=True)
            self.ui.plot1Widget.plot(x=self.bender.t, y=self.bender.vel, pen='r')

            self.show_stim(self.bender.t, self.bender.pos, 'Time (sec)', self.ui.plot1Widget)

            self.ui.plot2Widget.setXLink(self.ui.plot1Widget)

            self.updateFileName()

    def updateFileName(self):
        pattern = self.ui.fileNamePatternEdit.text()
        filename = self.getFileName(pattern)
        self.ui.nextFileNameLabel.setText(filename)

    def updateOutputFrequency(self):
        pass

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
                             'num': self.ui.nextFileNumberBox.value()})
            try:
                data['v'] = stim['Activation', 'Voltage']
            except Exception:
                pass

            try:
                data['lv'] = stim['Activation', 'Left voltage']
                data['rv'] = stim['Activation', 'Right voltage']
            except Exception:
                pass

            if not stim['Activation', 'On']:
                data['lv'] = 0
                data['rv'] = 0
                data['v'] = 0
        elif stimtype == 'Frequency Sweep':
            data = SafeDict({'tp': 'freqsweep',
                             'a': stim['Amplitude'],
                             'f0': stim['Start frequency'],
                             'f1': stim['End frequency'],
                             'num': self.ui.nextFileNumberBox.value()})
        elif stimtype == 'Ramp':
            data = SafeDict({'tp': 'ramp',
                             'a': stim['Amplitude'],
                             'r': stim['Rate'],
                             'num': self.ui.nextFileNumberBox.value()})
            try:
                data['v'] = stim['Activation', 'Voltage']
            except Exception:
                pass
            try:
                data['v'] = stim['Activation', 'Stim voltage']
                data['s'] = stim['Activation', 'Stim side']
            except Exception:
                pass
        else:
            assert False

        # default formats for the different types
        fmt = {'f': '.2f',
               'a': '.0f',
               'ph': '02.0f',
               'lv': '.0f',
               'rv': '.0f',
               'v': '.0f',
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
                if stimtype in self.stimParameterDefs:
                    stimParamGroup = self.params.child('Stimulus', 'Parameters')
                    stimParamGroup.clearChildren()
                    stimParamGroup.addChildren(self.stimParameterDefs[stimtype])
                else:
                    assert False
                self.curStimType = stimtype

            self.readParameters(settings, self.params)
        finally:
            self.connectParameterSlots()

        settings.endGroup()

        try:
            self.updateOutputFrequency()
            self.changeStimType(self.params.child('Stimulus','Type'), self.params['Stimulus', 'Type'])
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
    print("Run the run_bender.py file instead")


if __name__ == '__main__':
    sys.exit(main())


