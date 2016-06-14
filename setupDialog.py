from __future__ import print_function, unicode_literals
import sys
import os
import logging
from PyQt4 import QtGui, QtCore
from xml.etree import ElementTree
import numpy as np

import itertools

from setup_ui import Ui_SetupDialog
from settings import SETTINGS_FILE

class SetupDialog(QtGui.QDialog):
    def __init__(self):
        super(SetupDialog, self).__init__()

        self.initUI()

        self.filename = None
        self.curFileNum = None

    def initUI(self):
        ui = Ui_SetupDialog()
        ui.setupUi(self)
        self.ui = ui

        self.settingsWidgets = [('InputChannels', [ui.inputFrequencyBox,
                                                   ui.xForceEdit, ui.yForceEdit, ui.zForceEdit, ui.xTorqueEdit,
                                                   ui.yTorqueEdit, ui.zTorqueEdit,
                                                   ui.encoderChannelEdit, ui.countsPerRevBox]),
                                ('Calibration', [ui.calibrationBox_1_1, ui.calibrationBox_1_2, ui.calibrationBox_1_3,
                                                 ui.calibrationBox_1_4, ui.calibrationBox_1_5, ui.calibrationBox_1_6,
                                                 ui.calibrationBox_2_1, ui.calibrationBox_2_2, ui.calibrationBox_2_3,
                                                 ui.calibrationBox_2_4, ui.calibrationBox_2_5, ui.calibrationBox_2_6,
                                                 ui.calibrationBox_3_1, ui.calibrationBox_3_2, ui.calibrationBox_3_3,
                                                 ui.calibrationBox_3_4, ui.calibrationBox_3_5, ui.calibrationBox_3_6,
                                                 ui.calibrationBox_4_1, ui.calibrationBox_4_2, ui.calibrationBox_4_3,
                                                 ui.calibrationBox_4_4, ui.calibrationBox_4_5, ui.calibrationBox_4_6,
                                                 ui.calibrationBox_5_1, ui.calibrationBox_5_2, ui.calibrationBox_5_3,
                                                 ui.calibrationBox_5_4, ui.calibrationBox_5_5, ui.calibrationBox_5_6,
                                                 ui.calibrationBox_6_1, ui.calibrationBox_6_2, ui.calibrationBox_6_3,
                                                 ui.calibrationBox_6_4, ui.calibrationBox_6_5, ui.calibrationBox_6_6]),
                                ('Output', [ui.outputFrequencyBox, ui.leftStimEdit, ui.rightStimEdit, ui.digitalPortEdit]),
                                ('Motor', [ui.motorMaxSpeedBox, ui.motorMinPulseFreqBox, ui.motorMaxPulseFreqBox]),
                                ('Stimulus', [ui.stimulusTypeTab, ui.bendFrequencyBox, ui.bendAmplitudeBox,
                                              ui.bendCyclesBox, ui.actStartCycleBox, ui.stimPhaseBox, ui.stimPulseRateBox,
                                              ui.leftStimCheck, ui.leftVoltageBox, ui.leftVoltageScale,
                                              ui.rightStimCheck, ui.rightVoltageBox, ui.rightVoltageScale,
                                              ui.freqSweepStartFreqBox, ui.freqSweepEndFreqBox,
                                              ui.freqSweepTypeBox, ui.freqSweepDurationBox,
                                              ui.freqSweepAmplitudeBox, ui.freqSweepFreqExponentBox,
                                              ui.waitPreBox, ui.waitPostBox]),
                                ('Geometry', [ui.doutVertBox, ui.doutHorizBox, ui.dinBox, ui.dclampBox,
                                              ui.widthBox, ui.heightBox]),
                                ('FileOutput', [ui.outputPathEdit, ui.fileNamePatternEdit, ui.curFileNumberBox])]

        self.calibrationBoxes = [[self.ui.calibrationBox_1_1, self.ui.calibrationBox_1_2, self.ui.calibrationBox_1_3,
                                  self.ui.calibrationBox_1_4, self.ui.calibrationBox_1_5, self.ui.calibrationBox_1_6],
                                 [self.ui.calibrationBox_2_1, self.ui.calibrationBox_2_2, self.ui.calibrationBox_2_3,
                                  self.ui.calibrationBox_2_4, self.ui.calibrationBox_2_5, self.ui.calibrationBox_2_6],
                                 [self.ui.calibrationBox_3_1, self.ui.calibrationBox_3_2, self.ui.calibrationBox_3_3,
                                  self.ui.calibrationBox_3_4, self.ui.calibrationBox_3_5, self.ui.calibrationBox_3_6],
                                 [self.ui.calibrationBox_4_1, self.ui.calibrationBox_4_2, self.ui.calibrationBox_4_3,
                                  self.ui.calibrationBox_4_4, self.ui.calibrationBox_4_5, self.ui.calibrationBox_4_6],
                                 [self.ui.calibrationBox_5_1, self.ui.calibrationBox_5_2, self.ui.calibrationBox_5_3,
                                  self.ui.calibrationBox_5_4, self.ui.calibrationBox_5_5, self.ui.calibrationBox_5_6],
                                 [self.ui.calibrationBox_6_1, self.ui.calibrationBox_6_2, self.ui.calibrationBox_6_3,
                                  self.ui.calibrationBox_6_4, self.ui.calibrationBox_6_5, self.ui.calibrationBox_6_6]]
        self.setWindowTitle('Setup')
        self.ui.stackedWidget.setCurrentIndex(0)

        self.readSettings()

        self.ui.loadCalibrationButton.clicked.connect(self.loadCalibration)
        self.ui.nextButton.clicked.connect(self.nextPage)
        self.ui.backButton.clicked.connect(self.previousPage)

        self.ui.outputPathBrowseButton.clicked.connect(self.browseOutputPath)
        self.ui.fileNamePatternEdit.editingFinished.connect(self.fileNamePatternEdited)

    def loadCalibration(self):
        calfilename = QtGui.QFileDialog.getOpenFileName(self, "Choose calibration file", "",
                                                        "Calibration files (*.cal);;All files (*.*)")

        try:
            tree = ElementTree.parse(calfilename)
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

        for row, boxes in zip(mat, self.calibrationBoxes):
            for val, box in zip(row, boxes):
                box.setValue(val)

    def readSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        for groupname, widgets in self.settingsWidgets:
            settings.beginGroup(groupname)
            for widget in widgets:
                try:
                    v = settings.value(widget.objectName()).toBool()
                    widget.setChecked(v)
                    continue
                except AttributeError:
                    pass

                try:
                    widget.setText(settings.value(widget.objectName()).toString())
                    continue
                except AttributeError:
                    pass

                try:
                    v, _ = settings.value(widget.objectName()).toDouble()
                    widget.setValue(v)
                    continue
                except AttributeError:
                    pass

                try:
                    v, _ = settings.value(widget.objectName()).toInt()
                    widget.setCurrentIndex(v)
                    continue
                except AttributeError:
                    logging.warning('Could not read in settings for {}'.format(widget.objectName()))
            settings.endGroup()

    def writeSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        for groupname, widgets in self.settingsWidgets:
            settings.beginGroup(groupname)
            for widget in widgets:
                try:
                    v = widget.isChecked()
                    settings.setValue(widget.objectName(), v)
                    continue
                except AttributeError:
                    pass

                try:
                    settings.setValue(widget.objectName(), widget.value())
                    continue
                except AttributeError:
                    pass

                try:
                    v = widget.text()
                    settings.setValue(widget.objectName(), v)
                    continue
                except AttributeError:
                    pass

                try:
                    v = widget.currentIndex()
                    settings.setValue(widget.objectName(), v)
                    continue
                except AttributeError:
                    logging.warning('Could not save settings for {}'.format(widget.objectName()))

            settings.endGroup()

    def setCurrentPage(self, page):
        if page == self.ui.stackedWidget.count()-1:
            self.ui.nextButton.setText("Done")
        else:
            self.ui.nextButton.setText("Next")

        if page == 0:
            self.ui.backButton.setEnabled(False)
        else:
            self.ui.backButton.setEnabled(True)

        self.ui.stackedWidget.setCurrentIndex(page)

    def nextPage(self):
        nextindex = self.ui.stackedWidget.currentIndex()+1
        if nextindex < self.ui.stackedWidget.count():
            self.setCurrentPage(nextindex)
        else:
            self.accept()

    def previousPage(self):
        previndex = self.ui.stackedWidget.currentIndex()-1
        if previndex >= 0:
            self.setCurrentPage(previndex)

    def browseOutputPath(self):
        oldoutpath = self.ui.outputPathEdit.text()
        outpath = QtGui.QFileDialog.getExistingDirectory(self, "Choose output directory", directory=oldoutpath)

        if outpath:
            self.ui.outputPathEdit.setText(outpath)

    def fileNamePatternEdited(self):
        filepattern = self.ui.fileNamePatternEdit.text()
        filename = self.getFileName(filepattern)
        self.ui.exampleNameEdit.setText(filename)

    def getFileName(self, filepattern):
        data = {'f': self.ui.bendFrequencyBox.value(),
                'a': self.ui.bendAmplitudeBox.value(),
                'ph': self.ui.stimPhaseBox.value(),
                'lv': self.ui.leftVoltageBox.value(),
                'rv': self.ui.rightVoltageBox.value(),
                'f0': self.ui.freqSweepStartFreqBox.value(),
                'f1': self.ui.freqSweepEndFreqBox.value(),
                'num': self.ui.curFileNumberBox.value()}

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
            filepattern = filepattern.replace('{'+key1+'}', '{'+key1+':'+fmt1+'}')

        if not self.ui.leftStimCheck.isChecked():
            data['lv'] = 0
        if not self.ui.rightStimCheck.isChecked():
            data['rv'] = 0

        filename = filepattern.format(**data)

        filename = filename.replace('.', '_')

        return filename

    def getValues(self):
        self.input = {'frequency': self.ui.inputFrequencyBox.value(),
                      'xForce': self.ui.xForceEdit.text(),
                      'yForce': self.ui.yForceEdit.text(),
                      'zForce': self.ui.zForceEdit.text(),
                      'xTorque': self.ui.xTorqueEdit.text(),
                      'yTorque': self.ui.yTorqueEdit.text(),
                      'zTorque': self.ui.zTorqueEdit.text(),
                      'encoder': self.ui.encoderChannelEdit.text(),
                      'encoderCountsPerRev': self.ui.countsPerRevBox.value()}

        cal = []
        for widgets in self.calibrationBoxes:
            row = []
            for widget in widgets:
                row.append(widget.value())
            cal.append(row)
        self.calibration = np.array(cal)

        self.output = {'frequency': self.ui.outputFrequencyBox.value(),
                       'leftStim': self.ui.leftStimEdit.text(),
                       'rightStim': self.ui.rightStimEdit.text(),
                       'digitalPort': self.ui.digitalPortEdit.text(),
                       'motorMaxSpeed': self.ui.motorMaxSpeedBox.value(),
                       'motorMinFreq': self.ui.motorMinPulseFreqBox.value(),
                       'motorMaxFreq': self.ui.motorMaxPulseFreqBox.value()}

        if self.ui.stimulusTypeTab.currentIndex() == 0:
            self.stim = {'type': 'sine',
                         'frequency': self.ui.bendFrequencyBox.value(),
                         'amplitude': self.ui.bendAmplitudeBox.value(),
                         'cycles': self.ui.bendCyclesBox.value(),
                         'actStartCycle': self.ui.actStartCycleBox.value(),
                         'stimPhase': self.ui.stimPhaseBox.value(),
                         'isLeftStim': self.ui.leftStimCheck.isChecked(),
                         'leftStimVolts': self.ui.leftVoltageBox.value(),
                         'leftVoltScale': self.ui.leftVoltageScale.value(),
                         'isRightStim': self.ui.rightStimCheck.isChecked(),
                         'rightStimVolts': self.ui.rightVoltageBox.value(),
                         'rightVoltScale': self.ui.rightVoltageScale.value(),
                         'waitPre': self.ui.waitPreBox.value(),
                         'waitPost': self.ui.waitPostBox()}
        elif self.ui.stimulusTypeTab.currentIndex() == 1:
            if self.ui.freqSweepTypeBox.currentIndex() == 0:
                fstype = 'exponential'
            elif self.ui.freqSweepTypeBox.currentIndex() == 1:
                fstype = 'linear'
            else:
                assert False

            self.stim = {'type': 'frequencysweep',
                         'startfreq': self.ui.freqSweepStartFreqBox.value(),
                         'endfreq': self.ui.freqSweepEndFreqBox.value(),
                         'frequencySweepType': fstype,
                         'duration': self.ui.freqSweepDurationBox.value(),
                         'amplitude': self.ui.freqSweepAmplitudeBox.value(),
                         'frequencyExponent': self.ui.freqSweepFreqExponentBox.value(),
                         'waitPre': self.ui.waitPreBox.value(),
                         'waitPost': self.ui.waitPostBox()}

        self.setupGeometry = {'dOutVert': self.ui.doutVertBox.value(),
                              'dOutHoriz': self.ui.doutHorizBox.value(),
                              'dIn': self.ui.dinBox.value(),
                              'dClamp': self.ui.dclampBox.value(),
                              'width': self.ui.widthBox.value(),
                              'height': self.ui.heightBox.value()}

        self.outputFilePath = self.ui.outputPathEdit.text()
        self.outputFilePattern = self.ui.fileNamePatternEdit.text()
        self.outputFileName = self.getFileName(self.outputFilePattern)
        self.outputFileNumber = self.ui.curFileNumberBox.value()

    def accept(self):
        logging.debug('Got accept')
        self.getValues()

        self.ui.curFileNumberBox.setValue(self.outputFileNumber+1)

        self.writeSettings()

        super(SetupDialog, self).accept()
