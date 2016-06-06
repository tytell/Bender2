from __future__ import print_function, unicode_literals
import sys
import os
import logging
from PyQt4 import QtGui, QtCore
from xml.etree import ElementTree

import itertools

from setup_ui import Ui_SetupDialog
from settings import SETTINGS_FILE

class SetupDialog(QtGui.QDialog):
    def __init__(self):
        super(SetupDialog, self).__init__()

        self.initUI()

    def initUI(self):
        ui = Ui_SetupDialog()
        ui.setupUi(self)
        self.ui = ui

        self.settingsWidgets = [('InputChannels', [ui.xForceEdit, ui.yForceEdit, ui.zForceEdit, ui.xTorqueEdit,
                                                   ui.yTorqueEdit, ui.zTorqueEdit, ui.leftStimEdit, ui.rightStimEdit]),
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
                                ('Motor', [ui.motorMaxSpeedBox, ui.motorMinPulseFreqBox, ui.motorMaxPulseFreqBox]),
                                ('Stimulus', [ui.stimulusTypeTab, ui.bendFrequencyBox, ui.bendAmplitudeBox])]

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
                    widget.setText(settings.value(widget.objectName()).toString())
                except AttributeError:
                    try:
                        v, _ = settings.value(widget.objectName()).toDouble()
                        widget.setValue(v)
                    except AttributeError:
                        logging.warning('Could not read in settings for {}'.format(widget.objectName()))
            settings.endGroup()

        # settings.beginGroup("InputChannels")
        # self.ui.xForceEdit.setText(settings.value("xForce", "Dev1/ai0").toString())
        # self.ui.yForceEdit.setText(settings.value("yForce", "Dev1/ai1").toString())
        # self.ui.zForceEdit.setText(settings.value("zForce", "Dev1/ai2").toString())
        #
        # self.ui.xTorqueEdit.setText(settings.value("xTorque", "Dev1/ai3").toString())
        # self.ui.yTorqueEdit.setText(settings.value("yTorque", "Dev1/ai4").toString())
        # self.ui.zTorqueEdit.setText(settings.value("zTorque", "Dev1/ai5").toString())
        #
        # settings.beginReadArray("CalibrationMatrix")
        # for i, boxes in enumerate(self.calibrationBoxes):
        #     settings.setArrayIndex(i)
        #     settings.beginReadArray("row")
        #
        #     for j, box in enumerate(boxes):
        #         settings.setArrayIndex(j)
        #
        #         v, _ = settings.value("col", 0.0).toDouble()
        #         box.setValue(v)
        #     settings.endArray()
        #
        # settings.endArray()
        # settings.endGroup()
        #
        # settings.beginGroup("OutputChannels")
        # self.ui.leftStimEdit.setText(settings.value('LeftStim', 'Dev1/ao0').toString())
        # self.ui.rightStimEdit.setText(settings.value('RightStim', 'Dev1/ao1').toString())
        # settings.endGroup()
        #
        # settings.beginGroup("MotorParameters")
        # v, _ = settings.value("MaxSpeed", 400.0).toDouble()
        # self.ui.motorMaxSpeedBox.setValue(v)
        # v, _ = settings.value("MaxPulseFreq", 5.0).toDouble()
        # self.ui.motorMaxPulseFreqBox.setValue(v)
        # v, _ = settings.value("MinPulseFreq", "1.0").toDouble()
        # self.ui.motorMinPulseFreqBox.setValue(v)
        # settings.endGroup()

    def writeSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        for groupname, widgets in self.settingsWidgets:
            settings.beginGroup(groupname)
            for widget in widgets:
                try:
                    v = widget.value()
                except AttributeError:
                    try:
                        v = widget.text()
                    except AttributeError:
                        try:
                            v = widget.currentIndex()
                        except AttributeError:
                            logging.warning('Could not save settings for {}'.format(widget.objectName()))
                            continue

                settings.setValue(widget.objectName(), v)
            settings.endGroup()

        # settings.beginGroup('InputChannels')
        # settings.setValue('xForce', self.ui.xForceEdit.text())
        # settings.setValue('yForce', self.ui.yForceEdit.text())
        # settings.setValue('zForce', self.ui.zForceEdit.text())
        # settings.setValue('xTorque', self.ui.xTorqueEdit.text())
        # settings.setValue('yTorque', self.ui.yTorqueEdit.text())
        # settings.setValue('zTorque', self.ui.zTorqueEdit.text())
        #
        # settings.beginWriteArray("CalibrationMatrix", size=6)
        # for i, boxes in enumerate(self.calibrationBoxes):
        #     settings.setArrayIndex(i)
        #     settings.beginWriteArray("row", size=6)
        #     for j, box in enumerate(boxes):
        #         settings.setArrayIndex(j)
        #         settings.setValue("col", box.value())
        #     settings.endArray()
        # settings.endArray()
        # settings.endGroup()
        #
        # settings.beginGroup("MotorParameters")
        # settings.setValue("MaxSpeed", self.ui.motorMaxSpeedBox.value())
        # settings.setValue("MaxPulseFreq", self.ui.motorMaxPulseFreqBox.value())
        # settings.setValue("MinPulseFreq", self.ui.motorMinPulseFreqBox.value())
        # settings.endGroup()

    def nextPage(self):
        self.accept()

    def accept(self):
        logging.debug('Got accept')
        self.writeSettings()
        super(SetupDialog, self).accept()
