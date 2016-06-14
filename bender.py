from __future__ import print_function, unicode_literals
import sys
import os
import logging
import random
from PyQt4 import QtGui, QtCore
import numpy as np
from scipy import integrate, interpolate

from matplotlib.backends import qt_compat

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import PyDAQmx as daq

from setupDialog import SetupDialog
from settings import SETTINGS_FILE

assert qt_compat.QT_API == qt_compat.QT_API_PYQT

class BenderDAQ(object):
    def __init__(self, input, output, stim, geometry, outfilename):
        self.input = input
        self.output = output
        self.stim = stim
        self.geometry = geometry
        self.outfilename = outfilename

        self.t = None
        self.pos = None
        self.vel = None
        self.digital_out = None

    def make_stimulus(self):
        if self.stim.type == 'sine':
            self.make_sine_stimulus()
        elif self.stim.type == 'frequencySweep':
            self.make_freqsweep_stimulus()
        else:
            assert False

    def make_sine_stimulus(self):
        dur = self.stim.waitPre + self.stim.cycles/self.stim.frequency + \
            self.stim.waitPost
        self.duration = dur

        tout = np.arange(0.0,dur, 1/self.output.frequency) - self.stim.waitPre
        
        pos = self.stim.amplitude * np.sin(2*np.pi*self.stim.frequency*tout)
        pos[tout < 0] = 0
        pos[tout > self.stim.cycles/self.stim.frequency] = 0

        vel = 2.0*np.pi*self.stim.frequency * self.stim.amplitude * \
              np.sin(2 * np.pi * self.stim.frequency * tout)
        vel[tout < 0] = 0
        vel[tout > self.stim.cycles / self.stim.frequency] = 0

        phase = tout * self.stim.frequency
        phase[tout < 0] = -1
        phase[tout > self.stim.cycles / self.stim.frequency] = -1

        self.digital_out_data = self.make_motor_pulses(tout, vel)

        # make activation
        actburstdur = self.stim.actDuty / self.stim.frequency
        actburstdur = np.floor(actburstdur * self.stim.actPulseRate * 2) / (self.stim.actPulseRate * 2)
        actburstduty = actburstdur * self.stim.frequency

        actpulsephase = tout[np.logical_and(tout > 0, tout < actburstdur)] * self.stim.actPulseRate
        burst = (np.mod(actpulsephase, 1) < 0.5).astype(np.float)

        bendphase = phase - 0.25
        bendphase[phase == -1] = -1

        Lactcmd = np.zeros_like(tout)
        Ractcmd = np.zeros_like(tout)

        for c in range(int(self.stim.actStartCycle), int(self.stim.cycles)):
            tstart = (c - 0.25 + self.stim.actPhase) / self.stim.frequency
            tend = tstart + actburstdur
            np.place(Lactcmd, np.logical_and(bendphase >= c + self.stim.actPhase,
                                             bendphase < c + self.stim.actPhase + actburstduty),
                     burst)

            np.place(Ractcmd, np.logical_and(bendphase >= c + 0.5 + self.stim.actPhase,
                                             bendphase < c + 0.5 + self.stim.actPhase + actburstduty),
                     burst)

        Lactcmd = Lactcmd * self.stim.leftStimVolts / self.stim.leftVoltScale
        Ractcmd = Ractcmd * self.stim.rightStimVolts / self.stim.rightVoltScale

        self.analog_out_data = np.row_stack((Lactcmd, Ractcmd))

        self.t = np.arange(0.0,dur, 1/self.input.frequency) - self.stim.waitPre
        self.pos = interpolate.interp1d(tout, pos, assume_sorted=True)(self.t)
        self.vel = interpolate.interp1d(tout, vel, assume_sorted=True)(self.t)

        self.Lact = interpolate.interp1d(tout, Lactcmd, assume_sorted=True)(self.t)
        self.Ract = interpolate.interp1d(tout, Ractcmd, assume_sorted=True)(self.t)

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
                                  -10,10, daq.DAQmx_Val_Volts,None)
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

        nsamps =  int(self.duration * self.input.frequency)
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

        nsamps =  int(self.duration * self.input.frequency)

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

class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        # self.compute_initial_figure()

        #
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class BenderWindow(QtGui.QWidget):
    def __init__(self):
        super(BenderWindow, self).__init__()

        doneButton = QtGui.QPushButton("Done")
        doneButton.clicked.connect(self.close)

        self.forceAxes = MplCanvas(self)
        self.torqueAxes = MplCanvas(self)
        grid = QtGui.QGridLayout()
        grid.addWidget(self.forceAxes, 0,0)
        grid.addWidget(self.torqueAxes, 0,1)

        self.forceAxes.axes.plot([0, 1, 2, 3],[1, 2, 5, 1],'ro-')
        self.forceAxes.axes.set_ylabel('Force (N)')
        self.torqueAxes.axes.plot([0, 1, 2, 3],[8, 1, 2, 1],'gs-')
        self.torqueAxes.axes.set_ylabel('Torque (N m)')

        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(doneButton)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(grid)
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.readSettings()

    def readSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        settings.beginGroup("BenderWindow")
        self.resize(settings.value("size", QtCore.QSize(800, 600)).toSize())
        self.move(settings.value("position", QtCore.QSize(200, 200)).toPoint())
        settings.endGroup()

    def writeSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        logging.debug('Writing settings!')

        settings.beginGroup("BenderWindow")
        settings.setValue("size", self.size())
        settings.setValue("position", self.pos())
        settings.endGroup()

    def start(self):
        setup = SetupDialog()
        if not setup.exec_():
            return False

        self.show()
        return True

    def closeEvent(self, event):
        self.writeSettings()
        event.accept()


def main():
    logging.basicConfig(level=logging.DEBUG)
    app = QtGui.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    settings = QtCore.QSettings('bender.ini', QtCore.QSettings.IniFormat)

    bw = BenderWindow()
    if bw.start():
        return app.exec_()
    else:
        return False

if __name__ == '__main__':
    sys.exit(main())

