from __future__ import print_function, unicode_literals
import sys
import os
import string
import time
import logging
import numpy as np
from scipy import integrate, interpolate

from PyQt4 import QtCore

try:
    import PyDAQmx as daq
except ImportError:
    pass

from settings import SETTINGS_FILE, MOTOR_TYPE


class BenderDAQ(QtCore.QObject):
    sigUpdate = QtCore.Signal(np.ndarray, np.ndarray, np.ndarray)  ## analog input buffer, encoder input buffer
    sigDoneAcquiring = QtCore.Signal()

    def __init__(self):
        QtCore.QObject.__init__(self)

        self.params = None

        self.t = None
        self.pos = None
        self.vel = None
        self.duration = None
        self.digital_out = None

        if MOTOR_TYPE == 'stepper':
            self.make_motor_pulses = self.make_motor_stepper_pulses
        elif MOTOR_TYPE == 'velocity':
            self.make_motor_pulses = self.make_motor_velocity_pulses
        else:
            raise Exception("Unknown motor type %s", MOTOR_TYPE)

    def make_stimulus(self, parameters):
        self.params = parameters

        logging.debug('Stimulus.type = {}'.format(self.params['Stimulus', 'Type']))
        if self.params['Stimulus', 'Type'] == 'Sine':
            self.make_sine_stimulus()
        elif self.params['Stimulus', 'Type'] == 'Frequency Sweep':
            self.make_freqsweep_stimulus()
        elif self.params['Stimulus', 'Type'] == 'None':
            self.pos = None
            self.vel = None
            self.duration = self.params['Stimulus', 'Parameters', 'Duration']
            self.digital_out_data = None
            self.tout = None
            self.Lact = None
            self.Ract = None
            self.Lonoff = None
            self.Ronoff = None
        else:
            assert False

    def make_sine_stimulus(self):
        try:
            _ = self.params['Stimulus', 'Parameters', 'Cycles']
        except Exception:
            self.pos = None
            self.vel = None
            self.t = None
            return

        logging.debug('Stimulus.wait before = {}'.format(self.params['Stimulus', 'Wait before']))
        stim = self.params.child('Stimulus', 'Parameters')
        dur = self.params['Stimulus', 'Wait before'] + stim['Cycles'] / stim['Frequency'] + \
              self.params['Stimulus', 'Wait after']

        self.nupdates = int(np.ceil(dur * self.params['DAQ', 'Update rate']))
        dur = float(self.nupdates) / self.params['DAQ', 'Update rate']

        self.duration = dur

        t = np.arange(0.0, dur, 1 / self.params['DAQ', 'Input', 'Sampling frequency']) - \
               self.params['Stimulus', 'Wait before']

        pos = stim['Amplitude'] * np.sin(2 * np.pi * stim['Frequency'] * t)
        pos[t < 0] = 0
        pos[t > stim['Cycles'] / stim['Frequency']] = 0

        vel = 2.0 * np.pi * stim['Frequency'] * stim['Amplitude'] * \
              np.cos(2 * np.pi * stim['Frequency'] * t)
        vel[t < 0] = 0
        vel[t > stim['Cycles'] / stim['Frequency']] = 0

        tnorm = t * stim['Frequency']
        tnorm[t < 0] = -1
        tnorm[t > stim['Cycles'] / stim['Frequency']] = np.ceil(np.max(tnorm))

        # make activation
        actburstdur = stim['Activation','Duty']/100.0 / stim['Frequency']
        actburstdur = np.floor(actburstdur * stim['Activation','Pulse rate'] * 2) / (stim['Activation','Pulse rate'] * 2)
        actburstduty = actburstdur * stim['Frequency']

        actpulsephase = t[np.logical_and(t > 0, t < actburstdur)] * stim['Activation','Pulse rate']
        burst = (np.mod(actpulsephase, 1) < 0.5).astype(np.float)

        bendphase = tnorm - 0.25
        bendphase[tnorm == -1] = -1

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
                        Ronoff.append(np.array([tstart, tend]) + 0.5/stim['Frequency'])

                    np.place(Ractcmd, np.logical_and(bendphase >= c + 0.5 + actphase,
                                                     bendphase < c + 0.5 + actphase + actburstduty),
                             burst)

            Lactcmd = Lactcmd * stim['Activation','Left voltage'] / stim['Activation','Left voltage scale']
            Ractcmd = Ractcmd * stim['Activation','Right voltage'] / stim['Activation','Right voltage scale']

        # upsample analog out
        tout = np.arange(0, dur, 1.0 / self.params['DAQ', 'Output', 'Sampling frequency']) + t[0]

        Lacthi = interpolate.interp1d(t, Lactcmd, kind='linear', assume_sorted=True, bounds_error=False,
                                      fill_value=0.0)(tout)
        Racthi = interpolate.interp1d(t, Ractcmd, kind='linear', assume_sorted=True, bounds_error=False,
                                      fill_value=0.0)(tout)

        self.analog_out_data = np.row_stack((Lacthi, Racthi))
        self.digital_out_data = self.make_motor_pulses(t, pos, vel, tout)

        self.t = t
        self.pos = pos
        self.vel = vel
        self.tnorm = tnorm
        self.phase = np.mod(tnorm, 1)
        self.duration = dur

        self.tout = tout
        self.Lact = Lactcmd
        self.Ract = Ractcmd
        self.Lonoff = np.array(Lonoff)
        self.Ronoff = np.array(Ronoff)

    def make_freqsweep_stimulus(self):
        try:
            _ = self.params['Stimulus', 'Parameters', 'Duration']
        except Exception:
            self.pos = None
            self.vel = None
            self.t = None
            return

        stim = self.params.child('Stimulus', 'Parameters')

        dur = stim['Duration']

        totaldur = self.params['Stimulus', 'Wait before'] + stim['Duration'] + \
                   self.params['Stimulus', 'Wait after']
        self.duration = totaldur

        t = np.arange(0.0, totaldur, 1 / self.params['DAQ', 'Input', 'Sampling frequency']) - \
               self.params['Stimulus', 'Wait before']

        if stim['End frequency'] == stim['Start frequency']:
            # exponential sweep blows up if the frequencies are equal
            sweeptype = 'Linear'
        else:
            sweeptype = stim['Frequency change']

        if sweeptype == 'Exponential':
            lnk = 1/dur * (np.log(stim['End frequency']) - np.log(stim['Start frequency']))

            f = stim['Start frequency'] * np.exp(t * lnk)
            f[t < 0] = np.nan
            f[t > dur] = np.nan

            tnorm = 2*np.pi*stim['Start frequency'] * (np.exp(t * lnk) - 1)/lnk

            tnorm[t < 0] = -1
            tnorm[t > dur] = np.ceil(np.max(tnorm))

            logging.debug('stim children: {}'.format(stim.children()))
            a0 = stim['Start frequency'] ** stim['Frequency exponent']
            pos = stim['Amplitude']/a0 * (np.power(f, stim['Frequency exponent'])) * np.sin(tnorm)
            vel = stim['Amplitude']/a0 * np.exp(stim['Frequency exponent']*t*lnk) * lnk * \
                  (stim['Frequency exponent'] * np.sin(tnorm) +
                   2*np.pi/lnk * f * np.cos(tnorm))

        elif sweeptype == 'Linear':
            k = (stim['End frequency'] - stim['Start frequency']) / dur
            f = stim['Start frequency'] + k * t

            f[t < 0] = np.nan
            f[t > stim['Cycles'] / stim['Frequency']] = np.nan

            tnorm = 2*np.pi*(stim['Start frequency']*t + k/2 * np.power(t, 2))

            b = stim['Frequency exponent']
            a0 = stim['Start frequency'] ** b
            pos = stim['Amplitude']/a0 * np.power(f, b) * np.sin(tnorm)
            vel = stim['Amplitude']/a0 * (b * k * np.power(f, b-1) * np.sin(tnorm) +
                                          2*np.pi * np.power(f, b+1) * np.cos(tnorm))
        else:
            raise ValueError("Unrecognized frequency sweep type: {}".format(stim['Frequency change']))

        pos[t < 0] = 0
        pos[t > dur] = 0

        vel[t < 0] = 0
        vel[t > dur] = 0

        tnorm[t < 0] = -1
        tnorm[t > dur] = np.ceil(np.max(tnorm))

        if self.params['Stimulus', 'Wait after'] > 0.5:
            isramp = np.logical_and(t >= dur, t < dur+0.5)
            k = int((self.params['Stimulus', 'Wait before'] + dur) * self.params['DAQ', 'Input', 'Sampling frequency'])
        else:
            isramp = t >= totaldur - 0.5
            k = int((totaldur - 0.5) * self.params['DAQ', 'Input', 'Sampling frequency'])

        pend = pos[k]
        velend = (0 - pend) / 0.5

        ramp = pend + (t[isramp] - t[k])*velend

        np.place(vel, isramp, velend)
        np.place(vel, isramp, ramp)

        tout = np.arange(t[0], dur, 1.0 / self.params['DAQ', 'Output', 'Sampling frequency'])

        self.digital_out_data = self.make_motor_pulses(t, vel, tout)

        self.t = t
        self.pos = pos
        self.vel = vel
        self.tnorm = tnorm
        self.phase = np.mod(tnorm, 1)
        self.duration = totaldur

        self.tout = tout
        self.Lact = np.zeros_like(self.t)
        self.Ract = np.zeros_like(self.t)
        self.analog_out_data = np.zeros((2,len(tout)))

        self.Lonoff = np.array([])
        self.Ronoff = np.array([])

    def make_motor_stepper_pulses(self, t, pos, vel, tout):
        poshi = interpolate.interp1d(t, pos, kind='linear', assume_sorted=True, bounds_error=False,
                                     fill_value=0.0)(tout)
        velhi = interpolate.interp1d(t, vel, kind='linear', assume_sorted=True, bounds_error=False,
                                     fill_value=0.0)(tout)

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

        motorParams = self.params.child('Motor parameters')

        velfrac = velhi / (motorParams['Maximum speed'] / 60 * 360)
        if np.any(np.abs(velfrac) > 1):
            raise ValueError('Motion is too fast!')

        motorpulserate = np.abs(velfrac) * (motorParams['Maximum pulse frequency'] - motorParams['Minimum pulse frequency']) \
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

    def setup_channels(self):
        # analog input
        assert(self.duration is not None)

        self.analog_in = daq.Task()

        inputParams = self.params.child('DAQ', 'Input')
        outputParams = self.params.child('DAQ', 'Output')

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

        self.ninsamps = int(1.0/self.params['DAQ', 'Update rate'] * inputParams['Sampling frequency'])
        self.inputbufferlen = 2*self.ninsamps
        self.analog_in.CfgSampClkTiming("", inputParams['Sampling frequency'], daq.DAQmx_Val_Rising,
                                        daq.DAQmx_Val_ContSamps, self.inputbufferlen)

        # encoder input
        self.encoder_in = daq.Task()

        self.encoder_in.CreateCIAngEncoderChan(inputParams['Encoder'], 'encoder',
                                               daq.DAQmx_Val_X4, False,
                                               0, daq.DAQmx_Val_AHighBHigh,
                                               daq.DAQmx_Val_Degrees,
                                               inputParams['Counts per revolution'], 0, None)
        self.encoder_in.CfgSampClkTiming("ai/SampleClock", inputParams['Sampling frequency'], daq.DAQmx_Val_Rising,
                                         daq.DAQmx_Val_ContSamps, self.inputbufferlen)

        self.noutsamps = int(1.0/self.params['DAQ', 'Update rate'] * outputParams['Sampling frequency'])
        self.outputbufferlen = 2*self.noutsamps

        # split the output data into blocks of noutsamps
        assert(self.analog_out_data.shape[1] == self.noutsamps*self.nupdates)

        self.analog_out_buffers = []
        self.digital_out_buffers = []
        for i in range(self.nupdates):
            aobuf = np.zeros((self.analog_out_data.shape[0], self.noutsamps))
            aobuf[:, :] = self.analog_out_data[:, i*self.noutsamps + np.arange(self.noutsamps)]
            assert(aobuf.flags.c_contiguous)
            self.analog_out_buffers.append(aobuf)

            if self.digital_out_data is not None:
                dobuf = np.zeros((self.noutsamps,), dtype=np.uint8)
                dobuf[:] = self.digital_out_data[i * self.noutsamps + np.arange(self.noutsamps)]
                assert (dobuf.flags.c_contiguous)
                self.digital_out_buffers.append(dobuf)

        # write two additional buffers full of zeros
        aobuf = np.zeros((self.analog_out_data.shape[0], self.noutsamps))
        self.analog_out_buffers.append(aobuf)
        aobuf = np.zeros((self.analog_out_data.shape[0], self.noutsamps))
        self.analog_out_buffers.append(aobuf)

        if self.digital_out_data is not None:
            dobuf = np.zeros((self.noutsamps,), dtype=np.uint8)
            self.digital_out_buffers.append(dobuf)
            dobuf = np.zeros((self.noutsamps,), dtype=np.uint8)
            self.digital_out_buffers.append(dobuf)

        # analog output (stimulus)
        self.analog_out = daq.Task()
        aobyteswritten = daq.int32()

        self.analog_out.CreateAOVoltageChan(outputParams['Left stimulus'], 'Lstim',
                                            -10, 10, daq.DAQmx_Val_Volts, None)
        self.analog_out.CreateAOVoltageChan(outputParams['Right stimulus'], 'Rstim',
                                            -10, 10, daq.DAQmx_Val_Volts, None)

        # set the output sample frequency and number of samples to acquire
        self.analog_out.CfgSampClkTiming("", outputParams['Sampling frequency'], daq.DAQmx_Val_Rising,
                                         daq.DAQmx_Val_ContSamps, self.outputbufferlen)
        # make sure the output starts at the same time as the input
        self.analog_out.CfgDigEdgeStartTrig("ai/StartTrigger", daq.DAQmx_Val_Rising)

        # write the output data
        self.analog_out.SetWriteRegenMode(daq.DAQmx_Val_DoNotAllowRegen)
        self.analog_out.WriteAnalogF64(self.outputbufferlen, False, 10,
                                       daq.DAQmx_Val_GroupByChannel,
                                       np.column_stack(tuple(self.analog_out_buffers[0:2])),
                                       daq.byref(aobyteswritten), None)

        logging.debug('%d analog bytes written' % aobyteswritten.value)

        # digital output (motor)
        self.digital_out = daq.Task()
        dobyteswritten = daq.int32()

        self.digital_out.CreateDOChan(outputParams['Digital port'], '', daq.DAQmx_Val_ChanForAllLines)
        # use the analog output clock for digital output
        self.digital_out.CfgSampClkTiming("ao/SampleClock", outputParams['Sampling frequency'],
                                          daq.DAQmx_Val_Rising,
                                          daq.DAQmx_Val_ContSamps,
                                          self.outputbufferlen)

        # write the digital data
        self.digital_out.SetWriteRegenMode(daq.DAQmx_Val_DoNotAllowRegen)
        self.digital_out.WriteDigitalU8(self.outputbufferlen, False, 10,
                                        daq.DAQmx_Val_GroupByChannel,
                                        np.concatenate(tuple(self.digital_out_buffers[0:2])),
                                        daq.byref(dobyteswritten), None)

        logging.debug('%d digital bytes written' % dobyteswritten.value)

    def start(self):
        self.setup_channels()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)

        self.updateNum = 0

        # allocate input buffers
        self.t_buffer = np.reshape(self.t, (self.nupdates, -1))

        self.analog_in_data = np.zeros((self.nupdates, self.ninsamps, 6), dtype=np.float64)
        self.encoder_in_data = np.zeros((self.nupdates, self.ninsamps), dtype=np.float64)
        self.analog_in_buffer = np.zeros((6, self.ninsamps), dtype=np.float64)
        self.encoder_in_buffer = np.zeros((self.ninsamps,), dtype=np.float64)

        # start the digital and analog output tasks.  They won't
        # do anything until the analog input starts
        self.digital_out.StartTask()
        self.analog_out.StartTask()
        self.encoder_in.StartTask()
        self.analog_in.StartTask()

        self.timer.start(int(1000.0 / self.params['DAQ', 'Update rate']))

    def update(self):
        if self.updateNum < self.nupdates:
            aibytesread = daq.int32()
            encbytesread = daq.int32()
            aobyteswritten = daq.int32()
            dobyteswritten = daq.int32()

            interval = 1.0 / self.params['DAQ', 'Update rate']

            # read the input data
            try:
                self.analog_in.ReadAnalogF64(self.ninsamps, interval*0.1,
                                             daq.DAQmx_Val_GroupByChannel,
                                             self.analog_in_buffer, self.analog_in_buffer.size,
                                             daq.byref(aibytesread), None)
                self.analog_in_data[self.updateNum, :, :] = self.analog_in_buffer.T
                self.encoder_in.ReadCounterF64(self.ninsamps, interval*0.1, self.encoder_in_buffer,
                                               self.encoder_in_buffer.size, daq.byref(encbytesread), None)
                self.encoder_in_data[self.updateNum, :] = self.encoder_in_buffer

                logging.debug('%d: max = %f' % (self.updateNum, np.max(self.analog_out_buffers[self.updateNum+2])))
                self.analog_out.WriteAnalogF64(self.noutsamps, False, 10,
                                               daq.DAQmx_Val_GroupByChannel,
                                               self.analog_out_buffers[self.updateNum+2],
                                               daq.byref(aobyteswritten), None)
                self.digital_out.WriteDigitalU8(self.noutsamps, False, 10,
                                                daq.DAQmx_Val_GroupByChannel,
                                                self.digital_out_buffers[self.updateNum+2],
                                                daq.byref(dobyteswritten), None)
            except daq.DAQError as err:
                logging.debug('Error! {}'.format(err))
                self.abort()
                return

            self.sigUpdate.emit(self.t_buffer[0:self.updateNum+1, :], self.analog_in_data[0:self.updateNum+1, :, :],
                                self.encoder_in_data[0:self.updateNum+1, :])

            logging.debug('Read %d ai, %d enc' % (aibytesread.value, encbytesread.value))
            logging.debug('Wrote %d ao, %d dig' % (aobyteswritten.value, dobyteswritten.value))

            self.updateNum += 1
        else:
            self.analog_out.StopTask()
            self.digital_out.StopTask()
            self.analog_in.StopTask()
            self.encoder_in.StopTask()

            logging.debug('Stopping')
            self.timer.stop()
            self.timer.timeout.disconnect(self.update)

            self.analog_in_data = np.reshape(self.analog_in_data, (-1, 6))
            self.encoder_in_data = np.reshape(self.encoder_in_data, (-1,))

            del self.analog_in
            del self.encoder_in
            del self.analog_out
            del self.digital_out

            self.sigDoneAcquiring.emit()

    def abort(self):
        logging.debug('Aborting')
        self.timer.stop()
        self.timer.timeout.disconnect(self.update)

        # try to stop each of the tasks, and ignore any DAQ errors
        try:
            self.analog_out.StopTask()
        except daq.DAQError as err:
            logging.debug('Error stopping analog_out: {}'.format(err))

        try:
            self.digital_out.StopTask()
        except daq.DAQError as err:
            logging.debug('Error stopping analog_out: {}'.format(err))

        try:
            self.analog_in.StopTask()
        except daq.DAQError as err:
            logging.debug('Error stopping analog_out: {}'.format(err))

        try:
            self.encoder_in.StopTask()
        except daq.DAQError as err:
            logging.debug('Error stopping analog_out: {}'.format(err))

        self.analog_in_data = np.reshape(self.analog_in_data, (6, -1))
        self.encoder_in_data = np.reshape(self.encoder_in_data, (-1,))

        del self.analog_in
        del self.encoder_in
        del self.analog_out
        del self.digital_out

        self.sigDoneAcquiring.emit()
