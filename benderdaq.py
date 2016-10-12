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
    isfakedaq = False
except ImportError:
    import FakeDAQ as daq
    isfakedaq = True

from settings import SETTINGS_FILE, MOTOR_TYPE


def set_bits(arr, bit, val):
    """Set a particular bit in arr to equal val"""
    np.bitwise_or(arr, np.left_shift(val, bit), arr)


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

        rampdur = self.params['Stimulus', 'Ramp duration']
        before = self.params['Stimulus', 'Wait before'] + rampdur
        after = self.params['Stimulus', 'Wait after'] + rampdur

        logging.debug('Stimulus.wait before = {}'.format(self.params['Stimulus', 'Wait before']))
        stim = self.params.child('Stimulus', 'Parameters')
        dur = before + stim['Cycles'] / stim['Frequency'] + after

        self.nupdates = int(np.ceil(dur * self.params['DAQ', 'Update rate']))
        dur = float(self.nupdates) / self.params['DAQ', 'Update rate']

        self.duration = dur

        dt = 1 / self.params['DAQ', 'Input', 'Sampling frequency']
        t = np.arange(0.0, dur, dt) - before

        # generate rostral stimulus
        pos1 = stim['Rostral amplitude'] * np.sin(2 * np.pi * stim['Frequency'] * t)
        pos1[t < 0] = 0
        pos1[t > stim['Cycles'] / stim['Frequency']] = 0

        vel1 = 2.0 * np.pi * stim['Frequency'] * stim['Rostral amplitude'] * \
              np.cos(2 * np.pi * stim['Frequency'] * t)
        vel1[t < 0] = 0
        vel1[t > stim['Cycles'] / stim['Frequency']] = 0


        # generate caudal stimulus
        pos2 = stim['Caudal amplitude'] * np.sin(2 * np.pi * (stim['Frequency'] * t -
                                                              stim['Base phase offset'] -
                                                              stim['Additional phase offset']))
        p2start = stim['Caudal amplitude'] * np.sin(2 * np.pi * (-stim['Base phase offset'] -
                                                                 stim['Additional phase offset']))
        p2end = stim['Caudal amplitude'] * np.sin(2 * np.pi * (stim['Cycles'] - stim['Base phase offset'] -
                                                               stim['Additional phase offset']))

        # since it has a phase offset, it may not start at zero. Ramp up to the starting position and down to the
        # ending position
        np.place(pos2, np.logical_and(t >= -rampdur, t < 0),
                 np.linspace(0, p2start, int(round(rampdur/dt))))
        np.place(pos2, np.logical_and(t > stim['Cycles'] / stim['Frequency'],
                                      t <= stim['Cycles'] / stim['Frequency'] + rampdur),
                 np.linspace(p2end, 0, int(round(rampdur/dt))))

        pos2[t < -rampdur] = 0
        pos2[t > stim['Cycles'] / stim['Frequency'] + rampdur] = 0

        vel2 = 2.0 * np.pi * stim['Frequency'] * stim['Caudal amplitude'] * \
              np.cos(2 * np.pi * (stim['Frequency'] * t -
                                  stim['Base phase offset'] -
                                  stim['Additional phase offset']))
        np.place(vel2, np.logical_and(t >= -rampdur, t < 0), p2start / rampdur)
        np.place(vel2, np.logical_and(t > stim['Cycles'] / stim['Frequency'],
                                      t <= stim['Cycles'] / stim['Frequency'] + rampdur),
                 -p2end / rampdur)

        vel2[t < -rampdur] = 0
        vel2[t > stim['Cycles'] / stim['Frequency'] + rampdur] = 0

        tnorm = t * stim['Frequency']
        tnorm[t < 0] = -1
        tnorm[t > stim['Cycles'] / stim['Frequency']] = np.ceil(np.max(tnorm))

        tout = np.arange(0, dur, 1.0 / self.params['DAQ', 'Output', 'Sampling frequency']) + t[0]

        dig = self.make_motor_pulses(t, pos1, vel1, tout, bits=(1, 0, 2))
        self.make_motor_pulses(t, pos2, vel2, tout, out=dig, bits=(4, 3, 5))
        self.digital_out_data = dig

        self.t = t
        self.pos1 = pos1
        self.vel1 = vel1
        self.pos2 = pos2
        self.vel2 = vel2
        self.tnorm = tnorm
        self.phase = np.mod(tnorm, 1)
        self.duration = dur

        self.tout = tout

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

    def make_motor_stepper_pulses(self, t, pos, vel, tout, out=None, bits=(1, 0, 2)):
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

        if out is not None:
            dig = out
        else:
            dig = np.zeros_like(motorstep, dtype=np.uint32)
        set_bits(dig, bits[0], motorstep)
        set_bits(dig, bits[1], motordirection)
        set_bits(dig, bits[2], motorenable)

        if out is None:
            return dig

    def make_motor_velocity_pulses(self, t, pos, vel, tout, out=None, bits=(1, 0, 2)):

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

        if out is not None:
            dig = out
        else:
            dig = np.zeros_like(motorpulses, dtype=np.uint32)
        set_bits(dig, bits[0], motorpulses)
        set_bits(dig, bits[1], motordirection)
        set_bits(dig, bits[2], motorenable)

        if out is None:
            return dig

    def setup_channels(self):
        # analog input
        assert(self.duration is not None)

        self.analog_in = daq.Task()

        inputParams = self.params.child('DAQ', 'Input')
        outputParams = self.params.child('DAQ', 'Output')

        channels = self.params.child('DAQ', 'Input', 'Channels').children()

        for chan in channels:
            logging.debug('chan.name() = {}, chan.value() = {}'.format(chan.name(), chan.value()))
            self.analog_in.CreateAIVoltageChan(chan.name(), chan.value(), daq.DAQmx_Val_Cfg_Default,
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
        assert(self.digital_out_data.shape[0] == self.noutsamps*self.nupdates)

        self.digital_out_buffers = []
        for i in range(self.nupdates):
            dobuf = np.zeros((self.noutsamps,), dtype=np.uint8)
            dobuf[:] = self.digital_out_data[i * self.noutsamps + np.arange(self.noutsamps)]
            assert (dobuf.flags.c_contiguous)
            self.digital_out_buffers.append(dobuf)

        # write two additional buffers full of zeros
        dobuf = np.zeros((self.noutsamps,), dtype=np.uint8)
        self.digital_out_buffers.append(dobuf)
        dobuf = np.zeros((self.noutsamps,), dtype=np.uint8)
        self.digital_out_buffers.append(dobuf)

        # digital output (motor)
        self.digital_out = daq.Task()
        dobyteswritten = daq.int32()

        self.digital_out.CreateDOChan(outputParams['Digital port'], '', daq.DAQmx_Val_ChanForAllLines)
        # use the built in clock for digital output
        # TODO - figure out which clock to use here
        self.digital_out.CfgSampClkTiming("ao/SampleClock", outputParams['Sampling frequency'],
                                          daq.DAQmx_Val_Rising,
                                          daq.DAQmx_Val_ContSamps,
                                          self.outputbufferlen)
        # make sure the output starts at the same time as the input
        self.digital_out.CfgDigEdgeStartTrig("ai/StartTrigger", daq.DAQmx_Val_Rising)

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

        ninchan = len(self.params.child('DAQ', 'Input', 'Channels').children())
        self.analog_in_data = np.zeros((self.nupdates, self.ninsamps, ninchan), dtype=np.float64)
        self.encoder_in_data = np.zeros((self.nupdates, self.ninsamps), dtype=np.float64)
        self.analog_in_buffer = np.zeros((ninchan, self.ninsamps), dtype=np.float64)
        self.encoder_in_buffer = np.zeros((self.ninsamps,), dtype=np.float64)

        # start the digital and analog output tasks.  They won't
        # do anything until the analog input starts
        self.digital_out.StartTask()
        self.encoder_in.StartTask()
        self.analog_in.StartTask()

        delay = int(1000.0 / self.params['DAQ', 'Update rate'])
        if isfakedaq:
            delay = 1
        self.timer.start(delay)

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

                self.digital_out.WriteDigitalU8(self.noutsamps, False, 10,
                                                daq.DAQmx_Val_GroupByChannel,
                                                self.digital_out_buffers[self.updateNum+2],
                                                daq.byref(dobyteswritten), None)
            except daq.DAQError as err:
                logging.debug('Error! {}'.format(err))
                self.abort()
                return

            self.sigUpdate.emit(self.t_buffer[:self.updateNum+1, :], self.analog_in_data[:self.updateNum+1, :, :],
                                self.encoder_in_data[:self.updateNum+1, :])

            logging.debug('Read %d ai, %d enc' % (aibytesread.value, encbytesread.value))
            logging.debug('Wrote %d ao, %d dig' % (aobyteswritten.value, dobyteswritten.value))

            self.updateNum += 1
        else:
            self.digital_out.StopTask()
            self.analog_in.StopTask()
            self.encoder_in.StopTask()

            logging.debug('Stopping')
            self.timer.stop()
            self.timer.timeout.disconnect(self.update)

            ninchan = len(self.params.child('DAQ', 'Input', 'Channels').children())
            self.analog_in_data = np.reshape(self.analog_in_data, (-1, ninchan))
            self.encoder_in_data = np.reshape(self.encoder_in_data, (-1,))

            del self.analog_in
            del self.encoder_in
            del self.digital_out

            self.sigDoneAcquiring.emit()

    def abort(self):
        logging.debug('Aborting')
        self.timer.stop()
        self.timer.timeout.disconnect(self.update)

        # try to stop each of the tasks, and ignore any DAQ errors
        try:
            self.digital_out.StopTask()
        except daq.DAQError as err:
            logging.debug('Error stopping digital_out: {}'.format(err))

        try:
            self.analog_in.StopTask()
        except daq.DAQError as err:
            logging.debug('Error stopping analog_in: {}'.format(err))

        try:
            self.encoder_in.StopTask()
        except daq.DAQError as err:
            logging.debug('Error stopping encoder_in: {}'.format(err))

        ninchan = len(self.params.child('DAQ', 'Input', 'Channels').children())
        self.analog_in_data = np.reshape(self.analog_in_data, (-1, ninchan))
        self.encoder_in_data = np.reshape(self.encoder_in_data, (-1,))

        del self.analog_in
        del self.encoder_in
        del self.digital_out

        self.sigDoneAcquiring.emit()
