from __future__ import print_function, unicode_literals
import sys
import os
import string
import time
import logging
import numpy as np
from scipy import integrate, interpolate
from copy import copy

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

        self.pert = None
        self.pertvel = None

        self.angle_in = None
        self.angle_in_data = None
        self.digital_out = None
        self.digital_out_data = None

        self.ninchannels = None

    def make_stimulus(self, parameters):
        self.params = parameters

        logging.debug('Stimulus.type = {}'.format(self.params['Stimulus', 'Type']))
        if self.params['Stimulus', 'Type'] == 'Sine':
            self.make_sine_stimulus()
        elif self.params['Stimulus', 'Type'] == 'Frequency Sweep':
            self.make_freqsweep_stimulus()
        elif self.params['Stimulus', 'Type'] == 'Ramp':
            self.make_ramp_stimulus()
        elif self.params['Stimulus', 'Type'] == 'None':
            self.make_zero_stimulus()
        else:
            assert False

    def make_zero_stimulus(self):
        try:
            _ = self.params['Stimulus', 'Parameters', 'Duration']
        except Exception:
            self.pos = None
            self.vel = None
            self.t = None
            return

        dur = self.params['Stimulus', 'Parameters', 'Duration']

        self.nupdates = int(np.ceil(dur * self.params['DAQ', 'Update rate']))
        dur = float(self.nupdates) / self.params['DAQ', 'Update rate']

        self.duration = dur

        t = np.arange(0.0, dur, 1 / self.params['DAQ', 'Input', 'Sampling frequency']) - \
               self.params['Stimulus', 'Wait before']
        self.t = t

        self.pos = None
        self.vel = None

        self.t = t
        self.tnorm = None
        self.phase = None
        self.duration = dur

        self.digital_out_data = None

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
        self.t = t

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

        self.make_perturbations()
        if self.pert is not None:
            pos += self.pert
            vel += self.pertvel

        # upsample analog out
        self.tout = np.arange(0, dur, 1.0 / self.params['DAQ', 'Output', 'Sampling frequency']) + t[0]

        self.make_motor_signal(t, pos, vel)

        self.t = t
        self.pos = pos
        self.vel = vel
        self.tnorm = tnorm
        self.phase = np.mod(tnorm, 1)
        self.duration = dur

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
        self.nupdates = int(np.ceil(dur * self.params['DAQ', 'Update rate']))

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

            tnorm = 2*np.pi*stim['Start frequency'] * (np.exp(t * lnk) - 1)/lnk

            tnorm[t < 0] = -1
            tnorm[t > dur] = np.ceil(np.max(tnorm))

            logging.debug('stim children: {}'.format(stim.children()))
            a0 = stim['Start frequency'] ** stim['Frequency exponent']
            pos = stim['Amplitude']/a0 * (np.power(f, stim['Frequency exponent'])) * np.sin(tnorm)
            vel = stim['Amplitude']/a0 * np.exp(stim['Frequency exponent']*t*lnk) * lnk * \
                  (stim['Frequency exponent'] * np.sin(tnorm) +
                   2*np.pi/lnk * f * np.cos(tnorm))

            f[t < 0] = np.nan
            f[t > dur] = np.nan

        elif sweeptype == 'Linear':
            k = (stim['End frequency'] - stim['Start frequency']) / dur
            f = stim['Start frequency'] + k * t

            f[t < 0] = np.nan
            f[t > stim['Duration']] = np.nan

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

        self.tout = np.arange(t[0], dur, 1.0 / self.params['DAQ', 'Output', 'Sampling frequency'])

        self.make_motor_signal(t, pos, vel)

        self.t = t
        self.pos = pos
        self.vel = vel
        self.tnorm = tnorm
        self.phase = np.mod(tnorm, 1)
        self.duration = totaldur

        self.Lact = np.zeros_like(self.t)
        self.Ract = np.zeros_like(self.t)
        self.analog_out_data = np.zeros((2,len(self.tout)))

        self.Lonoff = np.array([])
        self.Ronoff = np.array([])

    def make_ramp_stimulus(self):
        try:
            _ = self.params['Stimulus', 'Parameters', 'Rate']
        except Exception:
            self.pos = None
            self.vel = None
            self.t = None
            return

        stim = self.params.child('Stimulus', 'Parameters')

        holddur = stim['Hold duration']
        amp = stim['Amplitude']
        rate = stim['Rate']
        rampdur = amp / rate

        totaldur = self.params['Stimulus', 'Wait before'] + rampdur*2 + holddur + \
                   self.params['Stimulus', 'Wait after']

        self.nupdates = int(np.ceil(totaldur * self.params['DAQ', 'Update rate']))

        totaldur = float(self.nupdates) / self.params['DAQ', 'Update rate']
        self.duration = totaldur

        dt = 1 / self.params['DAQ', 'Input', 'Sampling frequency']
        t = np.arange(0.0, totaldur, dt) - self.params['Stimulus', 'Wait before']
        pos = np.zeros_like(t)
        vel = np.zeros_like(t)

        nramp = np.ceil(rampdur / dt) + 1
        ramp = np.linspace(0, amp, nramp, endpoint=True)

        np.place(pos, np.logical_and(t >= 0, t < rampdur), ramp)
        pos[np.logical_and(t >= rampdur, t < rampdur+holddur)] = amp
        np.place(pos, np.logical_and(t >= rampdur+holddur, t < rampdur+holddur+rampdur), amp-ramp)

        vel[np.logical_and(t >= 0, t < rampdur)] = rate
        vel[np.logical_and(t >= rampdur+holddur, t < rampdur+holddur+rampdur)] = -rate

        # upsample analog out
        self.tout = np.arange(0, totaldur, 1.0 / self.params['DAQ', 'Output', 'Sampling frequency']) + t[0]

        self.make_motor_signal(t, pos, vel)

        self.t = t
        self.pos = pos
        self.vel = vel
        self.tnorm = t
        self.phase = np.zeros_like(t)
        self.duration = totaldur

    def make_perturbations(self):
        self.pertfreqs = []
        self.pertamps = []
        self.pertphases = []

        if self.t is None:
            return

        t = self.t
        dt = t[1] - t[0]

        self.pert = np.zeros_like(t)
        self.pertvel = np.zeros_like(t)

        try:
            pertinfo = self.params.child('Stimulus', 'Perturbations', 'Parameters')
            basefreq = self.params['Stimulus', 'Parameters', 'Frequency']
            ncycles = self.params['Stimulus', 'Parameters', 'Cycles']
        except Exception:
            return

        if self.params['Stimulus', 'Perturbations', 'Type'] == 'None':
            return
        elif self.params['Stimulus', 'Perturbations', 'Type'] == 'Sines':
            try:
                freqstr = pertinfo['Frequencies']
                phasestr = pertinfo['Phases']
            except Exception:
                return

            try:
                freqs = np.array([float(f) for f in freqstr.split()])
                phases = np.array([float(p) for p in phasestr.split()])
            except ValueError:
                logging.warning('Could not parse perturbation frequency string')
                return

            if len(freqs) == 0:
                return

            if len(phases) < len(freqs):
                newphases = np.random.random(size=(len(freqs)-len(phases),))
                phases = np.append(phases, newphases)

                phasestr = ' '.join(['{:.3f}'.format(p) for p in phases])
                pertinfo['Phases'] = phasestr
            elif len(phases) > len(freqs):
                phases = phases[:len(freqs)]
                phasestr = ' '.join(['{:.3f}'.format(p) for p in phases])
                pertinfo['Phases'] = phasestr

            if pertinfo['Amplitude scale'] == '% fundamental':
                maxamp = pertinfo['Max amplitude']/100.0 * self.params['Stimulus', 'Parameters', 'Amplitude']
            else:
                maxamp = pertinfo['Max amplitude']

            amps = np.power(freqs, -pertinfo['Amplitude frequency exponent'])
            amps = amps / amps[0] * maxamp

            startcycle = pertinfo['Start cycle']
            stopcycle = pertinfo['Stop cycle']
            if stopcycle <= 0:
                stopcycle += ncycles
            rampcycles = pertinfo['Ramp cycles']
            rampsamples = int(rampcycles/basefreq / dt)

            for a, f, p in zip(amps, freqs, phases):
                pert1 = a * np.cos(2*np.pi*(f*t - p))
                self.pert += pert1

                pertvel1 = -2*np.pi*f*a * np.sin(2*np.pi*(f*t - p))
                self.pertvel += pertvel1

            pertramp = np.ones_like(t)

            phase = t*basefreq
            pertramp[phase < startcycle - rampcycles] = 0.0
            np.place(pertramp, np.logical_and(phase >= startcycle - rampcycles, phase < startcycle),
                     np.linspace(0.0, 1.0, rampsamples))
            np.place(pertramp, np.logical_and(phase > stopcycle, phase <= stopcycle + rampcycles),
                     np.linspace(1.0, 0.0, rampsamples))
            pertramp[phase >= stopcycle + rampcycles] = 0.0

            self.pert *= pertramp
            self.pertvel *= pertramp

            self.pertfreqs = freqs
            self.pertamps = amps
            self.pertphases = phases

        elif self.params['Stimulus', 'Perturbations', 'Type'] == 'Triangles':
            startcycle = pertinfo['Start cycle']
            amp = pertinfo['Amplitude']
            dur = pertinfo['Duration']
            reps = pertinfo['Repetitions']
            phase = pertinfo['Phase']
            gap = pertinfo['Delay in between']

            totaltridur = startcycle/basefreq + (dur + gap/basefreq)*reps
            if totaltridur > ncycles/basefreq:
                reps = int(np.floor((ncycles/basefreq - startcycle/basefreq) / (dur + gap/basefreq)))
                logging.warning('Only time to do {} triangles'.format(reps))

                if reps == 0:
                    return

            ntri2 = round(dur/2 / dt)
            tri = np.zeros((2*ntri2,))

            tri[:ntri2] = np.linspace(0, 1, ntri2)
            tri[ntri2:] = np.linspace(1, 0, ntri2)
            tri *= amp

            trivel = np.zeros_like(tri)
            trivel[:ntri2] = amp/(dur/2)
            trivel[ntri2:] = -amp/(dur/2)

            pert = np.zeros_like(t)
            pertvel = np.zeros_like(t)

            pertt = []
            for i in range(reps):
                ttri1 = startcycle/basefreq + phase/basefreq + gap/basefreq*i
                istri = np.logical_and(t >= ttri1 - dur/2, t < ttri1 + dur/2)

                np.place(pert, istri, tri)
                np.place(pertvel, istri, trivel)
                pertt.append(ttri1)

            self.pert = pert
            self.pertvel = pertvel
            self.pertt = pertt
            self.pertamps = [amp]

    def make_motor_signal(self, t, pos, vel):
        assert False

    def setup_input_channels(self):
        assert False

    def get_analog_output_names(self):
        return [], []

    def setup_channels(self):
        self.setup_input_channels()

        if self.analog_out_data is not None:
            outputParams = self.params.child('DAQ', 'Output')

            self.noutsamps = int(1.0/self.params['DAQ', 'Update rate'] * outputParams['Sampling frequency'])
            self.outputbufferlen = 2*self.noutsamps

            # split the output data into blocks of noutsamps
            padlen = self.noutsamps*self.nupdates - self.analog_out_data.shape[1]
            if padlen > 0:
                self.analog_out_data = np.pad(self.analog_out_data, ((0, 0), (0, padlen)), mode='edge')
                if self.digital_out_data is not None:
                    self.digital_out_data = np.pad(self.digital_out_data, ((0,), (padlen,)), mode='edge')
            elif padlen < 0:
                n = self.noutsamps*self.nupdates
                self.analog_out_data = self.analog_out_data[:, :n]
                if self.digital_out_data is not None:
                    self.digital_out_data = self.digital_out_data[:n]

            assert(self.analog_out_data.shape[1] == self.noutsamps*self.nupdates)

            if len(self.t) != self.nupdates*self.ninsamps:
                self.t = np.arange(self.nupdates*self.ninsamps) / self.params['DAQ', 'Input', 'Sampling frequency']

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
        if self.analog_out_data is not None:
            self.analog_out = daq.Task()
            aobyteswritten = daq.int32()

            aochans, aonames = self.get_analog_output_names()

            for aochan, aoname in zip(aochans, aonames):
                self.analog_out.CreateAOVoltageChan(aochan, aoname,
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
        else:
            self.analog_out = None

        # digital output (motor)
        if self.digital_out_data is not None:
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
        else:
            self.digital_out = None

    def start(self):
        self.setup_channels()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)

        self.updateNum = 0

        # allocate input buffers
        self.t_buffer = np.reshape(self.t, (self.nupdates, -1))

        self.analog_in_data = np.zeros((self.nupdates, self.ninsamps, self.ninchannels), dtype=np.float64)
        self.analog_in_buffer = np.zeros((self.ninchannels, self.ninsamps), dtype=np.float64)
        if self.angle_in is not None:
            self.angle_in_data = np.zeros((self.nupdates, self.ninsamps), dtype=np.float64)
            self.angle_in_buffer = np.zeros((self.ninsamps,), dtype=np.float64)

        # start the digital and analog output tasks.  They won't
        # do anything until the analog input starts
        if self.digital_out is not None:
            self.digital_out.StartTask()
        if self.analog_out is not None:
            self.analog_out.StartTask()
        if self.angle_in is not None:
            self.angle_in.StartTask()
        self.analog_in.StartTask()

        self.timer.start(int(1000.0 / self.params['DAQ', 'Update rate']))

    def update(self):
        if self.updateNum < self.nupdates:
            aibytesread = daq.int32()
            angbytesread = daq.int32()
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

                if self.angle_in is not None:
                    self.angle_in.ReadCounterF64(self.ninsamps, interval*0.1, self.angle_in_buffer,
                                                   self.angle_in_buffer.size, daq.byref(angbytesread), None)
                    if self.params['DAQ', 'Input', 'Sign convention'] == 'Left is negative':
                        angsgn = -1
                    else:
                        angsgn = 1

                    self.angle_in_data[self.updateNum, :] = angsgn*self.angle_in_buffer

                if self.analog_out is not None:
                    logging.debug('%d: max = %f' % (self.updateNum, np.max(self.analog_out_buffers[self.updateNum+2])))
                    self.analog_out.WriteAnalogF64(self.noutsamps, False, 10,
                                                   daq.DAQmx_Val_GroupByChannel,
                                                   self.analog_out_buffers[self.updateNum+2],
                                                   daq.byref(aobyteswritten), None)
                if self.digital_out is not None:
                    self.digital_out.WriteDigitalU8(self.noutsamps, False, 10,
                                                    daq.DAQmx_Val_GroupByChannel,
                                                    self.digital_out_buffers[self.updateNum+2],
                                                    daq.byref(dobyteswritten), None)
            except daq.DAQError as err:
                logging.debug('Error! {}'.format(err))
                self.abort()
                return

            if self.angle_in_data is not None:
                angleup = self.angle_in_data[0:self.updateNum+1, :]
            else:
                angleup = np.array([])

            self.sigUpdate.emit(self.t_buffer[0:self.updateNum+1, :], self.analog_in_data[0:self.updateNum+1, :, :],
                                angleup)

            logging.debug('Read %d ai, %d ang' % (aibytesread.value, angbytesread.value))
            logging.debug('Wrote %d ao, %d dig' % (aobyteswritten.value, dobyteswritten.value))

            self.updateNum += 1
        else:
            if self.analog_out is not None:
                self.analog_out.StopTask()
            if self.digital_out is not None:
                self.digital_out.StopTask()
            self.analog_in.StopTask()
            if self.angle_in is not None:
                self.angle_in.StopTask()

            logging.debug('Stopping')
            self.timer.stop()
            self.timer.timeout.disconnect(self.update)

            self.analog_in_data = np.reshape(self.analog_in_data, (-1, self.ninchannels))
            if self.angle_in_data is not None:
                self.angle_in_data = np.reshape(self.angle_in_data, (-1,))

            del self.analog_in
            del self.angle_in
            del self.analog_out
            del self.digital_out

            self.sigDoneAcquiring.emit()

    def abort(self):
        logging.debug('Aborting')
        self.timer.stop()
        self.timer.timeout.disconnect(self.update)

        # try to stop each of the tasks, and ignore any DAQ errors
        if self.analog_out is not None:
            try:
                self.analog_out.StopTask()
            except daq.DAQError as err:
                logging.debug('Error stopping analog_out: {}'.format(err))

        if self.digital_out is not None:
            try:
                self.digital_out.StopTask()
            except daq.DAQError as err:
                logging.debug('Error stopping analog_out: {}'.format(err))

        try:
            self.analog_in.StopTask()
        except daq.DAQError as err:
            logging.debug('Error stopping analog_out: {}'.format(err))

        if self.angle_in is not None:
            try:
                self.angle_in.StopTask()
            except daq.DAQError as err:
                logging.debug('Error stopping analog_out: {}'.format(err))

        self.analog_in_data = np.reshape(self.analog_in_data, (-1, self.ninchannels))
        if self.angle_in_data is not None:
            self.angle_in_data = np.reshape(self.angle_in_data, (-1,))

        del self.analog_in
        del self.angle_in
        del self.analog_out
        del self.digital_out

        self.sigDoneAcquiring.emit()

