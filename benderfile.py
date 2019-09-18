from __future__ import print_function, unicode_literals
import sys
import os
import string
import h5py
import logging
import numpy as np
from scipy import integrate, interpolate

class BenderFile(object):
    def __init__(self, filename, allowoverwrite=False):
        if not allowoverwrite and os.path.exists(filename):
            raise IOError('File exists')

        self.filename = filename
        if allowoverwrite:
            mode = 'w'
        else:
            mode = 'w-'
        self.h5file = h5py.File(filename, mode=mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def setupFile(self, bender, params):
        # save the input data
        gin = self.h5file.create_group('RawInput')
        gin.attrs['SampleFrequency'] = params['DAQ', 'Input', 'Sampling frequency']

        # save the output data
        gout = self.h5file.create_group('Output')
        gout.attrs['SampleFrequency'] = params['DAQ', 'Output', 'Sampling frequency']

        dset = gout.create_dataset('DigitalOut', data=bender.digital_out_data)
        dset.attrs['HardwareChannel'] = params['DAQ', 'Output', 'Digital port']

        # save the parameters for generating the stimulus
        gout = self.h5file.create_group('NominalStimulus')
        gout.create_dataset('RostralPosition', data=bender.pos[0])
        gout.create_dataset('RostralVelocity', data=bender.vel[0])
        gout.create_dataset('CaudalPosition', data=bender.pos[1])
        gout.create_dataset('CaudalVelocity', data=bender.vel[1])
        gout.create_dataset('Phase', data=bender.phase)
        gout.create_dataset('t', data=bender.t)
        gout.create_dataset('tnorm', data=bender.tnorm)

        stim = params.child('Stimulus', 'Parameters')
        if params['Stimulus', 'Type'] == 'Sine':
            gout.attrs['RostralAmplitude'] = stim['Rostral amplitude']
            gout.attrs['CaudalAmplitude'] = stim['Caudal amplitude']
            try:
                gout.attrs['Frequency'] = stim['Frequency']
            except Exception:
                gout.attrs['RostralFrequency'] = stim['Rostral frequency']
                gout.attrs['CaudalFrequency'] = stim['Caudal frequency']

            gout.attrs['PhaseOffset'] = stim['Phase offset']
            gout.attrs['Cycles'] = stim['Cycles']
            gout.attrs['WaitPre'] = params['Stimulus', 'Wait before']
            gout.attrs['WaitPost'] = params['Stimulus', 'Wait after']
            gout.attrs['RampDuration'] = params['Stimulus', 'Ramp duration']

        elif params['Stimulus', 'Type'] == 'Frequency Sweep':
            gout.attrs['RostralAmplitude'] = stim['Rostral amplitude']
            gout.attrs['CaudalAmplitude'] = stim['Caudal amplitude']
            gout.attrs['PhaseOffset'] = stim['Phase offset']
            gout.attrs['StartFrequency'] = stim['Start frequency']
            gout.attrs['EndFrequency'] = stim['End frequency']
            gout.attrs['FrequencyChange'] = stim['Frequency change']
            gout.attrs['FrequencyExponent'] = stim['Frequency exponent']
            gout.attrs['Duration'] = stim['Duration']
            gout.attrs['WaitPre'] = params['Stimulus', 'Wait before']
            gout.attrs['WaitPost'] = params['Stimulus', 'Wait after']

        gout.attrs['Perturbations'] = params['Stimulus', 'Perturbations', 'Type']
        if params['Stimulus', 'Perturbations', 'Type'] == 'Sines':
            pertinfo = params.child('Stimulus', 'Perturbations', 'Parameters')

            pout = gout.create_group('Perturbations')

            for name1, freqs, amps, phases in zip(['Rostral', 'Caudal'], bender.pertfreqs,
                                                  bender.pertamps, bender.pertphases):
                pout.attrs[name1+'Frequencies'] = ' '.join(['{:.3f}'.format(p) for p in freqs])
                pout.attrs[name1+'Amplitudes'] = ' '.join(['{:.3f}'.format(p) for p in amps])
                pout.attrs[name1+'Phases'] = ' '.join(['{:.3f}'.format(p) for p in phases])

            pout.attrs['StartCycle'] = pertinfo['Start cycle']
            pout.attrs['StopCycle'] = pertinfo['Stop cycle']
            pout.attrs['RampCycle'] = pertinfo['Ramp cycles']
            pout.attrs['MaxAmplitude'] = pertinfo['Max amplitude']
            pout.attrs['AmplitudeScale'] = pertinfo['Amplitude scale']
            pout.attrs['AmplitudeFreqExp'] = pertinfo['Amplitude frequency exponent']

            pout.create_dataset('Position', data=bender.pert)
            pout.create_dataset('Velocity', data=bender.pertvel)

        # save the whole parameter tree, in case I change something and forget to add it above
        gparams = self.h5file.create_group('ParameterTree')
        self._writeParameters(gparams, params)

    def saveRawData(self, aidata, encdata, params):
        gin = self.h5file.require_group('RawInput')

        channels = params.child('DAQ', 'Input', 'Channels').children()

        for i, chan1 in enumerate(channels):
            logging.debug('chan1.name() = {}, chan1.value() = {}'.format(chan1.name(), chan1.value()))
            dset = gin.create_dataset(chan1.value(), data=aidata[:, i])
            dset.attrs['HardwareChannel'] = chan1.name()

        dset = gin.create_dataset('Encoder1', data=encdata[:, 0])
        dset.attrs['HardwareChannel'] = params['DAQ', 'Input', 'Encoder 1']
        dset.attrs['CountsPerRev'] = params['DAQ', 'Input', 'Counts per revolution']

        dset = gin.create_dataset('Encoder2', data=encdata[:, 1])
        dset.attrs['HardwareChannel'] = params['DAQ', 'Input', 'Encoder 2']
        dset.attrs['CountsPerRev'] = params['DAQ', 'Input', 'Counts per revolution']

    def close(self):
        self.h5file.close()

    def _writeParameters(self, group, params):
        for ch in params:
            if ch.hasChildren():
                sub = group.create_group(ch.name())
                self._writeParameters(sub, ch)
            elif ch.type() in ['float', 'int']:
                try:
                    group.attrs.create(ch.name(), ch.value())
                except TypeError as err:
                    logging.debug("Error saving {} = {}: {}".format(ch.name(), ch.value(), err))
                    continue
            elif ch.type() in ['list', 'str']:
                try:
                    group.attrs.create(ch.name(), str(ch.value()))
                except TypeError as err:
                    logging.debug("Error saving {} = {}: {}".format(ch.name(), ch.value(), err))
                    continue


