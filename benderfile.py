from __future__ import print_function, unicode_literals
import sys
import os
import string
import h5py
import logging
import numpy as np
from scipy import integrate, interpolate

class BenderFile(object):
    datasetNames = {'X force': 'xForce',
                    'Y force': 'yForce',
                    'Z force': 'zForce',
                    'X torque': 'xTorque',
                    'Y torque': 'yTorque',
                    'Z torque': 'zTorque'}

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

        gcal = self.h5file.create_group('Calibrated')

        # save the output data
        gout = self.h5file.create_group('Output')
        gout.attrs['SampleFrequency'] = params['DAQ', 'Output', 'Sampling frequency']

        # save the parameters for generating the stimulus
        gout = self.h5file.create_group('NominalStimulus')
        gout.create_dataset('Position', data=bender.pos)
        gout.create_dataset('Velocity', data=bender.vel)
        gout.create_dataset('Phase', data=bender.phase)
        gout.create_dataset('t', data=bender.t)
        gout.create_dataset('tnorm', data=bender.tnorm)

        try:
            pertinfo = params.child('Stimulus', 'Perturbation')
            dset = gout.create_dataset('Perturbation', data=bender.pert)
            dset.attrs['Frequencies'] = bender.pertfreqs
            dset.attrs['Amplitudes'] = bender.pertamps
            dset.attrs['Phases'] = bender.pertphases
            dset.attrs['MaxAmp'] = pertinfo['Max amplitude']
            dset.attrs['MaxAmpUnits'] = pertinfo['Amplitude scale']
            dset.attrs['AmplitudeFrequencyExponent'] = pertinfo['Amplitude frequency exponent']
            dset.attrs['StartCycle'] = pertinfo['Start cycle']
            dset.attrs['StopCycle'] = pertinfo['Stop cycle']
            dset.attrs['RampCycles'] = pertinfo['Ramp duration']
        except Exception:
            pass

        # save the stimulus info, but not the activation parameters, because those are different between the
        # whole body rig and the ergometer setup
        stim = params.child('Stimulus', 'Parameters')
        if params['Stimulus', 'Type'] == 'Sine':
            gout.attrs['Amplitude'] = stim['Amplitude']
            gout.attrs['Frequency'] = stim['Frequency']
            gout.attrs['Cycles'] = stim['Cycles']
            gout.attrs['WaitPre'] = params['Stimulus', 'Wait before']
            gout.attrs['WaitPost'] = params['Stimulus', 'Wait after']
        elif params['Stimulus', 'Type'] == 'Frequency Sweep':
            gout.attrs['Amplitude'] = stim['Amplitude']
            gout.attrs['StartFrequency'] = stim['Start frequency']
            gout.attrs['EndFrequency'] = stim['End frequency']
            gout.attrs['FrequencyChange'] = stim['Frequency change']
            gout.attrs['FrequencyExponent'] = stim['Frequency exponent']
            gout.attrs['Duration'] = stim['Duration']
            gout.attrs['WaitPre'] = params['Stimulus', 'Wait before']
            gout.attrs['WaitPost'] = params['Stimulus', 'Wait after']
        elif params['Stimulus', 'Type'] == 'Ramp':
            gout.attrs['Amplitude'] = stim['Amplitude']
            gout.attrs['Rate'] = stim['Rate']
            gout.attrs['HoldDur'] = stim['Hold duration']

        # save the whole parameter tree, in case I change something and forget to add it above
        gparams = self.h5file.create_group('ParameterTree')
        self._writeParameters(gparams, params)

    def saveRawData(self, aidata, angdata, params):
        assert False

    def saveCalibratedData(self, data, calibration, params):
        assert False

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


