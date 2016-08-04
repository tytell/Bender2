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

        gcal = self.h5file.create_group('Calibrated')

        # save the output data
        gout = self.h5file.create_group('Output')
        gout.attrs['SampleFrequency'] = params['DAQ', 'Output', 'Sampling frequency']

        dset = gout.create_dataset('Lact', data=bender.Lact)
        dset.attrs['HardwareChannel'] = params['DAQ', 'Output', 'Left stimulus']
        try:
            dset.attrs['VoltScale'] = params['Stimulus', 'Parameters', 'Activation', 'Left voltage scale']
        except KeyError:
            pass

        dset = gout.create_dataset('Ract', data=bender.Ract)
        dset.attrs['HardwareChannel'] = params['DAQ', 'Output', 'Right stimulus']
        try:
            dset.attrs['VoltScale'] = params['Stimulus', 'Parameters', 'Activation', 'Right voltage scale']
        except KeyError:
            pass

        dset = gout.create_dataset('DigitalOut', data=bender.digital_out_data)
        dset.attrs['HardwareChannel'] = params['DAQ', 'Output', 'Digital port']

        # save the parameters for generating the stimulus
        gout = self.h5file.create_group('NominalStimulus')
        gout.create_dataset('Position', data=bender.pos)
        gout.create_dataset('Velocity', data=bender.vel)
        gout.create_dataset('Phase', data=bender.phase)

        stim = params.child('Stimulus', 'Parameters')
        if params['Stimulus', 'Type'] == 'Sine':
            gout.attrs['Amplitude'] = stim['Amplitude']
            gout.attrs['Frequency'] = stim['Frequency']
            gout.attrs['Cycles'] = stim['Cycles']
            gout.attrs['WaitPre'] = params['Stimulus', 'Wait before']
            gout.attrs['WaitPost'] = params['Stimulus', 'Wait after']

            gout.attrs['ActivationOn'] = stim['Activation', 'On']
            gout.attrs['ActivationDuty'] = stim['Activation', 'Duty']
            gout.attrs['ActivationStartPhase'] = stim['Activation', 'Phase']
            gout.attrs['ActivationStartCycle'] = stim['Activation', 'Start cycle']
            gout.attrs['ActivationPulseFreq'] = stim['Activation', 'Pulse rate']
            gout.attrs['Left voltage'] = stim['Activation', 'Left voltage']
            gout.attrs['Right voltage'] = stim['Activation', 'Right voltage']
        elif params['Stimulus', 'Type'] == 'Frequency Sweep':
            gout.attrs['Amplitude'] = stim['Amplitude']
            gout.attrs['StartFrequency'] = stim['Start frequency']
            gout.attrs['EndFrequency'] = stim['End frequency']
            gout.attrs['FrequencyChange'] = stim['Frequency change']
            gout.attrs['FrequencyExponent'] = stim['Frequency exponent']
            gout.attrs['Duration'] = stim['Duration']
            gout.attrs['WaitPre'] = params['Stimulus', 'Wait before']
            gout.attrs['WaitPost'] = params['Stimulus', 'Wait after']

        # save the whole parameter tree, in case I change something and forget to add it above
        gparams = self.h5file.create_group('ParameterTree')
        self._writeParameters(gparams, params)

    def saveRawData(self, aidata, encdata, params):
        gin = self.h5file.require_group('RawInput')

        for i, aichan in enumerate(['xForce', 'yForce', 'zForce', 'xTorque', 'yTorque', 'zTorque']):
            dset = gin.create_dataset(aichan, data=aidata[:, i])
            dset.attrs['HardwareChannel'] = params['DAQ', 'Input', aichan]

        dset = gin.create_dataset('Encoder', data=encdata)
        dset.attrs['HardwareChannel'] = params['DAQ', 'Input', 'Encoder']
        dset.attrs['CountsPerRev'] = params['DAQ', 'Input', 'Counts per revolution']

    def saveCalibratedData(self, data, calibration, params):
        gin = self.h5file.require_group('Calibrated')

        for i, aichan in enumerate(['xForce', 'yForce', 'zForce', 'xTorque', 'yTorque', 'zTorque']):
            dset = gin.create_dataset(aichan, data=data[:, i])

        gin.create_dataset('CalibrationMatrix', data=calibration)

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


