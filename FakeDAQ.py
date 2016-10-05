from __future__ import print_function, unicode_literals
import logging
import h5py
import ctypes
import numpy as np

DAQmx_Val_Cfg_Default = 0
DAQmx_Val_Volts = 0
DAQmx_Val_Rising = 0
DAQmx_Val_ContSamps = 0

DAQmx_Val_X4 = 0
DAQmx_Val_AHighBHigh = 0
DAQmx_Val_Degrees = 0

DAQmx_Val_DoNotAllowRegen = 0

DAQmx_Val_GroupByChannel = 0
DAQmx_Val_ChanForAllLines = 0

int32 = ctypes.c_int32

class DAQError(IOError):
    pass

def byref(c_obj):
    return c_obj

class Task(object):
    def __init__(self):
        dt = None
        t = 0
        pass

    def CreateAIVoltageChan(self, hardwarechan, name, cfg, rangelo, rangehi, units, scale):
        pass

    def CreateCIAngEncoderChan(self, hardwarechan, name, enctype, zeropin, empty, abtype, units,
                               countsperrev, empty2, empty3):
        pass

    def CreateAOVoltageChan(self, hardwarechan, name, rangelo, rangehi, units, scale):
        pass

    def CreateDOChan(self, hardwarechan, name, grouping):
        pass

    def CfgSampClkTiming(self, clockname, freq, edge, samptype, bufferlength):
        self.dt = 1.0/freq

    def CfgDigEdgeStartTrig(self, triggername, edge):
        pass

    def SetWriteRegenMode(self, regenmode):
        pass

    def WriteAnalogF64(self, nsamps, start, timeout, grouping, data, byteswritten, empty):
        byteswritten.value = nsamps

    def WriteDigitalU8(self, nsamps, start, timout, grouping, data, byteswritten, empty):
        byteswritten.value = nsamps

    def ReadAnalogF64(self, nsamps, timeout, grouping, buffer, buffersize, bytesread, empty):
        t1 = self.t + np.arange(nsamps)*self.dt
        nchan = buffer.shape[0]
        v = np.array([np.random.randn(nsamps) * np.power(np.sin(2*np.pi*t1/2 - 0.2*c), 4) for c in range(nchan)])
        np.copyto(buffer, v)
        bytesread.value = nsamps
        self.t += nsamps * self.dt

    def ReadCounterF64(self, nsamps, timeout, buffer, buffersize, bytesread, empty):
        t1 = self.t + np.arange(nsamps)*self.dt
        v = np.sin(2*np.pi*t1/2)
        np.copyto(buffer, v)

        bytesread.value = nsamps
        self.t += nsamps * self.dt

    def StartTask(self):
        self.t = 0

    def StopTask(self):
        pass

