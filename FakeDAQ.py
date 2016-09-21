from __future__ import print_function, unicode_literals
import logging
import h5py
import ctypes

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
        freq = None
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
        self.freq = freq

    def CfgDigEdgeStartTrig(self, triggername, edge):
        pass

    def SetWriteRegenMode(self, regenmode):
        pass

    def WriteAnalogF64(self, nsamps, start, timeout, grouping, data, byteswritten, empty):
        byteswritten.value = nsamps

    def WriteDigitalU8(self, nsamps, start, timout, grouping, data, byteswritten, empty):
        byteswritten.value = nsamps

    def ReadAnalogF64(self, nsamps, timeout, grouping, buffer, buffersize, bytesread, empty):
        bytesread.value = nsamps

    def ReadCounterF64(self, nsamps, timeout, buffer, buffersize, bytesread, empty):
        bytesread.value = nsamps

    def StartTask(self):
        pass

    def StopTask(self):
        pass

