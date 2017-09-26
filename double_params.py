import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

class ChannelGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add channel"
        pTypes.GroupParameter.__init__(self, **opts)

    def addNew(self):
        newname = "Channel {}".format(len(self.childs)+1)
        hwchan = "ai{}".format(len(self.childs))

        self.addChild(
            dict(name=hwchan, type='str', value=newname, removable=True, renamable=True))


stimParameterDefs = {
    'None': [
        {'name': 'Duration', 'type': 'float', 'value': 5.0, 'step': 1.0, 'suffix': 'sec'}
    ],
    'Sine': [
        {'name': 'Type', 'type': 'list', 'values': ['Rostral only', 'Caudal only', 'Same amplitude',
                                                    'Same frequency', 'Different frequency']},
        {'name': 'Rostral amplitude', 'type': 'float', 'value': 15.0, 'step': 1.0, 'suffix': 'deg'},
        {'name': 'Caudal amplitude', 'type': 'float', 'value': 15.0, 'step':1.0, 'suffix': 'deg'},
        {'name': 'Distance between', 'type': 'int', 'value': 20, 'step': 1, 'suffix': 'seg'},
        {'name': 'Phase offset', 'type': 'float', 'value': 0.0},
        {'name': 'Frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'Caudal frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'Cycles', 'type': 'int', 'value': 10},
    ],
    'Frequency Sweep': [
        {'name': 'Start frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'End frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'Frequency change', 'type': 'list', 'values': ['Exponential','Linear'], 'value': 'Exponential'},
        {'name': 'Frequency exponent', 'type': 'float', 'value': 0.0, 'limits': (-1, 0)},
        {'name': 'Duration', 'type': 'float', 'value': 300.0, 'suffix': 'sec'},
        {'name': 'Caudal amplitude', 'type': 'float', 'value': 15.0, 'step': 1.0, 'suffix': 'deg'},
        {'name': 'Rostral amplitude', 'type': 'float', 'value': 15.0, 'step':1.0, 'suffix': 'deg'},
        {'name': 'Distance between', 'type': 'int', 'value': 20, 'step': 1, 'suffix': 'seg'},
        {'name': 'Phase offset', 'type': 'float', 'value': 0.0},
    ]
}

velocityDriverParams = [
    {'name': 'Maximum speed', 'type': 'float', 'value': 400.0, 'step': 50.0, 'suffix': 'RPM'},
    {'name': 'Minimum pulse frequency', 'type': 'float', 'value': 1000.0, 'step': 100.0, 'siPrefix': True,
     'suffix': 'Hz'},
    {'name': 'Maximum pulse frequency', 'type': 'float', 'value': 5000.0, 'step': 100.0, 'siPrefix': True,
     'suffix': 'Hz'},
]

stepperParams = [
    {'name': 'Steps per revolution', 'type': 'float', 'value': 6400}
]

perturbationDefs = {
    'None': [],
    'Sines': [
        {'name': 'Location', 'type': 'list', 'values': ['Caudal', 'Rostral'], 'value': 'Caudal'},
        {'name': 'Start cycle', 'type': 'float', 'value': 4.0, 'step': 0.5},
        {'name': 'Stop cycle', 'type': 'float', 'value': 0.0, 'step': 0.5,
         'tip': 'Stop perturbations at cycle number. Negative numbers are cycles relative to the last one'},
        {'name': 'Ramp cycles', 'type': 'float', 'value': 0.5, 'step': 0.1,
         'tip': 'Ramp perturbations in over this period of time'},
            {'name': 'Max amplitude', 'type': 'float', 'value': 5.0, 'suffix': '%'},
        {'name': 'Amplitude scale', 'type': 'list', 'values': ['mm', '% fundamental'], 'value': '% fundamental'},
            {'name': 'Amplitude frequency exponent', 'type': 'float', 'value': 0.25, 'step': 0.25,
         'tip': 'Divide amplitudes by frequency to this exponent. 0 = no frequency scaling'},
            {'name': 'Frequencies', 'type': 'str', 'value': ''},
        {'name': 'Load frequencies...', 'type': 'action'},
            {'name': 'Phases', 'type': 'str', 'value': ''},
        {'name': 'Randomize phases...', 'type': 'action'},
    ],
    'Triangles': [
        {'name': 'Location', 'type': 'list', 'values': ['Caudal', 'Rostral'], 'value': 'Caudal'},
        {'name': 'Duration', 'type': 'float', 'value': 0.05, 'step': 10.0, 'suffix': 'sec', 'siPrefix': True},
        {'name': 'Amplitude', 'type': 'float', 'value': 0.2, 'step':0.05, 'suffix': 'mm'},
        {'name': 'Phase', 'type': 'float', 'value': 0.0, 'step': 0.05},
        {'name': 'Repetitions', 'type': 'int', 'value': 2},
        {'name': 'Start cycle', 'type': 'float', 'value': 1.0, 'step': 0.25, 'suffix': 'cycles'},
        {'name': 'Delay in between', 'type': 'float', 'value': 1.0, 'step': 0.25, 'suffix': 'cycles'},
    ]
}

parameterDefinitions = [
    {'name': 'DAQ', 'type': 'group', 'children': [
        {'name': 'Device name', 'type': 'str', 'value': 'Dev1'},
        {'name': 'Input', 'type': 'group', 'children': [
            {'name': 'Sampling frequency', 'type': 'float', 'value': 1000.0, 'step': 500.0, 'siPrefix': True,
             'suffix': 'Hz'},
            ChannelGroup(name="Channels", children=[]),
            {'name': 'Encoder 1', 'type': 'str', 'value': 'ctr0'},
            {'name': 'Encoder 2', 'type': 'str', 'value': 'ctr2'},
            {'name': 'Counts per revolution', 'type': 'int', 'value': 10000, 'limits': (1, 100000)}
        ]},
        {'name': 'Output', 'type': 'group', 'children': [
            {'name': 'Sampling frequency', 'type': 'float', 'value': 10000.0, 'step': 1000.0, 'siPrefix': True,
             'suffix': 'Hz', 'readonly': True},
            {'name': 'Digital port', 'type': 'str', 'value': 'port0'}
        ]},
        {'name': 'Update rate', 'type': 'float', 'value': 10.0, 'suffix': 'Hz'}
    ]},
    {'name': 'Motor parameters', 'type': 'group', 'children': stepperParams},
    {'name': 'Stimulus', 'type': 'group', 'children': [
        {'name': 'Type', 'type': 'list', 'values': ['None', 'Sine', 'Frequency Sweep'], 'value': 'Sine'},
        {'name': 'Parameters', 'type': 'group', 'children': stimParameterDefs['Sine']},
        {'name': 'Ramp duration', 'type': 'float', 'value': 0.5, 'suffix': 's'},
        {'name': 'Perturbations', 'type': 'group', 'children': [
            {'name': 'Type', 'type': 'list', 'values': ['None', 'Sines', 'Triangles'], 'value': 'None'},
            {'name': 'Parameters', 'type': 'group', 'children': perturbationDefs['None']},
        ]},
        {'name': 'Wait before', 'type': 'float', 'value': 1.0, 'suffix': 's'},
        {'name': 'Wait after', 'type': 'float', 'value': 1.0, 'suffix': 's'},
    ]}
]

def setup_parameters(params):
    pass
