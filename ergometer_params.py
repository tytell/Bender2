import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

stimParameterDefs = {
    'None': [
        {'name': 'Duration', 'type': 'float', 'value': 5.0, 'step': 1.0, 'suffix': 'sec'}
    ],
    'Sine': [
        {'name': 'Amplitude', 'type': 'float', 'value': 15.0, 'step': 1.0, 'suffix': 'mm'},
        {'name': 'Frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'Cycles', 'type': 'int', 'value': 10},
        {'name': 'Activation', 'type': 'group', 'children': [
            {'name': 'On', 'type': 'bool', 'value': True},
            {'name': 'Start cycle', 'type': 'int', 'value': 3},
            {'name': 'Phase', 'type': 'float', 'value': 0.0, 'step': 10.0, 'suffix': '%'},
            {'name': 'Type', 'type': 'list', 'values': ['Sync pulse', 'Generate train'], 'value': 'Sync pulse'},
            {'name': 'Duty', 'type': 'float', 'value': 30.0, 'step': 10.0, 'suffix': '%'},
            {'name': 'Pulse rate', 'type': 'float', 'value': 75.0, 'step': 5.0, 'suffix': 'Hz'},
            {'name': 'Pulse duration', 'type': 'float', 'value': 0.001, 'step': 0.5, 'suffix': 'sec', 'siPrefix': True},
        ]}
    ],
    'Frequency Sweep': [
        {'name': 'Start frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'End frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'Frequency change', 'type': 'list', 'values': ['Exponential','Linear'], 'value': 'Exponential'},
        {'name': 'Duration', 'type': 'float', 'value': 300.0, 'suffix': 'sec'},
        {'name': 'Amplitude', 'type': 'float', 'value': 15.0, 'step': 1.0, 'suffix': 'mm'},
        {'name': 'Frequency exponent', 'type': 'float', 'value': 0.0, 'limits': (-1, 0)}
    ],
    'Ramp': [
        {'name': 'Amplitude', 'type': 'float', 'value': 10.0, 'step': 1.0, 'suffix': 'mm'},
        {'name': 'Rate', 'type': 'float', 'value': 50.0, 'step': 10.0, 'suffix': 'mm/s'},
        {'name': 'Hold duration', 'type': 'float', 'value': 2.0, 'step': 0.5, 'suffix': 'sec'},
        {'name': 'Activation', 'type': 'group', 'children': [
            {'name': 'During', 'type': 'list', 'values': ['Hold', 'Ramp'], 'value': 'Hold'},
            {'name': 'Duration', 'type': 'float', 'value': 0.3, 'step': 0.1, 'suffix': 'sec'},
            {'name': 'Delay', 'type': 'float', 'value': 0.05, 'step': 0.01, 'suffix': 'sec', 'siPrefix': True},
            {'name': 'Type', 'type': 'list', 'values': ['Sync pulse', 'Generate train'], 'value': 'Sync pulse'},
            {'name': 'Pulse rate', 'type': 'float', 'value': 75.0, 'step': 5.0, 'suffix': 'Hz'},
            {'name': 'Pulse duration', 'type': 'float', 'value': 0.001, 'step': 0.5, 'suffix': 'sec', 'siPrefix': True},
        ]}
    ]
}

perturbationDefs = {
    'None': [],
    'Sines': [
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
        {'name': 'Input', 'type': 'group', 'children': [
            {'name': 'Sampling frequency', 'type': 'float', 'value': 10000.0, 'step': 500.0, 'siPrefix': True,
             'suffix': 'Hz'},
            {'name': 'Length', 'type': 'str', 'value': 'Dev1/ai0'},
            {'name': 'Force', 'type': 'str', 'value': 'Dev1/ai1'},
            {'name': 'Force scale', 'type': 'float', 'value': 0.1, 'suffix': 'N/V'},
            {'name': 'Stim voltage', 'type': 'str', 'value': 'Dev1/ai2'},
            {'name': 'Stim voltage scale', 'type': 'float', 'value': 10, 'suffix': 'V/V',
             'tip': 'Stimulator output volts per volt on the monitor'},
            {'name': 'Stim current', 'type': 'str', 'value': 'Dev1/ai3'},
            {'name': 'Stim current scale', 'type': 'float', 'value': 0.1, 'suffix': 'A/V'},
            {'name': 'Sign convention', 'type': 'list',
             'values': ['Lengthening is positive', 'Lengthening is negative', 'None'],
             'value': 'Lengthening is positive'}
        ]},
        {'name': 'Output', 'type': 'group', 'children': [
            {'name': 'Sampling frequency', 'type': 'float', 'value': 10000.0, 'step': 1000.0, 'siPrefix': True,
             'suffix': 'Hz', 'readonly': False},
            {'name': 'Length', 'type': 'str', 'value': 'Dev1/ao0'},
            {'name': 'Stimulus', 'type': 'str', 'value': 'Dev1/ao1'},
        ]},
        {'name': 'Update rate', 'type': 'float', 'value': 10.0, 'suffix': 'Hz'}
    ]},
    {'name': 'Motor parameters', 'type': 'group', 'children': [
        {'name': 'Initial length', 'type': 'float', 'value': 7.0, 'suffix': 'mm', 'step': 0.01},
        {'name': 'Length scale', 'type': 'float', 'value': 0.5, 'suffix': 'mm/V'},
        {'name': 'Sign convention', 'type': 'list',
         'values': ['Lengthening is positive', 'Lengthening is negative', 'None'],
         'value': 'Lengthening is positive'}
    ]},
    {'name': 'Stimulus', 'type': 'group', 'children': [
        {'name': 'Type', 'type': 'list', 'values': ['None', 'Sine', 'Frequency Sweep', 'Ramp'], 'value': 'Sine'},
        {'name': 'Parameters', 'type': 'group', 'children': stimParameterDefs['Sine']},
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
