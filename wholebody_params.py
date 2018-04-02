import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

from settings import SETTINGS_FILE, MOTOR_TYPE, COUNTER_TYPE, EXPERIMENT_TYPE


stimParameterDefs = {
    'None': [
        {'name': 'Duration', 'type': 'float', 'value': 5.0, 'step': 1.0, 'suffix': 'sec'}
    ],
    'Sine': [
        {'name': 'Amplitude', 'type': 'float', 'value': 15.0, 'step': 1.0, 'suffix': 'deg'},
        {'name': 'Frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'Cycles', 'type': 'int', 'value': 10},
        {'name': 'Activation', 'type': 'group', 'children': [
            {'name': 'On', 'type': 'bool', 'value': True},
            {'name': 'Start cycle', 'type': 'int', 'value': 3},
            {'name': 'Phase', 'type': 'float', 'value': 0.0, 'step': 10.0, 'suffix': '%'},
            {'name': 'Duty', 'type': 'float', 'value': 30.0, 'step': 10.0, 'suffix': '%'},
            {'name': 'Left voltage', 'type': 'float', 'value': 2.0, 'step': 1.0, 'suffix': 'V'},
            {'name': 'Left voltage scale', 'type': 'float', 'value': 1.0, 'step': 1.0, 'suffix': 'V/V'},
            {'name': 'Right voltage', 'type': 'float', 'value': 2.0, 'step': 1.0, 'suffix': 'V'},
            {'name': 'Right voltage scale', 'type': 'float', 'value': 0.4, 'step': 1.0, 'suffix': 'V/V'},
            {'name': 'Pulse rate', 'type': 'float', 'value': 75.0, 'step': 5.0, 'suffix': 'Hz'},
        ]}
    ],
    'Frequency Sweep': [
        {'name': 'Start frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'End frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'Frequency change', 'type': 'list', 'values': ['Exponential','Linear'], 'value': 'Exponential'},
        {'name': 'Duration', 'type': 'float', 'value': 300.0, 'suffix': 'sec'},
        {'name': 'Amplitude', 'type': 'float', 'value': 15.0, 'step': 1.0, 'suffix': 'deg'},
        {'name': 'Frequency exponent', 'type': 'float', 'value': 0.0, 'limits': (-1, 0)}
    ],
    'Ramp': [
        {'name': 'Amplitude', 'type': 'float', 'value': 10.0, 'step': 1.0, 'suffix': 'deg'},
        {'name': 'Rate', 'type': 'float', 'value': 50.0, 'step': 10.0, 'suffix': 'deg/s'},
        {'name': 'Hold duration', 'type': 'float', 'value': 2.0, 'step': 0.5, 'suffix': 'sec'},
        {'name': 'Activation', 'type': 'group', 'children': [
            {'name': 'During', 'type': 'list', 'values': ['Hold', 'Ramp'], 'value': 'Hold'},
            {'name': 'Duration', 'type': 'float', 'value': 0.3, 'step': 0.1, 'suffix': 'sec'},
            {'name': 'Delay', 'type': 'float', 'value': 0.05, 'step': 0.01, 'suffix': 'sec', 'siPrefix': True},
            {'name': 'Stim side', 'type': 'list', 'values': ['Left', 'Right'], 'value': 'Left'},
            {'name': 'Stim voltage', 'type': 'float', 'value': 2.0, 'step': 1.0, 'suffix': 'V'},
            {'name': 'Left voltage scale', 'type': 'float', 'value': 1.0, 'step': 1.0, 'suffix': 'V/V'},
            {'name': 'Right voltage scale', 'type': 'float', 'value': 0.4, 'step': 1.0, 'suffix': 'V/V'},
            {'name': 'Pulse rate', 'type': 'float', 'value': 75.0, 'step': 5.0, 'suffix': 'Hz'},
        ]}
    ]
}

velocityDriverParams = [
    {'name': 'Maximum speed', 'type': 'float', 'value': 400.0, 'step': 50.0, 'suffix': 'RPM'},
    {'name': 'Minimum pulse frequency', 'type': 'float', 'value': 1000.0, 'step': 100.0, 'siPrefix': True,
     'suffix': 'Hz'},
    {'name': 'Maximum pulse frequency', 'type': 'float', 'value': 5000.0, 'step': 100.0, 'siPrefix': True,
     'suffix': 'Hz'},
    {'name': 'Sign convention', 'type': 'list', 'values': ['Left is positive', 'Left is negative', 'None'],
     'value': 'Left is positive'}
]

stepperParams = [
    {'name': 'Steps per revolution', 'type': 'float', 'value': 6400},
    {'name': 'Scale factor', 'type': 'float', 'value': 6,
     'tip': 'Output speed reduction. Number of output teeth divided by number of input teeth'},
    {'name': 'Sign convention', 'type': 'list', 'values': ['Left is positive', 'Left is negative', 'None'],
     'value': 'Left is positive'}
]

encoderParams = [
    {'name': 'Encoder', 'type': 'str', 'value': 'Dev1/ctr0'},
    {'name': 'Counts per revolution', 'type': 'int', 'value': 10000, 'limits': (1, 100000)},
    {'name': 'Sign convention', 'type': 'list', 'values': ['Left is positive', 'Left is negative', 'None'],
     'value': 'Left is positive'}
]

pwmParams = [
    {'name': 'Pulse width counter', 'type': 'str', 'value': 'Dev1/ctr0'},
    {'name': 'Base frequency', 'type': 'float', 'value': 500, 'suffix': 'Hz'}
]

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
            {'name': 'Sampling frequency', 'type': 'float', 'value': 1000.0, 'step': 500.0, 'siPrefix': True,
             'suffix': 'Hz'},
            {'name': 'xForce', 'type': 'str', 'value': 'Dev1/ai0'},
            {'name': 'yForce', 'type': 'str', 'value': 'Dev1/ai1'},
            {'name': 'zForce', 'type': 'str', 'value': 'Dev1/ai2'},
            {'name': 'xTorque', 'type': 'str', 'value': 'Dev1/ai3'},
            {'name': 'yTorque', 'type': 'str', 'value': 'Dev1/ai4'},
            {'name': 'zTorque', 'type': 'str', 'value': 'Dev1/ai5'},
            {'name': 'Left stim', 'type': 'str', 'value': 'Dev1/ai6'},
            {'name': 'Right stim', 'type': 'str', 'value': 'Dev1/ai7'},
            {'name': 'Get calibration...', 'type': 'action'},
            {'name': 'Calibration file', 'type': 'str', 'readonly': True}
        ]},
        {'name': 'Output', 'type': 'group', 'children': [
            {'name': 'Sampling frequency', 'type': 'float', 'value': 10000.0, 'step': 1000.0, 'siPrefix': True,
             'suffix': 'Hz', 'readonly': True},
            {'name': 'Left stimulus', 'type': 'str', 'value': 'Dev1/ao0'},
            {'name': 'Right stimulus', 'type': 'str', 'value': 'Dev1/ao1'},
            {'name': 'Digital port', 'type': 'str', 'value': 'Dev1/port0'}
        ]},
        {'name': 'Update rate', 'type': 'float', 'value': 10.0, 'suffix': 'Hz'}
    ]},
    {'name': 'Motor parameters', 'type': 'group', 'children': []},
    {'name': 'Geometry', 'type': 'group', 'children': [
        {'name': 'doutvert', 'tip': 'Vertical distance from transducer to center of pressure', 'type': 'float',
         'value': 0.011, 'step': 0.001, 'siPrefix': True, 'suffix': 'm'},
        {'name': 'douthoriz', 'tip': 'Horizontal distance from transducer to center of pressure', 'type': 'float',
         'value': 0.0, 'step': 0.001, 'siPrefix': True, 'suffix': 'm'},
        {'name': 'din', 'tip': 'Horizontal distance from center of pressure to center of rotation', 'type': 'float',
         'value': 0.035, 'step': 0.001, 'siPrefix': True, 'suffix': 'm'},
        {'name': 'dclamp', 'tip': 'Horizontal distance between the edges of the clamps', 'type': 'float',
         'value': 0.030, 'step': 0.001, 'siPrefix': True, 'suffix': 'm'},
        {'name': 'Cross-section', 'type': 'group', 'children': [
            {'name': 'width', 'type': 'float', 'value': 0.021, 'step': 0.001, 'siPrefix': True, 'suffix': 'm'},
            {'name': 'height', 'type': 'float', 'value': 0.021, 'step': 0.001, 'siPrefix': True, 'suffix': 'm'},
        ]}
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


