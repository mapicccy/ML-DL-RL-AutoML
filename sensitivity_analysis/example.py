from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
import math


problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[-math.pi, math.pi],
               [-math.pi, math.pi],
               [-math.pi, math.pi]],
}

params_values = saltelli.sample(problem, 1000)
# print(params_values.shape)

Y = Ishigami.evaluate(params_values)
Si = sobol.analyze(problem, Y, print_to_console=True)

print(Si['S1'])