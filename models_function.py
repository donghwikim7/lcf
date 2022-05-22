import numpy as np
import scipy as sp
from scipy.optimize import basinhopping

class ModelsFunction:
    def __init__(self, model_id):
        self.id = model_id
        self.model_func = self.model_function
        self.n_param = 1 if model_id == 'last1' else (2 if model_id in ['pow2', 'log2', 'exp2', 'lin2', 'ilog2'] else
                                                      (3 if model_id in ['pow3', 'exp3', 'vap3', 'expp3', 'expd3',
                                                                         'logpower3'] else
                                                       (4 if model_id in ['mmf4', 'wbl4', 'exp4', 'pow4'] else 0)))


    def model_function(self, beta, x):
        match self.id:
            case 'pow2':
                return -beta[0] * x ** (-beta[1])
            case 'pow3':
                return beta[0] - beta[1] * x ** (-beta[2])
            case 'log2':
                return -beta[0] * np.log(x) + beta[1]
            case 'pow4':
                return beta[0] - beta[1] * (x + beta[3]) ** (-beta[2])
            case 'exp3':
                return beta[2] + beta[0] * np.exp(-beta[1] * x)
            case 'exp2':
                return beta[0] * np.exp(- beta[1] * x)
            case 'lin2':
                return beta[0] * x * 1.0 + beta[1]
            case 'vap3':
                return np.exp(beta[0] + beta[1] / x + beta[2] * np.log(x))
            case 'mmf4':
                return (beta[0] * beta[1] + beta[2] * x ** beta[3]) / (beta[1] + x ** beta[3])
            case 'wbl4':
                return beta[2] - beta[1] * np.exp((-1.0 *  beta[0] * x ** beta[3]))
            case 'exp4':
                return beta[2]  - np.exp(-beta[0] * ((x) ** beta[3]) + beta[1])
            case 'expp3':
                # fun = lambda x: a * np.exp(-b*x) + c
                return beta[2] - np.exp(((x - beta[1]) ** beta[0]))
            case 'ilog2':
                return beta[1] - (beta[0] / np.log(x*10000))
            case 'expd3':
                return beta[2] - (beta[2] - beta[0]) * np.exp(-beta[1] * x)
            case 'logpower3':
                return beta[0] / (1 + (x / np.exp(beta[1])) ** beta[2])
            case 'last1':
                return (beta[0] + x) - x
            case _:
                return None
