#%%
import numpy as np
import pandas as pd
import glob
import scipy.stats as st
from matplotlib import pyplot as plt

import potion.visualization.notebook_utils as notebook_utils
from experiments import plot_utils
from expsuite import PyExperimentSuite

mysuite = PyExperimentSuite(config='experiments.cfg')

exps = mysuite.get_exps('results')
for exp in exps:
    exp_params = mysuite.get_params(exp)

    plt.figure()
    plt.title(exp_params['name'])
    plt.ylabel('Deterministic Performance')
    plt.ylabel('Return')
    plt.xlabel('Trajectories')

    m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp, 'TestPerf')
    plt.plot(np.cumsum(mysuite.get_history(exp,0,'BatchSize')), m)
    plt.fill_between(np.cumsum(mysuite.get_history(exp,0,'BatchSize')), ci_lb, ci_ub, alpha=.1)

    plt.show()
