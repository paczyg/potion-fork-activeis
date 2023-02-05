#%%
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt

from experiments import plot_utils
from expsuite import PyExperimentSuite

mysuite = PyExperimentSuite(config='experiments.cfg')

exps = mysuite.get_exps('results/onpolicy')
for exp in exps:
    exp_params = mysuite.get_params(exp)

    plt.figure()
    plt.title(exp)
    plt.ylabel('Deterministic Performance')
    plt.ylabel('Return')
    plt.xlabel('Trajectories')

    plt.plot(mysuite.get_history(exp,0,'TestPerf'))

    plt.show()
