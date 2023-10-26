#%%
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt

#%%
from expsuite import PyExperimentSuite
mysuite = PyExperimentSuite(config='experiments_offpolicy.cfg')

def plot_experiments(exps, legend_params=[], overlapping=True, **plt_kwargs):
    # NOTE: no handling of experiments repetitions
    def get_random_marker():
        import matplotlib.markers as mmarkers
        import random
        return random.choice(list(mmarkers.MarkerStyle.filled_markers))
    
    def make_list(x):
        if type(x) is list:
            return x
        else:
            return [x]
    exps = make_list(exps)
    legend_params = make_list(legend_params)
    
    if overlapping:
        plt.figure()
        plt.ylabel('Return')
        plt.xlabel('Trajectories')

    for exp in exps:
        exp_params = mysuite.get_params(exp)

        if not overlapping:
            plt.figure()
            plt.ylabel('Return')
            plt.xlabel('Trajectories')

        xx = mysuite.get_history(exp,0,'Batch_total')
        xx = [50]*len(xx)
        if xx:
            # offpolicy
            xx = np.cumsum(xx)
        else:
            # onpolicy
            xx = np.cumsum(mysuite.get_history(exp,0,'BatchSize'))

        plt.plot(xx,
                 mysuite.get_history(exp,0,'TestPerf'),
                 label = [f'{x} = {exp_params[x]}' for x in legend_params],
                 marker = get_random_marker(),
                 **plt_kwargs)

        if not overlapping and legend_params:
            plt.legend()

    if overlapping and legend_params:
        plt.legend()
    plt.tight_layout()
    plt.show()

#%% Plot offpolicy
exps = mysuite.get_exps('results/offpolicy/offpolicy_debug_target_Adam')
plot_experiments(exps, overlapping = True,
                 legend_params = ['ce_initialize_behavioural_policies', 'ce_tol_grad'])

exps = mysuite.get_exps('results/offpolicy/offpolicy')
plot_experiments(exps, overlapping = True, legend_params='batchsize')

exps = mysuite.get_exps('results/offpolicy/offpolicy_616')
plot_experiments(exps, overlapping = True, legend_params=['ce_mis_rescale', 'defensive_batch'])

exps = mysuite.get_exps('results/offpolicy/offpolicy_630')
plot_experiments(exps, overlapping = False)

exps = mysuite.get_exps('results/offpolicy/offpolicy_706')
plot_experiments(exps)

exps = mysuite.get_exps('results/offpolicy/offpolicy_707')
plot_experiments(exps)

exps = mysuite.get_exps('results/offpolicy/lr_0720')
plot_experiments(exps, legend_params=['stepper', 'ce_mis_clip'])

exps = mysuite.get_exps('results/offpolicy/horizon_0810')
plot_experiments(exps, legend_params = ['ce_mis_clip', 'stepper', 'horizon'])

#%% Plot onpolicy
exps = mysuite.get_exps('results/onpolicy_adam')
plot_experiments(exps, overlapping = True, legend_params = ['batchsize'])

exps = mysuite.get_exps('results/onpolicy_616')
plot_experiments(exps, overlapping = False, legend_params = ['batchsize'])

exps = mysuite.get_exps('results/onpolicy/lr_0727')
plot_experiments(exps, overlapping = True, legend_params = ['stepper'])