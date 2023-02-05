#%%
import numpy as np
import pandas as pd
import os
import scipy.stats as st
from matplotlib import pyplot as plt

import potion.visualization.notebook_utils as notebook_utils
from experiments import plot_utils
from expsuite import PyExperimentSuite

#%% === CARTPOLE ================================================================================
mysuite  = PyExperimentSuite(config='experiments.cfg')
dir_results = 'results' #TODO: leggere il path dal file di configurazione?

# on_batchsize_list           = [10, 20, 50, 100]
# off_batchsize_list          = [10, 20, 50, 100]
# off_defensive_batch         = [0, 0, 0, 0]
# stormpg_init_batchsize_list = [20, 40, 100, 200]
# stormpg_mini_batchsize_list = [10, 20, 50, 100]

on_batchsize_list           = [50, 100, 200]
off_batchsize_list          = [50, 100, 200]
off_defensive_batch         = [0, 0, 0, 0]

fig, axs = plt.subplots(len(on_batchsize_list), figsize=(12,5))
for i,ax in enumerate(axs):

    exp_on       = f"{dir_results}/onpolicy/batchsize{on_batchsize_list[i]}"
    exp_off      = f'{dir_results}/offpolicy/batchsize{off_batchsize_list[i]}defensive_batch{off_defensive_batch[i]}'
    # exp_stormpg  = f"{dir_results}/stormpg/init_batchsize{stormpg_init_batchsize_list[i]}mini_batchsize{stormpg_mini_batchsize_list[i]}"

    ax.set_title(f'Batchsize {on_batchsize_list[i]}')
    ax.set_ylabel('Deterministic Performance')
    ax.set_ylabel('Return')
    ax.set_xlabel('Trajectories')

    # Onpolicy
    m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_on, 'TestPerf')
    ax.plot(np.cumsum(mysuite.get_history(exp_on,0,'BatchSize')), m, label='onpolicy')
    ax.fill_between(np.cumsum(mysuite.get_history(exp_on,0,'BatchSize')), ci_lb, ci_ub, alpha=.1)

    # Offpolicy
    m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_off, 'TestPerf')
    ax.plot(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total')), m, label='offpolicy', linestyle='dashed')
    ax.fill_between(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total')), ci_lb, ci_ub, alpha=.1)

    # Stormpg
    # dfs = [pd.read_csv(file, index_col=False) for file in glob.glob(exp_stormpg+"/*.csv")]
    # mean_df, std_df = notebook_utils.moments(dfs)
    # ci_lb, ci_ub = st.norm.interval(0.68, loc=mean_df['TestPerf'], scale=std_df['TestPerf'])
    # ax.plot(np.cumsum(mean_df['BatchSize']), mean_df['TestPerf'], label=f'stormpg', linestyle='dotted')
    # ax.fill_between(np.cumsum(mean_df['BatchSize']), ci_lb, ci_ub, alpha=.1)

    ax.legend()
plt.show()

#%% === CARTPOLE STD ================================================================================
mysuite  = PyExperimentSuite('experiments_std.cfg')
dir_results = 'results_std' #TODO: leggere il path dal file di configurazione

on_batchsize_list           = [50, 100, 200, 300, 500, 700]
off_batchsize_list          = [50, 100, 200, 300, 500, 700]
off_defensive_batch         = [0, 0, 0, 0, 0, 0]

fig, axs = plt.subplots(len(on_batchsize_list), figsize=(12,5))
for i in range(len(on_batchsize_list)):

    exp_on       = f'{dir_results}/onpolicy/batchsize{on_batchsize_list[i]}'
    exp_off      = f'{dir_results}/offpolicy/batchsize{off_batchsize_list[i]}defensive_batch{off_defensive_batch[i]}'

    plt.figure()
    plt.title(f'Batchsize {on_batchsize_list[i]}')
    plt.ylabel('Deterministic Performance')
    plt.ylabel('Return')
    plt.xlabel('Trajectories')

    # Onpolicy
    m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_on, 'TestPerf')
    plt.plot(np.cumsum(mysuite.get_history(exp_on,0,'BatchSize'))[0:len(m)], m, label='onpolicy')
    plt.fill_between(np.cumsum(mysuite.get_history(exp_on,0,'BatchSize'))[0:len(m)], ci_lb, ci_ub, alpha=.1)

    # Offpolicy
    m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_off, 'TestPerf')
    plt.plot(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total'))[0:len(m)], m, label='offpolicy', linestyle='dashed')
    plt.fill_between(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total'))[0:len(m)], ci_lb, ci_ub, alpha=.1)

    plt.legend()
plt.show()
