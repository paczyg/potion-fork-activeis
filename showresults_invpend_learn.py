#%%
import numpy as np
import pandas as pd
import os
import glob
import scipy.stats as st
from matplotlib import pyplot as plt

import potion.visualization.notebook_utils as notebook_utils
from experiments import plot_utils
from expsuite import PyExperimentSuite

# === CARTPOLE ================================================================================
mysuite  = PyExperimentSuite(config = 'experiments_cart_learn.cfg')
dir_results = 'invertedpendulum/results'


on_batchsize_list           = [10, 10, 10, 10, 10]
on_bscent_batchsize_list    = [100, 100, 100, 100, 100]
off_batchsize_list          = [80, 64, 48, 32, 16 ]
off_defensive_batch         = [ 0,  16,  32, 48, 64]
my_batchsize_list           = [10, 8, 6, 4, 2]
my_defensive_batch          = [ 0,  2,  4, 6, 8]
#my_batchsize_list           = [5, 4, 3, 2, 1]
#my_defensive_batch          = [ 0,  1,  2, 3, 4]
ce_batchsize_list           = [20, 20, 20, 20, 20]
my_ce_batchsize_list        = [5, 5, 5, 5, 5]

"""
on_batchsize_list = [40, 40, 40, 40, 40]
off_batchsize_list = [40, 32, 24, 16, 8]
off_defensive_batch = [0, 8, 16, 24, 32]
"""
#stormpg_init_batchsize_list = [20, 40, 100, 200]
#stormpg_mini_batchsize_list = [10, 20, 50, 100]

for i in range(len(on_batchsize_list)):
    
    exp_on       = f"{dir_results}/onpolicy/616/batchsize{on_batchsize_list[i]}"
    exp_onbscent = f"{dir_results}/onpolicy_bs100/616/batchsize{on_bscent_batchsize_list[i]}"
    exp_off      = f'{dir_results}/offpolicy/not_offline_target_0720/batchsize{off_batchsize_list[i]}defensive_batch{off_defensive_batch[i]}'
    #exp_stormpg  = f"{dir_results}/my/init_batchsize{stormpg_init_batchsize_list[i]}mini_batchsize{stormpg_mini_batchsize_list[i]}"
    exp_my = f"{dir_results}/my/not_offline_target_0720/batchsize{my_batchsize_list[i]}defensive_batch{my_defensive_batch[i]}"
    
    """
    exp_on       = f"{dir_results}/onpolicy/batchsize{on_batchsize_list[i]}"
    exp_off      = f'{dir_results}/offpolicy/batchsize{off_batchsize_list[i]}defensive_batch{off_defensive_batch[i]}'
    #exp_stormpg  = f"{dir_results}/my/init_batchsize{stormpg_init_batchsize_list[i]}mini_batchsize{stormpg_mini_batchsize_list[i]}"
    exp_my = f"{dir_results}/my/defensive_batch{off_defensive_batch[i]}batchsize{off_batchsize_list[i]}"
    """


    plt.figure()
    #plt.title(f'Def_Batchsize {off_defensive_batch[i]}')
    plt.title(f"Inverted Pendulum Defensive alpha = {off_defensive_batch[i]/80}" )
    plt.ylabel('Deterministic Performance')
    plt.ylabel('Return')
    plt.xlabel('Trajectories')

    # Onpolicy
    m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_onbscent, 'TestPerf')
    #print(m.shape)
    #print(len(mysuite.get_history(exp_on,0,'Batch_total')))
    plt.plot(np.cumsum([elem for elem in mysuite.get_history(exp_onbscent,0,'BatchSize')]), m, label='onpolicy batch size 100')
    plt.fill_between(np.cumsum([elem  for elem in mysuite.get_history(exp_onbscent,0,'BatchSize')]), ci_lb, ci_ub, alpha=.1)

    #onpol_bs100
    
    m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_on, 'TestPerf')
    #print(m.shape)
    #print(len(mysuite.get_history(exp_on,0,'Batch_total')))
    plt.plot(np.cumsum([elem for elem in mysuite.get_history(exp_on,0,'BatchSize')]), m, label='onpolicy batch size 10')
    plt.fill_between(np.cumsum([elem  for elem in mysuite.get_history(exp_on,0,'BatchSize')]), ci_lb, ci_ub, alpha=.1)

    
    # Offpolicy
    
    m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_off, 'TestPerf')
    plt.plot(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total')), m, label='offpolicy')
    plt.fill_between(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total')), ci_lb, ci_ub, alpha=.1)
    

    # My
    m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_my, 'TestPerf')
    """
    plt.plot(np.cumsum([elem + my_ce_batchsize_list[i] for elem in  mysuite.get_history(exp_my,0,'Batch_total')]), m, label='my')
    plt.fill_between(np.cumsum([elem + my_ce_batchsize_list[i] for elem in  mysuite.get_history(exp_my,0,'Batch_total')]), ci_lb, ci_ub, alpha=.1)
    """
    plt.plot(np.cumsum([elem  for elem in  mysuite.get_history(exp_my,0,'Batch_total')])[:240], m[:240], label='DAIS-PG')
    plt.fill_between(np.cumsum([elem  for elem in  mysuite.get_history(exp_my,0,'Batch_total')])[:240], ci_lb[:240], ci_ub[:240], alpha=.1)

    """
    # Stormpg
    dfs = [pd.read_csv(file, index_col=False) for file in glob.glob(exp_stormpg+"/*.csv")]
    mean_df, std_df = notebook_utils.moments(dfs)
    ci_lb, ci_ub = st.norm.interval(0.68, loc=mean_df['TestPerf'], scale=std_df['TestPerf'])
    plt.plot(np.cumsum(mean_df['BatchSize']), mean_df['TestPerf'], label=f'stormpg', linestyle='dotted')
    plt.fill_between(np.cumsum(mean_df['BatchSize']), ci_lb, ci_ub, alpha=.1)
    """
    plt.legend()
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
