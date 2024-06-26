#%%
import numpy as np
import pandas as pd
import glob
import os
import scipy.stats as st
from matplotlib import pyplot as plt

import potion.visualization.notebook_utils as notebook_utils
from experiments import plot_utils
from expsuite import PyExperimentSuite

#%% ===================================================================================
mysuite  = PyExperimentSuite(config = 'experiments_s1.cfg')
dir_results = 'results_s1' #TODO: leggere il path dal file di configurazione

exp_off      = f"{dir_results}/offpolicy/batchsize{5}defensive_batch{0}"
exp_on       = f"{dir_results}/onpolicy/batchsize{5}"
exp_stormpg  = f"{dir_results}/stormpg/init_batchsize{5}mini_batchsize{5}"


plt.ylabel('Deterministic Performance')
plt.ylabel('Return')
plt.xlabel('Trajectories')

m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_on, 'TestPerf')
plt.plot(np.cumsum(mysuite.get_history(exp_on,0,'BatchSize')), m, label='onpolicy')
plt.fill_between(np.cumsum(mysuite.get_history(exp_on,0,'BatchSize')), ci_lb, ci_ub, alpha=.1)

m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_off, 'TestPerf')
plt.plot(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total')), m, label='offpolicy')
plt.fill_between(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total')), ci_lb, ci_ub, alpha=.1)

dfs = [pd.read_csv(file, index_col=False) for file in glob.glob(exp_stormpg+"/*.csv")]
mean_df, std_df = notebook_utils.moments(dfs)
ci_lb, ci_ub = st.norm.interval(0.68, loc=mean_df['TestPerf'], scale=std_df['TestPerf'])
plt.plot(np.cumsum(mean_df['BatchSize']), mean_df['TestPerf'], label=f'stormpg')
plt.fill_between(np.cumsum(mean_df['BatchSize']), ci_lb, ci_ub, alpha=.1)

plt.legend()
plt.show()

#%% =====================================================================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    'text.latex.preamble': r"\usepackage{amsmath}"
})

mysuite  = PyExperimentSuite(config='experiments_s1.cfg')
dir_results = 'results_s1'

# 5 trajectories
fig, ax = plt.subplots(1, dpi = 200)
exp_on  = f"{dir_results}/onpolicy/batchsize{5}"
m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_on, 'TestPerf')
plt.plot(np.cumsum(mysuite.get_history(exp_on,0,'BatchSize')), m, '--', label = 'on-policy')
plt.fill_between(np.cumsum(mysuite.get_history(exp_on, 0, 'BatchSize')), ci_lb, ci_ub, alpha=.1)

batchsizes = [3,5]
defensive_batches = [2,0]

for (b,db) in zip(batchsizes,defensive_batches):
    exp_off  = f"{dir_results}/offpolicy/batchsize{b}defensive_batch{db}"
    m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_off, 'TestPerf')
    plt.plot(np.cumsum(mysuite.get_history(exp_off, 0, 'Batch_total')), m, label = r'off-policy $\beta =$' + f' {b}/{db}')
    plt.fill_between(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total')), ci_lb, ci_ub, alpha=.1)
plt.legend()

ax.set(xlabel = 'Trajectories', ylabel = 'Return')
fig.tight_layout()
plt.show()

# 10 trajectories
fig, ax = plt.subplots(1, dpi = 200)
exp_on = f"{dir_results}/onpolicy/batchsize{10}"
m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_on, 'TestPerf')
plt.plot(np.cumsum(mysuite.get_history(exp_on,0,'BatchSize')), m, '--', label = 'on-policy')
plt.fill_between(np.cumsum(mysuite.get_history(exp_on,0,'BatchSize')), ci_lb, ci_ub, alpha = 0.1)

batchsizes = [5,8,10]
defensive_batches = [2,2,0]
for (b,db) in zip(batchsizes,defensive_batches):
    exp_off  = f"{dir_results}/offpolicy/batchsize{b}defensive_batch{db}"
    m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_off, 'TestPerf')
    plt.plot(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total')), m, label = r'off-policy $\beta =$' + f' {b}/{db}')
    plt.fill_between(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total')), ci_lb, ci_ub, alpha = 0.1)
plt.legend()

ax.set(xlabel = 'Trajectories', ylabel = 'Return')
fig.tight_layout()
plt.show()

# 20 trajectories
fig, ax = plt.subplots(1, dpi = 200)
exp_on  = f"{dir_results}/onpolicy/batchsize{20}"
m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_on, 'TestPerf')
plt.plot(np.cumsum(mysuite.get_history(exp_on,0,'BatchSize')), m, '--', label = 'on-policy')
plt.fill_between(np.cumsum(mysuite.get_history(exp_on,0,'BatchSize')), ci_lb, ci_ub, alpha = 0.1)

batchsizes = [10,15,15]
defensive_batches = [5,0,5]
for (b,db) in zip(batchsizes,defensive_batches):
    exp_off  = f"{dir_results}/offpolicy/batchsize{b}defensive_batch{db}"
    m, ci_lb, ci_ub = plot_utils.get_ci(mysuite, exp_off, 'TestPerf')
    plt.plot(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total')), m, label = r'off-policy $\beta =$' + f' {b}/{db}')
    plt.fill_between(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total')), ci_lb, ci_ub, alpha = 0.1)
plt.legend()

ax.set(xlabel = 'Trajectories', ylabel = 'Return')
fig.tight_layout()
plt.show()
