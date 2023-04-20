#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import torch

from potion.algorithms.ce_optimization import algo, optimize_behavioural

# Environment
# ===========
from potion.envs.lq import LQ

ds = 1
env = LQ(ds,1)
env.horizon=1
state_dim  = sum(env.observation_space.shape)
action_dim = sum(env.action_space.shape)

# Policy
# ======
from potion.actors.continuous_policies import ShallowGaussianPolicy

# Linear policy in the state
policy = ShallowGaussianPolicy(
    state_dim,                      # input size
    action_dim,                     # output size
    mu_init = 0.0*torch.ones(ds),   # initial mean parameters
    logstd_init = 0.0,              # log of standard deviation
    learn_std = False               # We are NOT going to learn the variance parameter
)

# Generate batch
# ==============
from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import clip, seed_all_agent

# seed = None
seed = 42
env.seed(seed)
seed_all_agent(seed)

batchsize = 100

# states, actions, rewards, mask, infos
batch = generate_batch(env, policy, env.horizon, batchsize, 
                       action_filter=None, 
                       seed=seed, 
                       n_jobs=False)

#%% Test Cross-Entropy
# ===================
target_policy = policy
mis_policies  = 1 * [policy]
mis_batches   = 1 * [batch]

q = optimize_behavioural(env, target_policy, mis_policies, mis_batches, optimize_mean=True, optimize_variance=True)
print(f"theta* = {q.get_loc_params()}, logstd* = {q.logstd}")

#%% Test Algorithm
# ================
target_policy = policy
N_per_it = 20
n_ce_opt = 5
results, stats, info = algo(env, target_policy, N_per_it, n_ce_opt, 
                            estimator='gpomdp',
                            baseline='zero',
                            action_filter=None,
                            window=None,
                            optimize_mean=True,
                            optimize_variance=True,
                            run_mc_comparison = True)
print(f"{results}, \n {pd.DataFrame(stats)}")

stats = pd.DataFrame(stats)
fig,ax = plt.subplots()
ax.plot(stats["opt_policy_scale"], color='red')
ax.set_ylabel('$\log\sigma^\star$', color='red')

if target_policy.num_params() < 2:  #TODO: plottare anche caso >2 parametri
    ax2=ax.twinx()
    ax2.plot(stats["opt_policy_loc"].apply(np.asarray), color='blue')
    ax2.set_ylabel('$\mu^\star$',color='blue')

plt.xlabel('n_ce_iterations')
plt.show()

#%% PLOT Variance reduction VS parameters change
N_per_it_list = [100]
n_ce_opt_list = [1]
nExperiments = 5

target_policy = policy
mis_policies  = 1 * [policy]
mis_batches   = 1 * [batch]

res = []
for exp in range(nExperiments):
    for N_per_it in N_per_it_list:
        for n_ce_opt in n_ce_opt_list:
            _results, _, _info = algo(env, target_policy, N_per_it, n_ce_opt, 
                                      estimator='gpomdp',
                                      baseline='zero',
                                      action_filter=None,
                                      window=None,
                                      optimize_mean=True,
                                      optimize_variance=True,
                                      run_mc_comparison = True)
            _results.update({'exp': exp})
            res.append({**_results, **_info})

res = pd.DataFrame(res)
fig, ax = plt.subplots()
res.loc[:,'var_grad_is/var_grad_mc'] = res['var_grad_is'] / res['var_grad_mc']
for N_per_it in res['N_per_it'].unique():
    tmp = res.loc[res['N_per_it']==N_per_it, ['N_tot','var_grad_is/var_grad_mc']]
    stats = tmp.groupby('N_tot')['var_grad_is/var_grad_mc'].agg(['mean', 'std', 'count'])

    xps = tmp['N_tot'].unique()
    yps = stats['mean']
    ax.plot(xps, yps, '--o', linewidth=2.0)
    
    ci_lb, ci_ub = st.norm.interval(0.68, loc=stats['mean'], scale=stats['std']/np.sqrt(stats['count']))
    ax.fill_between(xps, ci_lb, ci_ub, alpha=.5, linewidth=0)

plt.grid()
plt.legend([f"N_per_it={n}" for n in res['N_per_it'].unique()])
plt.xlabel('N_tot')
plt.ylabel('Var_IS / Var_MC')
plt.show()
