#%%
import os
import glob
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import torch

from potion.envs.lq import LQ
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent
from potion.meta.steppers import ConstantStepper
from potion.actors.continuous_policies import ShallowGaussianPolicy
from potion.algorithms.reinforce import reinforce

seed = 42
seed_all_agent(seed)

# Environment
# ===========
ds = 1
env = LQ(ds,1,random=False)
env.seed(seed)
# env.A = np.array([[1,2], [3,4]])
# env.B = np.array([[1,0.5]]).T
# env.Q = np.eye(ds)
# env.R = 1
# env.computeOptimalK() #FIXME
state_dim  = sum(env.observation_space.shape)
action_dim = sum(env.action_space.shape)
horizon    = 10

# Logger
# ======
main_dir = os.getcwd()
root_log_dir = 'logs'
# log_name = str(np.random.rand())
main_log_name = 'learning'

#%% Run algorithm
# ===============
stepper = ConstantStepper(0.05)
batchsize = 100

for n_offpolicy_opt in [0,1]:
    for rep in range(2):

        logger = Logger(directory=root_log_dir+'/offpolicy'+str(n_offpolicy_opt),
                        name = main_log_name+'_rep'+str(rep), modes=['csv'])

        policy = ShallowGaussianPolicy(
            state_dim, # input size
            action_dim, # output size
            mu_init = 0*torch.ones(ds), # initial mean parameters
            logstd_init = 0.0, # log of standard deviation
            learn_std = False # We are NOT going to learn the variance parameter
        )
        
        reinforce(env = env,
                n_offpolicy_opt=n_offpolicy_opt,
                defensive = True,
                biased_offpolicy = True,
                policy = policy,
                horizon = horizon,
                stepper = stepper,
                batchsize = batchsize,
                test_batchsize = batchsize,
                log_params = True,
                disc = env.gamma,
                iterations = 20,
                action_filter = None, # clip(env)
                estimate_var=True,
                seed = seed,
                logger = logger,
                save_params = False,
                shallow = True,
                estimator = 'gpomdp', #gpomdp, reinforce
                baseline = 'zero' #peters, zero
                )

#%% Plot
# ======
import potion.visualization.notebook_utils as nu

experiments_dirs = os.listdir(root_log_dir)
experiments_infos = []
experiments_dfs = []
for exp_id,exp_dir in enumerate(experiments_dirs):
    # Load results
    os.chdir(os.path.join(root_log_dir,exp_dir+'/'))
    reps_files = [x for x in glob.glob("*.csv") if x.startswith(main_log_name + '_')]
    reps = [pd.read_csv(x, index_col=False) for x in reps_files]
    experiments_dfs.append(reps)
    # Load infos
    info_files = [x for x in glob.glob("*.json") if x.startswith(main_log_name + '_')]
    with open(info_files[-1], 'r') as f:
        experiments_infos.append(json.load(f))
    os.chdir(main_dir)
    # for f in info_files:
    #     with open(f, 'r') as f:
    #         algo_infos.append(json.load(f))

for dfs,info in zip(experiments_dfs,experiments_infos):
    nu.plot_ci(dfs,key='TestPerf', name=f"n_ce_opt={info['n_offpolicy_opt']}")
    plt.xlabel('Learning Epochs')
    plt.ylabel('Test Performance')
plt.legend()
plt.show()

for dfs,info in zip(experiments_dfs,experiments_infos):
    nu.plot_ci(dfs,key='param0', name=f"n_ce_opt={info['n_offpolicy_opt']}")
    plt.xlabel('Learning Epochs')
    plt.ylabel('Policy parameter #0')
plt.legend()
plt.show()

#%%
K = []
for i in range(policy.num_params()):
    K.append(run[f'param{i}'].iloc[-1])
K = np.array(K).reshape(1,-1)
eig, _ = np.linalg.eig(env.A + env.B@K)
print(f"abs(eigs) = {np.abs(eig)}")