#%%
import os
import glob
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import torch

from potion.algorithms.reinforce import reinforce
from potion.meta.steppers import ConstantStepper
from potion.actors.continuous_policies import ShallowGaussianPolicy
from potion.common.logger import Logger
from potion.common.misc_utils import seed_all_agent

# Environment
# ===========
from potion.envs.lq import LQ

env = LQ(1,1)
state_dim  = sum(env.observation_space.shape)
action_dim = sum(env.action_space.shape)
horizon    = 10

# Logger
# ======
main_dir = os.getcwd()
log_dir = 'logs'
log_name = str(np.random.rand())

#%% Run algorithm
# ===============
seed = 42
env.seed(seed)
seed_all_agent(seed)
stepper = ConstantStepper(0.1)
batchsize = 1000

for n_offpolicy_opt in [1]:
    policy = ShallowGaussianPolicy(
        state_dim, # input size
        action_dim, # output size
        mu_init = 0*torch.ones(1), # initial mean parameters
        logstd_init = 0.0, # log of standard deviation
        learn_std = False # We are NOT going to learn the variance parameter
    )
    logger = Logger(directory=log_dir, name = log_name, modes=['csv'])
    reinforce(env = env,
              n_offpolicy_opt=n_offpolicy_opt,
              defensive = False,
              biased_offpolicy = False,
              policy = policy,
              horizon = horizon,
              stepper = stepper,
              batchsize = batchsize,
              test_batchsize = batchsize,
              log_params = True,
              disc = env.gamma,
              iterations = 20,
              seed = seed,
              logger = logger,
              save_params = False,
              shallow = True,
              estimator = 'gpomdp', #gpomdp, reinforce
              baseline = 'zero' #peters, zero
            )

#%% Plot
# ======
os.chdir(log_dir)
# Load results
run_files = [x for x in glob.glob("*.csv") if x.startswith(log_name + '_')]
runs = [pd.read_csv(x, index_col=False) for x in run_files]
# Load infos
info_files = [x for x in glob.glob("*.json") if x.startswith(log_name + '_')]
algo_infos = []
for f in info_files:
    with open(f, 'r') as f:
        algo_infos.append(json.load(f))
os.chdir(main_dir)

for run,info in zip(runs,algo_infos):
    perf = run['Perf']
    plt.plot(range(len(perf)), perf, label=f"n_ce_opt={info['n_offpolicy_opt']}")
    plt.xlabel('Learning Epochs')
    plt.ylabel('Performance')
plt.legend()
plt.show()

for run,info in zip(runs,algo_infos):
    perf = run['TestPerf']
    plt.plot(range(len(perf)), perf, label=f"n_ce_opt={info['n_offpolicy_opt']}")
    plt.xlabel('Learning Epochs')
    plt.ylabel('Test Performance')
plt.legend()
plt.show()

for run,info in zip(runs,algo_infos):
    param_plot = run['param0']
    plt.plot(range(len(param_plot)), param_plot, label=f"n_ce_opt={info['n_offpolicy_opt']}")
    plt.xlabel('Learning Epochs')
    plt.ylabel('Policy parameter')
plt.legend()
plt.show()
