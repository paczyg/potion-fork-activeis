#%%
import os
import torch

#%% Environment
# ===========
from potion.envs.lq import LQ

env = LQ()
state_dim  = sum(env.observation_space.shape)
action_dim = sum(env.action_space.shape)
horizon    = env.horizon

#%% Policy
# ======
from potion.actors.continuous_policies import ShallowGaussianPolicy

# Linear policy in the state
policy = ShallowGaussianPolicy(state_dim, # input size
                               action_dim, # output size
                               mu_init = 0*torch.ones(1), # initial mean parameters
                               logstd_init = 0.0, # log of standard deviation
                               learn_std = False # We are NOT going to learn the variance parameter
                              )
params_init = policy.get_flat()


#%% Learn
# =======
from potion.algorithms.reinforce import reinforce
from potion.meta.steppers import ConstantStepper

## Log
from potion.common.logger import Logger
main_dir = os.getcwd()
log_dir = 'logs'
log_name = 'REINFORCE'
logger = Logger(directory=log_dir, name = log_name, modes=['csv'])

## Algorithm settings
stepper = ConstantStepper(0.1)
batchsize = 200

seed = 42
env.seed(seed)

## Algorithm
reinforce(env = env,
          n_offpolicy_opt=10,
          policy = policy,
          horizon = horizon,
          stepper = stepper,
          batchsize = batchsize,
          test_batchsize = batchsize,
          log_params = True,
          disc = env.gamma,
          iterations = 10,
          seed = seed,
          logger = logger,
          save_params = 50, #Policy parameters will be saved on disk each 5 iterations
          shallow = True, #Use optimized code for shallow policies
          estimator = 'gpomdp', #Use the G(PO)MDP refined estimator
          baseline = 'peters' #Use Peter's variance-minimizing baseline
         )
params_opt = policy.get_flat()
print(f'Optimized policy parameters = {params_opt}')

#%% Plot
# ======
import os
import glob
import pandas as pd
import json
import matplotlib.pyplot as plt

os.chdir(log_dir)
# Load all runs
# files = [x for x in glob.glob("*.csv") if x.startswith(log_name + '_')]
# runs = [pd.read_csv(x, index_col=False) for x in files]
# Load last run
file = [x for x in glob.glob("*.csv") if x.startswith(log_name + '_')][-1]
run = pd.read_csv(file, index_col=False)
# Load info files
files = [x for x in glob.glob("*.txt") if x.startswith(log_name + '_')]
algo_infos = []
for f in files:
    with open(f, 'r') as f:
        algo_infos.append(json.load(f))
os.chdir(main_dir)

perf = run['Perf']
plt.plot(range(len(perf)), perf)
plt.xlabel('Iterations')
plt.ylabel('Performance')
plt.show()

perf = run['TestPerf']
plt.plot(range(len(perf)), perf)
plt.xlabel('Iterations')
plt.ylabel('Test Performance')
plt.show()

param_plot = run['param0']
plt.plot(range(len(param_plot)), param_plot)
plt.xlabel('Iterations')
plt.ylabel('Policy parameter')
plt.show()

# #%% Confronto policy
# #NOTE: debug
# # ==================
# from potion.simulation.trajectory_generators import generate_batch
# from potion.common.misc_utils import performance, avg_horizon, mean_sum_info

# # params = 0*torch.ones(1)
# # policy.set_from_flat(params)
# # batch = generate_batch(env, policy, horizon, batchsize)
# # print(f"Performance of manual parameters: {performance(batch, env.gamma)}")

# policy.set_from_flat(params_init)
# # batch = generate_batch(env, policy, horizon, batchsize)
# batch = generate_batch(env, policy, horizon, batchsize,deterministic=True)
# print(f"Performance of initial parameters: {performance(batch, env.gamma)}")

# policy.set_from_flat(params_opt)
# # batch = generate_batch(env, policy, horizon, batchsize)
# batch = generate_batch(env, policy, horizon, batchsize,deterministic=True)
# print(f"Performance of optimized parameters: {performance(batch, env.gamma)}")

# policy.set_from_flat(env.computeOptimalK())
# # batch = generate_batch(env, policy, horizon, batchsize)
# batch = generate_batch(env, policy, horizon, batchsize,deterministic=True)
# print(f"Performance of star parameters: {performance(batch, env.gamma)}")