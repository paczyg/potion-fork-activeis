import numpy as np
import torch
import gym

from expsuite import PyExperimentSuite
from potion.envs.lq import LQ
from potion.common.misc_utils import clip, seed_all_agent
from potion.meta.steppers import ConstantStepper, Adam
from potion.actors.continuous_policies import ShallowGaussianPolicy
from potion.algorithms.reinforce_offpolicy import reinforce_offpolicy_step
from potion.algorithms.reinforce import reinforce_step
from potion.simulation.trajectory_generators import generate_batch

class MySuite(PyExperimentSuite):

    def reset(self, params, rep):

        self.seed = params['seed']*rep
        seed_all_agent(self.seed)

        # Environment
        # ===========
        if params['path'] == 'results_lq_s1':
            self.env = LQ(1,1,max_pos=10, max_action = float('inf'), sigma_noise=params['sigma_noise'], horizon=params["horizon"])
        elif params['path'] == 'results_lq_s5':
            self.env = LQ(5,1,max_pos=10, max_action = float('inf'), sigma_noise=params['sigma_noise'], horizon=params["horizon"])
        elif params['path'] == 'results_cartpole':
            self.env = gym.make('ContCartPole-v0')
            self.env.gamma = 1
        else:
            raise NotImplementedError
        self.env.horizon = params['horizon']
        self.env.seed(self.seed)
        state_dim  = sum(self.env.observation_space.shape)
        action_dim = sum(self.env.action_space.shape)

        # Policy
        # ======
        self.policy = ShallowGaussianPolicy(
            state_dim, # input size
            action_dim, # output size
            mu_init     = params["mu_init"]*torch.ones(state_dim),
            logstd_init = params["logstd_init"]*torch.ones(action_dim),
            learn_std   = params["learn_std"]
        )
        
        self.stepper = eval(params["stepper"])

        # Algorithm
        # =========
        if 'offpolicy/' in params['name']:
            # Initial data for first offline CE estimation
            if params['ce_use_offline_data']:
                self.offline_policies = [self.policy]
                self.offline_batches  = [generate_batch(self.env, self.policy, self.env.horizon, params['batchsize'], seed=self.seed)]
            else:
                self.offline_policies = None
                self.offline_batches  = None

    def iterate(self, params, rep, n):
        # Offpolicy algorithm
        if 'offpolicy' in params['name']:
            if isinstance(params['ce_batchsizes'], str):
                ce_batchsizes = eval(params['ce_batchsizes'])
            else:
                ce_batchsizes = params['ce_batchsizes']

            log, self.offline_policies, self.offline_batches = reinforce_offpolicy_step(
                self.env, self.policy, self.env.horizon, self.offline_policies, self.offline_batches,
                batchsize        = params['batchsize'], 
                baseline         = params['baseline'],
                biased_offpolicy = params['biased_offpolicy'],
                ce_batchsizes    = ce_batchsizes,
                disc             = self.env.gamma,
                defensive_batch  = params['defensive_batch'],
                estimator        = params['estimator'],
                seed             = params['seed']+n,
                shallow          = isinstance(self.policy, ShallowGaussianPolicy),
                stepper          = self.stepper,
                test_batchsize   = 100)
        
        # Onpolicy algorithm
        elif 'onpolicy' in params['name']:
            log = reinforce_step(
                self.env, self.policy, self.env.horizon,
                batchsize      = params['batchsize'], 
                disc           = self.env.gamma,
                stepper        = self.stepper,
                estimator      = params['estimator'],
                baseline       = params['baseline'],
                shallow        = isinstance(self.policy, ShallowGaussianPolicy),
                seed           = params['seed']+n,
                test_batchsize = 100)
        return log

if __name__ == "__main__":
    # Interactive window
    # mysuite = MySuite(config='lq_s1.cfg', numcores=1)
    
    # Command line
    mysuite = MySuite()
    
    mysuite.start()