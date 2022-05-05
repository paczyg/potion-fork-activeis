import numpy as np
import torch
import gym

from expsuite import PyExperimentSuite
from potion.envs.lq import LQ
from potion.common.misc_utils import clip, seed_all_agent
from potion.meta.steppers import ConstantStepper
from potion.actors.continuous_policies import ShallowGaussianPolicy
from potion.algorithms.reinforce_step import reinforce_step

class MySuite(PyExperimentSuite):

    def reset(self, params, rep):

        self.seed = params['seed']*rep
        seed_all_agent(self.seed)

        # Environment
        if params['path'] == 'results_lq_s1':
            self.env = LQ(1,1,max_pos=10, max_action = float('inf'), sigma_noise=params['sigma_noise'])
        elif params['path'] == 'results_lq_s5':
            self.env = LQ(5,1,max_pos=10, max_action = float('inf'), sigma_noise=params['sigma_noise'])
        elif params['path'] == 'results_cartpole':
            self.env = gym.make('ContCartPole-v0')
        else:
            raise NotImplementedError
        self.env.seed(self.seed)
        state_dim  = sum(self.env.observation_space.shape)
        action_dim = sum(self.env.action_space.shape)

        # Policy
        self.policy = ShallowGaussianPolicy(
            state_dim, # input size
            action_dim, # output size
            mu_init = 0*torch.ones(state_dim), # initial mean parameters
            logstd_init = 0.0, # log of standard deviation
            learn_std = False # We are NOT going to learn the variance parameter
        )
        
        # Algorithm
        self.stepper = ConstantStepper(0.0001)

    def iterate(self, params, rep, n):
        log = reinforce_step(
            self.env, self.policy, params['horizon'],
            n_offpolicy_opt     = params['n_offpolicy_opt'],
            estimator           = params['estimator'],
            baseline            = params['baseline'],
            batchsize           = params['batchsize'],
            test_batchsize      = params['batchsize'],
            action_filter       = params['action_filter'],
            defensive           = params['defensive'],
            biased_offpolicy    = params['biased_offpolicy'],
            disc                = self.env.gamma,
            stepper             = self.stepper,
            seed                = self.seed,
            log_params          = True,
            estimate_var        = True,
            shallow             = True
        )
        
        return log

if __name__ == "__main__":
    mysuite = MySuite()
    mysuite.start()