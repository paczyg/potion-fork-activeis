import numpy as np
import torch
import gym

from expsuite import PyExperimentSuite
from potion.envs.lq import LQ
from potion.common.misc_utils import clip, seed_all_agent
from potion.meta.steppers import ConstantStepper, Adam
from potion.actors.continuous_policies import ShallowGaussianPolicy
from potion.algorithms.reinforce_step import reinforce_step

class MySuite(PyExperimentSuite):

    def reset(self, params, rep):

        self.seed = params['seed']*rep
        seed_all_agent(self.seed)

        # Environment
        if params['path'] == 'results_lq_s1':
            self.env = LQ(1,1,max_pos=10, max_action = float('inf'), sigma_noise=params['sigma_noise'], horizon=params["horizon"])
        elif params['path'] == 'results_lq_s5':
            self.env = LQ(5,1,max_pos=10, max_action = float('inf'), sigma_noise=params['sigma_noise'], horizon=params["horizon"])
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
            mu_init = params["mu_init"]*torch.ones(state_dim),
            logstd_init = params["logstd_init"]*torch.ones(action_dim),
            learn_std = params["learn_std"]
        )
        
        self.stepper = eval(params["stepper"])

    def iterate(self, params, rep, n):
        if isinstance(params['ce_batchsizes'], str):
            ce_batchsizes = eval(params['ce_batchsizes'])
        else:
            ce_batchsizes = params['ce_batchsizes']

        log = reinforce_step(
            self.env, self.policy, params['horizon'],
            ce_batchsizes       = ce_batchsizes,
            estimator           = params['estimator'],
            baseline            = params['baseline'],
            batchsize           = params['batchsize'],
            test_batchsize      = 500,
            action_filter       = params['action_filter'],
            defensive           = params['defensive'],
            biased_offpolicy    = params['biased_offpolicy'],
            disc                = self.env.gamma,
            stepper             = self.stepper,
            seed                = self.seed+n,
            log_params          = True,
            log_grad            = True,
            log_ce_params       = True,
            estimate_var        = True,
            shallow             = True
        )
        
        return log

if __name__ == "__main__":
    # Interactive window
    # mysuite = MySuite(config='lq_s1.cfg', experiment='grad_comparison', numcores=1)
    
    # Command line
    mysuite = MySuite()
    
    mysuite.start()