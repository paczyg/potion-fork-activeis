import numpy as np
import os
import torch
import gym
import time

from expsuite import PyExperimentSuite
from potion.envs.lq import LQ
from potion.common.misc_utils import clip, seed_all_agent
from potion.meta.steppers import ConstantStepper, Adam
from potion.actors.continuous_policies import ShallowGaussianPolicy
from potion.algorithms.variance_reduced import stormpg
from potion.common.logger import Logger

class MySuite(PyExperimentSuite):

    def reset(self, params, rep):

        self.seed = params['seed']*rep
        seed_all_agent(self.seed)

        # Environment
        # ===========
        if params['environment'] == 'lq':
            self.env = LQ(params['state_dim'],1,max_pos=10, max_action = float('inf'), sigma_noise=params['sigma_noise'], horizon=params["horizon"])
        elif params['environment'] == 'cartpole':
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
        
        # Algorithm
        # =========
        self.stepper = eval(params["stepper"])

        # Logger
        # ======
        self.logger = Logger(directory=os.path.join(params['path'],params['name']), name = str(rep))

    def iterate(self, params, rep, n):

        stormpg(
            self.env,
            self.policy,
            horizon         = self.env.horizon,
            stepper         = self.stepper,
            init_batchsize  = params['init_batchsize'],
            mini_batchsize  = params['mini_batchsize'],
            decay           = 0.9,
            iterations      = params['stormpg_iterations'],
            disc            = self.env.gamma,
            seed            = self.seed,
            logger          = self.logger,
            shallow         = isinstance(self.policy,ShallowGaussianPolicy),
            estimator       = params['estimator'],
            baseline        = params['baseline'],
            test_batchsize  = 100,
            log_params      = False)

        return {'logger_directory':self.logger.directory, 'logger_name':self.logger.name}

if __name__ == "__main__":
    # Interactive window
    mysuite = MySuite()
    
    mysuite.start()