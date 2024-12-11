import numpy as np
import os
import logging
import copy
import torch
import gymnasium as gym

from potion.common.debug_logger import setup_debug_logger
from expsuite import PyExperimentSuite
from potion.envs.lq import LQ
from potion.common.misc_utils import clip, seed_all_agent
from potion.meta.steppers import ConstantStepper, Adam
from potion.actors.continuous_policies import ShallowGaussianPolicy, DeepGaussianPolicy
from potion.algorithms.reinforce_offpolicy import reinforce_offpolicy_step
from potion.algorithms.reinforce import reinforce_step
from potion.simulation.trajectory_generators import generate_batch

class MySuite(PyExperimentSuite):

    def reset(self, params, rep):

        self.seed = params['seed']*rep
        seed_all_agent(self.seed)

        # Environment
        # ===========
        if params['environment'] == 'lq':
            self.env = LQ(params['state_dim'],1,max_pos=10, max_action = float('inf'), sigma_noise=params['sigma_noise'], horizon=params["horizon"])
            self.env.horizon = params['horizon']
            self.env.seed(self.seed)
        elif params['environment'] == 'cartpole':
            self.env = gym.make('ContCartPole-v0')
            self.env.gamma = 1
            self.env.horizon = params['horizon']
            self.env.seed(self.seed)
        elif params['environment'] == 'swimmer':
            self.env = gym.make('Swimmer-v4')
            self.env.horizon = params['horizon']
            self.env.gamma = params['gamma']
        elif params['environment'] == 'halfcheetah':
            self.env = gym.make('HalfCheetah-v4')
            self.env.horizon = params['horizon']
            self.env.gamma = params['gamma']
        elif params['environment'] == 'invertedpendulum':
            self.env = gym.wrappers.ClipAction(gym.make('InvertedPendulum-v4'))
            self.env.horizon = params['horizon']
            self.env.gamma = params['gamma']
        elif params['environment'] == 'pusher':
            self.env = gym.wrappers.ClipAction(gym.make('Pusher-v4'))
            self.env.horizon = params['horizon']
            self.env.gamma = params['gamma']
        

        else:
            raise NotImplementedError
        state_dim  = sum(self.env.observation_space.shape)
        action_dim = sum(self.env.action_space.shape)

        # Policy
        # ======
        if params['environment'] == 'lq' or params['environment'] == 'cartpole':
            self.policy = ShallowGaussianPolicy(
                state_dim, # input size
                action_dim, # output size
                mu_init     = params["mu_init"]*torch.ones(state_dim),
                logstd_init = params["logstd_init"]*torch.ones(action_dim),
                learn_std   = params["learn_std"]
            )
        elif params['environment'] == 'swimmer' or params['environment'] == 'halfcheetah':
            self.policy = DeepGaussianPolicy(
                state_dim,
                action_dim,
                hidden_neurons  = [32,32],
                mu_init         = None,
                logstd_init     = params["logstd_init"]*torch.ones(action_dim),
                learn_std       = params["learn_std"]
            )
        elif params['environment'] == 'invertedpendulum' or params['environment'] == 'pusher':
            self.policy = DeepGaussianPolicy(
                state_dim,
                action_dim,
                hidden_neurons  = [8,8],
                mu_init         = None,
                logstd_init     = params["logstd_init"]*torch.ones(action_dim),
                learn_std       = params["learn_std"]
            )
        
        self.stepper = eval(params["stepper"])

        # Logger
        # ======
        self.debug_logger = setup_debug_logger(
            name        = str(rep),
            log_file    = os.path.join(params['path'],params['name'],'') + str(rep) + '_DEBUG' + '.log',
            level       = logging.DEBUG,
            stream      ='stderr'
        )


    def iterate(self, params, rep, n):
        
        log = reinforce_step(
            self.env, self.policy, self.env.horizon,
            batchsize = params['batchsize'],
            debug_logger = self.debug_logger,
            disc = self.env.gamma,
            stepper = self.stepper,
            estimator = params['estimator'],
            baseline = params['baseline'],
            shallow = isinstance(self.policy, ShallowGaussianPolicy),
            seed = params['seed']+n,
            test_batchsize = params['batchsize'])
        return log

if __name__ == "__main__":
    
    # Command line
    mysuite = MySuite()
    
    mysuite.start()