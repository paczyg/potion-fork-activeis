import os
import torch
import gym

from expsuite import PyExperimentSuite
from potion.envs.lq import LQ
from potion.common.misc_utils import seed_all_agent
from potion.actors.continuous_policies import ShallowGaussianPolicy, DeepGaussianPolicy
from potion.algorithms.variance_reduced import pagepg
from potion.common.logger import Logger

class MySuite(PyExperimentSuite):

    def reset(self, params, rep):

        self.seed = params['seed']*rep
        seed_all_agent(self.seed)

        # Environment
        # ===========
        if params['environment'] == 'lq':
            self.env = LQ(params['state_dim'], 1, max_pos = 10, max_action = float('inf'), sigma_noise = params['sigma_noise'], horizon = params["horizon"])
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

        else:
            raise NotImplementedError
        
        state_dim  = sum(self.env.observation_space.shape)
        action_dim = sum(self.env.action_space.shape)

        # Policy
        # ======
        if params['environment'] == 'lq' or params['environment'] == 'cartpole':
            self.policy = ShallowGaussianPolicy(
                state_dim,
                action_dim,
                mu_init = params["mu_init"]*torch.ones(state_dim),
                logstd_init = params["logstd_init"]*torch.ones(action_dim),
                learn_std = params["learn_std"]
            )
        elif params['environment'] == 'swimmer':
            self.policy = DeepGaussianPolicy(
                state_dim,
                action_dim,
                hidden_neurons = [32,32],
                mu_init = params["mu_init"]*torch.ones(state_dim*32+32*32+32*action_dim),
                logstd_init = params["logstd_init"]*torch.ones(action_dim),
                learn_std = params["learn_std"]
            )
        
        # Algorithm
        # =========
        self.stepper = eval(params["stepper"])

        # Logger
        # ======
        self.logger = Logger(directory = os.path.join(params['path'], params['name']), name = str(rep))

    def iterate(self, params, rep, n):

        pagepg(
            self.env,
            self.policy,
            self.env.horizon,
            init_batchsize = params['init_batchsize'],
            mini_batchsize = params['mini_batchsize'], 
            snapshot_prob = .1,
            iterations = params['pagepg_iterations'],
            disc = self.env.gamma,
            stepper = self.stepper,
            action_filter = None,
            estimator = params['estimator'],
            baseline = params['baseline'],
            logger = self.logger,
            shallow = isinstance(self.policy,ShallowGaussianPolicy),
            seed = self.seed,
            test_batchsize = 100,
            log_params = False,
        )


        return {'logger_directory': self.logger.directory, 'logger_name': self.logger.name}

if __name__ == "__main__":
    mysuite = MySuite()
    mysuite.start()