import os
import logging
import copy
import torch
import gym

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
        elif params['environment'] == 'swimmer':
            self.policy = DeepGaussianPolicy(
                state_dim,
                action_dim,
                hidden_neurons  = [32,32],
                mu_init         = None,
                logstd_init     = params["logstd_init"]*torch.ones(action_dim),
                learn_std       = params["learn_std"]
            )
        
        self.stepper = eval(params["stepper"])

        # Offpolicy data
        # ==============
        # Initial data for first offline CE estimation
        if params['ce_use_offline_data']:
            self.offline_policies = [self.policy]
            self.offline_batches  = [generate_batch(self.env, self.policy, self.env.horizon, params['batchsize'], seed=self.seed)]
        else:
            self.offline_policies = None
            self.offline_batches  = None

        # Prepare behavioural policies
        if isinstance(params['ce_batchsizes'], str):
            self.ce_batchsizes = eval(params['ce_batchsizes'])
        else:
            self.ce_batchsizes = params['ce_batchsizes']
        if self.ce_batchsizes is None:
            self.behavioural_policies = [copy.deepcopy(self.policy)]
        else:
            self.behavioural_policies = [copy.deepcopy(self.policy) for _ in range(len(self.ce_batchsizes)+1)]

        # Logger
        # ======
        self.debug_logger = setup_debug_logger(
            name        = str(rep),
            log_file    = os.path.join(params['path'],params['name'],'') + str(rep) + '_DEBUG' + '.log',
            level       = logging.DEBUG,
            stream      ='stderr'
        )


    def iterate(self, params, rep, n):
        if 'debug_target' == params['name']:

            log, self.offline_policies, self.offline_batches = reinforce_offpolicy_step(
                self.env, self.policy, self.env.horizon, self.behavioural_policies, self.offline_policies, self.offline_batches,
                batchsize = params['batchsize'], 
                baseline = params['baseline'],
                biased_offpolicy = params['biased_offpolicy'],
                ce_batchsizes = self.ce_batchsizes,
                disc = self.env.gamma,
                defensive_batch = params['defensive_batch'],
                debug_logger = self.debug_logger,
                estimator = params['estimator'],
                ce_tol_grad=params['ce_tol_grad'],
                ce_lr = params['ce_lr'],
                ce_initialize_behavioural_policies = params['ce_initialize_behavioural_policies'],
                ce_max_iter = params['ce_max_iter'],
                ce_weight_decay = params['ce_weight_decay'],
                ce_mis_normalize = params['ce_mis_normalize'],
                ce_mis_clip = params['ce_mis_clip'],
                ce_optimizer = params['ce_optimizer'],
                seed = params['seed']+n,
                shallow = isinstance(self.policy, ShallowGaussianPolicy),
                stepper = self.stepper,
                test_batchsize = params['batchsize'],
                log_grad = False,
                log_ce_params_norms = True,
                log_params_norms = True)
            
            # Uso la target corrente per la stima della CE
            self.offline_policies = [self.policy]
            self.offline_batches = [generate_batch(self.env, self.policy, self.env.horizon, params['batchsize'],
                                                    action_filter=None, 
                                                    seed=params['seed']+n, 
                                                    n_jobs=False)]
        else:
            log, self.offline_policies, self.offline_batches = reinforce_offpolicy_step(
                self.env, self.policy, self.env.horizon, self.behavioural_policies, self.offline_policies, self.offline_batches,
                batchsize = params['batchsize'], 
                baseline = params['baseline'],
                biased_offpolicy = params['biased_offpolicy'],
                ce_batchsizes = self.ce_batchsizes,
                disc = self.env.gamma,
                defensive_batch = params['defensive_batch'],
                debug_logger = self.debug_logger,
                estimator = params['estimator'],
                ce_tol_grad=params['ce_tol_grad'],
                ce_lr = params['ce_lr'],
                ce_initialize_behavioural_policies = params['ce_initialize_behavioural_policies'],
                ce_max_iter = params['ce_max_iter'],
                ce_weight_decay = params['ce_weight_decay'],
                ce_mis_normalize = params['ce_mis_normalize'],
                ce_mis_clip = params['ce_mis_clip'],
                ce_optimizer = params['ce_optimizer'],
                ce_optimize_mean = params['ce_optimize_mean'],
                ce_optimize_variance = params['ce_optimize_variance'],
                seed = params['seed']+n,
                shallow = isinstance(self.policy, ShallowGaussianPolicy),
                stepper = self.stepper,
                test_batchsize = params['batchsize'],
                log_grad = False,
                log_ce_params_norms = True,
                log_params_norms = True)
        
        return log

if __name__ == "__main__":
    mysuite = MySuite()
    mysuite.start()