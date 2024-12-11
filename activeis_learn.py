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
from potion.algorithms.reinforce_offpolicy_g import reinforce_offpolicy_step_g
from potion.algorithms.reinforce import reinforce_step
from potion.simulation.trajectory_generators import generate_batch

from gymnasium.spaces import Box
import numpy as np


class ClipActionOwn(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Clip the continuous action within the valid :class:`Box` observation space bound.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ClipAction
        >>> env = gym.make("Hopper-v4")
        >>> env = ClipAction(env)
        >>> env.action_space
        Box(-1.0, 1.0, (3,), float32)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([5.0, -2.0, 0.0]))
        ... # Executes the action np.array([1.0, -1.0, 0]) in the base environment
    """

    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

    def action(self, action):
        """Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """
        return np.clip(action, -1, 1)


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
            self.env = gym.wrappers.ClipAction(gym.make('ContCartPole-v0'))
            self.env.gamma = 1
            self.env.horizon = params['horizon']
            self.env.seed(self.seed)
        elif params['environment'] == 'swimmer':
            self.env = gym.wrappers.ClipAction(gym.make('Swimmer-v4'))
            self.env.horizon = params['horizon']
            self.env.gamma = params['gamma']
        elif params['environment'] == 'halfcheetah':
            self.env = ClipActionOwn(gym.make('HalfCheetah-v4'))
            self.env.horizon = params['horizon']
            self.env.gamma = params['gamma']
        elif params['environment'] == 'ant':
            self.env = gym.wrappers.ClipAction(gym.make('Ant-v4'))
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
        if params['environment'] == 'lq' or params['environment'] == 'cartpole'  or params['environment'] == "halfcheetah":
            self.policy = ShallowGaussianPolicy(
                state_dim, # input size
                action_dim, # output size
                mu_init     = params["mu_init"]*torch.ones([state_dim, action_dim]),
                logstd_init = params["logstd_init"]*torch.ones(action_dim),
                learn_std   = params["learn_std"]
            )
        elif params['environment'] == 'swimmer' or params['environment'] == "ant":
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

        # Offpolicy data
        # ==============
        # Initial data for first offline CE estimation
        if params['ce_use_offline_data']:
            self.offline_policies = [self.policy]
            self.offline_batches  = [generate_batch(self.env, self.policy, self.env.horizon, params['batchsize'], seed=self.seed)]
            self.njt_batches =  [generate_batch(self.env, self.policy, self.env.horizon, params['batchsize'], seed=self.seed)]
        else:
            self.offline_policies = None
            self.offline_batches  = None
            self.njt_batches = None

        # Prepare behavioural policies
        if isinstance(params['ce_batchsizes'], str):
            self.ce_batchsizes = eval(params['ce_batchsizes'])
        else:
            self.ce_batchsizes = params['ce_batchsizes']
        if self.ce_batchsizes is None:
            self.behavioural_policies = [copy.deepcopy(self.policy)]
        else:
            print(self.ce_batchsizes)
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

            log, self.offline_policies, self.offline_batches, self.njt_batches = reinforce_offpolicy_step_g(
                self.env, self.policy, self.env.horizon, self.behavioural_policies, self.offline_policies, self.offline_batches, self.njt_batches,
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
                log_params_norms = True,
                njt_batchsize = params['njt_batchsize'],
                window=params['window'],)
            
            # Uso la target corrente per la stima della CE
            self.offline_policies = [self.policy]*params["batchsize"]
            self.offline_batches = [generate_batch(self.env, self.policy, self.env.horizon, params['batchsize'],
                                                    action_filter=None, 
                                                    seed=params['seed']+n, 
                                                    n_jobs=False)]
            self.njt_batches =  [generate_batch(self.env, self.policy, self.env.horizon, params['batchsize'],
                                                    action_filter=None, 
                                                    seed=params['seed']+n, 
                                                    n_jobs=False)]
        else:
            #self.offline_policies = self.offline_policies[-params["window"] * (params["batchsize"]+params["defensive_batch"]):]
            #self.offline_batches = self.offline_batches[-params["window"] * (params["batchsize"]+params["defensive_batch"]):]
            #self.njt_batches = self.njt_batches[-params["window"] * (params["batchsize"]+params["defensive_batch"]):]

            if params['ce_use_offline_data']:
                self.offline_policies = self.offline_policies[-params["window"]:]
                self.offline_batches = self.offline_batches[-params["window"]:]
                self.njt_batches = self.njt_batches[-params["window"]:]


            if params["ce_update_frequency"] is not None:
                if params["ce_update_frequency"] == 0 or n % params["ce_update_frequency"] == 0:


                    log, self.offline_policies, self.offline_batches, self.njt_batches = reinforce_offpolicy_step_g(
                        self.env, self.policy, self.env.horizon, self.behavioural_policies, self.offline_policies, self.offline_batches, self.njt_batches,
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
                        log_params_norms = True,
                        njt_batchsize = params['njt_batchsize'],
                        )
        

                else:
                    all_ce_batchsizes = sum(self.ce_batchsizes) if self.ce_batchsizes is not None else 0

                    log, self.offline_policies, self.offline_batches, self.njt_batches = reinforce_offpolicy_step_g(
                        self.env, self.policy, self.env.horizon, self.behavioural_policies, self.offline_policies, self.offline_batches, self.njt_batches,
                        batchsize = params['batchsize'],# + params["njt_batchsize"] + all_ce_batchsizes, 
                        baseline = params['baseline'],
                        biased_offpolicy = params['biased_offpolicy'],
                        ce_batchsizes = None,
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
                        test_batchsize = params['batchsize'], #+ params["njt_batchsize"] + all_ce_batchsizes,
                        log_grad = False,
                        log_ce_params_norms = True,
                        log_params_norms = True,
                        njt_batchsize = params['njt_batchsize'],
                        )
            else:
                log, self.offline_policies, self.offline_batches, self.njt_batches = reinforce_offpolicy_step_g(
                        self.env, self.policy, self.env.horizon, self.behavioural_policies, self.offline_policies, self.offline_batches, self.njt_batches,
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
                        log_params_norms = True,
                        njt_batchsize = params['njt_batchsize'],
                        )

        return log

if __name__ == "__main__":
    mysuite = MySuite()
    mysuite.start()