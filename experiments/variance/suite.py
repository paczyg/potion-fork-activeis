import numpy as np
import torch

from expsuite import PyExperimentSuite
from potion.envs.lq import LQ
from potion.algorithms.ce_optimization import algo
from potion.actors.continuous_policies import ShallowGaussianPolicy

class MySuite(PyExperimentSuite):

    def reset(self, params, rep):

        self.seed   = params['seed']*rep

        state_dim   = params['state_dim']
        mu_init     = params['mu_init']
        logstd_init = params['logstd_init']
        horizon     = params['horizon']

        # Environment
        # -----------
        self.env = LQ(state_dim, 1, max_pos=10, max_action = float('inf'), random=False)
        self.env.horizon = horizon
        self.env.seed(self.seed)

        # Target Policy
        # -------------
        self.target_policy = ShallowGaussianPolicy(
            state_dim, # input size
            1, # output size
            mu_init = mu_init*torch.ones(state_dim), # initial mean parameters
            logstd_init = logstd_init, # log of standard deviation
            learn_std = False # We are NOT going to learn the variance parameter
        )

    def iterate(self, params, rep, n):
        result, _, _ = algo(
            self.env,
            self.target_policy,
            n_per_it            = params['n_per_it'],
            n_ce_iterations     = params['n_ce_iterations'], 
            estimator           = params['estimator'],
            baseline            = params['baseline'],
            action_filter       = params['action_filter'],
            window              = None,
            optimize_mean       = True,
            optimize_variance   = True,
            reuse_samples       = True,
            run_mc_comparison   = True
        )

        return result

if __name__ == "__main__":
    mysuite = MySuite()
    print(mysuite.options)
    mysuite.start()