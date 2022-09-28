import numpy as np
import torch

from expsuite import PyExperimentSuite
from potion.envs.lq import LQ
from potion.algorithms.ce_optimization import ce_optimization, get_alphas, var_mean
from potion.actors.continuous_policies import ShallowGaussianPolicy
from potion.common.misc_utils import concatenate
from potion.simulation.trajectory_generators import generate_batch
from potion.estimation.offpolicy_gradients import multioff_gpomdp_estimator
from potion.estimation.gradients import gpomdp_estimator

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
        if isinstance(params['ce_batchsizes'], str):
            ce_batchsizes = eval(params['ce_batchsizes'])
        else:
            ce_batchsizes = params['ce_batchsizes']

        # Off-policy estimation
        # ---------------------

        ## Cross Entropy behavioural policy optimization, with trajectories collection
        opt_ce_policy, ce_policies, ce_batches = ce_optimization(
            self.env, self.target_policy, ce_batchsizes,
            estimator=params["estimator"], baseline=params["baseline"], seed=self.seed)

        ## Selection of batches and policies used during CE optimization
        defensive = True
        biased_offpolicy = True

        off_policies = []
        off_batches  = []
        if biased_offpolicy:
            # Resure CE samples for final offpolicy estimation
            if defensive:
                off_policies = ce_policies
                off_batches  = ce_batches
            else:
                off_policies = ce_policies[1:]
                off_batches  = ce_batches[1:]
        elif defensive:
            off_policies = ce_policies[0]
            off_batches  = ce_batches[0] 
        if params["batchsize"] > 0:
            off_policies.append(opt_ce_policy)
            off_batches.append(
                generate_batch(self.env, opt_ce_policy, self.env.horizon, params["batchsize"], seed=self.seed, n_jobs=False)
            )

        ## Gradients estimation
        if params["estimator"] == 'gpomdp':
            off_grad_samples = multioff_gpomdp_estimator(
                concatenate(off_batches), self.env.gamma, self.target_policy, off_policies, get_alphas(off_batches),
                baselinekind=params["baseline"], result='samples', is_shallow=isinstance(self.target_policy,ShallowGaussianPolicy))
        else:
            raise NotImplementedError

        # On-policy estimation
        # ---------------------
        
        ## Trajectories collection
        on_batch = generate_batch(self.env, self.target_policy, self.env.horizon, sum(ce_batchsizes) + params["batchsize"], seed=self.seed)

        ## Gradients estimation
        if params["estimator"] == 'gpomdp':
            on_grad_samples = gpomdp_estimator(
                on_batch, self.env.gamma, self.target_policy, baselinekind=params["baseline"],
                result='samples', shallow=isinstance(self.target_policy,ShallowGaussianPolicy))

        # Save results
        # ------------
        results = {}
        results['grad_is']      = torch.mean(off_grad_samples,0).tolist()
        results['var_grad_is']  = var_mean(off_grad_samples)[1]
        results['grad_mc']      = torch.mean(on_grad_samples,0).tolist()
        results['var_grad_mc']  = var_mean(on_grad_samples)[1]
        
        return results

if __name__ == "__main__":
    mysuite = MySuite(config="experiments.cfg", numcores=1)
    print("*** Test start ***")
    mysuite.start()
    print("*** Test end ***")