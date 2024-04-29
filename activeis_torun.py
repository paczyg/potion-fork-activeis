import numpy as np
import torch
import gym

from expsuite import PyExperimentSuite
from potion.envs.lq import LQ
from potion.algorithms.ce_optimization import get_alphas, var_mean
from potion.algorithms.ce_optimization_g import ce_optimization_g
from potion.actors.continuous_policies import ShallowGaussianPolicy, DeepGaussianPolicy
from potion.common.misc_utils import concatenate
from potion.simulation.trajectory_generators import generate_batch
from potion.estimation.offpolicy_gradients import multioff_gpomdp_estimator
from potion.estimation.gradients import gpomdp_estimator

class MySuite(PyExperimentSuite):

    def reset(self, params, rep):

        self.seed   = params['seed']*rep

        # Environment
        # -----------
        if params['env'] == 'lq':
            self.env = LQ(params['state_dim'], 1, max_pos=10, max_action = float('inf'), random=False)
            self.env.horizon = params['horizon']
            self.env.seed(self.seed)
        elif params['env'] == 'cartpole':
            self.env = gym.make('ContCartPole-v0')
            self.env.gamma = 1
            self.env.horizon = params['horizon']
            self.env.seed(self.seed)
        elif params['env'] == 'swimmer':
            self.env = gym.make('Swimmer-v4')
            self.env.horizon = params['horizon']
            self.env.gamma = params['gamma']
        else:
            raise NotImplementedError
        state_dim  = sum(self.env.observation_space.shape)
        action_dim = sum(self.env.action_space.shape)

        # Target Policy
        # -------------
        if params['env'] == 'lq' or params['env'] == 'cartpole':
            self.target_policy = ShallowGaussianPolicy(
                state_dim,
                action_dim,
                mu_init = params['mu_init']*torch.ones(state_dim),
                logstd_init = params['logstd_init']*torch.ones(action_dim),
                learn_std = params["learn_std"]
            )
        elif params['env'] == 'swimmer':
            self.target_policy = DeepGaussianPolicy(
                state_dim,
                action_dim,
                hidden_neurons  = [32,32],
                mu_init         = None,
                logstd_init     = params["logstd_init"]*torch.ones(action_dim),
                learn_std       = params["learn_std"]
            )

    def iterate(self, params, rep, n):
        if isinstance(params['ce_batchsizes'], str):
            ce_batchsizes = eval(params['ce_batchsizes'])
        else:
            ce_batchsizes = params['ce_batchsizes']

        # Off-policy estimation
        # ---------------------

        ## Cross Entropy behavioural policy optimization, with trajectories collection
        opt_ce_policy, ce_policies, ce_batches = ce_optimization_g(
            self.env, self.target_policy, ce_batchsizes,
            divergence = params['ce_divergence'],
            estimator = params['estimator'],
            baseline = params['baseline'],
            lr = params['ce_lr'],
            max_iter = params['ce_max_iter'],
            tol_grad = params['ce_tol_grad'],
            seed = self.seed
        )

        ## Selection of batches and policies used during CE optimization
        off_policies = []
        off_batches  = []
        
        if params['biased_offpolicy']:
            # Resure CE samples for final offpolicy estimation
                off_policies = off_policies + ce_policies
                off_batches = off_batches + ce_batches
        
        if params["batchsize"] > 0:
            
            if params['defensive_coeff'] > 0:
                off_policies.append(self.target_policy)
                off_batches.append(
                    generate_batch(
                        self.env,
                        self.target_policy,
                        self.env.horizon,
                        round(params["batchsize"]*params['defensive_coeff']),
                        seed = self.seed,
                        n_jobs = False) )
            
            off_policies.append(opt_ce_policy)
            off_batches.append(
                generate_batch(
                    self.env,
                    opt_ce_policy,
                    self.env.horizon,
                    round(params["batchsize"]*(1-params['defensive_coeff'])),
                    seed=self.seed,
                    n_jobs=False) )

        ## Gradients estimation
        if params["estimator"] == 'gpomdp':
            off_grad_samples, _ = multioff_gpomdp_estimator(
                concatenate(off_batches),
                self.env.gamma,
                self.target_policy,
                off_policies,
                get_alphas(off_batches),
                baselinekind = params["baseline"],
                result = 'samples',
                is_shallow = isinstance(self.target_policy, ShallowGaussianPolicy)
            )
        else:
            raise NotImplementedError

        # On-policy estimation
        # ---------------------
        
        ## Trajectories collection
        on_batch = generate_batch(
            self.env,
            self.target_policy,
            self.env.horizon,
            sum(ce_batchsizes) + params["batchsize"],
            seed = self.seed
        )

        ## Gradients estimation
        if params["estimator"] == 'gpomdp':
            on_grad_samples = gpomdp_estimator(
                on_batch,
                self.env.gamma,
                self.target_policy,
                baselinekind = params["baseline"],
                result = 'samples',
                shallow = isinstance(self.target_policy,ShallowGaussianPolicy)
            )
        else:
            raise NotImplementedError
        
        # Save results
        # ------------
        results = {}
        results['grad_is'] = torch.mean(off_grad_samples,0).tolist()
        results['var_grad_is'] = var_mean(off_grad_samples)[1]
        results['grad_mc'] = torch.mean(on_grad_samples,0).tolist()
        results['var_grad_mc'] = var_mean(on_grad_samples)[1]
        
        return results

if __name__ == "__main__":
    mysuite = MySuite()
    mysuite.start()