#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REINFORCE family of algorithms (actor-only policy gradient)
@author: Giorgio Manganini
"""
from re import S
import torch
import time
import copy

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon, mean_sum_info
from potion.estimation.offpolicy_gradients import multioff_gpomdp_estimator
from potion.common.logger import Logger
from potion.common.misc_utils import seed_all_agent, concatenate
from potion.meta.steppers import ConstantStepper, Adam
from potion.algorithms.ce_optimization import argmin_CE, get_alphas, var_mean

def make_list(x):
    if type(x) is list:
        return x
    else:
        return [x]

def reinforce_offpolicy(env, policy, horizon, *,
                 action_filter = None,
                 batchsize = 100, 
                 baseline = 'avg',
                 biased_offpolicy = True,
                 ce_batchsizes = None,
                 ce_use_offline_data = True,
                 disc = 0.99,
                 defensive_batch = 0,
                 entropy_coeff = 0.,
                 estimate_var = False,
                 estimator = 'gpomdp',
                 info_key = 'danger',
                 iterations = 50,
                 log_grad = False,
                 log_ce_params = False,
                 log_params = False,
                 logger = Logger(name='reinforce_ce'),
                 parallel = False,
                 seed = None,
                 shallow = False,
                 stepper = ConstantStepper(1e-2),
                 test_batchsize = False,
                 verbose = 1):
    """
    REINFORCE/G(PO)MDP algorithmn
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    batchsize: number of trajectories used to estimate policy gradient  #TODO spiegare i dati
    disc: discount factor
    stepper: step size criterion. A constant step size is used by default
    action_filter: function to apply to the agent's action before feeding it to 
        the environment, not considered in gradient estimation. By default,
        the action is clipped to satisfy evironmental boundaries
    estimator: either 'reinforce' or 'gpomdp' (default). The latter typically
        suffers from less variance
    ce_batchsizes: #TODO
    defensive_batch: number of samples from target policy to use for gradient estimation
    biased_offpolicy: whether to use the samples employed in the cross-entropy optimization in the gradient estimation
    baseline: control variate to be used in the gradient estimator. Either
        'avg' (average reward, default), 'peters' (variance-minimizing) or
        'zero' (no baseline)
    logger: for human-readable logs (standard output, csv, tensorboard...). If None, logs are returned by the function
    shallow: whether to employ pre-computed score functions (only available for
        shallow policies)
    seed: random seed (None for random behavior)
    estimate_var: whether to estimate the variance of the gradient samples and their average
    test_batchsize: number of test trajectories used to evaluate the 
        corresponding deterministic policy at each iteration. If 0 or False, no 
        test is performed
    log_params: whether to include policy parameters in the human-readable logs
    log_grad: whether to include gradients in the human-readable logs
    log_ce_params: whether to save the parameters of the CE potimized behavioural policies
    parallel: number of parallel jobs for simulation. If 0 or False, 
        sequential simulation is performed.
    render: how often (every x iterations) to render the agent's behavior
        on a sample trajectory. If False, no rendering happens
    verbose: level of verbosity (0: only logs; 1: normal; 2: maximum)

    next_offline_policies : list
        List with set of policies that can be reused at the next iteration for CE optimization
    next_offline_bathces : list
        List with set of batches that can be reused at the next iteration for CE optimization
    """
    
    # Saving algorithm information
    # ============================
    # Store function parameters (do not move it from here!)
    algo_params = copy.deepcopy(locals())
    if logger is not None:
        # Save algorithm parameters and policy info
        logger.write_info({**algo_params, **policy.info()})

    # Algorithm preparation
    # =====================

    # Initial data for first offline CE estimation
    if ce_use_offline_data:
        offline_policies = [policy]
        offline_batches = [generate_batch(env, policy, horizon, batchsize, 
                                        action_filter=action_filter,
                                        seed=seed,
                                        n_jobs=parallel)]
    else:
        offline_policies = None
        offline_batches = None
    
    # Prepare function arguments for iterations
    del algo_params['iterations']
    del algo_params['logger']
    del algo_params['ce_use_offline_data']

    # Run
    # ===
    results = []
    for it in range(iterations):

        log_row, offline_policies, offline_batches = reinforce_offpolicy_step(**algo_params, offline_policies=offline_policies, offline_batches=offline_batches)
        
        if logger is not None:
            if not logger.ready:
                logger.open(log_row.keys())
            logger.write_row(log_row, it)
        else:
            results.append(log_row)

    # Cleaning log
    # ============
    if logger is not None:
        logger.close()

    return results


def reinforce_offpolicy_step(env, policy, horizon, offline_policies, offline_batches, *,
                    action_filter = None,
                    batchsize = 100, 
                    baseline = 'avg',
                    biased_offpolicy = True,
                    ce_batchsizes = None,
                    disc = 0.99,
                    defensive_batch = 0,
                    entropy_coeff = 0.,
                    estimate_var = False,
                    estimator = 'gpomdp',
                    info_key = 'danger',
                    log_grad = False,
                    log_ce_params = False,
                    log_params = False,
                    parallel = False,
                    seed = None,
                    shallow = False,
                    stepper = ConstantStepper(1e-2),
                    test_batchsize = False,
                    verbose = 1):
    
    start = time.time()

    # Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    # Showing info
    params = policy.get_flat()
    if verbose > 0:
        print('Parameters:', params)
    
    # Preparing log
    log_row = {}

    # Testing the corresponding deterministic policy
    if test_batchsize:
        test_batch = generate_batch(env, policy, horizon, test_batchsize, 
                                    action_filter=action_filter,
                                    seed=seed,
                                    n_jobs=parallel,
                                    deterministic=True,
                                    key=info_key)
        log_row['TestPerf'] = performance(test_batch, disc)
        log_row['TestInfo'] = mean_sum_info(test_batch).item()
        log_row['UTestPerf'] = performance(test_batch, 1)
    
    # Cross-entropy optimization of behavioural policy (or policies)
    # ==============================================================
    # CE optimization with offline batches
    if offline_policies:
        try:
            opt_ce_policy = argmin_CE(env, policy, offline_policies, offline_batches, 
                                    estimator=estimator,
                                    baseline=baseline,
                                    optimize_mean=True,
                                    optimize_variance=True)
        except(RuntimeError):
            # If CE minimization is not possible, keep the atrget policy
            opt_ce_policy = policy
    else:
        opt_ce_policy = policy
    
    # Further iterative CE optimization, sampling from the current behavioural policies
    ce_policies   = []
    ce_batches    = []
    if ce_batchsizes is not None:
        for ce_batchsize in ce_batchsizes:
            ce_policies.append(opt_ce_policy)
            ce_batches.append(
                generate_batch(env, opt_ce_policy, horizon, ce_batchsize, 
                               action_filter=action_filter, 
                               seed=seed, 
                               n_jobs=False)
            )
            try:
                opt_ce_policy = argmin_CE(env, policy, ce_policies, ce_batches, 
                                          estimator=estimator,
                                          baseline=baseline,
                                          optimize_mean=True,
                                          optimize_variance=True)
            except(RuntimeError):
                # If CE minimization is not possible, keep the previous opt_ce_policy
                pass

    # Selection of batches and policies for gradient estimation
    # =========================================================
    gradient_estimation_policies = []
    gradient_estimation_batches  = []
    
    # Sampling from optimized behavioural policy
    gradient_estimation_policies.append(opt_ce_policy)
    gradient_estimation_batches.append(
        generate_batch(env, opt_ce_policy, horizon, batchsize,
                       action_filter=action_filter, 
                       seed=seed, 
                       n_jobs=False)
    )
    
    if defensive_batch > 0:
        # Sampling from target policy to use defensive trajecotires
        gradient_estimation_policies.append(policy)
        gradient_estimation_batches.append(
            generate_batch(env, policy, horizon, defensive_batch,
                           action_filter=action_filter, 
                           seed=seed, 
                           n_jobs=False)
        )

    if biased_offpolicy:
        # Reusing all the samples used for CE optimization
        if offline_policies:
            gradient_estimation_policies.append(*offline_policies)
            gradient_estimation_batches.append(*offline_batches)
        if ce_policies:
            gradient_estimation_policies.append(*ce_policies)
            gradient_estimation_batches.append(*ce_batches)
    
    # Off-policy gradient stimation
    # =============================
    if estimator == 'gpomdp' and entropy_coeff == 0:
        grad_samples = multioff_gpomdp_estimator(
            concatenate(gradient_estimation_batches), disc, policy, gradient_estimation_policies,
            get_alphas(gradient_estimation_batches), baselinekind=baseline, result='samples', is_shallow=shallow)
        grad = torch.mean(grad_samples,0)
    else:
        raise NotImplementedError

    if verbose > 1:
        print('Gradients: ', grad)
    
    # Update of policy parameters
    # ===========================
    stepsize = stepper.next(grad)
    if not torch.is_tensor(stepsize):
        stepsize = torch.tensor(stepsize)
    if isinstance(stepper,Adam):
        new_params = params + stepsize
    else:
        new_params = params + stepsize * grad
    policy.set_from_flat(new_params)
    
    # Logging
    # ========
    log_row['Time']     = time.time() - start
    log_row['StepSize'] = torch.norm(stepsize).item()
    log_row['GradNorm'] = torch.norm(grad).item()
    log_row['Batch_total']  = batchsize + defensive_batch
    if ce_batchsizes is not None:
        log_row['Batch_total'] += sum(ce_batchsizes)

    if estimate_var:
        # Variance of gradients samples
        n_samples = grad_samples.shape[0]
        centered = grad_samples - grad.unsqueeze(0)
        grad_cov = (n_samples/(n_samples - 1) * 
                    torch.mean(torch.bmm(centered.unsqueeze(2), 
                                        centered.unsqueeze(1)),0))
        grad_var = torch.sum(torch.diag(grad_cov)).item() #for humans
        log_row['SampleVar'] = grad_var
    
        # Variance of the sample mean
        log_row['VarMean'] = var_mean(grad_samples)[1]

    batch = concatenate(gradient_estimation_batches)
    log_row['Perf']         = performance(batch, disc)
    log_row['Info']         = mean_sum_info(batch).item()
    log_row['UPerf']        = performance(batch, disc=1.)
    log_row['AvgHorizon']   = avg_horizon(batch)
    log_row['Exploration']  = policy.exploration().item()
    log_row['Entropy']      = policy.entropy(0.).item()
    
    if log_params:
        for i in range(policy.num_params()):
            log_row['param%d' % i] = params[i].item()

    if log_grad:
        for i in range(policy.num_params()):
            log_row['grad%d' % i] = grad[i].item()
    
    if ce_batchsizes is not None:
        if log_ce_params:
            for ce_it, pol in enumerate(ce_policies):
                for i,el in enumerate(pol.get_loc_params().tolist()):
                    log_row[f"ce_policy_loc{i}_{ce_it}"] = el
                for i,el in enumerate(make_list(pol.get_scale_params().tolist())):
                    log_row[f"ce_policy_scale{i}_{ce_it}"] = el

    # Return values
    # =============
    # If offline data for CE estimation is provided, return the new offline data for the next iteration
    if offline_policies:
        next_offline_policies = [gradient_estimation_policies[0]]
        next_offline_bathces  = [gradient_estimation_batches[0]]
    else:
        next_offline_policies = None
        next_offline_bathces  = None
    return log_row, next_offline_policies, next_offline_bathces

"""Testing"""
if __name__ == '__main__':
    from potion.envs.lq import LQ
    from potion.actors.continuous_policies import ShallowGaussianPolicy
    from potion.common.logger import Logger

    env        = LQ(max_pos=10, max_action = float('inf'))
    state_dim  = sum(env.observation_space.shape)
    action_dim = sum(env.action_space.shape)

    policy = ShallowGaussianPolicy(
        state_dim, # input size
        action_dim, # output size
        mu_init = 0*torch.ones(1), # initial mean parameters
        logstd_init = 0.0, # log of standard deviation
        learn_std = False # We are NOT going to learn the variance parameter
    )

    stepper = ConstantStepper(0.0001)
    seed = 42
    env.seed(seed)

    # log_dir     = 'logs_test'
    # log_name    = 'reinforce_ce'
    # logger      = Logger(directory=log_dir, name = log_name, modes=['csv'])
    logger = None

    res = reinforce_offpolicy(env, policy, env.horizon,
                 action_filter = None,
                 batchsize = 100, 
                 baseline = 'peters',
                 biased_offpolicy = True,
                 ce_batchsizes = None,
                 ce_use_offline_data = True,
                 disc = env.gamma,
                 defensive_batch = 0,
                 entropy_coeff = 0.,
                 estimate_var = False,
                 estimator = 'gpomdp',
                 info_key = 'danger',
                 iterations = 30,
                 log_grad = False,
                 log_ce_params = False,
                 log_params = False,
                 logger = logger,
                 parallel = False,
                 seed = seed,
                 shallow = isinstance(policy,ShallowGaussianPolicy),
                 stepper = stepper,
                 test_batchsize = 100,
                 verbose = 1)
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    res = pd.DataFrame(res)
    plt.plot(np.cumsum(res['Batch_total']), res['TestPerf'])
    plt.xlabel('Trajectories')
    plt.ylabel('Return')
    plt.show()
