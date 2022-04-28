#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REINFORCE family of algorithms (actor-only policy gradient)
@author: Matteo Papini, Giorgio Manganini
"""
import math
import torch
import time

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon, mean_sum_info
from potion.estimation.gradients import gpomdp_estimator, reinforce_estimator, egpomdp_estimator
from potion.estimation.offpolicy_gradients import _shallow_multioff_gpomdp_estimator
from potion.estimation.importance_sampling import multiple_importance_weights
from potion.common.logger import Logger
from potion.common.misc_utils import seed_all_agent, concatenate
from potion.meta.steppers import ConstantStepper
from potion.algorithms.ce_optimization import argmin_CE, get_alphas, var_mean

def reinforce_step(env, policy, horizon, *,
                    batchsize = 100, 
                    disc = 0.99,
                    stepper = ConstantStepper(1e-2),
                    entropy_coeff = 0.,
                    action_filter = None,
                    estimator = 'gpomdp',
                    n_offpolicy_opt = 0,
                    defensive = True,
                    biased_offpolicy = True,
                    baseline = 'avg',
                    shallow = False,
                    seed = None,
                    estimate_var = False,
                    test_batchsize = False,
                    info_key = 'danger',
                    log_params = False,
                    log_grad = False,
                    parallel = False,
                    verbose = 1):
    """
    REINFORCE/G(PO)MDP algorithmn
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    batchsize: number of trajectories used to estimate policy gradient
    disc: discount factor
    stepper: step size criterion. A constant step size is used by default
    action_filter: function to apply to the agent's action before feeding it to 
        the environment, not considered in gradient estimation. By default,
        the action is clipped to satisfy evironmental boundaries
    estimator: either 'reinforce' or 'gpomdp' (default). The latter typically
        suffers from less variance
    n_offpolicy_opt: number of optimized behavioural policies
    defensive: whether to use the target policy in the gradient estimation
    biased_offpolicy: whether to use the samples employed in the cross-entropy optimization in the gradient estimation
    baseline: control variate to be used in the gradient estimator. Either
        'avg' (average reward, default), 'peters' (variance-minimizing) or
        'zero' (no baseline)
    logger: for human-readable logs (standard output, csv, tensorboard...)
    shallow: whether to employ pre-computed score functions (only available for
        shallow policies)
    seed: random seed (None for random behavior)
    estimate_var: whether to estimate the variance of the gradient samples and their average
    test_batchsize: number of test trajectories used to evaluate the 
        corresponding deterministic policy at each iteration. If 0 or False, no 
        test is performed
    log_params: whether to include policy parameters in the human-readable logs
    log_grad: whether to include gradients in the human-readable logs
    parallel: number of parallel jobs for simulation. If 0 or False, 
        sequential simulation is performed.
    render: how often (every x iterations) to render the agent's behavior
        on a sample trajectory. If False, no rendering happens
    verbose: level of verbosity (0: only logs; 1: normal; 2: maximum)
    """
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    log_keys = [
        'Perf', 
        'UPerf', 
        'AvgHorizon', 
        'StepSize', 
        'BatchSize',
        'GradNorm', 
        'Time',
        'StepSize',
        'Exploration',
        'Entropy',
        'Info'
    ]
    if estimate_var:
        log_keys.append('SampleVar')
        log_keys.append('VarMean')
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf', 'TestInfo']
    log_row = dict.fromkeys(log_keys)
    
    #Learning step
    start = time.time()
    params = policy.get_flat()
    if verbose > 0:
        print('Parameters:', params)
    
    #Test the corresponding deterministic policy
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
    
    # Collect trajectories
    # ====================
    if n_offpolicy_opt > 0:
        # Off-policy

        # Cross-Entropy optimization for off-policies
        if biased_offpolicy:
            # Resure CE samples for final offpolicy estimation
            N_per_it = math.ceil(batchsize / (n_offpolicy_opt+1))
        else:
            # Use N/2 samples for CE, and keep N/2 samples for offpolicy estimation
            N_per_it = math.ceil((batchsize/2) / n_offpolicy_opt)

        ce_policies = [policy]
        ce_batches  = [generate_batch(env, policy, horizon, N_per_it, 
                                        action_filter=action_filter, 
                                        seed=seed, 
                                        n_jobs=False)]
        for _ in range(n_offpolicy_opt):
            opt_ce_policy = argmin_CE(env, policy, ce_policies, ce_batches, 
                                    estimator=estimator,
                                    baseline=baseline,
                                    optimize_mean=True,
                                    optimize_variance=True)
            ce_policies.append(opt_ce_policy)
            ce_batches.append(generate_batch(env, opt_ce_policy, horizon, N_per_it, 
                                                action_filter=action_filter, 
                                                seed=seed, 
                                                n_jobs=False))

        # Selection of batches and off-policies used during CE optimization
        if biased_offpolicy:
            # Resure CE samples for final offpolicy estimation
            if defensive:
                off_policies = ce_policies
                off_batches  = ce_batches
            else:
                off_policies = ce_policies[1:]
                off_batches  = ce_batches[1:]
        else:
            # Use N/2 samples for offpolicy estimation (first N/2 were used for CE)
            off_batch = generate_batch(env, opt_ce_policy, horizon, batchsize/2,
                                        action_filter=action_filter, 
                                        seed=seed, 
                                        n_jobs=False)
            off_policies = [opt_ce_policy]
            off_batches  = [off_batch]
            if defensive:
                # Use also the first samples (<N/2) from target policy for the first estimation of CE
                off_policies = [policy, opt_ce_policy]
                off_batches  = [ce_batches[0], off_batch]
        batch = concatenate(off_batches)
        
    else:
        # On-policy
        batch = generate_batch(env, policy, horizon, batchsize, 
                                action_filter=action_filter, 
                                seed=seed, 
                                n_jobs=parallel,
                                key=info_key)
    
    log_row['Perf']         = performance(batch, disc)
    log_row['Info']         = mean_sum_info(batch).item()
    log_row['UPerf']        = performance(batch, disc=1.)
    log_row['AvgHorizon']   = avg_horizon(batch)
    log_row['Exploration']  = policy.exploration().item()
    log_row['Entropy']      = policy.entropy(0.).item()
    
    # Estimate policy gradient
    # ========================
    if n_offpolicy_opt == 0:
        # On-policy gradient estimation
        if estimator == 'gpomdp' and entropy_coeff == 0:
            grad_samples = gpomdp_estimator(batch, disc, policy, 
                                            baselinekind=baseline, 
                                            shallow=shallow,
                                            result='samples')
        elif estimator == 'gpomdp':
            grad_samples = egpomdp_estimator(batch, disc, policy, entropy_coeff,
                                            baselinekind=baseline,
                                            shallow=shallow,
                                            result='samples')
        elif estimator == 'reinforce':
            grad_samples = reinforce_estimator(batch, disc, policy, 
                                            baselinekind=baseline, 
                                            shallow=shallow,
                                            result='samples')
        else:
            raise ValueError('Invalid policy gradient estimator')
        grad = torch.mean(grad_samples, 0)
        
    elif n_offpolicy_opt > 0:
        # Off-policy estimation
        if estimator == 'gpomdp' and entropy_coeff == 0 and shallow:
            grad_samples = _shallow_multioff_gpomdp_estimator(
                batch, disc, policy, off_policies, get_alphas(off_batches),
                baselinekind=baseline, 
                result='samples'
            )
            grad = torch.mean(grad_samples,0)
        elif estimator == 'reinforce':
            #TODO offpolicy reinforce
            raise NotImplementedError
        else:
            raise NotImplementedError

    if estimate_var:
        # Variance of gradients samples
        centered = grad_samples - grad.unsqueeze(0)
        grad_cov = (batchsize/(batchsize - 1) * 
                    torch.mean(torch.bmm(centered.unsqueeze(2), 
                                        centered.unsqueeze(1)),0))
        grad_var = torch.sum(torch.diag(grad_cov)).item() #for humans
        log_row['SampleVar'] = grad_var
    
        # Variance of the sample mean
        log_row['VarMean'] = var_mean(grad_samples)[1]

    if verbose > 1:
        print('Gradients: ', grad)
    log_row['GradNorm'] = torch.norm(grad).item()
    
    #Select meta-parameters
    stepsize = stepper.next(grad)
    if not torch.is_tensor(stepsize):
        stepsize = torch.tensor(stepsize)
    log_row['StepSize'] = torch.norm(stepsize).item()
    log_row['BatchSize'] = batchsize
    
    #Update policy parameters
    new_params = params + stepsize * grad
    policy.set_from_flat(new_params)
    
    #Log
    log_row['Time'] = time.time() - start
    if log_params:
        for i in range(policy.num_params()):
            log_row['param%d' % i] = params[i].item()
    if log_grad:
        for i in range(policy.num_params()):
            log_row['grad%d' % i] = grad[i].item()
    
    return log_row