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

def make_list(x):
    if type(x) is list:
        return x
    else:
        return [x]

def reinforce_step(env, policy, horizon, *,
                    batchsize = 100, 
                    disc = 0.99,
                    stepper = ConstantStepper(1e-2),
                    entropy_coeff = 0.,
                    action_filter = None,
                    estimator = 'gpomdp',
                    ce_batchsizes = None,
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
                    log_ce_params = False,
                    parallel = False,
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
    log_ce_params: whether to save the parameters of the CE potimized behavioural policies
    parallel: number of parallel jobs for simulation. If 0 or False, 
        sequential simulation is performed.
    render: how often (every x iterations) to render the agent's behavior
        on a sample trajectory. If False, no rendering happens
    verbose: level of verbosity (0: only logs; 1: normal; 2: maximum)
    """
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    # Prepare the log
    # ================
    log_keys = [
        'Perf', 
        'UPerf', 
        'AvgHorizon', 
        'StepSize', 
        'BatchSize',
        'Batch_total',
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
    if ce_batchsizes is None:
        log_keys.append('CE_batchsize')
    else:
        ce_batchsizes = make_list(ce_batchsizes)
        log_keys += ['CE_batchsize_%d' % i for i,_ in enumerate(ce_batchsizes)]
        if log_ce_params:
            # save means and diagonal scales of behavioural policies
            log_keys += [f"ce_policy_loc{i}_{j}" for j in range(ce_batchsizes) for i in range(policy.num_loc_params()) ]
            log_keys += [f"ce_policy_scale{i}_{j}" for j in range(ce_batchsizes) for i in range(policy.num_scale_params()) ]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf', 'TestInfo']
    log_row = dict.fromkeys(log_keys)
    
    log_row['BatchSize'] = batchsize
    if ce_batchsizes is not None:
        for i,el in enumerate(ce_batchsizes):
            log_row[f"CE_batchsize_{i}"] = el
        log_row['Batch_total'] = sum(ce_batchsizes) + batchsize
    else:
        log_row['CE_batchsize'] = None
        log_row['Batch_total']  = batchsize

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
    if ce_batchsizes is not None:
        # Off-policy

        # Cross-Entropy optimization for off-policies
        opt_ce_policy = policy
        ce_policies   = []
        ce_batches    = []
        for _,ce_batchsize in enumerate(ce_batchsizes):
            ce_policies.append(opt_ce_policy)
            ce_batches.append(
                generate_batch(env, opt_ce_policy, horizon, ce_batchsize, 
                                            action_filter=action_filter, 
                                            seed=seed, 
                                            n_jobs=False)
            )
            opt_ce_policy = argmin_CE(env, policy, ce_policies, ce_batches, 
                                    estimator=estimator,
                                    baseline=baseline,
                                    optimize_mean=True,
                                    optimize_variance=True)

        # Selection of batches and off-policies used during CE optimization
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
        if batchsize > 0:
            off_policies.append(opt_ce_policy)
            off_batches.append(
                generate_batch(env, opt_ce_policy, horizon, batchsize,
                                        action_filter=action_filter, 
                                        seed=seed, 
                                        n_jobs=False)
            )
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
    if ce_batchsizes is None:
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
        
    else:
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
        n_samples = grad_samples.shape[0]
        centered = grad_samples - grad.unsqueeze(0)
        grad_cov = (n_samples/(n_samples - 1) * 
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
    
    if ce_batchsizes is not None:
        if log_ce_params:
            for ce_it, pol in enumerate(ce_policies[1:]):   # Skip the target policy
                for i,el in enumerate(pol.get_loc_params().tolist()):
                    log_row[f"ce_policy_loc{i}_{ce_it}"] = el
                for i,el in enumerate(make_list(pol.get_scale_params().tolist())):
                    log_row[f"ce_policy_scale{i}_{ce_it}"] = el

    return log_row