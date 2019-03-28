#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe Policy Gradient (actor-only)
@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon
from potion.estimation.gradients import gpomdp_estimator, reinforce_estimator
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent
import torch
import time
import math
import scipy.stats as sts

def incr_safepg(env, policy, horizon, lip_const, var_bound, *,
                    conf = 0.2,
                    max_batchsize = 10000,
                    iterations = 1000,
                    disc = 0.99,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    logger = Logger(name='safepg'),
                    shallow = True,
                    seed = None,
                    test_batchsize = False,
                    save_params = 100,
                    log_params = False,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    SafePG algorithm, incremental version
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    max_batchsize: maximum number of trajectories to estimate policy gradient
    iterations: number of policy updates
    disc: discount factor
    action_filter: function to apply to the agent's action before feeding it to 
        the environment, not considered in gradient estimation. By default,
        the action is clipped to satisfy evironmental boundaries
    estimator: either 'reinforce' or 'gpomdp' (default). The latter typically
        suffers from less variance
    baseline: control variate to be used in the gradient estimator. Either
        'avg' (average reward, default), 'peters' (variance-minimizing) or
        'zero' (no baseline)
    logger: for human-readable logs (standard output, csv, tensorboard...)
    shallow: whether to employ pre-computed score functions (only available for
        shallow policies)
    seed: random seed (None for random behavior)
    test_batchsize: number of test trajectories used to evaluate the 
        corresponding deterministic policy at each iteration. If 0 or False, no 
        test is performed
    save_params: how often (every x iterations) to save the policy 
        parameters to disk. Final parameters are always saved for 
        x>0. If False, they are never saved.
    log_params: whether to include policy parameters in the human-readable logs
    log_grad: whether to include gradients in the human-readable logs
    parallel: number of parallel jobs for simulation. If 0 or False, 
        sequential simulation is performed.
    render: how often (every x iterations) to render the agent's behavior
        on a sample trajectory. If False, no rendering happens
    verbose: level of verbosity (0: only logs; 1: normal; 2: maximum)
    """
    #Defaults
    if action_filter is None:
        action_filter = clip(env)
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'REINFORCE',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'Disc': disc,
                   'ConfidenceParam': conf,
                   'LipschitzConstant': lip_const,
                   'VarianceBound': var_bound,
                   'Seed': seed,
                   }
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 
                'UPerf', 
                'AvgHorizon', 
                'StepSize', 
                'GradNorm', 
                'Time',
                'StepSize',
                'BatchSize']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    #Learning loop
    it = 0
    low_samples = True
    unsafe = False
    while(it < iterations and not unsafe):
        #Begin iteration
        start = time.time()
        if verbose:
            print('\nIteration ', it)
        params = policy.get_flat()
        if verbose > 1:
            print('Parameters:', params)
        
        #Test the corresponding deterministic policy
        if test_batchsize:
            test_batch = generate_batch(env, policy, horizon, test_batchsize, 
                                        action_filter=action_filter,
                                        seed=seed,
                                        parallel=parallel,
                                        deterministic=True)
            log_row['TestPerf'] = performance(test_batch, disc)
            log_row['UTestPerf'] = performance(test_batch, 1)
        
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon, 
                           episodes=1, 
                           action_filter=action_filter, 
                           render=True)
    
        #Sampling loop
        batch = []
        while True:
            #Collect one trajectory
            batch += generate_batch(env, policy, horizon, 1, 
                                   action_filter=action_filter, 
                                   seed=seed, 
                                   parallel=parallel, 
                                   n_jobs=parallel)
            
            #Estimate policy gradient
            if estimator == 'gpomdp':
                grad = gpomdp_estimator(batch, disc, policy, 
                                    baselinekind=baseline, 
                                    shallow=shallow)
            elif estimator == 'reinforce':
                grad = reinforce_estimator(batch, disc, policy, 
                                       baselinekind=baseline, 
                                       shallow=shallow)
            else:
                raise ValueError('Invalid policy gradient estimator')
        
            
            optimal_batchsize = torch.ceil(4 * var_bound / 
                               (conf * torch.norm(grad)**2)).item()
            batchsize = len(batch)
            print('%d -> %d' % (batchsize, optimal_batchsize))
            if batchsize >= optimal_batchsize or batchsize >= max_batchsize:
                break
        
        if verbose > 1:
            print('Gradients: ', grad)
        log_row['GradNorm'] = torch.norm(grad).item()
        log_row['Perf'] = performance(batch, disc)
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
            
        #Select meta-parameters
        #Step size
        if low_samples:    
            stepsize = 1. / lip_const * (1 - 
                                          math.sqrt(var_bound / (conf * batchsize)) / 
                                          torch.norm(grad).item())
        else:
            stepsize = 1. / (2 * lip_const)  
        log_row['StepSize'] = torch.norm(torch.tensor(stepsize)).item()
        
        log_row['BatchSize'] = batchsize #current
        low_samples = (optimal_batchsize >= max_batchsize)
        min_safe_batchsize = torch.ceil(var_bound / (conf * torch.norm(grad)**2)).item()
        if batchsize < min_safe_batchsize:
            print('Unsafe, stopping. Would require %d samples' % min_safe_batchsize)
            unsafe = True
        
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
        logger.write_row(log_row, it)
        
        #Save parameters
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        #Next iteration
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()

def safepg(env, policy, horizon, lip_const, var_bound, *,
                    conf = 0.2,
                    init_batchsize = 500,
                    max_batchsize = 1000,
                    iterations = 1000,
                    disc = 0.99,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    logger = Logger(name='safepg'),
                    shallow = True,
                    seed = None,
                    test_batchsize = False,
                    save_params = 100,
                    log_params = False,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    SafePG algorithm
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    batchsize: number of trajectories used to estimate policy gradient
    iterations: number of policy updates
    disc: discount factor
    action_filter: function to apply to the agent's action before feeding it to 
        the environment, not considered in gradient estimation. By default,
        the action is clipped to satisfy evironmental boundaries
    estimator: either 'reinforce' or 'gpomdp' (default). The latter typically
        suffers from less variance
    baseline: control variate to be used in the gradient estimator. Either
        'avg' (average reward, default), 'peters' (variance-minimizing) or
        'zero' (no baseline)
    logger: for human-readable logs (standard output, csv, tensorboard...)
    shallow: whether to employ pre-computed score functions (only available for
        shallow policies)
    seed: random seed (None for random behavior)
    test_batchsize: number of test trajectories used to evaluate the 
        corresponding deterministic policy at each iteration. If 0 or False, no 
        test is performed
    save_params: how often (every x iterations) to save the policy 
        parameters to disk. Final parameters are always saved for 
        x>0. If False, they are never saved.
    log_params: whether to include policy parameters in the human-readable logs
    log_grad: whether to include gradients in the human-readable logs
    parallel: number of parallel jobs for simulation. If 0 or False, 
        sequential simulation is performed.
    render: how often (every x iterations) to render the agent's behavior
        on a sample trajectory. If False, no rendering happens
    verbose: level of verbosity (0: only logs; 1: normal; 2: maximum)
    """
    #Defaults
    if action_filter is None:
        action_filter = clip(env)
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'REINFORCE',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'Disc': disc,
                   'ConfidenceParam': conf,
                   'LipschitzConstant': lip_const,
                   'VarianceBound': var_bound,
                   'InitialBatchSize': init_batchsize,
                   'Seed': seed,
                   }
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 
                'UPerf', 
                'AvgHorizon', 
                'StepSize', 
                'GradNorm', 
                'Time',
                'StepSize',
                'BatchSize']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    #Learning loop
    it = 0
    batchsize = init_batchsize
    low_samples = True
    unsafe = False
    while(it < iterations and not unsafe):
        #Begin iteration
        start = time.time()
        if verbose:
            print('\nIteration ', it)
        params = policy.get_flat()
        if verbose > 1:
            print('Parameters:', params)
        
        #Test the corresponding deterministic policy
        if test_batchsize:
            test_batch = generate_batch(env, policy, horizon, test_batchsize, 
                                        action_filter=action_filter,
                                        seed=seed,
                                        parallel=parallel,
                                        deterministic=True)
            log_row['TestPerf'] = performance(test_batch, disc)
            log_row['UTestPerf'] = performance(test_batch, 1)
        
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon, 
                           episodes=1, 
                           action_filter=action_filter, 
                           render=True)
    
        #Collect trajectories
        if verbose:
            print('Sampling %d trajectories' % batchsize)
        batch = generate_batch(env, policy, horizon, batchsize, 
                               action_filter=action_filter, 
                               seed=seed, 
                               parallel=parallel, 
                               n_jobs=parallel)
        log_row['Perf'] = performance(batch, disc)
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
    
        #Estimate policy gradient
        if estimator == 'gpomdp':
            grad = gpomdp_estimator(batch, disc, policy, 
                                    baselinekind=baseline, 
                                    shallow=shallow)
        elif estimator == 'reinforce':
            grad = reinforce_estimator(batch, disc, policy, 
                                       baselinekind=baseline, 
                                       shallow=shallow)
        else:
            raise ValueError('Invalid policy gradient estimator')
        if verbose > 1:
            print('Gradients: ', grad)
        log_row['GradNorm'] = torch.norm(grad).item()
        
        #Select meta-parameters
        #Step size
        if low_samples:    
            stepsize = 1. / lip_const * (1 - 
                                          math.sqrt(var_bound / (conf * batchsize)) / 
                                          torch.norm(grad).item())
        else:
            stepsize = 1. / (2 * lip_const)  
        log_row['StepSize'] = torch.norm(torch.tensor(stepsize)).item()
        
        log_row['BatchSize'] = batchsize #current
        batchsize = torch.ceil(4 * var_bound / 
                               (conf * torch.norm(grad)**2)).item() #next
        if batchsize > max_batchsize:
            low_samples = True
            batchsize = max_batchsize
        else:
            low_samples = False
        min_safe_batchsize = torch.ceil(var_bound / (conf * torch.norm(grad)**2)).item()
        if batchsize < min_safe_batchsize:
            print('Unsafe, stopping. Would require %d samples' % min_safe_batchsize)
            unsafe = True
        
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
        logger.write_row(log_row, it)
        
        #Save parameters
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        #Next iteration
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()
    