#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REINFORCE with stochastic variance reduction
@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon, mean_sum_info
from potion.estimation.gradients import gpomdp_estimator, reinforce_estimator
from potion.estimation.offpolicy_gradients import off_gpomdp_estimator, off_reinforce_estimator
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent
from potion.meta.steppers import ConstantStepper
import torch
import time
from functools import partial


def stormpg(env, policy, horizon, *,
                    init_batchsize = 100,
                    mini_batchsize = 10, 
                    decay = 0.9,
                    iterations = 1000,
                    disc = 0.99,
                    stepper = ConstantStepper(1e-1),
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'avg',
                    logger = Logger(name='stormpg'),
                    shallow = False,
                    seed = None,
                    test_batchsize = False,
                    info_key = 'grad_var',
                    save_params = 100000,
                    log_params = False,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    STORM-PG algorithm (Yue et al., 2020, https://arxiv.org/pdf/2003.04302.pdf) 
    Initial full gradient, recursion since start with exponential decay
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    init_batchsize: number of trajectories for initial "full" gradient
        estimate
    mini_batchsize: number of trajectories for each "stochastic" gradient 
        estimate
    decay: decay parameter for the recursive gradient term
    iterations: number of policy updates
    disc: discount factor
    stepper: step size criterion. A constant step size is used by default
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

    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'STORMPG',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'InitBatchSize': init_batchsize,
                   'MiniBatchSize': mini_batchsize,
                   'Disc': disc, 
                   'StepSizeCriterion': str(stepper), 
                   'Seed': seed
                   }
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 
                'UPerf', 
                'AvgHorizon', 
                'StepSize', 
                'BatchSize',
                'GradNorm', 
                'Time',
                'StepSize',
                'Exploration',
                'Entropy',
                'Info']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf', 'TestInfo']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    if estimator=='gpomdp':
        estimator_fun=gpomdp_estimator
        off_estimator_fun=off_gpomdp_estimator
    elif estimator=='reinforce':
        estimator_fun=reinforce_estimator
        off_estimator_fun=off_reinforce_estimator
    else: raise ValueError('Invalid policy gradient estimator')
    
    #dummy
    prev_params = torch.zeros_like(policy.get_flat())
    prev_grad = torch.zeros_like(policy.get_flat())
    
    #Learning loop
    it = 1
    while(it < iterations):
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
                                        n_jobs=parallel,
                                        deterministic=True,
                                        key=info_key)
            log_row['TestPerf'] = performance(test_batch, disc)
            log_row['TestInfo'] = mean_sum_info(test_batch).item()
            log_row['UTestPerf'] = performance(test_batch, 1)
        
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon, 
                           episodes=1, 
                           action_filter=action_filter, 
                           render=True,
                           key=info_key)
            
        
        if it==1: #"Full" gradient -------------------------------------------
            #collect large batch    
            batch = generate_batch(env, policy, horizon, init_batchsize, 
                                   action_filter=action_filter, 
                                   seed=seed, 
                                   n_jobs=parallel,
                                   key=info_key)
            
            #on-policy estimate from large batch
            grad = estimator_fun(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result='mean')
        else: #Stochastic gradient -------------------------------------------
            #collect minibatch
            batch = generate_batch(env, policy, horizon, mini_batchsize, 
                                   action_filter=action_filter, 
                                   seed=seed, 
                                   n_jobs=parallel,
                                   key=info_key)
            
            #on-policy estimate from minibatch
            stoc_grad = estimator_fun(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result='mean')
            
            #off-policy correction from minibatch (target is previous policy)
            correction = off_estimator_fun(batch, disc, policy, prev_params,
                                           baselinekind=baseline,
                                           shallow=shallow,
                                           result='mean')
            
            #storm estimate
            grad = (1 - decay) * (prev_grad - correction) + stoc_grad
        
        log_row['Perf'] = performance(batch, disc)
        log_row['Info'] = mean_sum_info(batch).item()
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['Exploration'] = policy.exploration().item()
        log_row['Entropy'] = policy.entropy(0.).item()
        
        if verbose > 1:
            print('Gradients: ', grad)
        log_row['GradNorm'] = torch.norm(grad).item()
        
        #Select meta-parameters
        stepsize = stepper.next(grad)
        if not torch.is_tensor(stepsize):
            stepsize = torch.tensor(stepsize)
        log_row['StepSize'] = torch.norm(stepsize).item()
        log_row['BatchSize'] = init_batchsize if it==1 else mini_batchsize
        
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
        prev_grad = grad
        prev_params = params
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()


def srvrpg(env, policy, horizon, *,
                    init_batchsize = 100,
                    mini_batchsize = 10, 
                    epoch_length = 10,
                    iterations = 1000,
                    disc = 0.99,
                    stepper = ConstantStepper(1e-1),
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'avg',
                    logger = Logger(name='srvrpg'),
                    shallow = False,
                    seed = None,
                    test_batchsize = False,
                    info_key = 'grad_var',
                    save_params = 100000,
                    log_params = False,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    SRVRPG algorithm (Xu et al., 2019, https://arxiv.org/abs/1909.08610)
    Periodic full gradient, recursion since last snapshot
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    init_batchsize: number of trajectories for occasional "full" gradient
        estimate
    mini_batchsize: number of trajectories for each "stochastic" gradient 
        estimate
    epoch_length: number of stochastic gradient updates for each "full" one
    iterations: number of policy updates
    disc: discount factor
    stepper: step size criterion. A constant step size is used by default
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

    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'SRVRPG',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'InitBatchSize': init_batchsize,
                   'MiniBatchSize': mini_batchsize,
                   'Disc': disc, 
                   'StepSizeCriterion': str(stepper), 
                   'Seed': seed
                   }
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 
                'UPerf', 
                'AvgHorizon', 
                'StepSize', 
                'BatchSize',
                'GradNorm', 
                'Time',
                'StepSize',
                'Exploration',
                'Entropy',
                'Info']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf', 'TestInfo']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    if estimator=='gpomdp':
        estimator_fun=gpomdp_estimator
        off_estimator_fun=off_gpomdp_estimator
    elif estimator=='reinforce':
        estimator_fun=reinforce_estimator
        off_estimator_fun=off_reinforce_estimator
    else: raise ValueError('Invalid policy gradient estimator')
    
    #dummy
    prev_params = torch.zeros_like(policy.get_flat())
    prev_grad = torch.zeros_like(policy.get_flat())
    
    #Learning loop
    it = 1
    while(it < iterations):
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
                                        n_jobs=parallel,
                                        deterministic=True,
                                        key=info_key)
            log_row['TestPerf'] = performance(test_batch, disc)
            log_row['TestInfo'] = mean_sum_info(test_batch).item()
            log_row['UTestPerf'] = performance(test_batch, 1)
        
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon, 
                           episodes=1, 
                           action_filter=action_filter, 
                           render=True,
                           key=info_key)
            
        
        if (it - 1) % epoch_length == 0: #"Full" gradient ---------------------
            #collect large batch    
            batch = generate_batch(env, policy, horizon, init_batchsize, 
                                   action_filter=action_filter, 
                                   seed=seed, 
                                   n_jobs=parallel,
                                   key=info_key)
            
            #on-policy estimate from large batch
            grad = estimator_fun(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result='mean')
            
        else: #Stochastic gradient -------------------------------------------
            #collect minibatch
            batch = generate_batch(env, policy, horizon, mini_batchsize, 
                                   action_filter=action_filter, 
                                   seed=seed, 
                                   n_jobs=parallel,
                                   key=info_key)
            
            #on-policy estimate from minibatch
            stoc_grad = estimator_fun(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result='mean')
            
            #off-policy correction from minibatch (target is previous policy)
            correction = off_estimator_fun(batch, disc, policy, prev_params,
                                           baselinekind=baseline,
                                           shallow=shallow,
                                           result='mean')
            
            #svrg estimate
            grad = prev_grad - correction + stoc_grad
        
        log_row['Perf'] = performance(batch, disc)
        log_row['Info'] = mean_sum_info(batch).item()
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['Exploration'] = policy.exploration().item()
        log_row['Entropy'] = policy.entropy(0.).item()
        
        if verbose > 1:
            print('Gradients: ', grad)
        log_row['GradNorm'] = torch.norm(grad).item()
        
        #Select meta-parameters
        stepsize = stepper.next(grad)
        if not torch.is_tensor(stepsize):
            stepsize = torch.tensor(stepsize)
        log_row['StepSize'] = torch.norm(stepsize).item()
        log_row['BatchSize'] = init_batchsize if it==1 else mini_batchsize
        
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
        prev_params = params
        prev_grad = grad
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()
    

def svrpg(env, policy, horizon, *,
                    init_batchsize = 100,
                    mini_batchsize = 10, 
                    epoch_length = 10,
                    iterations = 1000,
                    disc = 0.99,
                    stepper = ConstantStepper(1e-1),
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'avg',
                    logger = Logger(name='svrpg'),
                    shallow = False,
                    seed = None,
                    test_batchsize = False,
                    info_key = 'grad_var',
                    save_params = 100000,
                    log_params = False,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    SVRPG algorithm (Papini et al., 2018, https://arxiv.org/abs/1806.05618)
    Periodic full gradient, no recursion
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    init_batchsize: number of trajectories for occasional "full" gradient
        estimate
    mini_batchsize: number of trajectories for each "stochastic" gradient 
        estimate
    epoch_length: number of stochastic gradient updates for each "full" one
    iterations: number of policy updates
    disc: discount factor
    stepper: step size criterion. A constant step size is used by default
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

    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'SVRPG',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'InitBatchSize': init_batchsize,
                   'MiniBatchSize': mini_batchsize,
                   'Disc': disc, 
                   'StepSizeCriterion': str(stepper), 
                   'Seed': seed
                   }
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 
                'UPerf', 
                'AvgHorizon', 
                'StepSize', 
                'BatchSize',
                'GradNorm', 
                'Time',
                'StepSize',
                'Exploration',
                'Entropy',
                'Info']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf', 'TestInfo']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    if estimator=='gpomdp':
        estimator_fun=gpomdp_estimator
        off_estimator_fun=off_gpomdp_estimator
    elif estimator=='reinforce':
        estimator_fun=reinforce_estimator
        off_estimator_fun=off_reinforce_estimator
    else: raise ValueError('Invalid policy gradient estimator')
    
    #dummy
    snapshot_params = torch.zeros_like(policy.get_flat())
    snapshot_grad = torch.zeros_like(policy.get_flat())
    
    #Learning loop
    it = 1
    while(it < iterations):
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
                                        n_jobs=parallel,
                                        deterministic=True,
                                        key=info_key)
            log_row['TestPerf'] = performance(test_batch, disc)
            log_row['TestInfo'] = mean_sum_info(test_batch).item()
            log_row['UTestPerf'] = performance(test_batch, 1)
        
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon, 
                           episodes=1, 
                           action_filter=action_filter, 
                           render=True,
                           key=info_key)
            
        
        if (it - 1) % epoch_length == 0: #"Full" gradient ---------------------
            #collect large batch    
            batch = generate_batch(env, policy, horizon, init_batchsize, 
                                   action_filter=action_filter, 
                                   seed=seed, 
                                   n_jobs=parallel,
                                   key=info_key)
            
            #on-policy estimate from large batch
            grad = snapshot_grad = estimator_fun(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result='mean')
            snapshot_params = params
            
        else: #Stochastic gradient -------------------------------------------
            #collect minibatch
            batch = generate_batch(env, policy, horizon, mini_batchsize, 
                                   action_filter=action_filter, 
                                   seed=seed, 
                                   n_jobs=parallel,
                                   key=info_key)
            
            #on-policy estimate from minibatch
            stoc_grad = estimator_fun(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result='mean')
            
            #off-policy correction from minibatch (target is previous policy)
            correction = off_estimator_fun(batch, disc, policy, snapshot_params,
                                           baselinekind=baseline,
                                           shallow=shallow,
                                           result='mean')
            
            #svrg estimate
            grad = snapshot_grad - correction + stoc_grad
        
        log_row['Perf'] = performance(batch, disc)
        log_row['Info'] = mean_sum_info(batch).item()
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['Exploration'] = policy.exploration().item()
        log_row['Entropy'] = policy.entropy(0.).item()
        
        if verbose > 1:
            print('Gradients: ', grad)
        log_row['GradNorm'] = torch.norm(grad).item()
        
        #Select meta-parameters
        stepsize = stepper.next(grad)
        if not torch.is_tensor(stepsize):
            stepsize = torch.tensor(stepsize)
        log_row['StepSize'] = torch.norm(stepsize).item()
        log_row['BatchSize'] = init_batchsize if it==1 else mini_batchsize
        
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

def pagepg(env, policy, horizon, *,
                    init_batchsize = 100,
                    mini_batchsize = 10, 
                    snapshot_prob = .1,
                    iterations = 1000,
                    disc = 0.99,
                    stepper = ConstantStepper(1e-1),
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'avg',
                    logger = Logger(name='pagepg'),
                    shallow = False,
                    seed = None,
                    test_batchsize = False,
                    info_key = 'grad_var',
                    save_params = 100000,
                    log_params = False,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    PAGE-PG algorithm (Gargiani et al., 2022, https://arxiv.org/abs/2202.00308) 
    Occasional (random) full-gradient, recursion since last snapshot
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    init_batchsize: number of trajectories for occasional "full" gradient
        estimate
    mini_batchsize: number of trajectories for each "stochastic" gradient 
        estimate
    snapshot_prob: probability of computing a "full" gradient
    iterations: number of policy updates
    disc: discount factor
    stepper: step size criterion. A constant step size is used by default
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

    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'PAGEPG',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'InitBatchSize': init_batchsize,
                   'MiniBatchSize': mini_batchsize,
                   'Disc': disc, 
                   'StepSizeCriterion': str(stepper), 
                   'Seed': seed
                   }
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 
                'UPerf', 
                'AvgHorizon', 
                'StepSize', 
                'BatchSize',
                'GradNorm', 
                'Time',
                'StepSize',
                'Exploration',
                'Entropy',
                'Info']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf', 'TestInfo']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    if estimator=='gpomdp':
        estimator_fun=gpomdp_estimator
        off_estimator_fun=off_gpomdp_estimator
    elif estimator=='reinforce':
        estimator_fun=reinforce_estimator
        off_estimator_fun=off_reinforce_estimator
    else: raise ValueError('Invalid policy gradient estimator')
    
    #dummy
    prev_params = torch.zeros_like(policy.get_flat())
    prev_grad = torch.zeros_like(policy.get_flat())
    
    #Learning loop
    it = 1
    while(it < iterations):
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
                                        n_jobs=parallel,
                                        deterministic=True,
                                        key=info_key)
            log_row['TestPerf'] = performance(test_batch, disc)
            log_row['TestInfo'] = mean_sum_info(test_batch).item()
            log_row['UTestPerf'] = performance(test_batch, 1)
        
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon, 
                           episodes=1, 
                           action_filter=action_filter, 
                           render=True,
                           key=info_key)
        
        #do a snapshot with probability p
        do_snapshot = torch.bernoulli(torch.tensor(snapshot_prob)).item() > .5
        
        if it == 1 or do_snapshot: #"Full" gradient ---------------------------
            #collect large batch    
            batch = generate_batch(env, policy, horizon, init_batchsize, 
                                   action_filter=action_filter, 
                                   seed=seed, 
                                   n_jobs=parallel,
                                   key=info_key)
            
            #on-policy estimate from large batch
            grad = estimator_fun(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result='mean')
            
        else: #Stochastic gradient -------------------------------------------
            #collect minibatch
            batch = generate_batch(env, policy, horizon, mini_batchsize, 
                                   action_filter=action_filter, 
                                   seed=seed, 
                                   n_jobs=parallel,
                                   key=info_key)
            
            #on-policy estimate from minibatch
            stoc_grad = estimator_fun(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result='mean')
            
            #off-policy correction from minibatch (target is previous policy)
            correction = off_estimator_fun(batch, disc, policy, prev_params,
                                           baselinekind=baseline,
                                           shallow=shallow,
                                           result='mean')
            
            #svrg estimate
            grad = prev_grad - correction + stoc_grad
        
        log_row['Perf'] = performance(batch, disc)
        log_row['Info'] = mean_sum_info(batch).item()
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['Exploration'] = policy.exploration().item()
        log_row['Entropy'] = policy.entropy(0.).item()
        
        if verbose > 1:
            print('Gradients: ', grad)
        log_row['GradNorm'] = torch.norm(grad).item()
        
        #Select meta-parameters
        stepsize = stepper.next(grad)
        if not torch.is_tensor(stepsize):
            stepsize = torch.tensor(stepsize)
        log_row['StepSize'] = torch.norm(stepsize).item()
        log_row['BatchSize'] = init_batchsize if it==1 else mini_batchsize
        
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
        prev_grad = grad
        prev_params = params
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()