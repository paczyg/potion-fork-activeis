#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REINFORCE family of algorithms (actor-only policy gradient)
@author: Matteo Papini, Giorgio Manganini
"""

import torch
import time
import copy
from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon, mean_sum_info
from potion.estimation.gradients import gpomdp_estimator, reinforce_estimator, egpomdp_estimator
from potion.common.logger import Logger
from potion.common.misc_utils import seed_all_agent
from potion.meta.steppers import ConstantStepper, Adam

def reinforce(env, policy, horizon, *,
                    action_filter = None,
                    batchsize = 100, 
                    baseline = 'avg',
                    disc = 0.99,
                    entropy_coeff = 0.,
                    estimate_var = False,
                    estimator = 'gpomdp',
                    info_key = 'danger',
                    iterations = 1000,
                    logger = Logger(name='gpomdp'),
                    log_params = False,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    save_params = 100000,
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
    batchsize: number of trajectories used to estimate policy gradient
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

    # Saving algorithm information
    # ================
    # Store function parameters (do not move it from here!)
    algo_params = copy.deepcopy(locals())
    if logger is not None:
        # Save algorithm parameters and policy info
        logger.write_info({**algo_params, **policy.info()})

    # Prepare function arguments for iterations
    del algo_params['iterations']
    del algo_params['logger']
    del algo_params['render']
    del algo_params['save_params']

    # Learning loop
    # =============
    results = []
    for it in range(iterations):

        # Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon, 
                           episodes=1, 
                           action_filter=action_filter, 
                           render=True,
                           key=info_key)

        # Save parameters
        if logger is not None:
            if save_params and it % save_params == 0:
                logger.save_params(policy.get_flat(), it)

        # Update policy
        if verbose:
            print('\nIteration ', it)
        log_row = reinforce_step(**algo_params)
        if verbose:
            print(f'Iteration time {log_row['Time']}')
        
        if logger is not None:
            if not logger.ready:
                logger.open(log_row.keys())
            logger.write_row(log_row, it)
        else:
            results.append(log_row)

    # Cleaning logger
    # ===============
    if logger is not None:
        logger.close()

    return results


def reinforce_step(env, policy, horizon, *,
                    batchsize = 100, 
                    disc = 0.99,
                    stepper = ConstantStepper(1e-2),
                    entropy_coeff = 0.,
                    action_filter = None,
                    estimator = 'gpomdp',
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
    
    start = time.time()

    # Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    # Preparing the log
    log_row = {}
    
    # Showing info
    params = policy.get_flat()
    if verbose > 1:
        print('Parameters:', params)
    
    # Test the corresponding deterministic policy
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
    batch = generate_batch(env, policy, horizon, batchsize, 
                            action_filter=action_filter, 
                            seed=seed, 
                            n_jobs=parallel,
                            key=info_key)
    log_row['Perf'] = performance(batch, disc)
    log_row['Info'] = mean_sum_info(batch).item()
    log_row['UPerf'] = performance(batch, disc=1.)
    log_row['AvgHorizon'] = avg_horizon(batch)
    log_row['Exploration'] = policy.exploration().item()
    log_row['Entropy'] = policy.entropy(0.).item()
        
    # Estimate policy gradient
    result = 'samples' if estimate_var else 'mean'
    if estimator == 'gpomdp' and entropy_coeff == 0:
        grad_samples = gpomdp_estimator(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result=result)
    elif estimator == 'gpomdp':
        grad_samples = egpomdp_estimator(batch, disc, policy, entropy_coeff,
                                         baselinekind=baseline,
                                         shallow=shallow,
                                         result=result)
    elif estimator == 'reinforce':
        grad_samples = reinforce_estimator(batch, disc, policy, 
                                           baselinekind=baseline, 
                                           shallow=shallow,
                                           result=result)
    else:
        raise ValueError('Invalid policy gradient estimator')
    
    if estimate_var:
        grad = torch.mean(grad_samples, 0)
        centered = grad_samples - grad.unsqueeze(0)
        grad_cov = (batchsize/(batchsize - 1) * 
                    torch.mean(torch.bmm(centered.unsqueeze(2), 
                                            centered.unsqueeze(1)),0))
        grad_var = torch.sum(torch.diag(grad_cov)).item() #for humans
    else:
        grad = grad_samples
                
    if verbose > 1:
        print('Gradients: ', grad)
    log_row['GradNorm'] = torch.norm(grad).item()
    if estimate_var:
        log_row['SampleVar'] = grad_var
            
    # Select meta-parameters
    stepsize = stepper.next(grad)
    if not torch.is_tensor(stepsize):
        stepsize = torch.tensor(stepsize)
    log_row['StepSize'] = torch.norm(stepsize).item()
    log_row['BatchSize'] = batchsize
    
    # Update policy parameters
    if isinstance(stepper,Adam):
        new_params = params + stepsize
    else:
        new_params = params + stepsize * grad
    policy.set_from_flat(new_params)

    
    # Log
    log_row['Time'] = time.time() - start
    if log_params:
        for i in range(policy.num_params()):
            log_row['param%d' % i] = params[i].item()
    if log_grad:
        for i in range(policy.num_params()):
            log_row['grad%d' % i] = grad[i].item()
    
    return log_row    

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

    seed = 42
    env.seed(seed)

    # log_dir     = 'logs_test'
    # log_name    = 'reinforce'
    # logger      = Logger(directory=log_dir, name = log_name, modes=['csv'])
    logger = None

    res =  reinforce(env, policy, env.horizon,
                     action_filter = None,
                     batchsize = 100, 
                     baseline = 'peters',
                     disc = 0.99,
                     entropy_coeff = 0.,
                     estimate_var = False,
                     estimator = 'gpomdp',
                     info_key = 'danger',
                     iterations = 100,
                     logger = logger,
                     log_params = False,
                     log_grad = False,
                     parallel = False,
                     render = False,
                     save_params = 100000,
                     seed = None,
                     shallow = isinstance(policy,ShallowGaussianPolicy),
                     stepper = ConstantStepper(1e-4),
                     test_batchsize = 100,
                     verbose = 1)
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    res = pd.DataFrame(res)
    plt.plot(np.cumsum(res['BatchSize']), res['TestPerf'])
    plt.xlabel('Trajectories')
    plt.ylabel('Return')
    plt.show()