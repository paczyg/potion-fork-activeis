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
from potion.algorithms.ce_optimization import optimize_behavioural, get_alphas, var_mean
from potion.common.torch_utils import reset_all_weights, copy_params

def make_list(x):
    if type(x) is list:
        return x
    else:
        return [x]

def reinforce_offpolicy(
        env, policy, horizon, *,
        action_filter       = None,
        batchsize           = 100, 
        baseline            = 'avg',
        biased_offpolicy    = True,
        ce_batchsizes       = None,
        ce_use_offline_data = True,
        disc                = 0.99,
        defensive_batch     = 0,
        debug_logger        = None,
        entropy_coeff       = 0.,
        estimate_var        = False,
        estimator           = 'gpomdp',
        info_key            = 'danger',
        iterations          = 50,
        log_grad            = False,
        log_ce_params       = False,
        log_ce_params_norms = False,
        log_params          = False,
        log_params_norms    = False,
        logger              = Logger(name='reinforce_ce'),
        parallel            = False,
        seed                = None,
        shallow             = False,
        stepper             = ConstantStepper(1e-2),
        test_batchsize      = False,
        verbose             = 1):
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
    debug_logger: a Python logger to debug and report the main steps of the algorithm
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
    log_params_norms: whether to include policy parameters norms in the human-readable logs
    log_grad: whether to include gradients in the human-readable logs
    log_ce_params: whether to save the parameters of the CE potimized behavioural policies
    log_ce_params_norms: whether to save the parameters norms of the CE potimized behavioural policies
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
        if debug_logger is not None:
            debug_logger.debug('Data collection for first offline CE estimation...')
        offline_policies = [policy]
        offline_batches = [generate_batch(env, policy, horizon, batchsize, 
                                        action_filter=action_filter,
                                        seed=seed,
                                        n_jobs=parallel)]
        if debug_logger is not None:
            debug_logger.debug('done')
    else:
        offline_policies = None
        offline_batches = None
    
    # Prepare function arguments for iterations
    del algo_params['iterations']
    del algo_params['logger']
    del algo_params['ce_use_offline_data']

    # Prepare behavioural policies
    if ce_batchsizes is None:
        behavioural_policies = [copy.deepcopy(policy)]
    else:
        behavioural_policies = [copy.deepcopy(policy) for _ in range(len(ce_batchsizes)+1)]

    # Run
    # ===
    results = []
    for it in range(iterations):

        if verbose:
            print('\nIteration ', it)
        log_row, offline_policies, offline_batches = reinforce_offpolicy_step(
            **algo_params,
            behavioural_policies=behavioural_policies,
            offline_policies=offline_policies,
            offline_batches=offline_batches)
        
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


def reinforce_offpolicy_step(
        env, policy, horizon, behavioural_policies, offline_policies, offline_batches, *,
        action_filter=None,
        batchsize=100, 
        baseline='avg',
        biased_offpolicy=True,
        ce_batchsizes= None,
        disc=0.99,
        defensive_batch=0,
        debug_logger=None,
        entropy_coeff=0.,
        estimate_var=False,
        estimator='gpomdp',
        grad_norm_threshold=10,
        ce_tol_grad=100,
        ce_lr=1e-5,
        ce_max_iter=1e5,
        ce_weight_decay=100,
        ce_optimizer='adam',
        ce_initialize_behavioural_policies='target',
        ce_mis_rescale=False,
        ce_mis_normalize=False,
        ce_mis_clip=None,
        info_key='danger',
        log_grad= False,
        log_ce_params=False,
        log_ce_params_norms=False,
        log_params_norms=False,
        log_params=False,
        parallel=False,
        seed=None,
        shallow=False,
        stepper=ConstantStepper(1e-2),
        test_batchsize=False,
        verbose=1):
    
    start = time.time()

    # Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    # Showing info
    params = policy.get_flat()
    if verbose > 1:
        print('Parameters (norm):', torch.norm(params))
    if debug_logger is not None:
        debug_logger.debug(f'Parameters (norm):{torch.norm(params)}')
    
    # Preparing log
    log_row = {}

    # Testing the corresponding deterministic policy
    if test_batchsize:
        if debug_logger is not None:
            debug_logger.debug('Testing the corresponding deterministic policy...')
        test_batch = generate_batch(env, policy, horizon, test_batchsize, 
                                    action_filter=action_filter,
                                    seed=seed,
                                    n_jobs=parallel,
                                    deterministic=True,
                                    key=info_key)
        if debug_logger is not None:
            debug_logger.debug('done')
        log_row['TestPerf'] = performance(test_batch, disc)
        log_row['TestInfo'] = mean_sum_info(test_batch).item()
        log_row['UTestPerf'] = performance(test_batch, 1)
    
    # Cross-entropy optimization of behavioural policy (or policies)
    # ==============================================================
    # Reinizializzation of behavioural policies
    if ce_initialize_behavioural_policies == 'reset':
        for behavioural_policy in behavioural_policies:
            reset_all_weights(behavioural_policy)
    elif ce_initialize_behavioural_policies == 'target':
        for behavioural_policy in behavioural_policies:
            copy_params(policy, behavioural_policy)
    else:
        ce_itialize_behavioural_policies_values = ['reset', 'target']
        raise ValueError(f"Invalid ce_itialize_behavioural_policies type. Expected one of: {ce_itialize_behavioural_policies_values}")

    
    # (First) Behavioural policy optimization with offline batches
    if offline_policies:
        if debug_logger is not None:
            debug_logger.debug('Optimizing behavioural policy...')
        try:
            optimize_behavioural(
                behavioural_policies[0], env, policy, offline_policies, offline_batches, 
                estimator=estimator,
                baseline=baseline,
                optimize_mean=True,
                optimize_variance=True,
                tol_grad=ce_tol_grad,
                lr=ce_lr,
                max_iter=ce_max_iter,
                mis_normalize = ce_mis_normalize,
                mis_clip = ce_mis_clip,
                weight_decay=ce_weight_decay,
                optimizer=ce_optimizer)
            #NOTE: la policy ottimizzata può avere più parametri con requires_grad=true della policy target
            if debug_logger is not None:
                debug_logger.debug('done.')
                debug_logger.debug(f'Behavioural parameters (norm): {behavioural_policies[0].get_flat().norm().item()}')
        except(RuntimeError):
            # If CE minimization is not possible, keep the target policy
            if debug_logger is not None:
                debug_logger.exception('An exception was thrown!')
            copy_params(policy, behavioural_policies[0])

    else:
        copy_params(policy, behavioural_policies[0])
    
    # Further iterative behavioural policies optimization, sampling from the last behavioural policy
    behavioural_batches = []
    if ce_batchsizes is not None:
        for i,ce_batchsize in enumerate(ce_batchsizes):
            if debug_logger is not None:
                debug_logger.debug('Generating new online batch for iterative CE optimization...')
            # Generate samples from the last optimized behavioural policy
            behavioural_batches.append(
                generate_batch(env, behavioural_policies[i], horizon, ce_batchsize, 
                               action_filter=action_filter, 
                               seed=seed, 
                               n_jobs=False)
            )
            if debug_logger is not None:
                debug_logger.debug('done')
            try:
                if debug_logger is not None:
                    debug_logger.debug('Iteratively optimizing behavioural policy...')
                optimize_behavioural(
                    behavioural_policies[i+1], env, policy, behavioural_policies[:i+1], behavioural_batches, 
                    estimator=estimator,
                    baseline=baseline,
                    optimize_mean=True,
                    optimize_variance=True,
                    tol_grad=ce_tol_grad,
                    lr=ce_lr,
                    max_iter=ce_max_iter,
                    weight_decay=ce_weight_decay,
                    optimizer=ce_optimizer)

                if debug_logger is not None:
                    debug_logger.debug('done.')
                    debug_logger.debug(f'Behavioural parameters (norm): {behavioural_policies[i+1].get_flat().norm().item()}')
            except(RuntimeError):
                # If CE minimization is not possible, keep the previous behavioural policy
                if debug_logger is not None:
                    debug_logger.exception('An exception was thrown!')
                copy_params(behavioural_policies[i], behavioural_policies[i+1])

    # Selection of batches and policies for gradient estimation
    # =========================================================
    gradient_estimation_policies = []
    gradient_estimation_batches  = []
    
    # Sampling from optimized behavioural policy
    gradient_estimation_policies.append(behavioural_policies[-1])
    if debug_logger is not None:
        debug_logger.debug('Generating batch of trajectories from behavioural policy...')
    gradient_estimation_batches.append(
        generate_batch(env, behavioural_policies[-1], horizon, batchsize,
                       action_filter=action_filter, 
                       seed=seed, 
                       n_jobs=False)
    )
    if debug_logger is not None:
        debug_logger.debug('done')
    
    if defensive_batch > 0:
        # Sampling from target policy to use defensive trajecotires
        gradient_estimation_policies.append(policy)
        if debug_logger is not None:
            debug_logger.debug('Generating defensive batch of trajectories ...')
        gradient_estimation_batches.append(
            generate_batch(env, policy, horizon, defensive_batch,
                           action_filter=action_filter, 
                           seed=seed, 
                           n_jobs=False)
        )
        if debug_logger is not None:
            debug_logger.debug('done')

    if biased_offpolicy:
        # Reusing all the samples used for behavioural policies optimization
        if offline_policies:
            gradient_estimation_policies += offline_policies
            gradient_estimation_batches  += offline_batches
        if behavioural_batches:
            # The lastly optimized behavioural has been used already for generating trajectories 
            gradient_estimation_policies += behavioural_policies[:-1]
            gradient_estimation_batches  += behavioural_batches
    
    # Off-policy gradient stimation
    # =============================
    if debug_logger is not None:
        debug_logger.debug('Estimating off-policy gradients...')
    if estimator == 'gpomdp' and entropy_coeff == 0:
        grad_samples, iws = multioff_gpomdp_estimator(
            concatenate(gradient_estimation_batches), disc, policy, gradient_estimation_policies,
            get_alphas(gradient_estimation_batches), baselinekind=baseline, result='samples', is_shallow=shallow)
        grad = torch.mean(grad_samples,0)
    else:
        raise NotImplementedError
    if debug_logger is not None:
        debug_logger.debug('done')

    if verbose > 1:
        print('Gradients (norm): ', torch.norm(grad))
        print('Importance weights (norm): ', iws.norm().item())
    if debug_logger is not None:
        debug_logger.debug(f'Gradients (norm): {torch.norm(grad).item()}')
        debug_logger.debug(f'Importance weights (norm): {iws.norm().item()}')
    
    # Update of policy parameters
    # ===========================
    if debug_logger is not None:
        debug_logger.debug('Update parameters')
    stepsize = stepper.next(grad)
    if not torch.is_tensor(stepsize):
        stepsize = torch.tensor(stepsize)
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
    
    if log_params_norms:
        log_row['policy_loc_norm'] = policy.get_loc_params().norm().item()
        log_row['policy_scale_norm'] = policy.get_scale_params().norm().item()
    elif log_params:
        for i in range(policy.num_params()):
            log_row['param%d' % i] = params[i].item()

    if log_grad:
        for i in range(policy.num_params()):
            log_row['grad%d' % i] = grad[i].item()
    
    if log_ce_params_norms:
        for p, policy in enumerate(behavioural_policies):
            log_row[f"ce_policy_loc_{p}_norm"] = policy.get_loc_params().norm().item()
            log_row[f"ce_policy_scale_{p}_norm"] = policy.get_scale_params().norm().item()
    elif log_ce_params:
        for p, policy in enumerate(behavioural_policies):
            for i,el in enumerate(policy.get_loc_params().tolist()):
                log_row[f"ce_policy_loc{i}_{p}"] = el
            for i,el in enumerate(make_list(policy.get_scale_params().tolist())):
                log_row[f"ce_policy_scale{i}_{p}"] = el

    # Return values
    # =============
    # If offline data for CE estimation is provided, return the new offline data for the next iteration
    if offline_policies:
        next_offline_policies = [copy.deepcopy(x) for x in gradient_estimation_policies]
        next_offline_bathces  = gradient_estimation_batches
    else:
        next_offline_policies = []
        next_offline_bathces  = []
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
