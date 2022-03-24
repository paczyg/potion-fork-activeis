#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REINFORCE family of algorithms (actor-only policy gradient)
@author: Matteo Papini, Giorgio Manganini
"""
import math
import torch
import time
import copy

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon, mean_sum_info
from potion.estimation.gradients import gpomdp_estimator, reinforce_estimator, egpomdp_estimator
from potion.estimation.offpolicy_gradients import off_gpomdp_estimator, _shallow_multioff_gpomdp_estimator
from potion.estimation.importance_sampling import multiple_importance_weights
from potion.common.logger import Logger
from potion.common.misc_utils import clip, seed_all_agent, concatenate
from potion.meta.steppers import ConstantStepper
from potion.algorithms.ce_optimization import argmin_CE, get_alphas

def reinforce(env, policy, horizon, *,
                    batchsize = 100, 
                    iterations = 1000,
                    disc = 0.99,
                    stepper = ConstantStepper(1e-2),
                    entropy_coeff = 0.,
                    action_filter = None,
                    estimator = 'gpomdp',
                    n_offpolicy_opt = 0,
                    defensive = True,
                    biased_offpolicy = True,
                    baseline = 'avg',
                    logger = Logger(name='gpomdp'),
                    shallow = False,
                    seed = None,
                    estimate_var = False,
                    test_batchsize = False,
                    info_key = 'danger',
                    save_params = 100000,
                    log_params = False,
                    log_grad = False,
                    parallel = False,
                    render = False,
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
    n_offpolicy_opt: #TODO
    defensive: #TODO
    biased_offpolicy: #TODO
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
    algo_info = {
        'Algorithm': 'REINFORCE',
        'Estimator': estimator,
        'n_offpolicy_opt': n_offpolicy_opt,
        'defensive': defensive,
        'biased_offpolicy': biased_offpolicy,
        'Baseline': baseline,
        'Env': str(env), 
        'Horizon': horizon,
        'BatchSize': batchsize, 
        'Disc': disc, 
        'StepSizeCriterion': str(stepper), 
        'Seed': seed,
        'EntropyCoefficient': entropy_coeff
    }
    logger.write_info({**algo_info, **policy.info()})
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
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf', 'TestInfo']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
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
    
        #Collect trajectories
        # ===================
        if n_offpolicy_opt > 0:
            # Off-policy optimization

            if biased_offpolicy:
                # Resure CE samples for final offpolicy estimation
                N_per_it = math.ceil(batchsize / (n_offpolicy_opt+1))
            else:
                # Use N/2 samples for CE, and keep N/2 samples for offpolicy estimation
                N_per_it = math.ceil((batchsize/2) / n_offpolicy_opt)

            mis_policies = [policy]
            mis_batches  = [generate_batch(env, policy, horizon, N_per_it, 
                                            action_filter=None, 
                                            seed=seed, 
                                            n_jobs=False)]
            for _ in range(n_offpolicy_opt):
                opt_policy = argmin_CE(env, policy, mis_policies, mis_batches, 
                                        estimator=estimator,
                                        optimize_mean=True,
                                        optimize_variance=False)    #TODO: ottimizzare variance (ma non usare stimatore off-policy Matteo per il gradiente)
                mis_policies.append(opt_policy)
                mis_batches.append(generate_batch(env, opt_policy, horizon, N_per_it, 
                                                  action_filter=None, 
                                                  seed=seed, 
                                                  n_jobs=False))

            if biased_offpolicy:
                if not defensive:
                    del mis_policies[0]
                    del mis_batches[0]
                # Resure CE samples for final offpolicy estimation
                batch   = concatenate(mis_batches)
                iws     = multiple_importance_weights(batch, policy, mis_policies, get_alphas(mis_batches))  #[N]
            else:
                # Use N/2 samples for offpolicy estimation (first N/2 were used for CE)
                batch = generate_batch(env, opt_policy, horizon, batchsize/2,
                                       action_filter=None, 
                                       seed=seed, 
                                       n_jobs=False)
                iws     = multiple_importance_weights(batch, policy, opt_policy, 1)  #[N]
                if defensive:
                    # Use also the first samples (<N/2) from target policy for the first estimation of CE
                    batch = concatenate([mis_batches[0], batch])
                    iws   = multiple_importance_weights(batch, policy, [policy, opt_policy], get_alphas([mis_batches[0], batch]))  #[N]
            
        else:
            # On-policy
            batch = generate_batch(env, policy, horizon, batchsize, 
                                   action_filter=action_filter, 
                                   seed=seed, 
                                   n_jobs=parallel,
                                   key=info_key)
            iws = torch.ones(batchsize)
        
        # TODO: dove le mettiamo queste? Devono valutae il batch raccolto con IS?
        log_row['Perf']         = performance(batch, disc)  #FIXME: queste dovrebbero essere le performance della policy target (?)
        log_row['Info']         = mean_sum_info(batch).item()
        log_row['UPerf']        = performance(batch, disc=1.)
        log_row['AvgHorizon']   = avg_horizon(batch)
        log_row['Exploration']  = policy.exploration().item()
        log_row['Entropy']      = policy.entropy(0.).item()
        
        #Estimate policy gradient
        # =======================
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
                # if estimator == 'gpomdp' and entropy_coeff == 0:
                #     #TODO: aggiungere gpomdp offpolicy estimator
                #     #TODO: aggiungere baseline
                #     grad_samples = gpomdp_estimator(batch, disc, policy, 
                #                                         baselinekind='zero', 
                #                                         shallow=shallow,
                #                                         result='samples')
                #     grad = torch.mean(torch.einsum('i,ij->ij', iws, grad_samples), 0)
                # else:
                #     raise NotImplementedError
            # if biased_offpolicy:
            #     # TODO: offpolicy estimation with correct baseline
            #     grad = torch.mean(torch.einsum('i,ij->ij', iws, grad_samples), 0)
            # elif not biased_offpolicy and defensive:
            #     grad = torch.mean(torch.einsum('i,ij->ij', iws, grad_samples), 0)
            if not biased_offpolicy and not defensive and n_offpolicy_opt==1:
                #NOTE funziona solo con policy con stessa varianza della target
                # il batch è stato generato dalla policy ottimizzata CE
                target_params = policy.get_loc_params()
                grad = off_gpomdp_estimator(batch, disc, opt_policy, target_params,
                                            baselinekind=baseline, 
                                            result='mean',
                                            shallow=shallow)
            elif not biased_offpolicy and defensive:
                grad = _shallow_multioff_gpomdp_estimator(batch, disc, policy, [policy, opt_policy], get_alphas([mis_batches[0], batch]),
                                                    baselinekind=baseline, 
                                                    result='mean')
            elif not biased_offpolicy and not defensive:
                grad = _shallow_multioff_gpomdp_estimator(batch, disc, policy, opt_policy, 1,
                                                    baselinekind=baseline, 
                                                    result='mean')
        
        # Variance of gradients samples
        if estimate_var:
            centered = torch.einsum('i,ij->ij', iws, grad_samples) - grad.unsqueeze(0)
            grad_cov = (batchsize/(batchsize - 1) * 
                        torch.mean(torch.bmm(centered.unsqueeze(2), 
                                            centered.unsqueeze(1)),0))
            grad_var = torch.sum(torch.diag(grad_cov)).item() #for humans
        
        # Variance of the sample mean
        # TODO: E? utile? Da salvare?
        # log_row['VarMean'] = var_mean(grad_samples,iws)

        if verbose > 1:
            print('Gradients: ', grad)
        log_row['GradNorm'] = torch.norm(grad).item()
        if estimate_var:
            log_row['SampleVar'] = grad_var
        
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
