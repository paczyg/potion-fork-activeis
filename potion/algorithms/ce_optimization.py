#%%
import numpy as np
import copy
import pandas as pd
import torch

from potion.estimation.importance_sampling import multiple_importance_weights
from potion.common.misc_utils import unpack, concatenate
from potion.estimation.gradients import gpomdp_estimator, reinforce_estimator
from potion.simulation.trajectory_generators import generate_batch
from potion.estimation.offpolicy_gradients import _shallow_multioff_gpomdp_estimator

def gradients(batch, discount, policy, estimator='reinforce', baseline='zero'):
    """
    Compute on-policy per-trajectory policy gradients:
        grad_\theta(log policy(trajectory;\theta) * reward(trajectory))
    
    Parameters
    ----------
    batch: list of N trajectories. Each trajectory is a tuple (states, actions, rewards, mask, info).
    discount: discount factor
    policy: the one used to collect the data
    estimator: either 'reinforce' or 'gpomdp' (default). The latter typically suffers from less variance
    baseline: control variate to be used in the gradient estimator. Either
        'avg' (average reward, default), 'peters' (variance-minimizing) or 'zero' (no baseline)
        
    Returns
    -------
    grad_samples: (#trajectories,#parameters) tensor with gradients
    """

    is_shallow = True   # TODO: For now we only deal with shallow policies
    if estimator == 'gpomdp':
        grad_samples = gpomdp_estimator(batch, discount, policy, 
                            baselinekind=baseline, 
                            shallow=is_shallow,
                            result='samples')
    elif estimator == 'reinforce':
        grad_samples = reinforce_estimator(batch, discount, policy, 
                            baselinekind=baseline, 
                            shallow=is_shallow,
                            result='samples')
    else:
        raise ValueError('Invalid policy gradient estimator')

    return grad_samples

def get_alphas(mis_batches):
    return list(np.array([len(b) for b in mis_batches]) / sum([len(b) for b in mis_batches]))

def argmin_CE(env, target_policy, mis_policies, mis_batches, *,
              estimator='gpomdp',
              baseline = 'avg',
              optimize_mean = True,
              optimize_variance=True):
    """
    TODO: funziona solo per policy gaussiane, lineari nella media (ShallowGaussianPolicy e DeepGaussianPolicy).
    TODO: Controllare tipo di target_policy e mis_policies
    """

    # Parse parameters
    # ----------------
    if not isinstance(mis_policies, list):
        mis_policies = [mis_policies]
    if not isinstance(mis_batches,list):
        mis_batches = [mis_batches]
    assert len(mis_policies)==len(mis_batches), "Parameters mis_policies and mis_batches do not have same lenght"

    # Data for CE estimation
    # ----------------------
    batch                       = concatenate(mis_batches)
    states, actions, _, mask, _ = unpack(batch)         #[N,H,*]
    horizons                    = torch.sum(mask,1)     #[N]
    if target_policy.feature_fun is not None:
        mu_features = target_policy.feature_fun(states) #[N,H,FS]
    else:
        mu_features = states                            #[N,H,FS]
    
    grad_samples = gradients(batch, env.gamma, target_policy, estimator=estimator, baseline=baseline)
    coefficients = multiple_importance_weights(batch, target_policy, mis_policies, get_alphas(mis_batches)) \
                    * torch.linalg.norm(grad_samples,dim=1)    #[N]

    # Maximize CE
    # -----------
    opt_policy = copy.deepcopy(target_policy)
    #TODO: detacha grad_fn dal tensore di media e varianza??
    if 1 == target_policy.n_actions:
        if optimize_mean:
            # Mean
            ## num[FS] = sum_n W[n] * sum_t a[n,t] phi(s[n,t]).T
            num = torch.einsum('n,nj->j', coefficients, (actions*mu_features).sum(1) )  #[FS]
            ## den[N,FS,FS] = sum_t phi[n,t,FS]*phi[n,t,FS].T
            den = torch.einsum('nti,ntj->ntij',mu_features,mu_features).sum(1)          #[N,FS,FS]
            ## den[FS,FS] = sum_n W[n] den[n,FS,FS]
            den = torch.einsum('n,nij->ij',coefficients,den)                            #[FS,FS]
            opt_mean_params = num @ torch.inverse(den)                                  #[FS]
            opt_policy.set_loc_params(opt_mean_params)

            if any(opt_mean_params.isnan()):
                raise ValueError

        if optimize_variance:
            # Variance
            mu = opt_policy.mu(states)                                            #[N,H,A=1]
            ## num[N] = sum_t (a[N,t] - mu[N,t])^2
            num = ((actions - mu)**2).sum(1)                                      #[N,A=1]
            ## sum_n W[n]*num[n] / sum_n horizon[n]*W[n]
            opt_var = (coefficients@num) / (horizons@coefficients)                #[1]
            opt_policy.set_scale_params(torch.log(torch.sqrt(opt_var)).item())

    else:
        raise NotImplementedError

    return opt_policy

def var_mean(X,iws=None):
    """
    Compute variance of sample mean of X, either via Monte Carlo (iws=None) or Importance Sampling (iws).

    Parameters
    ----------
    X : tensor (N,D)-shape
    iws : tensor (N)-shape or (N,1)-shape

    Returns
    -------
    cov : 2D tensor (D,D)-shape of covariance matrix
    var : trace of cov
    """
    
    N = X.shape[0]
    if iws is None:
        #Monte Carlo
        iws = torch.ones(N,1)               #[N,1]
    else:
        # Importance Sampling
        iws = iws.ravel().reshape((-1,1))   #[N,1]

    X_mean   = torch.mean(iws*X,0)
    centered = iws*X - X_mean
    cov      = 1/N**2 * torch.sum(torch.bmm(centered.unsqueeze(2), centered.unsqueeze(1)),0)
    var      = torch.sum(torch.diag(cov)).item()

    return cov, var

def algo(env, target_policy, n_per_it, n_ce_iterations, *,
         estimator='gpomdp',
         baseline='zero',
         action_filter=None,
         window=None,
         optimize_mean=True,
         optimize_variance=True,
         reuse_samples = True,
         run_mc_comparison = True):
    """
    Algorithm 2
    """

    # Check parameters
    # ----------------
    if window is not None:
        window = -window

    # Logs, statistics and information
    # --------------------------------
    algo_info = {
        'N_per_it':         n_per_it,
        'n_ce_iterations':  n_ce_iterations,
        'N_tot':            n_per_it*(n_ce_iterations+1),
        'Estimator':        estimator,
        'Baseline':         baseline,
        'action_filter':    action_filter,
        'reuse_samples':    reuse_samples,
        'Env':              str(env)
    }
    stats = []
    results = dict.fromkeys(['grad_mc', 'grad_is', 'var_grad_mc', 'var_grad_is'])

    # TODO: seed è da risettare ogni volta alla creazione di un batch? Forse no
    seed = None

    # Compute optimal importance sampling distributions
    # ------------------------------------------------
    mis_policies = [target_policy]
    mis_batches  = [generate_batch(env, target_policy, env.horizon, n_per_it, 
                                   action_filter=action_filter, 
                                   seed=seed, 
                                   n_jobs=False)]
    for _ in range(n_ce_iterations):
        opt_policy = argmin_CE(env, target_policy, mis_policies[window:], mis_batches[window:], 
                               estimator=estimator, baseline=baseline, optimize_mean=optimize_mean, optimize_variance=optimize_variance)
        mis_policies.append(opt_policy)
        mis_batches.append(
            generate_batch(env, opt_policy, env.horizon, n_per_it, 
                           action_filter=action_filter, 
                           seed=seed, 
                           n_jobs=False))
        stats.append({
            'opt_policy_loc'    : opt_policy.get_loc_params().tolist(),
            'opt_policy_scale'  : opt_policy.get_scale_params().tolist()
        })

    # Estimate IS mean, and its variance
    # ----------------------------------
    if not reuse_samples:
        del mis_batches[:-1]
        del mis_policies[:-1]

    if estimator == 'gpomdp':
        grad_samples    = _shallow_multioff_gpomdp_estimator(
                            concatenate(mis_batches), env.gamma, target_policy, mis_policies, get_alphas(mis_batches),
                            baselinekind=baseline, 
                            result='samples'
                        )   #[N,D]
        results['grad_is']      = torch.mean(grad_samples,0).tolist()
        results['var_grad_is']  = var_mean(grad_samples)[1]
        # GPOMDP with importance ratios
        # grad_samples = gpomdp_estimator(batch, env.gamma, target_policy, 
        #                                         baselinekind=baseline, 
        #                                         shallow=True,
        #                                         result='samples')
        # results['grad_is']      = torch.mean(iws[:,None]*grad_samples,0).tolist()
        # results['var_grad_is']  = var_mean(grad_samples,iws)[1]
    elif estimator == 'reinforce':
        grad_samples = reinforce_estimator(concatenate(mis_batches), env.gamma, target_policy, 
                                           baselinekind=baseline, 
                                           shallow=True,
                                           result='samples') #[N]
        iws = multiple_importance_weights(concatenate(mis_batches), target_policy, mis_policies, get_alphas(mis_batches))  #[N]
        results['grad_is']      = torch.mean(iws[:,None]*grad_samples,0).tolist()
        results['var_grad_is']  = var_mean(grad_samples,iws)[1]
    else:
        raise NotImplementedError

    # Estimate MC mean, and its variance (for further comparisons)
    # ------------------------------------------------------------
    if run_mc_comparison:
        mc_batch = generate_batch(env, target_policy, env.horizon, n_per_it*(n_ce_iterations+1), 
                                  action_filter=action_filter, 
                                  seed=seed, 
                                  n_jobs=False)
        grad_samples            = gradients(mc_batch, env.gamma, target_policy, estimator=estimator, baseline=baseline)
        results['grad_mc']      = torch.mean(grad_samples,0).tolist()
        results['var_grad_mc']  = var_mean(grad_samples)[1]

    return results, stats, algo_info

def ce_optimization(env, target_policy, batchsizes, *,
         estimator='gpomdp',
         baseline='zero',
         action_filter=None,
         optimize_mean=True,
         optimize_variance=True,
         reuse_samples = True):
    """
    Parameters
    ----------
    ce_batchsizes : list
        List with the number of samples to be used at every CE optimization epoch
    optimize_mean : boolean
        Whether or not to optimize the mean of the behavioural policy
    optimize_variance : boolean
        Whether or not to optimize the variance of the behavioural policy
    reuse_samples : boolean
        Whether to reuse or not samples collected during the iterations of CE optimization
        for the current behavioural policy optimization.
        If False, only the last batch is used to estimate and optimize the CE loss for the current behavioural policy

    Returns
    -------
    opt_ce_policy : list
        The cross-entropy optimized behavioural policy
    ce_policies : list
        The list of optimized policies during the cross entropy epochs
    ce_batches : list
        The list of trajectories batches collected during the cross entropy epochs
    """

    # Parse parameters
    # ----------------
    window = None if reuse_samples else -1

    seed = None

    # Compute optimal importance sampling distributions
    # ------------------------------------------------
    opt_ce_policy = target_policy
    ce_policies   = []
    ce_batches    = []
    for batchsize in batchsizes:
        ce_policies.append(opt_ce_policy)
        ce_batches.append(
            generate_batch(env, opt_ce_policy, env.horizon, batchsize, 
                           action_filter=action_filter, 
                           seed=seed, 
                           n_jobs=False)
        )
        try:
            opt_ce_policy = argmin_CE(env, target_policy, ce_policies[window:], ce_batches[window:], 
                                    estimator=estimator,
                                    baseline=baseline,
                                    optimize_mean=optimize_mean,
                                    optimize_variance=optimize_variance)
        except(RuntimeError):
            # If CE minimization is not possible, keep the previous opt_ce_policy
            pass
        
    return opt_ce_policy, ce_policies, ce_batches
