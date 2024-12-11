#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:55:39 2019

@author: matteo, giorgio
"""
import math
import functools
import torch
import sys

from potion.common.misc_utils import unpack

def importance_weights(batch, policy, target_params, normalize=False, clip=None):
    #Samples
    states, actions, _, mask, _ = unpack(batch) #NxHx*
    
    #Proposal log-probability
    proposal = policy.log_pdf(states, actions) #NxH
        
    #Target log-probability
    params = policy.get_flat()
    policy.set_from_flat(target_params)
    target = policy.log_pdf(states, actions) #NxH
        
    #Restore proposal
    policy.set_from_flat(params)
    
    #Importance weights
    mask = mask.view(target.shape)
    iws = torch.exp(torch.sum((target - proposal) * mask, 1)) #N
    
    if clip is not None:
        iws = torch.clamp(iws, 1-clip, 1+clip)
    
    #Self-normalization
    if normalize:
        iws /= torch.sum(iws)
    
    return iws

def multiple_importance_weights(batch, policy, proposal_policies, alphas, rescale=False, normalize=False, clip=None):
    """
    Compute Multiple Importance Sampling (MIS) weights, one for each trajectory in the batch, i.e:
        policy(tau) / sum_j alpha_j*proposal_policies_j(tau)
    Balance heuristic is employed for the determination of the weights functions.
    See Owen, Art B. "Monte Carlo theory, methods and examples." (2013).
    See Metelli, Alberto Maria, et al. "Importance sampling techniques for policy optimization." The Journal of Machine Learning Research 21.1 (2020): 5552-5626.

    Parameters
    ----------
    batch: list of N trajectories. Each trajectory is a tuple (states, actions, rewards, mask, info).
    policy: the target distribution
    proposal_policies: list with proposal policies distributions
    alphas: list of data fractions from each proposal distribution
    rescale: wether to minmax rescale the importance weights
    normalize: whether to normalize the imortance weights
    clip: clamp importance weights to the specified value
    """

    # Check parameters
    if not isinstance(proposal_policies,list): 
        proposal_policies = [proposal_policies]
    if not isinstance(alphas,list): 
        alphas = [alphas]

    assert len(proposal_policies)==len(alphas), (
        "Parameters proposal_policies and alphas do not have same lenght")
    assert abs(sum(alphas) - 1) <= 1e-5, (
        "Parameter alphas do not sum to 1")

    # Utlity function
    # Usage: log(x+y) = smoothmax(log(x),log(y)) := log(x) + log(1+exp(log(y)-log(x)))
    smoothmax = lambda x,y: x + torch.log(1+torch.exp(y-x))

    # Samples
    states, actions, _, mask, _ = unpack(batch) #NxHx*
    
    with torch.no_grad():

        # Target probability
        log_target = policy.log_pdf(states, actions) #NxH
        mask = mask.view(log_target.shape)
        log_target = log_target * mask
        sum_log_target = torch.sum(log_target,1) #N
        
        # Proposal probability
        log_proposals = [None]*len(proposal_policies)
        for i,p in enumerate(proposal_policies):
            if alphas[i]>0:
                log_proposals[i] = math.log(alphas[i]) + torch.sum(p.log_pdf(states, actions)*mask, 1)
            else:
                log_proposals[i] = math.log(sys.float_info.min) + torch.sum(p.log_pdf(states, actions)*mask, 1)
        sum_log_proposals = functools.reduce(smoothmax, log_proposals) #N

        # Importance weights
        log_iws = sum_log_target - sum_log_proposals

        if normalize:
            log_iws = log_iws - torch.log(torch.exp(log_iws).sum())

        iws = torch.exp(log_iws)
        
        if clip is not None:
            iws = torch.clamp(iws, 1-clip, 1+clip)

    if not all(iws.isfinite()):
        raise ValueError

    return iws

"""Testing"""
if __name__ == '__main__':
    from potion.actors.continuous_policies import ShallowGaussianPolicy as Gauss
    from potion.simulation.trajectory_generators import generate_batch
    from potion.common.misc_utils import seed_all_agent
    import potion.envs
    import gym.spaces
    env = gym.make('ContCartPole-v0')
    env.seed(0)
    seed_all_agent(0)
    N = 100
    H = 100
    disc = 0.99
    pol = Gauss(4,1, mu_init=[0.,0.,0.,0.], learn_std=True)
    pol2 = Gauss(4,1, mu_init=[1.,1.,1.,1.], learn_std=True)
    
    batch = generate_batch(env, pol, H, N)
    print(importance_weights(batch, pol, pol.get_flat()))
    print(multiple_importance_weights(batch, pol, pol,1))
    print(multiple_importance_weights(batch, pol, [pol,pol], [0.5,0.5]))

    print(importance_weights(batch, pol, pol2.get_flat()))
    print(multiple_importance_weights(batch, pol2, pol,1))

    print(importance_weights(batch, pol, pol2.get_flat(), normalize = True))
    print(multiple_importance_weights(batch, pol2, pol, 1, normalize = True))
