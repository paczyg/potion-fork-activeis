#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy gradient stimators
@author: Matteo Papini
"""

import torch
import potion.common.torch_utils as tu
from potion.common.misc_utils import unpack, discount
from potion.common.torch_utils import tensormat, jacobian
from potion.estimation.moments import incr_mean, incr_var


def gpomdp_estimator_def(batch, disc, policy,  baselinekind='avg', result='mean',
                     shallow=False, alpha=0.5):
    """G(PO)MDP policy gradient estimator
       
    batch: list of N trajectories. Each trajectory is a tuple 
        (states, actions, rewards, mask). Each element of the tuple is a 
        tensor where the first dimension is time.
    disc: discount factor
    policy: the one used to collect the data
    baselinekind: kind of baseline to employ in the estimator. 
        Either 'avg' (average reward, default), 'peters' 
        (variance-minimizing),  or 'zero' (no baseline)
    result: whether to return the final estimate ('mean', default), or the 
        single per-trajectory estimates ('samples')
    shallow: whether to use precomputed score functions (only available
        for shallow policies)
    """
    if shallow:
        return _shallow_gpomdp_estimator(batch, disc, policy, baselinekind, result)
    
    N = len(batch)
    states, actions, rewards, mask, _ = unpack(batch) #NxHxd_s, NxHxd_a, NxH, NxH
    H = rewards.shape[1]
    m = policy.num_params()
    
    disc_rewards = discount(rewards, disc) #NxH
    logps = policy.log_pdf(states, actions) * mask #NxH
    cm_logps = torch.cumsum(logps, 1) #NxH

    #Behavioral
    
    
    
    #Target
    
    target_logps = policy.log_pdf(states, actions) * mask #NxH
    
    behavioral_logps = policy.log_pdf(states, actions) #NxH
    iw_num = torch.exp(behavioral_logps)
    iw_den = alpha * iw_num + (1-alpha) *  torch.exp(target_logps)
    log_iws = torch.cumsum(torch.log(iw_num / iw_den) * mask, 1)


    
    if baselinekind == 'peters':
        jac = jacobian(policy, cm_logps.view(-1)).reshape((N,H,m)) #NxHxm   
        b_num = torch.sum(tensormat(jac**2, 
                                    disc_rewards), 0) #Hxm
        b_den = torch.sum(jac**2, 0) #Hxm
        baseline = b_num / b_den #Hxm
        baseline[baseline != baseline] = 0
        values = disc_rewards.unsqueeze(2) - baseline.unsqueeze(0) #NxHxm
        _samples = torch.sum(tensormat(values * jac, mask), 1) #Nxm
    else:
        if baselinekind == 'avg':
            baseline = torch.mean(disc_rewards, 0) #H
        else:
            baseline = torch.zeros(1) #1
        values = (disc_rewards - baseline) * mask #NxH
        
        _samples = torch.stack([tu.flat_gradients(policy, cm_logps[i,:], 
                                              values[i,:]* torch.exp(log_iws)[i,:])
                                       for i in range(N)], 0) #Nxm
    if result == 'samples':
        return _samples #Nxm
    else:
        return torch.mean(_samples, 0) #m
    
#entropy-augmented version
def egpomdp_estimator(batch, disc, policy, coeff, baselinekind='avg', result='mean',
                     shallow=False):
    """G(PO)MDP policy gradient estimator
       
    batch: list of N trajectories. Each trajectory is a tuple 
        (states, actions, rewards, mask). Each element of the tuple is a 
        tensor where the first dimension is time.
    disc: discount factor
    policy: the one used to collect the data
    coeff: entropy bonus coefficient
    baselinekind: kind of baseline to employ in the estimator. 
        Either 'avg' (average reward, default), 'peters' 
        (variance-minimizing),  or 'zero' (no baseline)
    result: whether to return the final estimate ('mean', default), or the 
        single per-trajectory estimates ('samples')
    shallow: whether to use precomputed score functions (only available
        for shallow policies)
    """
    if shallow:
        return _shallow_egpomdp_estimator(batch, disc, policy, coeff, baselinekind, result)    
    else:
        raise NotImplementedError

def reinforce_estimator_def(batch, disc, policy, baselinekind='avg', 
                        result='mean', shallow=False, alpha=0.5):
    """REINFORCE policy gradient estimator
       
    batch: list of N trajectories. Each trajectory is a tuple 
        (states, actions, rewards, mask). Each element of the tuple is a 
        tensor where the first dimension is time.
    disc: discount factor
    policy: the one used to collect the data
    baselinekind: kind of baseline to employ in the estimator. 
        Either 'avg' (average reward, default), 'peters' 
        (variance-minimizing),  or 'zero' (no baseline)
    result: whether to return the final estimate ('mean', default), or the 
        single per-trajectory estimates ('samples')
    shallow: whether to use precomputed score functions (only available
        for shallow policies)
    """
    if shallow:
        return _shallow_reinforce_estimator(batch, disc, policy, baselinekind, result)
    
    N = len(batch)
    states, actions, rewards, mask, _ = unpack(batch) #NxHxm, NxHxd, NxH, NxH
    
    disc_rewards = discount(rewards, disc) #NxH
    rets = torch.sum(disc_rewards, 1) #N
    logps = policy.log_pdf(states, actions) * mask #NxH

    target_logps = policy.log_pdf(states, actions) * mask #NxH
    behavioral_logps = policy.log_pdf(states, actions) #NxH
    iw_num = torch.exp(behavioral_logps)
    iw_den = alpha * iw_num + (1-alpha) *  torch.exp(target_logps)
    log_iws = torch.cumsum(torch.log(iw_num / iw_den) * mask, 1)
    
    if baselinekind == 'peters':
        logp_sums = torch.sum(logps, 1) #N
        jac = jacobian(policy, logp_sums) #Nxm   
        b_num = torch.sum(jac ** 2 * rets.unsqueeze(1), 0) #m
        b_den = torch.sum(jac **2, 0) #m
        baseline = b_num / b_den #m
        baseline[baseline != baseline] = 0
        values = rets.unsqueeze(1) - baseline.unsqueeze(0) #Nxm
        _samples = jac * values
    else:
        if baselinekind == 'avg':
            baseline = torch.mean(rets, 0) #1
        else:
            baseline = torch.zeros(1) #1
        baseline[baseline != baseline] = 0
        values = rets - baseline #N
        
        if result == 'mean':
            logp_sums = torch.sum(logps, 1)
            return tu.flat_gradients(policy, logp_sums, values) / N
        
        _samples = torch.stack([tu.flat_gradients(policy, logps[i,:]) * 
                                                     values[i,:]* torch.exp(log_iws)[i,:]
                                       for i in range(N)], 0) #Nxm
    if result == 'samples':
        return _samples #Nxm
    else:
        return torch.mean(_samples, 0) #m

def _shallow_gpomdp_estimator(batch, disc, policy, baselinekind='peters', result='mean', alpha = 0.5):
    with torch.no_grad():        
        states, actions, rewards, mask, _ = unpack(batch) # NxHxm, NxHxd, NxH, NxH
        
        disc_rewards = discount(rewards, disc) #NxH
        scores = policy.score(states, actions) #NxHxM
        G = torch.cumsum(tensormat(scores, mask), 1) #NxHxm
        n_k = torch.sum(mask, dim=0) #H
        n_k[n_k==0.] = 1.
        target_logps = policy.log_pdf(states, actions) * mask #NxH
        behavioral_logps = policy.log_pdf(states, actions) #NxH
        iw_num = torch.exp(behavioral_logps)
        iw_den = alpha * iw_num + (1-alpha) *  torch.exp(target_logps)
        log_iws = torch.cumsum(torch.log(iw_num / iw_den) * mask, 1)
        
        if baselinekind == 'avg':
            baseline = (torch.sum(disc_rewards, 0) / n_k).unsqueeze(-1) #Hx1
        elif baselinekind == 'peters':
            baseline = torch.sum(tensormat(G ** 2, disc_rewards), 0) / \
                            torch.sum(G ** 2, 0) #Hxm
        elif baselinekind == 'peters2':
            G2 = torch.cumsum(tensormat(scores ** 2, mask), 1) #NxHxm
            baseline = torch.sum(tensormat(G ** 2, disc_rewards), 0) / \
                            torch.sum(G2, 0) #Hxm
        elif baselinekind == 'optimal':
            g = tensormat(scores, mask) #NxHxm
            vanilla = tensormat(G, disc_rewards) #NxHxm
            term1 = torch.flip(torch.cumsum(torch.flip(vanilla, (1,)), 1), (1,)) #NxHxm (reverse cumsum)
            m2 = torch.sum(g ** 2, dim=0, keepdim=True) / n_k.unsqueeze(0).unsqueeze(-1) #(N)xHxm
            scores_on_m2 = g / m2 #NxHxm
            scores_on_m2[g==0] = 0
            shifted = torch.zeros_like(scores_on_m2) #NxHxm
            shifted[:,:-1,:] = scores_on_m2[:,1:,:]
            term2 = scores_on_m2 - shifted #NxHxm
            baseline = torch.sum(term1 * term2, dim=0) / n_k.unsqueeze(-1) #Hxm
        elif baselinekind == 'system':
            A = torch.mean(torch.einsum('ijk,ihk->ijhk', (G, G)), 0).transpose(0,2) #mxHxH
            vanilla = torch.sum(tensormat(G, disc_rewards), dim=1, keepdim=True) #Nx(H)xm
            c = torch.mean(vanilla * G, 0).transpose(0,1).unsqueeze(-1) #Hxmx1
            #print(scores[0,:,1], G[0,:,1])
            x, _ = torch.solve(c, A) #mxHx1
            baseline = x.squeeze(-1).transpose(0,1)
        elif baselinekind == 'zero':
            baseline = torch.zeros(1,1) #1x1
        else:
            raise ValueError('Unknown baseline')
        baseline[baseline != baseline] = 0 #removes non-real values
        values = disc_rewards.unsqueeze(2) - baseline.unsqueeze(0) #NxHxm

        #print(values.size(), log_iws.size(), G.size())

        # create a 100 x 500 x 4 torch tensor from log_iws, which is 100 x 500
        
        log_iws = torch.unsqueeze(log_iws, 2)
        log_iws = log_iws.repeat(1, 1, 4) # maybe a /4 is required
        # print(log_iws.size())
        # print(values.size())
        # print(G.size())
        #log_iws = log_iws / 4
        G = G * log_iws

        



        
        _samples = torch.sum(tensormat(G * values, mask), 1) #Nxm
        if result == 'samples':
            return _samples #Nxm
        else:
            return torch.mean(_samples, 0) #m
   
#entopy-augmented version     
def _shallow_egpomdp_estimator(batch, disc, policy, coeff, baselinekind='peters', result='mean', alpha = 0.5):
    with torch.no_grad():        
        states, actions, rewards, mask, _ = unpack(batch) # NxHxm, NxHxd, NxH, NxH
        
        rewards = (1-coeff) * rewards +  coeff * policy.entropy(states)
        disc_rewards = discount(rewards, disc) #NxH
        scores = policy.score(states, actions) #NxHxM
        G = torch.cumsum(tensormat(scores, mask), 1) #NxHxm


        target_logps = policy.log_pdf(states, actions) * mask #NxH
        behavioral_logps = policy.log_pdf(states, actions) #NxH
        iw_num = torch.exp(behavioral_logps)
        iw_den = alpha * iw_num + (1-alpha) *  torch.exp(target_logps)
        log_iws = torch.cumsum(torch.log(iw_num / iw_den) * mask, 1)

        if baselinekind == 'avg':
            baseline = torch.mean(disc_rewards, 0).unsqueeze(1) #Hx1
        elif baselinekind == 'peters':
            baseline = torch.sum(tensormat(G ** 2, disc_rewards), 0) / \
                            torch.sum(G ** 2, 0) #Hxm
        else:
            baseline = torch.zeros(1,1) #1x1
        baseline[baseline != baseline] = 0
        values = disc_rewards.unsqueeze(2) - baseline.unsqueeze(0) #NxHxm
        
        ent_bonus = torch.mean(policy.entropy_grad(states), 1) #Nxm
        _samples = (1-coeff) * torch.sum(tensormat(G * values*log_iws, mask), 1) + coeff * ent_bonus #Nxm
        if result == 'samples':
            return _samples #Nxm
        else:
            return torch.mean(_samples, 0) #m
        
def _shallow_reinforce_estimator(batch, disc, policy, baselinekind='peters', result='mean', alpha=0.5):
    with torch.no_grad():        
        states, actions, rewards, mask, _ = unpack(batch) #NxHxm, NxHxd, NxH, NxH
        
        scores = policy.score(states, actions) #NxHxm
        scores = tensormat(scores, mask) #NxHxm
        G = torch.sum(scores, 1) #Nxm
        disc_rewards = discount(rewards, disc) #NxH
        rets = torch.sum(disc_rewards, 1) #N
        
        target_logps = policy.log_pdf(states, actions) * mask #NxH
    
        behavioral_logps = policy.log_pdf(states, actions) #NxH
        iw_num = torch.exp(behavioral_logps)
        iw_den = alpha * iw_num + (1-alpha) *  torch.exp(target_logps)
        log_iws = torch.cumsum(torch.log(iw_num / iw_den) * mask, 1)



        if baselinekind == 'avg':
            baseline = torch.mean(rets, 0) #1
        elif baselinekind == 'peters':
            baseline = torch.mean(G ** 2 * rets.unsqueeze(1), 0) /\
                torch.mean(G ** 2, 0) #m
        else:
            baseline = torch.zeros(1) #1
        baseline[baseline != baseline] = 0
        values = rets.unsqueeze(1) - baseline.unsqueeze(0) #Nxm
        
        _samples = G * values * log_iws #Nxm
        if result == 'samples':
            return _samples #Nxm
        else:
            return torch.mean(_samples, 0) #m

def _incr_shallow_gpomdp_estimator(traj, disc, policy, baselinekind='peters', result='mean', cm_1 = 0., cm_2 = 0., cm_3 = 0., tot_trajs = 1):
    with torch.no_grad():
        states, actions, rewards, mask, _ = traj #Hxm, Hxd, H, H
    
        disc_rewards = discount(rewards, disc) #H
        scores = policy.score(states, actions).squeeze() #Hxm
        G = torch.cumsum(scores * mask.unsqueeze(1), 0) #Hxm
        if baselinekind == 'avg':
            baseline = incr_mean(cm_2, disc_rewards, tot_trajs) #H
            res_2 = baseline
            res_3 = 0.
            values = (disc_rewards - baseline).unsqueeze(1)
        elif baselinekind == 'peters':
            num = incr_mean(cm_2, G**2 * disc_rewards.unsqueeze(1), tot_trajs) #Hxm
            den = incr_mean(cm_3, G**2, tot_trajs) #Hxm
            baseline = num / den #Hxm
            res_2 = num
            res_3 = den
            baseline[baseline!=baseline] = 0
            values = disc_rewards.unsqueeze(1) - baseline
        else:
            values = disc_rewards.unsqueeze(1)
            res_2 = 0.
            res_3 = 0.
        
        _sample = torch.sum(G * values * mask.unsqueeze(1), 0)
        if result == 'samples':
            return _sample, res_2, res_3
        else:
            return incr_mean(cm_1, _sample, tot_trajs), res_2, res_3
        
"""Testing"""
if __name__ == '__main__':
    pass
    