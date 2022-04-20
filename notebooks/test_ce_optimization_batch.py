import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import torch

from potion.algorithms.ce_optimization import algo
from potion.envs.lq import LQ
from potion.actors.continuous_policies import ShallowGaussianPolicy

"""
Test definitions
"""
def test_variable_mu_logstd(env, N_per_it, n_ce_iterations, mu_init_list, logstd_init_list, *,
                            estimator='gpomdp', baseline='zero', action_filter=None):

    test_info={
        'env'               :str(env),
        'N_per_it'          :N_per_it, 
        'n_ce_iterations'   :n_ce_iterations,
        'mu_init_list'      :mu_init_list,
        'logstd_init_list'  :logstd_init_list,
        'estimator'         :estimator,
        'baseline'          :baseline,
        'action_filter'     :action_filter
    }

    results = []
    for mu_init in mu_init_list:
        for logstd_init in logstd_init_list:

            target_policy = ShallowGaussianPolicy(
                env.ds,                      # input size
                env.da,                      # output size
                mu_init = mu_init,           # initial mean parameters
                logstd_init = logstd_init,   # log of standard deviation
                learn_std = False            # We are NOT going to learn the variance parameter
            )

            _result, _, _ = algo(
                env,
                target_policy,
                N_per_it,
                n_ce_iterations, 
                estimator=estimator,
                baseline=baseline,
                action_filter=action_filter,
                window=None,
                optimize_mean=True,
                optimize_variance=True,
                run_mc_comparison = True
            )

            _result.update({'grad_mc': _result['grad_mc'][0]})
            _result.update({'grad_is': _result['grad_is'][0]})
            _result.update({'mu_init': mu_init.item()})
            _result.update({'logstd_init': logstd_init.item()})
            results.append(_result)
     
    return pd.DataFrame(results), test_info

def test_variable_horizon(env, N_per_it, n_ce_iterations, mu_init, logstd_init, horizon_list,*,
             estimator='gpomdp', baseline='zero', action_filter=None):

    test_info={
        'env'               :str(env),
        'N_per_it'          :N_per_it, 
        'n_ce_iterations'   :n_ce_iterations,
        'mu_init'           :mu_init.item(),
        'logstd_init'       :logstd_init.item(),
        'horizon_list'      :horizon_list,
        'estimator'         :estimator,
        'baseline'          :baseline,
        'action_filter'     :action_filter
    }

    target_policy = ShallowGaussianPolicy(
        env.ds,                      # input size
        env.da,                      # output size
        mu_init = mu_init,           # initial mean parameters
        logstd_init = logstd_init,   # log of standard deviation
        learn_std = False            # We are NOT going to learn the variance parameter
    )

    results = []
    for horizon in horizon_list:
        env.horizon = horizon
        _result, _, _ = algo(
            env,
            target_policy,
            N_per_it,
            n_ce_iterations, 
            estimator=estimator,
            baseline=baseline,
            action_filter=action_filter,
            window=None,
            optimize_mean=True,
            optimize_variance=True,
            run_mc_comparison = True
        )

        _result.update({'grad_mc': _result['grad_mc'][0]})
        _result.update({'grad_is': _result['grad_is'][0]})
        _result.update({'horizon': horizon})
        results.append(_result)
     
    return pd.DataFrame(results), test_info

def test_variable_batch(env, N_per_it_list, n_ce_iterations, mu_init, logstd_init, *,
             estimator='gpomdp', baseline='zero', action_filter=None):

    test_info={
        'env'               :str(env),
        'N_per_it_list'     :N_per_it_list, 
        'n_ce_iterations'   :n_ce_iterations,
        'mu_init'           :mu_init.item(),
        'logstd_init'       :logstd_init.item(),
        'estimator'         :estimator,
        'baseline'          :baseline,
        'action_filter'     :action_filter
    }

    target_policy = ShallowGaussianPolicy(
        env.ds,                      # input size
        env.da,                      # output size
        mu_init = mu_init,           # initial mean parameters
        logstd_init = logstd_init,   # log of standard deviation
        learn_std = False            # We are NOT going to learn the variance parameter
    )

    results = []
    for N_per_it in N_per_it_list:
        _result, _, _ = algo(
            env,
            target_policy,
            N_per_it,
            n_ce_iterations, 
            estimator=estimator,
            baseline=baseline,
            action_filter=action_filter,
            window=None,
            optimize_mean=True,
            optimize_variance=True,
            run_mc_comparison = True
        )

        _result.update({'grad_mc': _result['grad_mc'][0]})
        _result.update({'grad_is': _result['grad_is'][0]})
        _result.update({'N_per_it': N_per_it})
        results.append(_result)
     
    return pd.DataFrame(results), test_info

"""
Utilities
""" 
def repeat_experiments(nExperiments, test_fun):
    results = []
    for exp in range(nExperiments):
        _result, test_info = test_fun()
        _result['exp'] = exp
        results.append(_result)
    return pd.concat(results), test_info


def plot_ci(df, key, xkey, ax=None, *plt_args, **plt_kwargs):
    if ax is None:
        ax=plt.gca()

    stats = df.groupby(xkey)[key].agg(['mean', 'std', 'count'])
    xs = df[xkey].unique()
    ys = stats['mean']
    ci_lb, ci_ub = st.norm.interval(0.68, loc=stats['mean'], scale=stats['std']/np.sqrt(stats['count']))
    ax.plot(xs,ys, *plt_args, **plt_kwargs)
    ax.fill_between(xs, ci_lb, ci_ub, alpha=.1)

# fig.supxlabel('common_x')
# plt.savefig("image.png",bbox_inches='tight')

if __name__ == '__main__':
    do_plot = True

    # Set Environment
    # ---------------
    ds = 1
    da = 1
    env = LQ(ds,da)
    env.horizon=1

    # Test 1
    # Constant policy variance, varying initial policy mean
    # -----------------------------------------------------
    test1_res, test1_info = repeat_experiments(
        100,
        lambda: test_variable_mu_logstd(env,10,1,
                                        [x*torch.ones(ds) for x in [-1.0, -0.5, 0.0, 0.5, 1.0]],
                                        [torch.zeros(ds)])
    )

    if do_plot:
        fig,axes = plt.subplots(1,2)

        plot_ci(test1_res,'grad_is','mu_init', axes[0], 'o-')
        plot_ci(test1_res,'grad_mc','mu_init', axes[0], 's--')
        axes[0].set(xlabel='mu_init', ylabel='mean of gradients')
        axes[0].legend(["IS", "MC"])

        plot_ci(test1_res,'var_grad_is','mu_init', axes[1], 'o-')
        plot_ci(test1_res,'var_grad_mc','mu_init', axes[1], 's--')
        axes[1].set(xlabel='mu_init', ylabel='var of mean of gradients')
        axes[1].legend(["IS", "MC"])

        fig.tight_layout()
        plt.show()

    # Test 2
    # Constant policy mean, varying policy variance
    # ---------------------------------------------
    test2_res, test2_info = repeat_experiments(
        100,
        lambda: test_variable_mu_logstd(env,10,1,
                                        [torch.zeros(ds)],
                                        [x*torch.ones(ds) for x in [-1.0, -0.5, 0.0, 0.5, 1.0]])
    )

    if do_plot:
        fig,axes = plt.subplots(1,2)

        plot_ci(test2_res,'grad_is','logstd_init', axes[0], 'o-')
        plot_ci(test2_res,'grad_mc','logstd_init', axes[0], 's--')
        axes[0].set(xlabel='logstd_init', ylabel='mean of gradients')
        axes[0].legend(["IS", "MC"])

        plot_ci(test2_res,'var_grad_is','logstd_init', axes[1], 'o-')
        plot_ci(test2_res,'var_grad_mc','logstd_init', axes[1], 's--')
        axes[1].set(xlabel='logstd_init', ylabel='var of mean of gradients')
        axes[1].legend(["IS", "MC"])

        fig.tight_layout()
        plt.show()

    # Test 3
    # Constant policy mean and variance, varying horizon
    # ----------------------------------------------------
    test3_res, test3_info = repeat_experiments(
        100,
        lambda: test_variable_horizon(env,10,1, torch.zeros(ds), torch.zeros(ds),
                                      [1,2,5,10])
    )
    if do_plot:
        fig,axes = plt.subplots(1,2)

        plot_ci(test3_res,'grad_is','horizon', axes[0], 'o-')
        plot_ci(test3_res,'grad_mc','horizon', axes[0], 's--')
        axes[0].set(xlabel='horizon', ylabel='mean of gradients')
        axes[0].legend(["IS", "MC"])

        plot_ci(test3_res,'var_grad_is','horizon', axes[1], 'o-')
        plot_ci(test3_res,'var_grad_mc','horizon', axes[1], 's--')
        axes[1].set(xlabel='horizon', ylabel='var of mean of gradients')
        axes[1].legend(["IS", "MC"])

        fig.tight_layout()
        plt.show()

    # Test 4
    # Constant policy mean and variance, varying batchsize
    # ----------------------------------------------------
    test4_res, test4_info = repeat_experiments(
        100,
        lambda: test_variable_batch(env,[5,10,30,60],1, torch.zeros(ds), torch.zeros(ds))
    )
    if do_plot:
        fig,axes = plt.subplots(1,2)

        plot_ci(test4_res,'grad_is','N_per_it', axes[0], 'o-')
        plot_ci(test4_res,'grad_mc','N_per_it', axes[0], 's--')
        axes[0].set(xlabel='N_per_it', ylabel='mean of gradients')
        axes[0].legend(["IS", "MC"])

        plot_ci(test4_res,'var_grad_is','N_per_it', axes[1], 'o-')
        plot_ci(test4_res,'var_grad_mc','N_per_it', axes[1], 's--')
        axes[1].set(xlabel='N_per_it', ylabel='var of mean of gradients')
        axes[1].legend(["IS", "MC"])

        fig.tight_layout()
        plt.show()
