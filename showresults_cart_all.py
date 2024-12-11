# %% 
from matplotlib import pyplot as plt

from expsuite import PyExperimentSuite
from experiments.plot_utils import plot_ci, get_dataframe

import numpy as np
import pandas as pd
import scipy.stats as st

"""
***************************************************************************************************
                                            Plot 
***************************************************************************************************
"""
mysuite = PyExperimentSuite(config='exp_done/experiments_cartpole_my.cfg')

origsuite = PyExperimentSuite(config='exp_done/experiments_cartpole_orig.cfg')

# mysuite = PyExperimentSuite(config='experiments_cartpole_chi2.cfg')
experiments = ['means_20', 'stds_20']

'''
# Command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', '-c', help='Experiment configuration file', type=str,
                    default='experiments.cfg')
parser.add_argument('--experiment', '-e', help='Experiments to be shown', dest='experiments', action = 'append',type=str,
                    default=None)

# Parse arguments
args = parser.parse_known_args()[0]
mysuite = PyExperimentSuite(config = args.config)
if args.experiments is not None:
    experiments = args.experiments
else:
    experiments = mysuite.cfgparser.sections()
'''


def plot_ci_curve(df, key, xkey, ax=None, *plt_args, **plt_kwargs):
    if ax is None:
        ax=plt.gca()

    print(df.head())

    df_grouped = df.groupby(xkey)[key]
    xs = list(df_grouped.indices.keys())
    ys = df_grouped.mean()
    ci_lb, ci_ub = st.norm.interval(0.95, loc = ys, scale = df_grouped.agg(st.sem))
    ci_lb[np.isnan(ci_lb)] = ys[np.isnan(ci_lb)]
    ci_ub[np.isnan(ci_ub)] = ys[np.isnan(ci_ub)]
    lines = ax.plot(xs,ys, *plt_args, **plt_kwargs)
    ax.set_xticks(xs)
    ax.fill_between(xs, ci_lb, ci_ub, alpha=.1)

    return lines


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    'text.latex.preamble': r"\usepackage{amsmath}"
})

# Varying initial policy mean
# -----------------------------------------------------
if 'means_20' in experiments:
    df = get_dataframe(mysuite, experiment_name='means_20', xkey='mu_init', cos_sim=True)
    df_orig = get_dataframe(origsuite, experiment_name='means_20', xkey='mu_init', cos_sim=True)

    fig,axes = plt.subplots(1, 2, dpi = 200)

    line_is = plot_ci(df,'grad_cos_sim','mu_init', axes[0], 'o-')
    line_origis = plot_ci(df_orig,'grad_cos_sim','mu_init', axes[0], 'r-')
    axes[0].set(xlabel = r'Policy mean $\boldsymbol{\theta}_\mu$', ylabel = 'Cosine similarity')
    axes[0].set_ylim([-1.2, 1.2])

    line_is = plot_ci(df,'var_grad_is','mu_init', axes[1], 'o-')
    line_origis = plot_ci(df_orig,'var_grad_is','mu_init', axes[1], 'r-')
    line_mc = plot_ci(df_orig,'var_grad_mc','mu_init', axes[1], 's--')
    axes[1].set(xlabel = r'Policy mean $\boldsymbol{\theta}_\mu$', ylabel = 'Variance of gradient estimates')
    axes[1].legend([line_is[0], line_origis[0], line_mc[0]], ["my", "off-policy", "on-policy"])

    fig.tight_layout()
    plt.show()

    figk, axesk = plt.subplots(1, 1, dpi=200)
    axesk.plot(df['ret'], 'b-')
    axesk.plot(df_orig['ret'], 'r-')

    axesk.set(xlabel='Iterations', ylabel='Return')
    axesk.legend(["my", "orig"])
    plt.show()

    #plot_ci(df, 'ret', 'Iterations', axesk, 'o-')
    #plot_ci(df_orig, 'ret', 'Iterations', axesk, 'r-')


# Varying initial policy variance
# -----------------------------------------------------
if 'stds_20' in experiments:
    df = get_dataframe(mysuite, experiment_name='stds_20', xkey='logstd_init', cos_sim=True)
    df_orig = get_dataframe(origsuite, experiment_name='stds_20', xkey='logstd_init', cos_sim=True)

    fig,axes = plt.subplots(1, 2, dpi = 200)

    line_is = plot_ci(df,'grad_cos_sim','logstd_init', axes[0], 'o-')
    line_origis = plot_ci(df_orig,'grad_cos_sim','logstd_init', axes[0], 'r-')
    axes[0].set(xlabel = 'Policy log standard deviation', ylabel = 'Cosine similarity')
    axes[0].set_ylim([-1.2, 1.2])

    line_is = plot_ci(df,'var_grad_is','logstd_init', axes[1], 'o-')
    line_origis = plot_ci(df_orig,'var_grad_is','logstd_init', axes[1], 'r-')
    line_mc = plot_ci(df,'var_grad_mc','logstd_init', axes[1], 's--')
    axes[1].set(xlabel = 'Policy log standard deviation', ylabel = 'Variance of gradient estimates')
    axes[1].legend([line_is[0], line_origis[0], line_mc[0]], ["my", "off-policy", "on-policy"])

    fig.tight_layout()
    plt.show()