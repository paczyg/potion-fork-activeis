import os
import argparse
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt

from expsuite import PyExperimentSuite

"""
***************************************************************************************************
                                            Utilities 
***************************************************************************************************
"""
def plot_ci(df, key, xkey, ax=None, *plt_args, **plt_kwargs):
    if ax is None:
        ax=plt.gca()

    stats = df.groupby(xkey)[key].agg(['mean', 'std', 'count'])
    xs = stats.index.values
    ys = stats['mean']
    ci_lb, ci_ub = st.norm.interval(0.68, loc=stats['mean'], scale=stats['std'])
    ax.plot(xs,ys, *plt_args, **plt_kwargs)
    ax.fill_between(xs, ci_lb, ci_ub, alpha=.1)

def get_dataframe(suite, experiment_name, xkey):
    assert experiment_name in suite.cfgparser.sections(), \
        "The experiment name is not present in the chosen configuration file"

    experiment_path = os.path.join(suite.cfgparser.defaults()['path'], experiment_name)
    exps = suite.get_exps(path=experiment_path)
    params = suite.get_params(experiment_path)

    df = pd.DataFrame()
    for rep in range(params['repetitions']):
        for exp in exps:
            _df = pd.DataFrame(suite.get_value(exp, rep, 'all', 'last'))
            _df[xkey] = suite.get_params(exp)[xkey]
            df = pd.concat([df, _df],ignore_index = True)
    
    return df

"""
***************************************************************************************************
                                            Plot 
***************************************************************************************************
"""
mysuite = PyExperimentSuite(config='experiments_lq.cfg')
# mysuite = PyExperimentSuite(config='experiments_lq_chi2.cfg')
# mysuite = PyExperimentSuite(config='experiments_cartpole_chi2.cfg')
experiments = ['means', 'stds', 'horizons', 'dimensions']

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


# Varying initial policy mean
# -----------------------------------------------------
if 'means' in experiments:
    df = get_dataframe(mysuite, experiment_name='means', xkey='mu_init')

    fig,axes = plt.subplots(1,2)

    plot_ci(df,'grad_is','mu_init', axes[0], 'o-')
    plot_ci(df,'grad_mc','mu_init', axes[0], 's--')
    axes[0].set(xlabel='mu_init', ylabel='mean of gradients')
    axes[0].legend(["IS", "MC"])

    plot_ci(df,'var_grad_is','mu_init', axes[1], 'o-')
    plot_ci(df,'var_grad_mc','mu_init', axes[1], 's--')
    axes[1].set(xlabel='mu_init', ylabel='var of mean of gradients')
    axes[1].legend(["IS", "MC"])

    fig.tight_layout()
    plt.show()

# Varying initial policy variance
# -----------------------------------------------------
if 'stds' in experiments:
    df = get_dataframe(mysuite, experiment_name='stds', xkey='logstd_init')

    fig,axes = plt.subplots(1,2)

    plot_ci(df,'grad_is','logstd_init', axes[0], 'o-')
    plot_ci(df,'grad_mc','logstd_init', axes[0], 's--')
    axes[0].set(xlabel='logstd_init', ylabel='mean of gradients')
    axes[0].legend(["IS", "MC"])

    plot_ci(df,'var_grad_is','logstd_init', axes[1], 'o-')
    plot_ci(df,'var_grad_mc','logstd_init', axes[1], 's--')
    axes[1].set(xlabel='logstd_init', ylabel='var of mean of gradients')
    axes[1].legend(["IS", "MC"])

    fig.tight_layout()
    plt.show()

# Varying horizon
# -----------------------------------------------------
if 'horizons' in experiments:
    df = get_dataframe(mysuite, experiment_name='horizons', xkey='horizon')

    fig,axes = plt.subplots(1,2)

    plot_ci(df,'grad_is','horizon', axes[0], 'o-')
    plot_ci(df,'grad_mc','horizon', axes[0], 's--')
    axes[0].set(xlabel='horizon', ylabel='mean of gradients')
    axes[0].legend(["IS", "MC"])

    plot_ci(df,'var_grad_is','horizon', axes[1], 'o-')
    plot_ci(df,'var_grad_mc','horizon', axes[1], 's--')
    axes[1].set(xlabel='horizon', ylabel='var of mean of gradients')
    axes[1].legend(["IS", "MC"])

    fig.tight_layout()
    plt.show()

# Varying state dimensions
# -----------------------------------------------------
if 'dimensions' in experiments:
    df = get_dataframe(mysuite, experiment_name='dimensions', xkey='state_dim')

    fig,axes = plt.subplots(1,2)

    plot_ci(df,'grad_is','state_dim', axes[0], 'o-')
    plot_ci(df,'grad_mc','state_dim', axes[0], 's--')
    axes[0].set(xlabel='state_dim', ylabel='mean of gradients')
    axes[0].legend(["IS", "MC"])

    plot_ci(df,'var_grad_is','state_dim', axes[1], 'o-')
    plot_ci(df,'var_grad_mc','state_dim', axes[1], 's--')
    axes[1].set(xlabel='state_dim', ylabel='var of mean of gradients')
    axes[1].legend(["IS", "MC"])

    fig.tight_layout()
    plt.show()