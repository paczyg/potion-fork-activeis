from suite import MySuite
import potion.visualization.notebook_utils as nu

import os
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt

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
    ci_lb, ci_ub = st.norm.interval(0.68, loc=stats['mean'], scale=stats['std']/np.sqrt(stats['count']))
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
mysuite = MySuite(config='experiments.cfg')

# Test 1
# Varying initial policy mean
# -----------------------------------------------------
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

# Test 2
# Varying initial policy variance
# -----------------------------------------------------
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

# Test 3
# Varying horizon
# -----------------------------------------------------
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

# Test 4
# Varying horizon
# -----------------------------------------------------
df = get_dataframe(mysuite, experiment_name='batchsizes', xkey='n_per_it')

fig,axes = plt.subplots(1,2)

plot_ci(df,'grad_is','n_per_it', axes[0], 'o-')
plot_ci(df,'grad_mc','n_per_it', axes[0], 's--')
axes[0].set(xlabel='n_per_it', ylabel='mean of gradients')
axes[0].legend(["IS", "MC"])

plot_ci(df,'var_grad_is','n_per_it', axes[1], 'o-')
plot_ci(df,'var_grad_mc','n_per_it', axes[1], 's--')
axes[1].set(xlabel='n_per_it', ylabel='var of mean of gradients')
axes[1].legend(["IS", "MC"])

fig.tight_layout()
plt.show()

# Test 5
# Varying state dimensions
# -----------------------------------------------------
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