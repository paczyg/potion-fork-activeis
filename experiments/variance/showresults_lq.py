from matplotlib import pyplot as plt

from expsuite import PyExperimentSuite
from experiments.plot_utils import plot_ci, get_dataframe, plot_boxplot

"""
***************************************************************************************************
                                            Plot 
***************************************************************************************************
"""
mysuite = PyExperimentSuite(config='experiments_lq.cfg')
# mysuite = PyExperimentSuite(config='experiments_lq_chi2.cfg')
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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    'text.latex.preamble': r"\usepackage{amsmath}"
})

# Varying initial policy mean
# -----------------------------------------------------
if 'means' in experiments:
    df = get_dataframe(mysuite, experiment_name='means', xkey='mu_init', cos_sim = True)
    # df = get_dataframe(mysuite, experiment_name='means', xkey='mu_init')

    fig,axes = plt.subplots(1, 2, dpi = 200)

    plot_ci(df,'grad_cos_sim','mu_init', axes[0])
    axes[0].set(xlabel = r'Policy mean $\boldsymbol{\theta}_\mu$', ylabel = 'Cosine similarity')
    axes[0].set_ylim([-1.2, 1.2])
    # line_is = plot_ci(df,'grad_is','mu_init', axes[0], 'o-')
    # line_mc = plot_ci(df,'grad_mc','mu_init', axes[0], 's--')
    # axes[0].set(xlabel='mu_init', ylabel='mean of gradients')
    # axes[0].legend([line_is[0], line_mc[0]], ["off-policy", "on-policy"])

    line_is = plot_ci(df,'var_grad_is','mu_init', axes[1], 'o-')
    line_mc = plot_ci(df,'var_grad_mc','mu_init', axes[1], 's--')
    axes[1].set(xlabel = r'Policy mean $\boldsymbol{\theta}_\mu$', ylabel = 'Variance of gradient estimates')
    axes[1].legend([line_is[0], line_mc[0]], ["off-policy", "on-policy"])

    fig.tight_layout()
    plt.show()

# Varying initial policy variance
# -----------------------------------------------------
if 'stds' in experiments:
    df = get_dataframe(mysuite, experiment_name='stds', xkey='logstd_init', cos_sim = True)
    # df = get_dataframe(mysuite, experiment_name='stds', xkey='logstd_init')

    fig,axes = plt.subplots(1, 2, dpi = 200)

    plot_ci(df,'grad_cos_sim','logstd_init', axes[0])
    axes[0].set(xlabel = 'Policy log standard deviation', ylabel = 'Cosine similarity')
    axes[0].set_ylim([-1.2, 1.2])
    # line_is = plot_ci(df,'grad_is','logstd_init', axes[0], 'o-')
    # line_mc = plot_ci(df,'grad_mc','logstd_init', axes[0], 's--')
    # axes[0].set(xlabel='logstd_init', ylabel='mean of gradients')
    # axes[0].legend([line_is[0], line_mc[0]], ["off-policy", "on-policy"])

    line_is = plot_ci(df,'var_grad_is','logstd_init', axes[1], 'o-')
    line_mc = plot_ci(df,'var_grad_mc','logstd_init', axes[1], 's--')
    axes[1].set(xlabel = 'Policy log standard deviation', ylabel = 'Variance of gradient estimates')
    axes[1].legend([line_is[0], line_mc[0]], ["off-policy", "on-policy"])

    fig.tight_layout()
    plt.show()

# Varying horizon
# -----------------------------------------------------
if 'horizons' in experiments:
    df = get_dataframe(mysuite, experiment_name='horizons', xkey='horizon', cos_sim = True)
    # df = get_dataframe(mysuite, experiment_name='horizons', xkey='horizon')

    fig,axes = plt.subplots(1, 2, dpi = 200)

    plot_ci(df,'grad_cos_sim','horizon', axes[0])
    axes[0].set(xlabel = 'Horizon', ylabel = 'Cosine similarity')
    axes[0].set_ylim([-1.2, 1.2])
    # line_is = plot_ci(df,'grad_is','horizon', axes[0], 'o-')
    # line_mc = plot_ci(df,'grad_mc','horizon', axes[0], 's--')
    # axes[0].set(xlabel='horizon', ylabel='mean of gradients')
    # axes[0].legend([line_is[0], line_mc[0]], ["off-policy", "on-policy"])

    line_is = plot_ci(df,'var_grad_is','horizon', axes[1], 'o-')
    line_mc = plot_ci(df,'var_grad_mc','horizon', axes[1], 's--')
    axes[1].set(xlabel = 'Horizon', ylabel = 'Variance of gradient estimates')
    axes[1].legend([line_is[0], line_mc[0]], ["off-policy", "on-policy"])

    fig.tight_layout()
    plt.show()

# Varying state dimensions
# -----------------------------------------------------
if 'dimensions' in experiments:
    df = get_dataframe(mysuite, experiment_name='dimensions', xkey='state_dim', cos_sim = True)

    fig,axes = plt.subplots(1, 2, dpi = 200)

    line_is = plot_ci(df,'grad_cos_sim','state_dim', axes[0])
    axes[0].set(xlabel = r'State dimension $d$', ylabel = 'Cosine similarity')
    axes[0].set_ylim([-1.2, 1.2])

    line_is = plot_ci(df,'var_grad_is','state_dim', axes[1], 'o-')
    line_mc = plot_ci(df,'var_grad_mc','state_dim', axes[1], 's--')
    axes[1].set(xlabel = r'State dimension $d$', ylabel = 'Variance of gradient estimates')
    axes[1].legend([line_is[0], line_mc[0]], ["off-policy", "on-policy"])

    fig.tight_layout()
    plt.show()