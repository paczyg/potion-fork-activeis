from matplotlib import pyplot as plt

from expsuite import PyExperimentSuite
from experiments.plot_utils import plot_ci, get_dataframe

"""
***************************************************************************************************
                                            Plot 
***************************************************************************************************
"""
mysuite = PyExperimentSuite(config='experiments_cartpole.cfg')
# mysuite = PyExperimentSuite(config='experiments_cartpole_chi2.cfg')
experiments = ['means', 'stds']

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

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })

# Varying initial policy mean
# -----------------------------------------------------
if 'means' in experiments:
    df = get_dataframe(mysuite, experiment_name='means', xkey='mu_init', cos_sim=True)

    fig,axes = plt.subplots(1,2)

    line_is = plot_ci(df,'grad_cos_sim','mu_init', axes[0])
    axes[0].set(xlabel='mu_init', ylabel='grad cosine similarity')

    line_is = plot_ci(df,'var_grad_is','mu_init', axes[1], 'o-')
    line_mc = plot_ci(df,'var_grad_mc','mu_init', axes[1], 's--')
    axes[1].set(xlabel='mu_init', ylabel='var of mean of gradients')
    axes[1].legend([line_is[0], line_mc[0]], ["IS", "MC"])

    fig.tight_layout()
    plt.show()

# Varying initial policy variance
# -----------------------------------------------------
if 'stds' in experiments:
    df = get_dataframe(mysuite, experiment_name='stds', xkey='logstd_init', cos_sim=True)

    fig,axes = plt.subplots(1, 2, dpi = 200)

    line_is = plot_ci(df,'grad_cos_sim','logstd_init', axes[0])
    axes[0].set(xlabel='logstd_init', ylabel='grad cosine similarity')

    line_is = plot_ci(df,'var_grad_is','logstd_init', axes[1], 'o-')
    line_mc = plot_ci(df,'var_grad_mc','logstd_init', axes[1], 's--')
    axes[1].set(xlabel='logstd_init', ylabel='var of mean of gradients')
    axes[1].legend([line_is[0], line_mc[0]], ["IS", "MC"])

    fig.tight_layout()
    plt.show()