from matplotlib import pyplot as plt

from expsuite import PyExperimentSuite
from experiments.plot_utils import plot_ci, get_dataframe

"""
***************************************************************************************************
                                            Plot 
***************************************************************************************************
"""
mysuite = PyExperimentSuite(config='experiments_swimmer.cfg')
experiments = ['batches', 'iterations']

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
    df = get_dataframe(mysuite, experiment_name='means', xkey='mu_init')

    fig,axes = plt.subplots(1,2)

    line_is = plot_ci(df,'grad_is','mu_init', axes[0], 'o-')
    line_mc = plot_ci(df,'grad_mc','mu_init', axes[0], 's--')
    axes[0].set(xlabel='mu_init', ylabel='mean of gradients')
    axes[0].legend([line_is[0], line_mc[0]], ["IS", "MC"])

    line_is = plot_ci(df,'var_grad_is','mu_init', axes[1], 'o-')
    line_mc = plot_ci(df,'var_grad_mc','mu_init', axes[1], 's--')
    axes[1].set(xlabel='mu_init', ylabel='var of mean of gradients')
    axes[1].legend([line_is[0], line_mc[0]], ["IS", "MC"])

    fig.tight_layout()
    plt.show()

# Varying iterations
# ------------------
if 'iterations' in experiments:
    df = get_dataframe(mysuite, experiment_name='iterations', xkey='ce_max_iter', cos_sim = True)

    fig,axes = plt.subplots(1, 2, dpi = 200)

    plot_ci(df,'grad_cos_sim','ce_max_iter', axes[0])
    axes[0].set(xlabel='ce_max_iter', ylabel='grad cosine similarity')

    line_is = plot_ci(df,'var_grad_is','ce_max_iter', axes[1], 'o-')
    line_mc = plot_ci(df,'var_grad_mc','ce_max_iter', axes[1], 's--')
    axes[1].set(xlabel='ce_max_iter', ylabel='var of mean of gradients')
    axes[1].legend([line_is[0], line_mc[0]], ["IS", "MC"])

    fig.tight_layout()
    plt.show()

# Varying batches
# ---------------
if 'batches' in experiments:
    df = get_dataframe(mysuite, experiment_name='batches', xkey='ce_batchsizes', cos_sim = True)

    fig,axes = plt.subplots(1, 2, dpi = 200)

    plot_ci(df,'grad_cos_sim','ce_batchsizes', axes[0], 'o-')
    axes[0].set(xlabel='ce_batchsizes', ylabel='grad cosine similarity')

    line_is = plot_ci(df,'var_grad_is','ce_batchsizes', axes[1], 'o-')
    line_mc = plot_ci(df,'var_grad_mc','ce_batchsizes', axes[1], 's--')
    axes[1].set(xlabel='ce_batchsizes', ylabel='var of mean of gradients')
    axes[1].legend([line_is[0], line_mc[0]], ["IS", "MC"])

    fig.tight_layout()
    plt.show()
