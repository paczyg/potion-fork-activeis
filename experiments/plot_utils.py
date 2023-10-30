import os
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt
from expsuite import PyExperimentSuite

def plot_ci(df, key, xkey, ax=None, *plt_args, **plt_kwargs):
    if ax is None:
        ax=plt.gca()

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

def plot_boxplot(df, key, xkey, ax=None, *plt_args, **plt_kwargs):
    if ax is None:
        ax=plt.gca()

    df.boxplot(column = key, by = xkey, ax = ax)

def get_dataframe(suite, experiment_name, xkey, cos_sim=False):
    assert experiment_name in suite.cfgparser.sections(), \
        "The experiment name is not present in the chosen configuration file"

    experiment_path = os.path.join(suite.cfgparser.defaults()['path'], experiment_name)
    exps = suite.get_exps(path=experiment_path)
    params = suite.get_params(experiment_path)

    df = pd.DataFrame()
    for rep in range(params['repetitions']):
        for exp in exps:
            
            _dict = suite.get_value(exp, rep, 'all', 'last')

            # Use cosine similarity to compare gradients
            if cos_sim:
                grad_cos_sim = np.dot(_dict['grad_is'], _dict['grad_mc'])/(np.linalg.norm(_dict['grad_is'])*np.linalg.norm(_dict['grad_mc']))
                _dict['grad_cos_sim'] = grad_cos_sim
                del _dict['grad_is'], _dict['grad_mc']

            _df = pd.DataFrame(_dict, index=[rep])
            _df[xkey] = suite.get_params(exp)[xkey]
            df = pd.concat([df, _df],ignore_index = True)
    
    return df

def get_dataframes_over_repetitions(suite, exp, tags="all"):
    # Retrieve the actual repetitions of a given experiment tags
    # If a repetition has less iterations than the prescibed ones, NaN values are used to fill the dataframe

    dfs = []
    params = suite.get_params(exp)
    for rep in range(params['repetitions']):
        log = suite.get_history(exp, rep, tags)
        if len(log) == 0:
            continue
        if tags != "all" and type(tags) is not list:
            log = pd.DataFrame(log,columns=[tags])
        else:
            log = pd.DataFrame(log)
        # Fill rows with the expected number of iterations
        if log.shape[0] < params["iterations"]:
            log = pd.concat([log,pd.DataFrame(np.nan, index=np.arange(params["iterations"]-log.shape[0]), columns=log.columns)])
        dfs.append(log)
    return dfs

def get_ci(suite, exp, key):
    m = suite.get_histories_over_repetitions(exp, key, np.mean)
    s = suite.get_histories_over_repetitions(exp, key, st.sem)
    s[s==0] = "nan"
    lb, ub = st.norm.interval(0.95, loc = m, scale = s)
    return m, lb, ub

def find_exps(suite: PyExperimentSuite, query_dict: dict, dir: str, experiment: str=None) -> tuple(list[str], list[dict]):
    """
    Search all the experiments paths corresponding to a particular combination of parameters.
    The search of the experiments is performed from the specified directory, and within an (optional) experiment set.

    Args:
        suite (PyExperimentSuite): The suite containing the experiments (and, in principle, their configuration file)
        query_dict (dict): The dictionary containing the parameters configuration used to find the corresponding experiments
        dir (str): The main directory containing all the experiments you want to search in
        experiment (str, optional): The name of the experiment set, within the specified suite

    Returns:
        list[str]: The list with the experiments paths matching the queried parameters configurations
        list[dict]: The list with the parameters dictionaries of the retrieved experiments
    """
    
    if experiment is None:
        exps = suite.get_exps(dir)
    else:
        # exps = suite.get_exps(os.path.join(dir, experiment))
        exps = suite.get_exps(os.path.join(dir, experiment))

    names = []
    params = []
    for dic in [suite.get_params(exp) for exp in exps]:
        select = True
        for k,v in query_dict.items():
            select = False if dic[k] != v else select
        if select:
            names.append(os.path.join(dir, dic['name']))
            params.append(dic)
        
    if len(names) == 1:
        return names[0], params[0]
    else:
        return names, params


def plot_iterations_key(dir, key='TestPerf'):
    """
    Plot iterations VS confidence intervals of specified key.
    Each experiment found in the directory is plotted into a separate figure.

    dir: directory with an experiment.cfg file describing the experiments in the directory
    key: key values to plot
    """
    suite = PyExperimentSuite(config = os.path.join(dir,'experiment.cfg'))
    exps = suite.get_exps(dir)
    for exp in exps:

        plt.figure()
        plt.title(suite.get_params(exp)['name'])
        plt.ylabel('Deterministic Performance')
        plt.ylabel('Return')
        plt.xlabel('Iterations')

        m, ci_lb, ci_ub = get_ci(suite, exp, key)
        plt.plot(range(len(m)), m, label='onpolicy')
        plt.fill_between(range(len(m)), ci_lb, ci_ub, alpha=.1)

        plt.show()

#TODO Styling with cycler
# https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html#sphx-glr-tutorials-intermediate-color-cycle-py

#TODO: un po' inutile, perchè stormpg deve caricare diversamente i dati
# Idealmente dovrebbe servire per plottare una sezione degli esperimenti
def plot_experiments(ax, x_key, label_keys, dir = '.'):
    mysuite = PyExperimentSuite(config = os.path.join(dir,'experiment.cfg'))
    exps = mysuite.get_exps(dir)
    for exp in exps:
        exp_params = mysuite.get_params(exp)
        exp_str = ','.join([f"{k}={exp_params[k]}" for k in label_keys])

        m, ci_lb, ci_ub = get_ci(mysuite, exp, 'TestPerf')
        ax.plot(np.cumsum(mysuite.get_history(exp,0,x_key)), m, label=exp_str)
        ax.fill_between(np.cumsum(mysuite.get_history(exp,0,x_key)), ci_lb, ci_ub, alpha=.1)
    ax.legend()