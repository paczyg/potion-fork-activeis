#%%
from suite import MySuite

import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt

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
    params = suite.get_params(exp)
    n = params['repetitions']
    m = suite.get_histories_over_repetitions(exp, key, np.mean)
    s = suite.get_histories_over_repetitions(exp, key, np.std)
    s[s==0] = "nan"
    lb, ub = st.norm.interval(0.68, loc=m, scale=s/np.sqrt(n))
    return m, lb, ub

def plot_ci(suite, exp, key="UTestPerf", label=None):
    params = suite.get_params(exp)
    m, ci_lb, ci_ub = get_ci(suite, exp, key)
    plt.plot(np.arange(params['iterations']), m, label=label)
    plt.fill_between(np.arange(params['iterations']), ci_lb, ci_ub, alpha=.1)

#%% ===================================================================================
mysuite = MySuite(config='lq_s1.cfg')
main_dir = "results_lq_s1"

exp_off  = f"{main_dir}/offpolicy/batchsize{15}defensive_batch{5}"
exp_on  = f"{main_dir}/onpolicy/batchsize{20}"

plt.ylabel('Deterministic Performance')
plt.ylabel('Return')
plt.xlabel('Trajectories')

m, ci_lb, ci_ub = get_ci(mysuite, exp_on, 'TestPerf')
plt.plot(np.cumsum(mysuite.get_history(exp_on,0,'BatchSize')), m, label='onpolicy')
plt.fill_between(np.cumsum(mysuite.get_history(exp_on,0,'BatchSize')), ci_lb, ci_ub, alpha=.1)

m, ci_lb, ci_ub = get_ci(mysuite, exp_off, 'TestPerf')
plt.plot(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total')), m, label='offpolicy')
plt.fill_between(np.cumsum(mysuite.get_history(exp_off,0,'Batch_total')), ci_lb, ci_ub, alpha=.1)

plt.legend()
plt.show()

#%% === LQ 1 11Agosto================================================================================
mysuite = MySuite(config='lq_s1.cfg')
main_dir = "results_lq_s1_11Agosto"

stepper      = "ConstantStepper1e4" #"ConstantStepper1e4", "Adam1e1", "Adam1e2", "Adam1e4"
horizon      = "10"                 #10, 50
sigma_noise  = "0"                  #0, 2
mu_init      = "0"                  #0, 0.5, 1
logstd_init  = "30"                 #00, 10, 20, 30

exp_on  = f"{main_dir}/onpolicy/sigma_noise{sigma_noise}stepper{stepper}horizon{horizon}mu_init{mu_init}logstd_init{logstd_init}"
exp_off  = f"{main_dir}/offpolicy/sigma_noise{sigma_noise}stepper{stepper}horizon{horizon}mu_init{mu_init}logstd_init{logstd_init}"

plt.ylabel('Deterministic Performance')
plt.xlabel('Itrations')
plot_ci(mysuite, exp_on, label="onpolicy")
plot_ci(mysuite, exp_off, label="offpolicy")
plt.title(f"stepper={stepper}, horizon={horizon}, sigma_noise={sigma_noise}, mu_init={mu_init}, logstd_init={logstd_init}")
plt.legend()
plt.show()

# Query behavioural policy parameters
# fig,ax = plt.subplots()
# ax.plot(
#     mysuite.get_histories_fix_params('{main_dir}/sigma_noise0', 0, 'ce_policy_loc00', n_offpolicy_opt=1)[0][0],
#     color='red'
# )
# ax.set_ylabel('ce policy loc', color='red')
# ax2=ax.twinx()
# ax2.plot(
#     mysuite.get_histories_fix_params('{main_dir}/sigma_noise0', 0, 'param0', n_offpolicy_opt=1)[0][0],
#     color='blue'
# )
# ax2.set_ylabel('target policy loc', color='blue')
# plt.xlabel('Itrations')
# plt.title(f"Sigma noise = 0")
# plt.show()

# plt.plot(
#     mysuite.get_histories_fix_params('{main_dir}/sigma_noise0', 0, 'ce_policy_scale00', n_offpolicy_opt=1)[0][0]
# )
# plt.xlabel('Itrations')
# plt.ylabel('ce policy scale')
# plt.title("Sigma noise = 0")
# plt.show()


#%% === LQ 5 ================================================================================
mysuite = MySuite(config='lq_s5.cfg')

stepper      = "Adam1e4" #"ConstantStepper1e4", "Adam1e1", "Adam1e2", "Adam1e4"
horizon      = "10" #10, 50
sigma_noise  = "0" #0, 2
mu_init      = "0" #0, 05, 1
logstd_init  = "10"

exp_on  = f"results_lq_s5/onpolicy"
exp_off  = f"results_lq_s5/offpolicy"

# plt.subplot(2, 1, 1)
plot_ci(mysuite, exp_on, label="onpolicy")
plot_ci(mysuite, exp_off, label="offpolicy")
plt.ylabel('Deterministic Performance')
plt.xlabel('Itrations')
# plt.title(f"stepper={stepper}, horiz  on={horizon}, sigma_noise={sigma_noise}, mu_init={mu_init}")
plt.legend()
plt.show()

# ax = plt.subplot(2, 1, 2)
# plot_ci(mysuite, exp_on, key="VarMean", label="onpolicy 1000")
# plot_ci(mysuite, exp_off, key="VarMean", label="offpolicy 500+500")
# plt.ylabel('Variance of the gradient')
# plt.xlabel('Itrations')
# plt.title(f"stepper={stepper}, horizon={horizon}, sigma_noise={sigma_noise}, mu_init={mu_init}")
# ax.set_yscale('log')
# plt.legend()
# plt.show()


#%% === CARTPOLE ================================================================================
mysuite = MySuite(config='cartpole.cfg')

logstd_init  = "10" #"10", "30"
exp_on   = f"results_cartpole_0818/onpolicy/logstd_init{logstd_init}"
exp_off   = f"results_cartpole_0818/offpolicy/logstd_init{logstd_init}"

# plt.subplot(2, 1, 1)
plot_ci(mysuite, exp_on, label="onpolicy")
plot_ci(mysuite, exp_off, label="offpolicy")
plt.ylabel('Deterministic Performance')
plt.xlabel('Itrations')
plt.legend()
plt.show()

# ax = plt.subplot(2, 1, 2)
# plot_ci(mysuite, exp_on, key="VarMean", label="onpolicy 200")
# plot_ci(mysuite, exp_off, key="VarMean", label="offpolicy 100+100")
# plt.ylabel('Variance of the gradient')
# plt.xlabel('Itrations')
# plt.title(f"stepper={stepper}")
# ax.set_yscale('log')
# plt.legend()
# plt.show()