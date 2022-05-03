from suite import MySuite

import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt

mysuite = MySuite(config='lq.cfg')

# sigma 0
exp_off = mysuite.get_exps(path='results_lq/offpolicy/sigma_noise0.0')[0]
params = mysuite.get_params(exp_off)
n = params['repetitions']
m = mysuite.get_histories_over_repetitions(exp_off, 'UTestPerf', np.mean)
s = mysuite.get_histories_over_repetitions(exp_off, 'UTestPerf', np.std)
ci_lb, ci_ub = st.norm.interval(0.68, loc=m, scale=s/np.sqrt(n))
plt.plot(np.arange(params['iterations']), m, label='off policy')
plt.fill_between(np.arange(params['iterations']), ci_lb, ci_ub, alpha=.1)

exp_on = mysuite.get_exps(path='results_lq/onpolicy/sigma_noise0.0')[0]
params = mysuite.get_params(exp_on)
n = params['repetitions']
m = mysuite.get_histories_over_repetitions(exp_on, 'UTestPerf', np.mean)
s = mysuite.get_histories_over_repetitions(exp_on, 'UTestPerf', np.std)
ci_lb, ci_ub = st.norm.interval(0.68, loc=m, scale=s/np.sqrt(n))
plt.plot(np.arange(params['iterations']), m, label='on policy')
plt.fill_between(np.arange(params['iterations']), ci_lb, ci_ub, alpha=.1)

plt.xlabel('Itrations')
plt.ylabel('Return')
plt.legend()
plt.title("Sigma noise = 0")
plt.show()


# sigma 0.5
exp_off = mysuite.get_exps(path='results_lq/offpolicy/sigma_noise0.50')[0]
params = mysuite.get_params(exp_off)
n = params['repetitions']
m = mysuite.get_histories_over_repetitions(exp_off, 'UTestPerf', np.mean)
s = mysuite.get_histories_over_repetitions(exp_off, 'UTestPerf', np.std)
ci_lb, ci_ub = st.norm.interval(0.68, loc=m, scale=s/np.sqrt(n))
plt.plot(np.arange(params['iterations']), m, label='off policy')
plt.fill_between(np.arange(params['iterations']), ci_lb, ci_ub, alpha=.1)

exp_on = mysuite.get_exps(path='results_lq/onpolicy/sigma_noise0.50')[0]
params = mysuite.get_params(exp_on)
n = params['repetitions']
m = mysuite.get_histories_over_repetitions(exp_on, 'UTestPerf', np.mean)
s = mysuite.get_histories_over_repetitions(exp_on, 'UTestPerf', np.std)
ci_lb, ci_ub = st.norm.interval(0.68, loc=m, scale=s/np.sqrt(n))
plt.plot(np.arange(params['iterations']), m, label='on policy')
plt.fill_between(np.arange(params['iterations']), ci_lb, ci_ub, alpha=.1)

plt.xlabel('Itrations')
plt.ylabel('Return')
plt.legend()
plt.title("Sigma noise = 0.5")
plt.show()

# sigma 1
exp_off = mysuite.get_exps(path='results_lq/offpolicy/sigma_noise1.0')[0]
params = mysuite.get_params(exp_off)
n = params['repetitions']
m = mysuite.get_histories_over_repetitions(exp_off, 'UTestPerf', np.mean)
s = mysuite.get_histories_over_repetitions(exp_off, 'UTestPerf', np.std)
ci_lb, ci_ub = st.norm.interval(0.68, loc=m, scale=s/np.sqrt(n))
plt.plot(np.arange(params['iterations']), m, label='off policy')
plt.fill_between(np.arange(params['iterations']), ci_lb, ci_ub, alpha=.1)

exp_on = mysuite.get_exps(path='results_lq/onpolicy/sigma_noise1.0')[0]
params = mysuite.get_params(exp_on)
n = params['repetitions']
m = mysuite.get_histories_over_repetitions(exp_on, 'UTestPerf', np.mean)
s = mysuite.get_histories_over_repetitions(exp_on, 'UTestPerf', np.std)
ci_lb, ci_ub = st.norm.interval(0.68, loc=m, scale=s/np.sqrt(n))
plt.plot(np.arange(params['iterations']), m, label='on policy')
plt.fill_between(np.arange(params['iterations']), ci_lb, ci_ub, alpha=.1)

plt.xlabel('Itrations')
plt.ylabel('Return')
plt.legend()
plt.title("Sigma noise = 1")
plt.show()