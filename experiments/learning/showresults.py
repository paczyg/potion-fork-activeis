from suite import MySuite
from matplotlib import pyplot as plt

mysuite = MySuite(config='lq.cfg')

exps = mysuite.get_exps(path='results_lq')
for exp in exps:
    params = mysuite.get_params(exp)
    ys = mysuite.get_history(exp, 0, 'TestPerf')

    plt.plot(ys, label=str(params['n_offpolicy_opt']))
    plt.xlabel('Learning Epochs')
    plt.ylabel('Test Performance')
plt.legend()
plt.show()
