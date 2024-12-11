# %% 

from expsuite import PyExperimentSuite
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

mysuite = PyExperimentSuite(config='experiments_new_cfull.cfg')


def get_data(suite, experiment_name, xkey, cos_sim=False):
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
            
            grad_cos_sim_off = np.dot(_dict['grad_is'], _dict['grad_mc'])/(np.linalg.norm(_dict['grad_is'])*np.linalg.norm(_dict['grad_mc']))
            _dict['grad_cos_sim_off'] = grad_cos_sim_off

            grad_cos_sim_own = np.dot(_dict['grad_g'], _dict['grad_mc'])/(np.linalg.norm(_dict['grad_g'])*np.linalg.norm(_dict['grad_mc']))
            _dict['grad_cos_sim_own'] = grad_cos_sim_own

            del _dict['grad_is'], _dict['grad_mc'], _dict["grad_g"]

            del _dict["var_grad_is"], _dict["var_grad_mc"], _dict["var_grad_g"]

            _df = pd.DataFrame(_dict, index=[rep])

            _df[xkey] = suite.get_params(exp)[xkey]

            #_df[xkey] = suite.get_params(exp)[xkey]
            df = pd.concat([df, _df],ignore_index = True)
    
    return df


data_20 = get_data(mysuite, "means_20", "defensive_coeff", None)

#print(data.groupby("defensive_coeff").mean())
#print(data.groupby("defensive_coeff").std())
print("----------------")


data_40 = get_data(mysuite, "means_40", "defensive_coeff", None)

#print(data.groupby("defensive_coeff").mean())
#print(data.groupby("defensive_coeff").std())
print("----------------")


data_60 = get_data(mysuite, "means_60", "defensive_coeff", None)

#print(data.groupby("defensive_coeff").mean())
#print(data.groupby("defensive_coeff").std())
#print("----------------")


data_80 = get_data(mysuite, "means_80", "defensive_coeff", None)

#print(data.groupby("defensive_coeff").mean())
#print(data.groupby("defensive_coeff").std())
#print("----------------")


def_0_g = []
def_4_g = []
def_8_g = []


def_0_o = []
def_4_o = []
def_8_o = []


def_0_g_a= data_20["grad_cos_sim_own"][data_20["defensive_coeff"] == 0]
def_0_o_a= data_20["grad_cos_sim_off"][data_20["defensive_coeff"] ==0]


def_4_g_a=data_20["grad_cos_sim_own"][data_20["defensive_coeff"] ==0.4]
def_4_o_a=data_20["grad_cos_sim_off"][data_20["defensive_coeff"] ==0.4]

def_8_g_a=data_20["grad_cos_sim_own"][data_20["defensive_coeff"] ==0.8]
def_8_o_a=data_20["grad_cos_sim_off"][data_20["defensive_coeff"] ==0.8]



def_0_g_b=data_40["grad_cos_sim_own"][data_40["defensive_coeff"] ==0]
def_0_o_b=data_40["grad_cos_sim_off"][data_40["defensive_coeff"] ==0]

def_4_g_b=data_40["grad_cos_sim_own"][data_40["defensive_coeff"] ==0.4]
def_4_o_b=data_40["grad_cos_sim_off"][data_40["defensive_coeff"] ==0.4]

def_8_g_b=data_40["grad_cos_sim_own"][data_40["defensive_coeff"] ==0.8]
def_8_o_b=data_40["grad_cos_sim_off"][data_40["defensive_coeff"] ==0.8]

def_0_g_c=data_60["grad_cos_sim_own"][data_60["defensive_coeff"] ==0]
def_0_o_c=data_60["grad_cos_sim_off"][data_60["defensive_coeff"] ==0]

def_4_g_c=data_60["grad_cos_sim_own"][data_60["defensive_coeff"] ==0.4]
def_4_o_c=data_60["grad_cos_sim_off"][data_60["defensive_coeff"] ==0.4]

def_8_g_c=data_60["grad_cos_sim_own"][data_60["defensive_coeff"] ==0.8]
def_8_o_c=data_60["grad_cos_sim_off"][data_60["defensive_coeff"] ==0.8]

def_0_g_d=data_80["grad_cos_sim_own"][data_80["defensive_coeff"] ==0]
def_0_o_d=data_80["grad_cos_sim_off"][data_80["defensive_coeff"] ==0]

def_4_g_d=data_80["grad_cos_sim_own"][data_80["defensive_coeff"] ==0.4]
def_4_o_d=data_80["grad_cos_sim_off"][data_80["defensive_coeff"] ==0.4]

def_8_g_d=data_80["grad_cos_sim_own"][data_80["defensive_coeff"] ==0.8]
def_8_o_d=data_80["grad_cos_sim_off"][data_80["defensive_coeff"] ==0.8]


def_all_g_a = [def_0_g_a, def_4_g_a, def_8_g_a][np.argmax([def_0_g_a.mean(), def_4_g_a.mean(), def_8_g_a.mean()])]
def_all_g_b = [def_0_g_b, def_4_g_b, def_8_g_b][np.argmax([def_0_g_b.mean(), def_4_g_b.mean(), def_8_g_b.mean()])]
def_all_g_c = [def_0_g_c, def_4_g_c, def_8_g_c][np.argmax([def_0_g_c.mean(), def_4_g_c.mean(), def_8_g_c.mean()])]
def_all_g_d = [def_0_g_d, def_4_g_d, def_8_g_d][np.argmax([def_0_g_d.mean(), def_4_g_d.mean(), def_8_g_d.mean()])]


def_all_o_a = [def_0_o_a, def_4_o_a, def_8_o_a][np.argmax([def_0_o_a.mean(), def_4_o_a.mean(), def_8_o_a.mean()])]
def_all_o_b = [def_0_o_b, def_4_o_b, def_8_o_b][np.argmax([def_0_o_b.mean(), def_4_o_b.mean(), def_8_o_b.mean()])]
def_all_o_c = [def_0_o_c, def_4_o_c, def_8_o_c][np.argmax([def_0_o_c.mean(), def_4_o_c.mean(), def_8_o_c.mean()])]
def_all_o_d = [def_0_o_d, def_4_o_d, def_8_o_d][np.argmax([def_0_o_d.mean(), def_4_o_d.mean(), def_8_o_d.mean()])]




def box_plot(data, edge_color, fill_color, label = ""):
    bp = ax.boxplot(data, patch_artist=True, labels = ["20", "40", "60", "80"] ,showfliers=False, label = label)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color, alpha=0.5)       
        
    return bp

"""
#fig = plt.figure()ű
fig, ax = plt.subplots()


bp2 = box_plot(def_0_o, 'blue', 'cyan', "offpolicy")
bp1 = box_plot(def_0_g, 'red', 'tan', "DAIS-PG")

plt.ylim(0.5,1)

#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['DAIS-PG', 'offpolicy'])

plt.xlabel("batch size")
plt.ylabel("cosine similarity")
plt.legend()

plt.show()

fig, ax = plt.subplots()

bp2 = box_plot(def_0_o, 'blue', 'cyan', "offpolicy")
bp1 = box_plot(def_0_g, 'red', 'tan', "DAIS-PG")
#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['DAIS-PG', 'offpolicy'])
plt.ylim(0.5,1)
plt.xlabel("batch size")
plt.ylabel("cosine similarity")
plt.legend()

plt.show()

fig, ax = plt.subplots()

bp2 = box_plot(def_0_o, 'blue', 'cyan', "offpolicy")
bp1 = box_plot(def_0_g, 'red', 'tan', "DAIS-PG")
#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['DAIS-PG', 'offpolicy'])
plt.ylim(0.5,1)
plt.xlabel("batch size")
plt.ylabel("cosine similarity")
plt.legend()

plt.show()
"""

from pylab import plot, show, savefig, xlim, figure, \
                 ylim, legend, boxplot, setp, axes

# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    #plt.setp(bp['fliers'][0], color='blue')
    #plt.setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    #plt.setp(bp['fliers'][2], color='red')
    #plt.setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')

# Some fake data to plot
#A= [[1, 2, 5,],  [7, 2]]
#B = [[5, 7, 2, 2, 5], [7, 2, 5]]
#C = [[3,2,5,7], [6, 7, 3]]

A = [def_all_g_a, def_all_o_a]
B = [def_all_g_b, def_all_o_b]
C = [def_all_g_c, def_all_o_c]
D = [def_all_g_d, def_all_o_d]
#D = [def_0_g, def_0_o]


fig = figure()
ax = axes()


# first boxplot pair
bp  = boxplot(A, positions = [0, 1], widths = 0.6)
setBoxColors(bp)

# second boxplot pair
bp = boxplot(B, positions = [3, 4], widths = 0.6)
setBoxColors(bp)

# thrid boxplot pair
bp = boxplot(C, positions = [6, 7], widths = 0.6)
setBoxColors(bp)

bp = boxplot(D, positions = [9, 10], widths = 0.6)
setBoxColors(bp)


# set axes limits and labels
#xlim(0,1)
ylim(0,1)

ax.set_xticks([0.5, 3.5, 6.5, 9.5])
ax.set_xticklabels(['20', '40', '60', "80"])

plt.xlabel("Batch size")
plt.ylabel("Cosine similarity")

# draw temporary red and blue lines and use them to create a legend
hB, = plot([1,1],'b-')
hR, = plot([1,1],'r-')
legend((hB, hR),('DAIS-PG', 'offpolicy'))
hB.set_visible(False)
hR.set_visible(False)

plt.title("Cartpole cosine similarity")

savefig('boxcompare.png')
show()