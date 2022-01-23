"""
Gets scores for backward model selection.
@author: SolVind
"""

from spikestats.toolkits import *
from spikestats.family import *
import spikestats.metrics as metrics
from spikestats.old_solver import OldSolver
from spikestats.calc_backwards_scores import *
import scipy.io
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import pickle
import numpy as np

da_folder = 'kavorka_190620_s2_intermediate_dark2'
file_folder = '/.../glm_res/kavorka/%s/' % da_folder

n = 0
final_model_res = {}
for da_file in os.listdir(file_folder):
    if da_file[-4:] == '.mat':
        n += 1
        print(da_file)
        mat = scipy.io.loadmat(file_folder + da_file)
        cell_name = mat['cellnames'][0]

        # file_name_sep = da_file.split('_')
        # file_name = file_name_sep[1:-1]
        # save_name = 'backwards_glmres_' + '_'.join(file_name) + '_full.mat'
        if mat['best-model'] == 'null_model':
            final_model_res[cell_name] = 'null'
        else:
            final_model_res[cell_name] = mat['%s-keys' % mat['best-model'][0]]
scipy.io.savemat('selected_model_%s.mat' % da_folder, final_model_res)

a_file = open(sys.argv[1], "rb")
data = pickle.load(a_file)
a_file.close()

data.keys()

print('start')

cell_index = int(sys.argv[2])

final_model = scipy.io.loadmat(sys.argv[3])

res = scipy.io.loadmat('glmres_roy_270520_s2_distal_weight_0000_imec0_cl0000_ch000_full.mat')
res.keys()
res['m114-scores']


a_file = open('Take 2020-05-27 07.38.10 PM_final.pkl', "rb")
data = pickle.load(a_file)
a_file.close()


a_file = open('1-HPC_GLM/glmdata_roy_270520_s2_distal_weight.pkl', "rb")
data = pickle.load(a_file)
a_file.close()

final_model = scipy.io.loadmat('1-HPC_GLM/selected_model_roy_270520_s2_distal_weight.mat')

family = 'bernoulli'
cell_index = 0

output_dict = calc_backwards_scores('bernoulli', data, cell_index, final_model)

output_dict = scipy.io.loadmat('backwards_glmres_roy_270520_s4_distal_light_0000_imec0_cl0000_ch000.mat')


y = output_dict['data'][0]
yhat = output_dict['full_model_predict']
output_dict['full_model_auc']
output_dict['sub_model_5_auc']
ynull = np.ones(len(y)) * np.mean(y)

nt = 100
tpath = np.linspace(0, 1, nt)
tp = np.zeros(nt)
tn = np.zeros(nt)
fp = np.zeros(nt)
fn = np.zeros(nt)
for i in range(nt):
    yt = (yhat >= tpath[i])*1
    # pval = np.where(yt == 1)[0]
    # nval = np.where(yt == 0)[0]
    tp[i] = np.sum(y[yt == 1])
    fn[i] = np.sum(y[yt == 0])
    fp[i] = len(np.where(y[yt == 1] == 0)[0])
    tn[i] = len(np.where(y[yt == 0] == 0)[0])

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

plt.plot(fpr, tpr)
plt.show()

metrics.auc(fpr, tpr)

output_dict.keys()
output_dict['full_model_keys']
output_dict['full_model_scores']
output_dict['full_model_new_scores']

output_dict['sub_model_0_scores']
output_dict['sub_model_1_scores']
output_dict['sub_model_2_scores']
output_dict['sub_model_3_scores']
output_dict['sub_model_4_scores']

# n_cell = len(data['cell_names'])
#
#
# for i in range(n_cell):
#     print((i, data['cell_names'][i])
#     output_dict = calc_backwards_scores('bernoulli', data, i, final_model)
