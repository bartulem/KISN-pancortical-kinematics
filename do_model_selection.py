from spikestats.toolkits import *
from spikestats.family import *
from spikestats.cross_validation import *
from spikestats.model_selection import *
from spikestats.process_data import *
import pickle
import matplotlib.pyplot as plt

# from spikestats.logit_1d import process_group_info
# data_raw = np.nan
a_file = open('/Users/jingyig/Work/Kavli/Data/data4ratemaps_v2_naned_XYZ/ok4rms_kavorka_190620_s2_intermediate_dark_reheaded_XYZeuler_notricks_1D2D_naned_evenodd_rotatedback.pkl', "rb")
data_raw = pickle.load(a_file)
a_file.close()

#
# # data_raw.keys()
# # data_raw['possiblecovariates'].keys()
# #
# # plt.scatter(data_raw['possiblecovariates']['L Ego3_Head_pitch'], data_raw['possiblecovariates']['M Ego3_Head_azimuth'])
# # plt.show()
#
#
#
data = prepare_data4glms(data_raw, use_imu=True)
data['file_name'] = 'kavorka_190620_s2_intermediate_dark_imu'

a_file = open('glmdata_kavorka_190620_s2_intermediate_dark_imu.pkl', 'wb')
pickle.dump(data, a_file)
a_file.close()


# a_file = open('1-HPC_GLM/glmdata_johnjohn_230520_s3_intermediate_dark.pkl', "rb")
# data = pickle.load(a_file)
# a_file.close()
#
#
# avoid_feature = ['R Ego2_head_azimuth', 'R Ego2_head_azimuth_1st_der','R Ego2_head_azimuth_2nd_der',
#                  'position_x', 'position_y', 'selfmotion_x', 'selfmotion_y',
#                  'C Body_direction_2nd_der', 'B Speeds_1st_der', 'D Allo_head_direction_2nd_der',
#                  'G Neck_elevation_2nd_der', 'K Ego3_Head_roll_2nd_der', 'L Ego3_Head_pitch_2nd_der',
#                  'M Ego3_Head_azimuth_2nd_der', 'N Back_pitch_2nd_der', 'O Back_azimuth_2nd_der',
#                  'P Ego2_head_roll_2nd_der', 'Q Ego2_head_pitch_2nd_der']
#
#
#
# ms = ForwardSelection(family='bernoulli', models=None, kappa=2., theta=1., tik_tau=None, group=None,
#                  nfold=10, fold_method='sublock', n_repeat=None, alpha=1, reg_lambda=0.0001,
#                  solver='L-BFGS', learning_rate=2e-1, max_iter=1000, xtol=1e-6,
#                  significance=0.01, fit_intercept=True, score_metric='llr',
#                  seed=142, verbose=False, save_file=True)
#
# ms.fit(data, cell_index=179, avoid_feature=avoid_feature)

#
#
# data.keys()
# data['spk_mat'].shape
#
# y = data['spk_mat'][0,:]
# data['features_mat'].keys()
#
# x1 = data['features_mat']['K Ego3_Head_roll']
# x2 = data['features_mat']['L Ego3_Head_pitch']
#
# x1[1] * x2[1]
#
# x1[0].shape
# x2[0].shape
#
# x_mat = np.column_stack([x1[0], x2[0]])
# np.shape(x_mat)
# group_index = np.concatenate([np.ones(15),2*np.ones(15)]).astype(int)
#
#
# group_dict = process_group_info(group_index, x_mat)
#



# ms = ForwardSelection()
# ms.fit(data, cell_index=0)
#
#
#
import scipy.io
mat = scipy.io.loadmat('glmres_frank_010620_s4_intermediate_light_0117_imec0_cl0177_ch135_full.mat')
mat.keys()
mat['best-model']
# ikeys = mat['full_model_keys']
# l0 = np.ravel(mat['null_loglik'])
# l1 = np.ravel(mat['full_model_loglik'])
# 
# pred = mat[]
# 
# McFadden = 1 - l1/l0



