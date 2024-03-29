"""
Estimate synaptic connectivity through spike-spike cross-correlations.
@author: bartulem
"""

import io
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse
from numba import njit
from tqdm import tqdm
from itertools import combinations
from random import shuffle
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson
import sessions2load
import select_clusters
import neural_activity
import quantify_ratemaps


def bin_spiking(cl_activity, bin_size, tracking_start, tracking_stop,
                to_jitter=True, jitter_size=.005, jitter_n=1000):
    """
    Description
    ----------
    This function takes a spike train (spike times in seconds), and bins it in a binary
    array (0/1) based on whether the spike occurred in any interval between tracking
    start and end with a bin size (e.g. 0.4ms) step. It can also do the same for jittered
    spikes.
    ----------

    Parameters
    ----------
    cl_activity : np.ndarray
        Spike times relative to tracking start.
    bin_size : float
        The size of bins for binning spikes.
    tracking_start : int/float
        Tracking start relative to tracking start (should be 0).
    tracking_stop : int/float
        Tracking stop relative to tracking start.
    to_jitter : bool
        Perform jittering; defaults to True.
    jitter_size : float
        The maximum a spike can jitter; defaults to 0.005 (seconds).
    jitter_n : int
        Number of times to jitter; defaults to 1000.
    ----------

    Returns
    ----------
    fr_arr : np.ndarray
        The binned spiking activity.
    jitter_arr : np.ndarray
        The binned jittered spiking activity.
    ----------
    """

    fr_arr = np.zeros(int(np.ceil((tracking_stop - tracking_start) / bin_size))).astype(np.float32)
    fr_arr[np.floor(cl_activity / bin_size).astype(np.int32)] = 1
    jitter_arr = sparse.csr_matrix((jitter_n, fr_arr.shape[0]), dtype=np.float32)
    if to_jitter:
        for sh in range(jitter_n):
            jitter_spikes = cl_activity.copy() + ((2 * jitter_size * np.random.random(cl_activity.shape[0])) - jitter_size)
            jitter_spikes = jitter_spikes[jitter_spikes <= tracking_stop]
            jitter_arr[sh, np.floor(jitter_spikes / bin_size).astype(np.int32)] = 1
    return fr_arr, jitter_arr


@njit(parallel=False)
def cross_correlate(big_x, big_x_mean, small_y, small_y_mean):
    """
    Parameters
    ----------
    big_x : np.ndarray
        The time-lagged spike trains.
    big_x_mean : np.ndarray
        Means of the time-lagged spike trains.
    small_y : np.ndarray
        The reference spike train.
    small_y_mean : np.float
        The mean of the reference spike train.
    ----------

    Returns
    ----------
    r : np.ndarray
        Pearson cross-correlations for every time shift.
    ----------
    """

    r_num = np.sum((big_x - big_x_mean) * (small_y - small_y_mean), axis=1)
    r_den = np.sqrt(np.sum((big_x - big_x_mean) ** 2, axis=1) * np.sum((small_y - small_y_mean) ** 2))
    r = r_num / r_den
    return r


@njit(parallel=False)
def dot_product(big_x, small_y):
    """
    Parameters
    ----------
    big_x : np.ndarray
        The time-lagged spike trains.
    small_y : np.ndarray
        The reference spike train.
    ----------

    Returns
    ----------
    r : np.ndarray
        Dot-product cross-correlations for every time shift.
    ----------
    """
    r = np.zeros(big_x.shape[0])
    for row in range(big_x.shape[0]):
        r[row] = np.dot(big_x[row, :], small_y)
    return r


@njit(parallel=False)
def hollowed_gaussian_kernel(cch, sigma=1, fraction_hollowed=.6):
    """
    Description
    ----------
    This function takes a cross-correlation histogram and convolves it with a large window
    "partially-hollowed" Gaussian.

    Detailed: To generate the low frequency baseline CCH, the observed CCG was convolved
    with a “partially hollow” Gaussian kernel (Stark and Abeles, JoNM, 2009), with a standard
    deviation of 10 ms, with a hollow fraction of 60% (i.e. 60% off the center bin).
    ----------

    Parameters
    ----------
    cch : np.ndarray
        The CCH array that should be smoothed.
    sigma : int
        The sigma for smoothing (in bins); defaults to 1.
    fraction_hollowed : float
        Proportion-wise, the amount of window hollowed; defaults to .6.
    ----------

    Returns
    ----------
    smoothed_cch : np.ndarray
        The hollow-Gaussian convolved CCH.
    ----------
    """

    smoothed_cch = np.zeros(cch.shape[0] * 3)
    input_array_reflected = np.concatenate((cch[::-1], cch, cch[::-1]))
    x_v = np.arange(smoothed_cch.shape[0])
    for idx in x_v:
        kernel_idx = np.exp(-(x_v - idx) ** 2 / (2 * sigma ** 2))
        kernel_idx[int(np.floor(kernel_idx.shape[0] / 2))] = kernel_idx[int(np.floor(kernel_idx.shape[0] / 2))] * (1 - fraction_hollowed)
        kernel_idx = kernel_idx / kernel_idx.sum()
        smoothed_cch[idx] = np.dot(kernel_idx, input_array_reflected)
    return smoothed_cch[cch.shape[0]:cch.shape[0] * 2]


def calculate_p_values(cch, slow_baseline):
    """
    Description
    ----------
    This function calculates the probabilities of obtaining an observed (or higher) synchrony count
    in the time lags of the observed CCG (in the offered range), given the expected, low frequency
    baseline rate in the same bins, estimated using the Poisson distribution (Fine-English et al.,
    Neuron, 2017) with a continuity correction.

    Extra: The absolute function at the end is there because when the factorials get very large,
    the calculation yields negative values.
    ----------

    Parameters
    ----------
    cch : np.ndarray
        The CCH array.
    slow_baseline : np.ndarray
        The convolved CCH array.
    ----------

    Returns
    ----------
    p_values : np.ndarray
        The probability of obtaining an observed (or higher) synchrony count in the mth time lag of
        the observed CCH, given the expected, low frequency baseline rate in the same bin.
    ----------
    """
    return np.abs(1 - poisson.cdf(k=cch - 1, mu=slow_baseline) - poisson.pmf(k=cch, mu=slow_baseline) * 0.5)


class FunctionalConnectivity:
    animal_ids = {'frank': '26473', 'johnjohn': '26471', 'kavorka': '26525',
                  'roy': '26472', 'bruno': '26148', 'jacopo': '26504', 'crazyjoe': '26507'}

    tuning_categories = {0: ['Unclassified'],
                         1: ['Z Position', 'C Body direction'],
                         2: ['Z Self_motion', 'B Speeds', 'C Body_direction_1st_der'],
                         3: ['K Ego3_Head_roll', 'M Ego3_Head_azimuth', 'L Ego3_Head_pitch'],
                         4: ['K Ego3_Head_roll_1st_der', 'M Ego3_Head_azimuth_1st_der', 'L Ego3_Head_pitch_1st_der'],
                         5: ['P Ego2_head_roll', 'D Allo_head_direction', 'Q Ego2_head_pitch'],
                         6: ['P Ego2_head_roll_1st_der', 'D Allo_head_direction_1st_der', 'Q Ego2_head_pitch_1st_der'],
                         7: ['O Back_azimuth', 'N Back_pitch'],
                         8: ['O Back_azimuth_1st_der', 'N Back_pitch_1st_der'],
                         9: ['G Neck_elevation'],
                         10: ['G Neck_elevation_1st_der']}

    def __init__(self, pkl_sessions_dir='', cluster_groups_dir='',
                 sp_profiles_csv='', pkl_file='', pkl_files=[], save_dir='',
                 mat_file_dirs=[], cch_data_dir=''):
        self.pkl_sessions_dir = pkl_sessions_dir
        self.cluster_groups_dir = cluster_groups_dir
        self.sp_profiles_csv = sp_profiles_csv
        self.pkl_files = pkl_files
        self.pkl_file = pkl_file
        self.save_dir = save_dir
        self.mat_file_dirs = mat_file_dirs
        self.cch_data_dir = cch_data_dir

    def noise_corr(self, **kwargs):
        """
        Description
        ----------
        This method calculates noise correlations for all selected clusters in a given session.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        bin_size (float)
            The size of bins for binning spikes; defaults to 0.0004 (seconds).
        bin_num (float)
            The one-sided number of bins for the CCG; defaults to 50.
        smooth_fr (bool)
            To smooth the firing rates; defaults to False.
        std_smooth (float)
            The std. deviation of the gaussian smoothing kernel; defaults to bin_size (seconds).
        to_jitter (bool)
            To jitter or not to jitter spikes; defaults to False.
        num_jitters (int)
            The number of times to jitter data; defaults to 1000.
        jitter_size (int)
            The one-sided bound of the spike jitter; defaults to 0.005 (ms).
        area_filter (list / bool)
            Areas to be included, you can pick specific areas or
            general (A - auditory, M - motor, P - parietal, S - somatosensory, V - visual); defaults to True.
        cluster_type_filter (str / bool)
            Cluster type to be included: 'good' or 'mua'; defaults to True.
        profile_filter (str / bool)
            Profile to be included: 'RS' or 'FS'; defaults to True.
        sort_ch_num (bool)
            If True, sorts clusters by channel number; defaults to False.
        combo_range (int)
            Range of combination of choice; defaults to [0, 1000].
        cross_corr (bool)
            Pearson correlate spike trains; defaults to False.
        convolve_sigma (int)
            Sigma for convolving the CCH; defaults to 0.01 (seconds).
        ----------

        Returns
        ----------
        noise_corrs (.mat file)
            .mat files containing cross-correlations of data/jitters.
        ----------
        """

        bin_size = kwargs['bin_size'] if 'bin_size' in kwargs.keys() and type(kwargs['bin_size']) == float else .0004
        bin_num = kwargs['bin_num'] if 'bin_num' in kwargs.keys() and type(kwargs['bin_num']) == int else 50
        smooth_fr = kwargs['smooth_fr'] if 'smooth_fr' in kwargs.keys() and type(kwargs['smooth_fr']) == bool else False
        std_smooth = kwargs['std_smooth'] if 'std_smooth' in kwargs.keys() and type(kwargs['std_smooth']) == float else bin_size
        to_jitter = kwargs['to_jitter'] if 'to_jitter' in kwargs.keys() and type(kwargs['to_jitter']) == bool else False
        num_jitters = kwargs['num_jitters'] if 'num_jitters' in kwargs.keys() and type(kwargs['num_jitters']) == int else 1000
        jitter_size = kwargs['jitter_size'] if 'jitter_size' in kwargs.keys() and type(kwargs['jitter_size']) == float else .005
        area_filter = kwargs['area_filter'] if 'area_filter' in kwargs.keys() and type(kwargs['area_filter']) == list else True
        cluster_type_filter = kwargs['cluster_type_filter'] if 'cluster_type_filter' in kwargs.keys() and type(kwargs['cluster_type_filter']) == str else True
        profile_filter = kwargs['profile_filter'] if 'profile_filter' in kwargs.keys() and type(kwargs['profile_filter']) == str else True
        sort_ch_num = kwargs['sort_ch_num'] if 'sort_ch_num' in kwargs.keys() and type(kwargs['sort_ch_num']) == bool else True
        combo_range = kwargs['combo_range'] if 'combo_range' in kwargs.keys() and type(kwargs['combo_range']) == list else True
        cross_corr = kwargs['cross_corr'] if 'cross_corr' in kwargs.keys() and type(kwargs['cross_corr']) == bool else False
        convolve_sigma = kwargs['convolve_sigma'] if 'convolve_sigma' in kwargs.keys() and type(kwargs['convolve_sigma']) == float else 0.01

        cluster_list = select_clusters.ClusterFinder(session=f'{self.pkl_sessions_dir}{os.sep}{self.pkl_file}',
                                                     cluster_groups_dir=self.cluster_groups_dir,
                                                     sp_profiles_csv=self.sp_profiles_csv).get_desired_clusters(filter_by_area=area_filter,
                                                                                                                filter_by_cluster_type=cluster_type_filter,
                                                                                                                filter_by_spiking_profile=profile_filter,
                                                                                                                sort_ch_num=sort_ch_num)

        # get spike data in seconds and tracking start and end time
        file_id, cluster_data = sessions2load.Session(session=f'{self.pkl_sessions_dir}{os.sep}{self.pkl_file}').data_loader(extract_clusters=cluster_list,
                                                                                                                             extract_variables=['tracking_ts'])

        # get all combinations of clusters
        cl_combinations = list(combinations(cluster_data['cluster_spikes'].keys(), 2))

        if combo_range is True:
            combo_start = 0
            combo_end = len(cl_combinations)
        else:
            combo_start = combo_range[0]
            combo_end = combo_range[1]

        output_dictionary = {}
        for combo_num in tqdm(range(combo_start, combo_end), position=0, leave=True):

            # pick a combination
            combo_name = f'{cl_combinations[combo_num][0]}-{cl_combinations[combo_num][1]}'

            # get activity of each cluster
            act1 = cluster_data['cluster_spikes'][cl_combinations[combo_num][0]]
            act2 = cluster_data['cluster_spikes'][cl_combinations[combo_num][1]]

            # eliminate spikes that happen prior to and post tracking
            act1 = neural_activity.purge_spikes_beyond_tracking(spike_train=act1, tracking_ts=cluster_data['tracking_ts'])
            act2 = neural_activity.purge_spikes_beyond_tracking(spike_train=act2, tracking_ts=cluster_data['tracking_ts'])

            # get firing rates
            if to_jitter:
                fr1, sh1 = bin_spiking(cl_activity=act1, bin_size=bin_size,
                                       tracking_start=cluster_data['tracking_ts'][0] - cluster_data['tracking_ts'][0],
                                       tracking_stop=cluster_data['tracking_ts'][1] - cluster_data['tracking_ts'][0],
                                       jitter_size=jitter_size, jitter_n=num_jitters)

                fr2, sh2 = bin_spiking(cl_activity=act2, bin_size=bin_size,
                                       tracking_start=cluster_data['tracking_ts'][0] - cluster_data['tracking_ts'][0],
                                       tracking_stop=cluster_data['tracking_ts'][1] - cluster_data['tracking_ts'][0],
                                       jitter_size=jitter_size, jitter_n=num_jitters)

                if smooth_fr:
                    fr1 = gaussian_filter1d(input=fr1, sigma=int(round(std_smooth / bin_size)))
                    sh1 = gaussian_filter1d(input=sh1.todense(), sigma=int(round(std_smooth / bin_size)), axis=1)
                    fr2 = gaussian_filter1d(input=fr2, sigma=int(round(std_smooth / bin_size)))
                    sh2 = gaussian_filter1d(input=sh2.todense(), sigma=int(round(std_smooth / bin_size)), axis=1)
            else:
                fr1, sh1 = bin_spiking(cl_activity=act1, bin_size=bin_size,
                                       tracking_start=cluster_data['tracking_ts'][0] - cluster_data['tracking_ts'][0],
                                       tracking_stop=cluster_data['tracking_ts'][1] - cluster_data['tracking_ts'][0],
                                       to_jitter=False, jitter_n=1)

                fr2, sh2 = bin_spiking(cl_activity=act2, bin_size=bin_size,
                                       tracking_start=cluster_data['tracking_ts'][0] - cluster_data['tracking_ts'][0],
                                       tracking_stop=cluster_data['tracking_ts'][1] - cluster_data['tracking_ts'][0],
                                       to_jitter=False, jitter_n=1)

                if smooth_fr:
                    fr1 = gaussian_filter1d(input=fr1, sigma=int(round(std_smooth / bin_size)))
                    fr2 = gaussian_filter1d(input=fr2, sigma=int(round(std_smooth / bin_size)))

            # cross-correlate
            all_bins = np.arange(-bin_num, bin_num + 1, 1)
            fr1_shape = fr1.shape[0]
            y_start = int(round(bin_num))
            y_end = int(round(fr1_shape - bin_num))

            x_bool = np.zeros((all_bins.shape[0], fr1_shape), dtype=bool)
            for bin_idx, one_bin in enumerate(all_bins):
                x_bool[bin_idx, int(round(bin_num + one_bin)):int(round(fr1_shape - bin_num + one_bin))] = True

            big_x = np.tile(A=fr1, reps=(all_bins.shape[0], 1))
            big_x = big_x[x_bool].reshape(all_bins.shape[0], y_end - y_start)
            big_x_mean = np.reshape(big_x.mean(axis=1), (big_x.shape[0], 1))
            y = fr2[y_start:y_end]
            y_mean = y.mean()

            if cross_corr:
                data = cross_correlate(big_x=big_x,
                                       big_x_mean=big_x_mean,
                                       small_y=y,
                                       small_y_mean=y_mean)
                del big_x, big_x_mean
                del y, y_mean

                if to_jitter:
                    sh_data = np.zeros((num_jitters, all_bins.shape[0]))
                    for sh in range(num_jitters):
                        big_x_sh = np.tile(A=sh1[sh, :], reps=(all_bins.shape[0], 1))
                        big_x_sh = big_x_sh[x_bool].reshape(all_bins.shape[0], y_end - y_start)
                        big_x_sh_mean = np.reshape(big_x_sh.mean(axis=1), (big_x_sh.shape[0], 1))
                        y_sh = sh2[sh, y_start:y_end]
                        y_sh_mean = y_sh.mean()
                        sh_data[sh, :] = cross_correlate(big_x=big_x_sh,
                                                         big_x_mean=big_x_sh_mean,
                                                         small_y=y_sh,
                                                         small_y_mean=y_sh_mean)
                        del big_x_sh, big_x_sh_mean
                        del y_sh, y_sh_mean

                if to_jitter:
                    output_dictionary['cch'] = data
                    output_dictionary['jitter_cch'] = sh_data
                else:
                    output_dictionary['cch'] = data
            else:
                output_array = np.zeros((all_bins.shape[0], 5))
                cch = dot_product(big_x=big_x,
                                  small_y=y)
                output_array[:, 0] = cch

                cch_convolved = hollowed_gaussian_kernel(cch=cch, sigma=int(round(convolve_sigma / bin_size)))
                output_array[:, 1] = cch_convolved

                cch_probabilities = calculate_p_values(cch=cch, slow_baseline=cch_convolved)
                output_array[:, 2] = cch_probabilities

                output_array[0, 3] = act1.shape[0]
                output_array[1, 3] = cluster_data['tracking_ts'][1] - cluster_data['tracking_ts'][0]
                output_array[0, 4] = act2.shape[0]
                output_array[1, 4] = cluster_data['tracking_ts'][1] - cluster_data['tracking_ts'][0]

                output_dictionary[combo_name] = output_array

        # save to file
        sio.savemat(f'{self.save_dir}{os.sep}cl{combo_start}-{combo_end}.mat', output_dictionary)

    def analyze_corr(self, **kwargs):
        """
        Description
        ----------
        This method analyzes noise correlation results. Specifically, it (1) ...
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        cch_time (float)
            The one-sided time of the CCG; defaults to 20 (ms).
        bin_num (float)
            The one-sided number of bins for the CCG; defaults to 50.
        relevant_cch_bounds (list)
            The CCH boundaries to check for significance; defaults to [0.8, 2.8] (ms).
        filter_by_area (bool / list)
            Areas of choice for the cluster pair; defaults to True (checks all pairs).
        filter_by_cluster_type (list)
            Cluster type to be included: 'good' or 'mua'; defaults to [True, True].
        filter_by_spiking_profile (list)
            Profile to be included: 'RS' or 'FS'; defaults to [True, True].
        filter_by_smi (list)
            Select clusters that have a significant SMI; defaults to [True, True].
        filter_by_lmi (list)
            Select clusters that have a significant LMI; defaults to [True, True].
        p_alpha (float)
            p-value boundary for accepting significance; defaults to .0001.
        json_file_names (list)
            The names of the json file containing putative synaptic connections/common input connections.
        ----------

        Returns
        ----------
        synaptic (.json file)
            .json file containing all putative synaptic connections.
        ----------
        """

        cch_time = kwargs['cch_time'] if 'cch_time' in kwargs.keys() and type(kwargs['cch_time']) == int else 20
        bin_num = kwargs['bin_num'] if 'bin_num' in kwargs.keys() and type(kwargs['bin_num']) == int else 50
        relevant_cch_bounds = kwargs['relevant_cch_bounds'] if 'relevant_cch_bounds' in kwargs.keys() and type(kwargs['relevant_cch_bounds']) == list else [1.6, 4.0]
        filter_by_area = kwargs['filter_by_area'] if 'filter_by_area' in kwargs.keys() and type(kwargs['filter_by_area']) == list else True
        filter_by_cluster_type = kwargs['filter_by_cluster_type'] if 'filter_by_cluster_type' in kwargs.keys() and type(kwargs['filter_by_cluster_type']) == list else [True, True]
        filter_by_spiking_profile = kwargs['filter_by_spiking_profile'] if 'filter_by_spiking_profile' in kwargs.keys() and type(kwargs['filter_by_spiking_profile']) == list else [True, True]
        filter_by_smi = kwargs['filter_by_smi'] if 'filter_by_smi' in kwargs.keys() and type(kwargs['filter_by_smi']) == list else [True, True]
        filter_by_lmi = kwargs['filter_by_lmi'] if 'filter_by_lmi' in kwargs.keys() and type(kwargs['filter_by_lmi']) == list else [True, True]
        p_alpha = kwargs['p_alpha'] if 'p_alpha' in kwargs.keys() and type(kwargs['p_alpha']) == float else .0001
        json_file_names = kwargs['json_file_names'] if 'json_file_names' in kwargs.keys() and type(kwargs['json_file_names']) == list else ['synaptic', 'common_input']

        # get relevant boundaries for calculating statistics
        cch_total_range = np.around(np.linspace(-cch_time, cch_time, (bin_num * 2) + 1), decimals=1)
        if relevant_cch_bounds[0] in cch_total_range \
                and relevant_cch_bounds[1] in cch_total_range:
            rel_cch_bounds_1 = relevant_cch_bounds[0]
            rel_cch_bounds_2 = relevant_cch_bounds[1]
        else:
            rel_cch_bounds_1 = cch_total_range[cch_total_range < relevant_cch_bounds[0]][-1]
            rel_cch_bounds_2 = cch_total_range[cch_total_range < relevant_cch_bounds[1]][-1]
        all_cch_values = np.around(np.concatenate((np.arange(-rel_cch_bounds_2, -rel_cch_bounds_1 + 0.4, 0.4),
                                                   np.arange(rel_cch_bounds_1, rel_cch_bounds_2 + 0.4, 0.4))), decimals=1)
        idx_array = np.where((np.isin(cch_total_range, all_cch_values)))[0]
        negative_end_bound_idx = np.where(cch_total_range == round(-(rel_cch_bounds_1 - .4), 1))[0][0]
        positive_start_bound_idx = np.where(cch_total_range == round(rel_cch_bounds_1 - .4, 1))[0][0]

        # load profile data with GLM categories
        if not os.path.exists(self.sp_profiles_csv):
            print(f"Invalid location for file {self.sp_profiles_csv}. Please try again.")
            sys.exit()
        else:
            profile_data = pd.read_csv(self.sp_profiles_csv)

        # find all clusters for brain areas of interest
        output_dict = {'synaptic': {}, 'common_input': {}}
        for pkl_idx, pkl_file in enumerate(tqdm(self.pkl_files)):
            if type(filter_by_area) == list:
                cl_group_1 = select_clusters.ClusterFinder(session=f'{self.pkl_sessions_dir}{os.sep}{pkl_file}',
                                                           cluster_groups_dir=self.cluster_groups_dir,
                                                           sp_profiles_csv=self.sp_profiles_csv).get_desired_clusters(filter_by_area=[filter_by_area[0]],
                                                                                                                      filter_by_cluster_type=filter_by_cluster_type[0],
                                                                                                                      filter_by_spiking_profile=filter_by_spiking_profile[0],
                                                                                                                      filter_by_lmi=filter_by_smi[0],
                                                                                                                      filter_by_smi=filter_by_lmi[0])
                cl_group_2 = select_clusters.ClusterFinder(session=f'{self.pkl_sessions_dir}{os.sep}{pkl_file}',
                                                           cluster_groups_dir=self.cluster_groups_dir,
                                                           sp_profiles_csv=self.sp_profiles_csv).get_desired_clusters(filter_by_area=[filter_by_area[1]],
                                                                                                                      filter_by_cluster_type=filter_by_cluster_type[1],
                                                                                                                      filter_by_spiking_profile=filter_by_spiking_profile[1],
                                                                                                                      filter_by_lmi=filter_by_smi[1],
                                                                                                                      filter_by_smi=filter_by_lmi[1])

            # get session id
            total_id_name = self.mat_file_dirs[pkl_idx].split('/')[-1]
            sp_session_id = total_id_name.split('_')[0] + '_' + total_id_name.split('_')[1] + '_' + total_id_name.split('_')[3]
            output_dict['synaptic'][sp_session_id] = {}
            output_dict['common_input'][sp_session_id] = {}

            # go through files and check whether there are any interesting CCH
            for mat_file in os.listdir(self.mat_file_dirs[pkl_idx]):
                data = sio.loadmat(f'{self.mat_file_dirs[pkl_idx]}{os.sep}{mat_file}')

                for cl_pair in data.keys():
                    pair_split = cl_pair.split('-')
                    if filter_by_area is True \
                            or (pair_split[0] in cl_group_1 and pair_split[1] in cl_group_2) \
                            or (pair_split[0] in cl_group_2 and pair_split[1] in cl_group_1):
                        baseline_subtracted_counts = data[cl_pair][:, 0] - data[cl_pair][:, 1]
                        most_aberrant_value_idx = np.argmax(np.abs(baseline_subtracted_counts))

                        common_input_peak = negative_end_bound_idx <= most_aberrant_value_idx < positive_start_bound_idx
                        synaptic_peak = most_aberrant_value_idx in idx_array
                        either_peak = synaptic_peak or common_input_peak
                        inhibition_p = False
                        excitation_p = False
                        connection_type = 'null'

                        if either_peak:
                            if synaptic_peak:
                                if data[cl_pair][most_aberrant_value_idx, 2] < p_alpha and \
                                        (data[cl_pair][most_aberrant_value_idx - 1, 2] < p_alpha or data[cl_pair][most_aberrant_value_idx + 1, 2] < p_alpha):
                                    if (data[cl_pair][negative_end_bound_idx + 1:positive_start_bound_idx, 2] > p_alpha).all():
                                        excitation_p = True
                                        connection_type = 'synaptic'
                                    else:
                                        excitation_p = True
                                        connection_type = 'common_input'
                                elif (1 - data[cl_pair][most_aberrant_value_idx, 2]) < p_alpha and \
                                        ((1 - data[cl_pair][most_aberrant_value_idx - 1, 2]) < p_alpha or (1 - data[cl_pair][most_aberrant_value_idx + 1, 2]) < p_alpha):
                                    if (1 - data[cl_pair][negative_end_bound_idx + 1:positive_start_bound_idx, 2] > p_alpha).all():
                                        inhibition_p = True
                                        connection_type = 'synaptic'
                                    else:
                                        inhibition_p = True
                                        connection_type = 'common_input'
                            else:
                                if data[cl_pair][most_aberrant_value_idx, 2] < p_alpha and \
                                        (data[cl_pair][most_aberrant_value_idx - 1, 2] < p_alpha or data[cl_pair][most_aberrant_value_idx + 1, 2] < p_alpha):
                                    excitation_p = True
                                    connection_type = 'common_input'
                                elif (1 - data[cl_pair][most_aberrant_value_idx, 2]) < p_alpha and \
                                        ((1 - data[cl_pair][most_aberrant_value_idx - 1, 2]) < p_alpha or (1 - data[cl_pair][most_aberrant_value_idx + 1, 2]) < p_alpha):
                                    inhibition_p = True
                                    connection_type = 'common_input'

                        if excitation_p or inhibition_p:
                            pair_1_category = profile_data.loc[(profile_data['session_id'] == sp_session_id)
                                                               & (profile_data['cluster_id'] == pair_split[0]), 'category'].values[0]
                            pair_2_category = profile_data.loc[(profile_data['session_id'] == sp_session_id)
                                                               & (profile_data['cluster_id'] == pair_split[1]), 'category'].values[0]

                            output_dict[connection_type][sp_session_id][cl_pair] = {'peak_time': 'nan',
                                                                                    'excitatory': True,
                                                                                    'sp_profiles': ['nan', 'nan'],
                                                                                    'broad_category': ['null', 'null'],
                                                                                    'first_covariate': ['null', 'null'],
                                                                                    'original_mat_file': 'nan',
                                                                                    'data': data[cl_pair].tolist()}

                            output_dict[connection_type][sp_session_id][cl_pair]['original_mat_file'] = mat_file

                            if inhibition_p:
                                output_dict[connection_type][sp_session_id][cl_pair]['excitatory'] = False

                            output_dict[connection_type][sp_session_id][cl_pair]['sp_profiles'][0] = profile_data.loc[(profile_data['session_id'] == sp_session_id)
                                                                                                                      & (profile_data['cluster_id'] == pair_split[0]), 'profile'].values[0]
                            output_dict[connection_type][sp_session_id][cl_pair]['sp_profiles'][1] = profile_data.loc[(profile_data['session_id'] == sp_session_id)
                                                                                                                      & (profile_data['cluster_id'] == pair_split[1]), 'profile'].values[0]

                            output_dict[connection_type][sp_session_id][cl_pair]['peak_time'] = cch_total_range[most_aberrant_value_idx]

                            if ~np.isnan(pair_1_category):
                                output_dict[connection_type][sp_session_id][cl_pair]['broad_category'][0] = self.tuning_categories[int(pair_1_category)]
                                output_dict[connection_type][sp_session_id][cl_pair]['first_covariate'][0] = profile_data.loc[(profile_data['session_id'] == sp_session_id)
                                                                                                                              & (profile_data['cluster_id'] == pair_split[0]),
                                                                                                                              'first_covariate'].values[0]

                            if ~np.isnan(pair_2_category):
                                output_dict[connection_type][sp_session_id][cl_pair]['broad_category'][1] = self.tuning_categories[int(pair_2_category)]
                                output_dict[connection_type][sp_session_id][cl_pair]['first_covariate'][1] = profile_data.loc[(profile_data['session_id'] == sp_session_id)
                                                                                                                              & (profile_data['cluster_id'] == pair_split[1]),
                                                                                                                              'first_covariate'].values[0]
        with io.open(f'{self.save_dir}{os.sep}{json_file_names[0]}.json', 'w', encoding='utf-8') as to_save_file:
            to_save_file.write(json.dumps(output_dict['synaptic'], ensure_ascii=False, indent=4))

        with io.open(f'{self.save_dir}{os.sep}{json_file_names[1]}.json', 'w', encoding='utf-8') as to_save_file_1:
            to_save_file_1.write(json.dumps(output_dict['common_input'], ensure_ascii=False, indent=4))

    def cross_correlations_summary(self, **kwargs):
        """
        Description
        ----------
        This method summarizes spiking cross-correlation results.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        cch_time (float)
            The one-sided time of the CCG; defaults to 20 (ms).
        bin_num (float)
            The one-sided number of bins for the CCG; defaults to 50.
        critical_p_value (float)
            The p_value below something is considered statistically significant; defaults to 0.05
        ----------

        Returns
        ----------
        cch_summary (.json file)
            A file with the CCH summary results.
        ----------
        """

        cch_time = kwargs['cch_time'] if 'cch_time' in kwargs.keys() and type(kwargs['cch_time']) == int else 20
        bin_num = kwargs['bin_num'] if 'bin_num' in kwargs.keys() and type(kwargs['bin_num']) == int else 50
        critical_p_value = kwargs['critical_p_value'] if 'critical_p_value' in kwargs.keys() and type(kwargs['critical_p_value']) == float else .05

        cch_total_range = np.around(np.linspace(-cch_time, cch_time, (bin_num * 2) + 1), decimals=1)

        spc = pd.read_csv(self.sp_profiles_csv)

        plotting_dict = {}
        for one_file in os.listdir(f'{self.cch_data_dir}'):
            pair_region_id = one_file[-7:-5]
            plotting_dict[pair_region_id] = {'distances': [],
                                             'timing': [],
                                             'strength': [],
                                             'type': {'excitatory': 0, 'inhibitory': 0},
                                             'profile': {'RS': 0, 'FS': 0},
                                             'SMI': {'excited': 0,
                                                     'suppressed': 0,
                                                     'ns': 0},
                                             'LMI': {'excited': 0,
                                                     'suppressed': 0,
                                                     'ns': 0},
                                             'behavior': {'Unclassified': 0,
                                                          'null': 0,
                                                          'Ego3_Head_roll_1st_der': 0,
                                                          'Ego3_Head_azimuth_1st_der': 0,
                                                          'Ego3_Head_pitch_1st_der': 0,
                                                          'Ego3_Head_roll': 0,
                                                          'Ego3_Head_azimuth': 0,
                                                          'Ego3_Head_pitch': 0,
                                                          'Ego2_head_roll_1st_der': 0,
                                                          'Allo_head_direction_1st_der': 0,
                                                          'Ego2_head_pitch_1st_der': 0,
                                                          'Ego2_head_roll': 0,
                                                          'Allo_head_direction': 0,
                                                          'Ego2_head_pitch': 0,
                                                          'Back_azimuth_1st_der': 0,
                                                          'Back_pitch_1st_der': 0,
                                                          'Back_azimuth': 0,
                                                          'Back_pitch': 0,
                                                          'Neck_elevation': 0,
                                                          'Neck_elevation_1st_der': 0,
                                                          'Position': 0,
                                                          'Body_direction': 0,
                                                          'Body_direction_1st_der': 0,
                                                          'Speeds': 0,
                                                          'Self_motion': 0}}
            with open(f'{self.cch_data_dir}{os.sep}{one_file}', 'r') as json_pairs_file:
                data = json.load(json_pairs_file)
                for animal_session in data.keys():
                    rat_id = [rat for rat in self.animal_ids.keys() if rat in animal_session][0]
                    if rat_id not in plotting_dict[pair_region_id].keys():
                        plotting_dict[pair_region_id][rat_id] = {}
                    if animal_session not in plotting_dict[pair_region_id][rat_id].keys():
                        plotting_dict[pair_region_id][rat_id][animal_session] = {'pairs': [],
                                                                                 'directionality': [],
                                                                                 'strength': [],
                                                                                 'type': [],
                                                                                 'clusters': {}}
                    for pair_id in data[animal_session].keys():
                        plotting_dict[pair_region_id][rat_id][animal_session]['pairs'].append(pair_id)
                        plotting_dict[pair_region_id][rat_id][animal_session]['directionality'].append(np.sign(data[animal_session][pair_id]['peak_time']))
                        if data[animal_session][pair_id]['excitatory'] is True:
                            plotting_dict[pair_region_id]['type']['excitatory'] += 1
                            plotting_dict[pair_region_id][rat_id][animal_session]['type'].append('excitatory')
                        else:
                            plotting_dict[pair_region_id]['type']['inhibitory'] += 1
                            plotting_dict[pair_region_id][rat_id][animal_session]['type'].append('inhibitory')
                        plotting_dict[pair_region_id]['distances'].append(
                            np.abs(np.linalg.norm(np.array(spc.iloc[spc[(spc['cluster_id'] == pair_id.split('-')[0]) & (spc['session_id'] == animal_session)].index, -3:]).ravel()
                                                  - np.array(spc.iloc[spc[(spc['cluster_id'] == pair_id.split('-')[1]) & (spc['session_id'] == animal_session)].index, -3:]).ravel())))
                        plotting_dict[pair_region_id]['timing'].append(np.abs(data[animal_session][pair_id]['peak_time']))
                        idx_of_interest = np.where(cch_total_range == data[animal_session][pair_id]['peak_time'])[0][0]
                        cross_corr_data = np.array(data[animal_session][pair_id]['data'])
                        strength = np.abs((cross_corr_data[idx_of_interest, 0] - cross_corr_data[idx_of_interest, 1]) / np.min([cross_corr_data[0, 3], cross_corr_data[0, 4]]))
                        plotting_dict[pair_region_id]['strength'].append(strength)
                        plotting_dict[pair_region_id][rat_id][animal_session]['strength'].append(strength)
                        for cl_idx, cl in enumerate(pair_id.split('-')):
                            if cl not in plotting_dict[pair_region_id][rat_id][animal_session]['clusters'].keys():
                                plotting_dict[pair_region_id][rat_id][animal_session]['clusters'][cl] = {'XYZ': 0,
                                                                                                         'profile': 0,
                                                                                                         'SMI': 'ns',
                                                                                                         'LMI': 'ns',
                                                                                                         'behavior': 0}
                                plotting_dict[pair_region_id][rat_id][animal_session]['clusters'][cl]['XYZ'] = \
                                    list(np.array(spc.iloc[spc[(spc['cluster_id'] == cl) & (spc['session_id'] == animal_session)].index, -3:]).ravel())
                                plotting_dict[pair_region_id]['profile'][data[animal_session][pair_id]['sp_profiles'][cl_idx]] += 1
                                plotting_dict[pair_region_id][rat_id][animal_session]['clusters'][cl]['profile'] = \
                                    data[animal_session][pair_id]['sp_profiles'][cl_idx]
                                smi_values = np.array(spc.iloc[spc[(spc['cluster_id'] == cl) & (spc['session_id'] == animal_session)].index, 8:10]).ravel()
                                if not np.isnan(smi_values).all():
                                    if smi_values[1] > critical_p_value:
                                        plotting_dict[pair_region_id][rat_id][animal_session]['clusters'][cl]['SMI'] = 'ns'
                                        plotting_dict[pair_region_id]['SMI']['ns'] += 1
                                    else:
                                        if smi_values[0] > 0:
                                            plotting_dict[pair_region_id][rat_id][animal_session]['clusters'][cl]['SMI'] = 'excited'
                                            plotting_dict[pair_region_id]['SMI']['excited'] += 1
                                        else:
                                            plotting_dict[pair_region_id][rat_id][animal_session]['clusters'][cl]['SMI'] = 'suppressed'
                                            plotting_dict[pair_region_id]['SMI']['suppressed'] += 1
                                else:
                                    plotting_dict[pair_region_id][rat_id][animal_session]['clusters'][cl]['SMI'] = 'ns'
                                    plotting_dict[pair_region_id]['SMI']['ns'] += 1

                                lmi_values = np.array(spc.iloc[spc[(spc['cluster_id'] == cl) & (spc['session_id'] == animal_session)].index, 10:13]).ravel()
                                if not np.isnan(lmi_values).all():
                                    if lmi_values[1] < critical_p_value < lmi_values[2]:
                                        if lmi_values[0] > 0:
                                            plotting_dict[pair_region_id][rat_id][animal_session]['clusters'][cl]['LMI'] = 'excited'
                                            plotting_dict[pair_region_id]['LMI']['excited'] += 1
                                        else:
                                            plotting_dict[pair_region_id][rat_id][animal_session]['clusters'][cl]['LMI'] = 'suppressed'
                                            plotting_dict[pair_region_id]['LMI']['suppressed'] += 1
                                    else:
                                        plotting_dict[pair_region_id]['LMI']['ns'] += 1
                                else:
                                    plotting_dict[pair_region_id]['LMI']['ns'] += 1

                                plotting_dict[pair_region_id]['behavior'][data[animal_session][pair_id]['first_covariate'][cl_idx]] += 1
                                plotting_dict[pair_region_id][rat_id][animal_session]['clusters'][cl]['behavior'] = data[animal_session][pair_id]['first_covariate'][cl_idx]

        with io.open(f'{self.save_dir}{os.sep}cch_summary_synaptic.json', 'w', encoding='utf-8') as to_save_file:
            to_save_file.write(json.dumps(plotting_dict, ensure_ascii=False, indent=4))

    def shuffling_connections(self, **kwargs):
        """
        Description
        ----------
        This method checks whether the CCH results were obtained by chance.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        areas_lst (list)
            Brain areas of interest; defaults to ['AA'].
        animal_dates (dict)
            Dictionary containing dates when animals were recorded: defaults to {'kavokra': '190620', 'frank': '010620', 'johnjohn': '210520'}.
        n_shuff (int)
            Number of shuffle to perform; defaults to 1000.
        mi (dict)
            Modulation index of choice: defaults to {'AA': 'SMI', 'VV': 'LMI', 'SS': 'LMI', 'MM': 'LMI'}.
        input_type (str)
            Analyze 'synaptic' data or 'common_input'; defaults to 'synaptic'.
        ----------

        Returns
        ----------
        data_and_random_connections (.pkl file)
            A file with the CCH data and random connections.
        ----------
        """

        areas_lst = kwargs['areas_lst'] if 'areas_lst' in kwargs.keys() and type(kwargs['areas_lst']) == list else ['AA']
        animal_dates = kwargs['animal_dates'] if 'animal_dates' in kwargs.keys() and type(kwargs['animal_dates']) == dict \
            else {'kavorka': {'distal': '190620', 'intermediate': '190620'}, 'frank': {'distal': '010620', 'intermediate': '010620'},
                  'johnjohn': {'distal': '210520', 'intermediate': '230520'}, 'roy': {'distal': '270520', 'intermediate': '270520'},
                  'jacopo': {'distal': '150620', 'intermediate': '150620'}, 'crazyjoe': {'distal': '170620', 'intermediate': '170620'}}
        n_shuff = kwargs['n_shuff'] if 'n_shuff' in kwargs.keys() and type(kwargs['n_shuff']) == int else 1000
        mi = kwargs['mi'] if 'mi' in kwargs.keys() and type(kwargs['mi']) == dict else {'AA': 'SMI', 'VV': 'LMI', 'SS': 'LMI', 'MM': 'LMI'}
        input_type = kwargs['input_type'] if 'input_type' in kwargs.keys() and kwargs['input_type'] in ['synaptic', 'common_input'] else 'synaptic'

        with open(f'{self.cch_data_dir}{os.sep}cch_summary_{input_type}.json', 'r') as summary_file:
            synaptic_data = json.load(summary_file)

        spc = pd.read_csv(self.sp_profiles_csv)

        data_dict = {area: {'data': {}, 'shuffled': {}} for area in areas_lst}
        for area in areas_lst:

            largest_connection_distance = 0
            smallest_connection_distance = 3.84

            area_dict = {connection_type: {'po_po': 0, 'po_mo': 0, 'mo_mo': 0,
                                           'mo_po': 0, f'mo_{mi[area]}': 0, f'po_{mi[area]}': 0,
                                           'unclass': 0, f'unclass_{mi[area]}': 0} for connection_type in ['exc', 'inh']}

            # data
            for key in synaptic_data[area].keys():
                if key in self.animal_ids.keys():
                    if key != 'bruno':
                        for session in synaptic_data[area][key]:
                            for idx, pair in enumerate(synaptic_data[area][key][session]['pairs']):
                                direction = synaptic_data[area][key][session]['directionality'][idx]
                                connection_type = synaptic_data[area][key][session]['type'][idx]
                                cl1, cl2 = pair.split('-')
                                if direction < 0:
                                    presynaptic_cell = cl1
                                    postsynaptic_cell = cl2
                                else:
                                    presynaptic_cell = cl2
                                    postsynaptic_cell = cl1
                                presynaptic_beh = synaptic_data[area][key][session]['clusters'][presynaptic_cell]['behavior']
                                postsynaptic_beh = synaptic_data[area][key][session]['clusters'][postsynaptic_cell]['behavior']
                                postsynaptic_mi = synaptic_data[area][key][session]['clusters'][postsynaptic_cell][mi[area]]

                                presynaptic_idx = spc[(spc['cluster_id'] == presynaptic_cell) & (spc['session_id'] == session)].index.tolist()[0]
                                postsynaptic_idx = spc[(spc['cluster_id'] == postsynaptic_cell) & (spc['session_id'] == session)].index.tolist()[0]
                                pair_euclidean_distance = np.abs(np.linalg.norm(np.array(spc.iloc[presynaptic_idx, -3:]) - np.array(spc.iloc[postsynaptic_idx, -3:])))
                                largest_connection_distance = max(largest_connection_distance, pair_euclidean_distance)
                                smallest_connection_distance = min(smallest_connection_distance, pair_euclidean_distance)

                                # classify by tuning of behavioral modulation
                                if ('der' in presynaptic_beh or 'Speeds' in presynaptic_beh or 'Self_motion' in presynaptic_beh) and \
                                        ('der' in postsynaptic_beh or 'Speeds' in postsynaptic_beh or 'Self_motion' in postsynaptic_beh) and \
                                        (presynaptic_beh != 'Unclassified' and presynaptic_beh != 'null') and \
                                        (postsynaptic_beh != 'Unclassified' and postsynaptic_beh != 'null'):
                                    if connection_type == 'excitatory':
                                        area_dict['exc']['mo_mo'] += 1
                                        if postsynaptic_mi != 'ns':
                                            area_dict['exc'][f'mo_{mi[area]}'] += 1
                                    else:
                                        area_dict['inh']['mo_mo'] += 1
                                        if postsynaptic_mi != 'ns':
                                            area_dict['inh'][f'mo_{mi[area]}'] += 1
                                elif not ('der' in presynaptic_beh or 'Speeds' in presynaptic_beh or 'Self_motion' in presynaptic_beh) and \
                                        not ('der' in postsynaptic_beh or 'Speeds' in postsynaptic_beh or 'Self_motion' in postsynaptic_beh) and \
                                        (presynaptic_beh != 'Unclassified' and presynaptic_beh != 'null') and \
                                        (postsynaptic_beh != 'Unclassified' and postsynaptic_beh != 'null'):
                                    if connection_type == 'excitatory':
                                        area_dict['exc']['po_po'] += 1
                                        if postsynaptic_mi != 'ns':
                                            area_dict['exc'][f'po_{mi[area]}'] += 1
                                    else:
                                        area_dict['inh']['po_po'] += 1
                                        if postsynaptic_mi != 'ns':
                                            area_dict['inh'][f'po_{mi[area]}'] += 1
                                elif ('der' in presynaptic_beh or 'Speeds' in presynaptic_beh or 'Self_motion' in presynaptic_beh) and \
                                        not ('der' in postsynaptic_beh or 'Speeds' in postsynaptic_beh or 'Self_motion' in postsynaptic_beh) and \
                                        (presynaptic_beh != 'Unclassified' and presynaptic_beh != 'null') and \
                                        (postsynaptic_beh != 'Unclassified' and postsynaptic_beh != 'null'):
                                    if connection_type == 'excitatory':
                                        area_dict['exc']['mo_po'] += 1
                                        if postsynaptic_mi != 'ns':
                                            area_dict['exc'][f'mo_{mi[area]}'] += 1
                                    else:
                                        area_dict['inh']['mo_po'] += 1
                                        if postsynaptic_mi != 'ns':
                                            area_dict['inh'][f'mo_{mi[area]}'] += 1
                                elif not ('der' in presynaptic_beh or 'Speeds' in presynaptic_beh or 'Self_motion' in presynaptic_beh) and \
                                        ('der' in postsynaptic_beh or 'Speeds' in postsynaptic_beh or 'Self_motion' in postsynaptic_beh) and \
                                        (presynaptic_beh != 'Unclassified' and presynaptic_beh != 'null') and \
                                        (postsynaptic_beh != 'Unclassified' and postsynaptic_beh != 'null'):
                                    if connection_type == 'excitatory':
                                        area_dict['exc']['po_mo'] += 1
                                        if postsynaptic_mi != 'ns':
                                            area_dict['exc'][f'po_{mi[area]}'] += 1
                                    else:
                                        area_dict['inh']['po_mo'] += 1
                                        if postsynaptic_mi != 'ns':
                                            area_dict['inh'][f'po_{mi[area]}'] += 1
                                elif (presynaptic_beh != 'Unclassified' and presynaptic_beh != 'null') or \
                                        (postsynaptic_beh != 'Unclassified' and postsynaptic_beh != 'null'):
                                    if connection_type == 'excitatory':
                                        area_dict['exc']['unclass'] += 1
                                        if postsynaptic_mi != 'ns':
                                            if not ('der' in presynaptic_beh or 'Speeds' in presynaptic_beh or 'Self_motion' in presynaptic_beh) \
                                                    and presynaptic_beh != 'Unclassified' and presynaptic_beh != 'null':
                                                area_dict['exc'][f'po_{mi[area]}'] += 1
                                            elif 'der' in presynaptic_beh or 'Speeds' in presynaptic_beh or 'Self_motion' in presynaptic_beh \
                                                    and presynaptic_beh != 'Unclassified' and presynaptic_beh != 'null':
                                                area_dict['exc'][f'mo_{mi[area]}'] += 1
                                            else:
                                                area_dict['exc'][f'unclass_{mi[area]}'] += 1
                                    else:
                                        area_dict['inh']['unclass'] += 1
                                        if postsynaptic_mi != 'ns':
                                            if not ('der' in presynaptic_beh or 'Speeds' in presynaptic_beh or 'Self_motion' in presynaptic_beh) \
                                                    and presynaptic_beh != 'Unclassified' and presynaptic_beh != 'null':
                                                area_dict['inh'][f'po_{mi[area]}'] += 1
                                            elif 'der' in presynaptic_beh or 'Speeds' in presynaptic_beh or 'Self_motion' in presynaptic_beh \
                                                    and presynaptic_beh != 'Unclassified' and presynaptic_beh != 'null':
                                                area_dict['inh'][f'mo_{mi[area]}'] += 1
                                            else:
                                                area_dict['inh'][f'unclass_{mi[area]}'] += 1
                                else:
                                    if connection_type == 'excitatory':
                                        area_dict['exc']['unclass'] += 1
                                        if postsynaptic_mi != 'ns':
                                            area_dict['exc'][f'unclass_{mi[area]}'] += 1
                                    else:
                                        area_dict['inh']['unclass'] += 1
                                        if postsynaptic_mi != 'ns':
                                            area_dict['inh'][f'unclass_{mi[area]}'] += 1

            data_dict[area]['data'] = area_dict

            # shuffled data
            exc_n = synaptic_data[area]['type']['excitatory']
            inh_n = synaptic_data[area]['type']['inhibitory']

            cl_dict = quantify_ratemaps.RatemapCharacteristics(ratemap_mat_dir='/home/.../ratemap_mats',
                                                               pkl_sessions_dir=self.pkl_sessions_dir,
                                                               specific_date={'bruno': ['020520', '030520'],
                                                                              'roy': True,
                                                                              'jacopo': True,
                                                                              'crazyjoe': True,
                                                                              'frank': True,
                                                                              'johnjohn': ['210520', '230520'],
                                                                              'kavorka': True},
                                                               area_filter=area[0],
                                                               session_id_filter=True,
                                                               session_non_filter='johnjohn_210520_s2',
                                                               session_type_filter=['dark'],
                                                               cluster_type_filter='good',
                                                               cluster_groups_dir=self.cluster_groups_dir,
                                                               sp_profiles_csv=self.sp_profiles_csv).file_finder(return_clusters=True)

            if 'MM' in areas_lst or 'SS' in areas_lst:
                if 'roy' not in cl_dict.keys():
                    cl_dict['roy'] = {}
                cl_dict['roy']['intermediate'] = select_clusters.ClusterFinder(session=f'{self.pkl_sessions_dir}{os.sep}clean_data_roy_270520_s1_intermediate_light_reheaded_XYZeuler_notricks.pkl',
                                                                               cluster_groups_dir=self.cluster_groups_dir,
                                                                               sp_profiles_csv=self.sp_profiles_csv).get_desired_clusters(filter_by_area=[area[0]],
                                                                                                                                          filter_by_cluster_type='good')

            shuffled_dict = {connection_type: {'po_po': np.zeros(n_shuff), 'po_mo': np.zeros(n_shuff), 'mo_mo': np.zeros(n_shuff),
                                               'mo_po': np.zeros(n_shuff), f'mo_{mi[area]}': np.zeros(n_shuff), f'po_{mi[area]}': np.zeros(n_shuff),
                                               'unclass': np.zeros(n_shuff), f'unclass_{mi[area]}': np.zeros(n_shuff)} for connection_type in ['exc', 'inh']}


            for shuffle_num in tqdm(range(n_shuff)):
                exc_n_shuff = 0
                inh_n_shuff = 0
                while exc_n_shuff < exc_n or inh_n_shuff < inh_n:
                    animals_lst = list(cl_dict.keys())
                    shuffle(animals_lst)
                    temp_animal = animals_lst[0]
                    banks_lst = list(cl_dict[temp_animal].keys())
                    shuffle(banks_lst)
                    temp_bank = banks_lst[0]
                    shuffle(cl_dict[temp_animal][temp_bank])

                    cl1, cl2 = cl_dict[temp_animal][temp_bank][0], cl_dict[temp_animal][temp_bank][1]
                    spc1 = spc[(spc['cluster_id'] == cl1) & (spc['session_id'] == f'{temp_animal}_{animal_dates[temp_animal][temp_bank]}_{temp_bank}')].index.tolist()[0]
                    spc2 = spc[(spc['cluster_id'] == cl2) & (spc['session_id'] == f'{temp_animal}_{animal_dates[temp_animal][temp_bank]}_{temp_bank}')].index.tolist()[0]
                    spc1_profile = spc.loc[spc1, 'profile']
                    spc2_profile = spc.loc[spc2, 'profile']
                    spc1_tuning = spc.loc[spc1, 'first_covariate']
                    spc2_tuning = spc.loc[spc2, 'first_covariate']

                    spc_pair_euclidean_distance = np.abs(np.linalg.norm(np.array(spc.iloc[spc1, -3:]) - np.array(spc.iloc[spc2, -3:])))

                    exc_bool = ((spc1_profile == 'RS' and spc2_profile == 'RS') or (spc1_profile == 'RS' and spc2_profile == 'FS')) and exc_n_shuff < exc_n
                    inh_bool = ((spc1_profile == 'FS' and spc2_profile == 'RS') or (spc1_profile == 'FS' and spc2_profile == 'FS')) and inh_n_shuff < inh_n
                    distance_bool = smallest_connection_distance <= spc_pair_euclidean_distance <= largest_connection_distance

                    if (exc_bool or inh_bool) and distance_bool and type(spc1_tuning) == str and type(spc2_tuning) == str:
                        mi2_bool = False
                        if mi == 'SMI':
                            if spc.loc[spc2, 'pSMI'] < .05:
                                mi2_bool = True
                        else:
                            if spc.loc[spc2, 'pLMI'] < .05 and spc.loc[spc2, 'pLMIcheck'] > .05:
                                mi2_bool = True

                        # classify by behavioral tuning and sensory modulation
                        if ('der' in spc1_tuning or 'Speeds' in spc1_tuning or 'Self_motion' in spc1_tuning) and \
                                ('der' in spc2_tuning or 'Speeds' in spc2_tuning or 'Self_motion' in spc2_tuning) and \
                                spc1_tuning != 'Unclassified' and spc2_tuning != 'Unclassified':
                            if exc_bool:
                                shuffled_dict['exc']['mo_mo'][shuffle_num] += 1
                                exc_n_shuff += 1
                                if mi2_bool is True:
                                    shuffled_dict['exc'][f'mo_{mi[area]}'][shuffle_num] += 1
                            else:
                                shuffled_dict['inh']['mo_mo'][shuffle_num] += 1
                                inh_n_shuff += 1
                                if mi2_bool is True:
                                    shuffled_dict['inh'][f'mo_{mi[area]}'][shuffle_num] += 1
                        elif not ('der' in spc1_tuning or 'Speeds' in spc1_tuning or 'Self_motion' in spc1_tuning) and \
                                not ('der' in spc2_tuning or 'Speeds' in spc2_tuning or 'Self_motion' in spc2_tuning) and \
                                spc1_tuning != 'Unclassified' and spc2_tuning != 'Unclassified':
                            if exc_bool:
                                shuffled_dict['exc']['po_po'][shuffle_num] += 1
                                exc_n_shuff += 1
                                if mi2_bool is True:
                                    shuffled_dict['exc'][f'po_{mi[area]}'][shuffle_num] += 1
                            else:
                                shuffled_dict['inh']['po_po'][shuffle_num] += 1
                                inh_n_shuff += 1
                                if mi2_bool is True:
                                    shuffled_dict['inh'][f'po_{mi[area]}'][shuffle_num] += 1
                        elif ('der' in spc1_tuning or 'Speeds' in spc1_tuning or 'Self_motion' in spc1_tuning) and \
                                not ('der' in spc2_tuning or 'Speeds' in spc2_tuning or 'Self_motion' in spc2_tuning) and \
                                spc1_tuning != 'Unclassified' and spc2_tuning != 'Unclassified':
                            if exc_bool:
                                shuffled_dict['exc']['mo_po'][shuffle_num] += 1
                                exc_n_shuff += 1
                                if mi2_bool is True:
                                    shuffled_dict['exc'][f'mo_{mi[area]}'][shuffle_num] += 1
                            else:
                                shuffled_dict['inh']['mo_po'][shuffle_num] += 1
                                inh_n_shuff += 1
                                if mi2_bool is True:
                                    shuffled_dict['inh'][f'mo_{mi[area]}'][shuffle_num] += 1
                        elif not ('der' in spc1_tuning or 'Speeds' in spc1_tuning or 'Self_motion' in spc1_tuning) and \
                                ('der' in spc2_tuning or 'Speeds' in spc2_tuning or 'Self_motion' in spc2_tuning) and \
                                spc1_tuning != 'Unclassified' and spc2_tuning != 'Unclassified':
                            if exc_bool:
                                shuffled_dict['exc']['po_mo'][shuffle_num] += 1
                                exc_n_shuff += 1
                                if mi2_bool is True:
                                    shuffled_dict['exc'][f'po_{mi[area]}'][shuffle_num] += 1
                            else:
                                shuffled_dict['inh']['po_mo'][shuffle_num] += 1
                                inh_n_shuff += 1
                                if mi2_bool is True:
                                    shuffled_dict['inh'][f'po_{mi[area]}'][shuffle_num] += 1
                        elif spc1_tuning != 'Unclassified' or spc2_tuning != 'Unclassified':
                            if exc_bool:
                                shuffled_dict['exc']['unclass'][shuffle_num] += 1
                                exc_n_shuff += 1
                                if mi2_bool is True:
                                    if not ('der' in spc1_tuning or 'Speeds' in spc1_tuning or 'Self_motion' in spc1_tuning) \
                                            and spc1_tuning != 'Unclassified':
                                        shuffled_dict['exc'][f'po_{mi[area]}'][shuffle_num] += 1
                                    elif 'der' in spc1_tuning or 'Speeds' in spc1_tuning or 'Self_motion' in spc1_tuning \
                                            and spc1_tuning != 'Unclassified':
                                        shuffled_dict['exc'][f'mo_{mi[area]}'][shuffle_num] += 1
                                    else:
                                        shuffled_dict['exc'][f'unclass_{mi[area]}'][shuffle_num] += 1
                            else:
                                shuffled_dict['inh']['unclass'][shuffle_num] += 1
                                inh_n_shuff += 1
                                if mi2_bool is True:
                                    if not ('der' in spc1_tuning or 'Speeds' in spc1_tuning or 'Self_motion' in spc1_tuning) \
                                            and spc1_tuning != 'Unclassified':
                                        shuffled_dict['inh'][f'po_{mi[area]}'][shuffle_num] += 1
                                    elif 'der' in spc1_tuning or 'Speeds' in spc1_tuning or 'Self_motion' in spc1_tuning \
                                            and spc1_tuning != 'Unclassified':
                                        shuffled_dict['inh'][f'mo_{mi[area]}'][shuffle_num] += 1
                                    else:
                                        shuffled_dict['inh'][f'unclass_{mi[area]}'][shuffle_num] += 1
                        else:
                            if exc_bool:
                                shuffled_dict['exc']['unclass'][shuffle_num] += 1
                                exc_n_shuff += 1
                                if mi2_bool is True:
                                    shuffled_dict['exc'][f'unclass_{mi[area]}'][shuffle_num] += 1
                            else:
                                shuffled_dict['inh']['unclass'][shuffle_num] += 1
                                inh_n_shuff += 1
                                if mi2_bool is True:
                                    shuffled_dict['inh'][f'unclass_{mi[area]}'][shuffle_num] += 1

                    else:
                        continue

            data_dict[area]['shuffled'] = shuffled_dict

        # save as .pkl file
        with open(f'{self.save_dir}{os.sep}{input_type}_data_and_random_connections', 'wb') as save_dict:
            pickle.dump(data_dict, save_dict)
