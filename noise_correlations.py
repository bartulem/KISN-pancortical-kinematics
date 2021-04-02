# -*- coding: utf-8 -*-

"""

@author: bartulem

Calculate noise correlations.

"""

import os
import numpy as np
import scipy.io as sio
from scipy import sparse
from numba import njit
from tqdm import tqdm
from itertools import combinations
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson
import sessions2load
import select_clusters
import neural_activity


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

    fr_arr = np.zeros(int(np.ceil((tracking_stop-tracking_start)/bin_size))).astype(np.float32)
    fr_arr[np.round(cl_activity/bin_size).astype(np.int32)] = 1
    jitter_arr = sparse.csr_matrix((jitter_n, fr_arr.shape[0]), dtype=np.float32)
    if to_jitter:
        for sh in range(jitter_n):
            jitter_spikes = cl_activity.copy() + ((2 * jitter_size * np.random.random(cl_activity.shape[0])) - jitter_size)
            jitter_spikes = jitter_spikes[jitter_spikes <= tracking_stop]
            jitter_arr[sh, np.round(jitter_spikes/bin_size).astype(np.int32)] = 1
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

    r_num = np.sum((big_x-big_x_mean)*(small_y-small_y_mean), axis=1)
    r_den = np.sqrt(np.sum((big_x-big_x_mean)**2, axis=1)*np.sum((small_y-small_y_mean)**2))
    r = r_num/r_den
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
        Tracking start relative to tracking start (should be 0).
    ----------

    Returns
    ----------
    smoothed_cch : np.ndarray
        The hollow-Gaussian convolved CCH.
    ----------
    """

    smoothed_cch = np.zeros(cch.shape[0]*3)
    input_array_reflected = np.concatenate((cch[::-1], cch, cch[::-1]))
    x_v = np.arange(smoothed_cch.shape[0])
    for idx in x_v:
        kernel_idx = np.exp(-(x_v - idx) ** 2 / (2 * sigma ** 2))
        kernel_idx[int(np.floor(kernel_idx.shape[0]/2))] = kernel_idx[int(np.floor(kernel_idx.shape[0]/2))] * (1-fraction_hollowed)
        kernel_idx = kernel_idx / kernel_idx.sum()
        smoothed_cch[idx] = np.dot(kernel_idx, input_array_reflected)
    return smoothed_cch[cch.shape[0]:cch.shape[0]*2]

def calculate_p_values(cch, slow_baseline):
    """
    Description
    ----------
    This function calculates the probabilities of obtaining an observed (or higher) synchrony count
    in the time lags of the observed CCG (in the offered range), given the expected, low frequency
    baseline rate in the same bins, estimated using the Poisson distribution (Fine-English et al.,
    Neuron, 2009) with a continuity correction.

    Extra: The absolute function at the end is there because when the factorials get very large,
    the calculation yields negative values.
    ----------

    Parameters
    ----------
    cch : np.ndarray
        The CCH array that should be smoothed.
    slow_baseline : np.ndarray
        The convolved CCH array.
    ----------

    Returns
    ----------
    p_values : np.ndarray
        The hollow-Gaussian convolved CCH.
    ----------
    """
    return np.abs(1 - poisson.cdf(k=cch - 1, mu=slow_baseline) - poisson.pmf(k=cch, mu=slow_baseline)*0.5)


class FunctionalConnectivity:

    def __init__(self, pkl_sessions_dir='', cluster_groups_dir='',
                 sp_profiles_csv='', pkl_file='', save_dir=''):
        self.pkl_sessions_dir = pkl_sessions_dir
        self.cluster_groups_dir = cluster_groups_dir
        self.sp_profiles_csv = sp_profiles_csv
        self.pkl_file = pkl_file
        self.save_dir=save_dir

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
        sort_ch_num = kwargs['sort_ch_num'] if 'sort_ch_num' in kwargs.keys() and type(kwargs['sort_ch_num']) == bool else False
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
                                       tracking_start=cluster_data['tracking_ts'][0]-cluster_data['tracking_ts'][0],
                                       tracking_stop=cluster_data['tracking_ts'][1]-cluster_data['tracking_ts'][0],
                                       jitter_size=jitter_size, jitter_n=num_jitters)

                fr2, sh2 = bin_spiking(cl_activity=act2, bin_size=bin_size,
                                       tracking_start=cluster_data['tracking_ts'][0]-cluster_data['tracking_ts'][0],
                                       tracking_stop=cluster_data['tracking_ts'][1]-cluster_data['tracking_ts'][0],
                                       jitter_size=jitter_size, jitter_n=num_jitters)

                if smooth_fr:
                    fr1 = gaussian_filter1d(input=fr1, sigma=int(round(std_smooth/bin_size)))
                    sh1 = gaussian_filter1d(input=sh1.todense(), sigma=int(round(std_smooth/bin_size)), axis=1)
                    fr2 = gaussian_filter1d(input=fr2, sigma=int(round(std_smooth/bin_size)))
                    sh2 = gaussian_filter1d(input=sh2.todense(), sigma=int(round(std_smooth/bin_size)), axis=1)
            else:
                fr1, sh1 = bin_spiking(cl_activity=act1, bin_size=bin_size,
                                       tracking_start=cluster_data['tracking_ts'][0]-cluster_data['tracking_ts'][0],
                                       tracking_stop=cluster_data['tracking_ts'][1]-cluster_data['tracking_ts'][0],
                                       to_jitter=False, jitter_n=1)

                fr2, sh2 = bin_spiking(cl_activity=act2, bin_size=bin_size,
                                       tracking_start=cluster_data['tracking_ts'][0]-cluster_data['tracking_ts'][0],
                                       tracking_stop=cluster_data['tracking_ts'][1]-cluster_data['tracking_ts'][0],
                                       to_jitter=False, jitter_n=1)

                if smooth_fr:
                    fr1 = gaussian_filter1d(input=fr1, sigma=int(round(std_smooth/bin_size)))
                    fr2 = gaussian_filter1d(input=fr2, sigma=int(round(std_smooth/bin_size)))

            # cross-correlate
            all_bins = np.arange(-bin_num, bin_num+1, 1)
            fr1_shape = fr1.shape[0]
            y_start = int(round(bin_num))
            y_end = int(round(fr1_shape-bin_num))

            x_bool = np.zeros((all_bins.shape[0], fr1_shape), dtype=bool)
            for bin_idx, one_bin in enumerate(all_bins):
                x_bool[bin_idx, int(round(bin_num+one_bin)):int(round(fr1_shape-bin_num+one_bin))] = True

            big_x = np.tile(A=fr1, reps=(all_bins.shape[0], 1))
            big_x = big_x[x_bool].reshape(all_bins.shape[0], y_end-y_start)
            big_x_mean=np.reshape(big_x.mean(axis=1), (big_x.shape[0], 1))
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
                        big_x_sh = big_x_sh[x_bool].reshape(all_bins.shape[0], y_end-y_start)
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
                    output_dictionary[combo_name]['cch'] = data
                    output_dictionary[combo_name]['jitter_cch'] = sh_data
                else:
                    output_dictionary[combo_name]['cch'] = data
            else:
                output_array = np.zeros((all_bins.shape[0], 3))
                cch = dot_product(big_x=big_x,
                                  small_y=y)
                output_array[:, 0] = cch

                cch_convolved = hollowed_gaussian_kernel(cch=cch, sigma=int(round(convolve_sigma/bin_size)))
                output_array[:, 1] = cch_convolved

                cch_probabilities = calculate_p_values(cch=cch, slow_baseline=cch_convolved)
                output_array[:, 2] = cch_probabilities

                output_dictionary[combo_name] = output_array

        # save to file
        sio.savemat(f'{self.save_dir}{os.sep}cl{combo_start}-{combo_end}.mat', output_dictionary)
