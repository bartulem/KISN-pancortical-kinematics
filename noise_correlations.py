# -*- coding: utf-8 -*-

"""

@author: bartulem

Calculate noise correlations.

"""

import os
import numpy as np
from numba import njit
from sessions2load import Session
from select_clusters import ClusterFinder


@njit(parallel=False)
def get_firing_rate(cl_activity, binned_time, smoothing_dev, gauss_a, gauss_b):
    smoothed_fr = np.zeros(binned_time.shape[0])
    for idx in range(binned_time.shape[0]):
        dd = np.abs(cl_activity-binned_time[idx])
        dd = dd[dd<6*smoothing_dev]
        smoothed_fr[idx] = np.sum(gauss_a*np.exp( (dd**2) * gauss_b ))
    return smoothed_fr


class FunctionalConnectivity:

    def __init__(self, pkl_sessions_dir='', cluster_groups_dir='',
                 sp_profiles_csv='', pkl_file=''):
        self.session = pkl_sessions_dir
        self.cluster_groups_dir = cluster_groups_dir
        self.sp_profiles_csv = sp_profiles_csv
        self.pkl_file = pkl_file

    def noise_corr(self, **kwargs):
        """
        Description
        ----------
        This method calculates noise correlations for all clusters in a given session.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        bin_size (float)
            The size of bins for binning spikes; defaults to .5 (ms).
        bin_num (float)
            The one-sided number of bins for the cross-correlogram; defaults to 40.
        std_smooth (float)
            The std. deviation of the gaussian smoothing kernel; defaults to 0.001 (ms).
        num_jitters (int)
            The number of times to jitter data; defaults to 1000.
        area_filter (list / bool)
            Areas to be included, you can pick specific areas or
            general (A - auditory, M - motor, P - parietal, S - somatosensory, V - visual); defaults to True.
        cluster_type_filter (str / bool)
            Cluster type to be included: 'good' or 'mua'; defaults to True.
        profile_filter (str / bool)
            Profile to be included: 'RS' or 'FS'; defaults to True.
        sort_ch_num (bool)
            If True, sorts clusters by channel number; defaults to False.
        ----------

        Returns
        ----------
        noise_corrs ()
            A list with the name-sorted filtered clusters.
        ----------
        """

        bin_size = kwargs['bin_size'] if 'bin_size' in kwargs.keys() and type(kwargs['bin_size']) == float else .0005
        bin_num = kwargs['bin_num'] if 'bin_num' in kwargs.keys() and type(kwargs['bin_num']) == int else 40
        std_smooth = kwargs['std_smooth'] if 'std_smooth' in kwargs.keys() and type(kwargs['std_smooth']) == float else .001
        num_jitters = kwargs['num_jitters'] if 'num_jitters' in kwargs.keys() and type(kwargs['num_jitters']) == float else 1000
        area_filter = kwargs['area_filter'] if 'area_filter' in kwargs.keys() and type(kwargs['area_filter']) == list else True
        cluster_type_filter = kwargs['cluster_type_filter'] if 'cluster_type_filter' in kwargs.keys() and type(kwargs['cluster_type_filter']) == str else True
        profile_filter = kwargs['profile_filter'] if 'profile_filter' in kwargs.keys() and type(kwargs['profile_filter']) == str else True
        sort_ch_num = kwargs['sort_ch_num'] if 'sort_ch_num' in kwargs.keys() and type(kwargs['sort_ch_num']) == bool else False

        gauss_a = 1./np.sqrt(2.*np.pi*std_smooth**2)
        gauss_b = -1./(2.*std_smooth**2)

        cluster_list = ClusterFinder(session=f'{self.pkl_sessions_dir}{os.sep}{self.pkl_file}',
                                     cluster_groups_dir=self.cluster_groups_dir,
                                     sp_profiles_csv=self.sp_profiles_csv).get_desired_clusters(filter_by_area=area_filter,
                                                                                                filter_by_cluster_type=cluster_type_filter,
                                                                                                filter_by_spiking_profile=profile_filter,
                                                                                                sort_ch_num=sort_ch_num)
        # get spike data in seconds and tracking start and end time
        file_id, cluster_data = Session(session=f'{self.pkl_sessions_dir}{os.sep}{self.pkl_file}').data_loader(extract_clusters=cluster_list, extract_variables=['tracking_ts'])

        binned_activity_range = np.arange(cluster_data['tracking_ts'][0], cluster_data['tracking_ts'][1], bin_size)
