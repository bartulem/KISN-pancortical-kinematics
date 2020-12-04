# -*- coding: utf-8 -*-

"""

@author: bartulem

Make clean ratemaps for any session/combination of sessions.

"""

import os
import sys
import numpy as np
import scipy.io as sio


# data[0, :] = xvals (bin centers)
# data[1, :] = raw rate map (ratemap / no smoothing)
# data[2, :] = occupancy (occupancy / no smoothing)
# data[3, :] = smoothed rate map
# data[4, :] = shuffled mean
# data[5, :] = shuffled std
# data[6, :] = smoothed occupancy
# data[7, :] = rawrm_p1 (even minutes ratemap / no smoothing)
# data[8, :] = smrm_p1 (even minutes ratemap / smoothed)
# data[9, :] = occ_p1 (even minutes occupancy / no smoothing)
# data[10, :] = smocc_p1 (even minutes occupancy / smoothed)
# data[11, :] = rawrm_p2 (odd minutes ratemap / no smoothing)
# data[12, :] = smrm_p2 (odd minutes ratemap / smoothed)
# data[13, :] = occ_p2 (odd minutes occupancy / no smoothing)
# data[14, :] = smocc_p2 (odd minutes occupancy / smoothed)


class Ratemap:
    feature_colors = {'Allo_head_pitch': '#C91517',
                      'Allo_head_azimuth': '#ED6C6D',
                      'Allo_head_roll': '#F1A6B1',
                      'Back_pitch': '#3052A0',
                      'Back_azimuth': '#77AEDF',
                      'Neck_elevation': '#F07F00'}

    def __init__(self, ratemap_mat_dir='', feature_filter=None):
        if feature_filter is None:
            feature_filter = {'cell_id': '',
                              'animal_id': '',
                              'bank': '',
                              'sessions': True,
                              'feature': ''}
        self.ratemap_mat_dir = ratemap_mat_dir
        self.feature_filter = feature_filter

    def make_clean_plots(self, **kwargs):
        """
        Description
        ----------
        This method enables plotting of clean ratemaps.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        use_smoothed (bool)
            Use raw or smoothed values to make ratemaps; defaults to True.
        min_acceptable_occ (float)
            The minimum acceptable occupancy; defaults to 0.4 (ms).
        """

        use_smoothed = 6 if 'raw_or_smoothed' in kwargs.keys() and kwargs['raw_or_smoothed'] is True else 2
        min_acceptable_occ = kwargs['min_acceptable_occ'] if 'min_acceptable_occ' in kwargs.keys() and type(kwargs['min_acceptable_occ']) == float else 0.4

        file_names = []
        if os.path.exists(self.ratemap_mat_dir):
            for file_name in os.listdir(self.ratemap_mat_dir):
                if all(fe_filter in file_name for fe_filter in [self.feature_filter['cell_id'],
                                                                self.feature_filter['animal_id'],
                                                                self.feature_filter['bank']]):
                    if self.feature_filter['sessions'] is True or any(se_filter in file_name for se_filter in self.feature_filter['sessions']):
                        file_names.append(file_name)
        else:
            print(f"Invalid location for ratemap directory {self.ratemap_mat_dir}. Please try again.")
            sys.exit()

        # get ranges with sufficient occupancy
        good_ranges = {}
        for file_idx, chosen_file in enumerate(sorted(file_names)):
            chosen_file_mat = sio.loadmat(f'{self.ratemap_mat_dir}{os.sep}{chosen_file}')
            for feature_key in chosen_file_mat.keys():
                if self.feature_filter['feature'] in feature_key and 'data' in feature_key:
                    good_ranges[file_idx] = np.array([idx for idx, occ in enumerate(chosen_file_mat[feature_key][use_smoothed, :]) if occ > min_acceptable_occ])

        print(good_ranges)
