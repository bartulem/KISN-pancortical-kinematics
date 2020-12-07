# -*- coding: utf-8 -*-

"""

@author: bartulem

Make clean ratemaps for any session/combination of sessions.

"""

import os
import re
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


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

    def __init__(self, ratemap_mat_dir='', animals=None,
                 session_type_labels=None,
                 session_num_labels=None,
                 save_fig=False, fig_format='png', save_dir='/home/bartulm/Downloads', feature_filter=None):
        if session_num_labels is None:
            session_num_labels = ['s1', 's2', 's3', 's4', 's5', 's6']
        if session_type_labels is None:
            session_type_labels = ['light', 'dark', 'weight', 'sound']
        if animals is None:
            animals = ['bruno', 'johnjohn', 'roy', 'frank', 'jacopo', 'crazyjoe', 'kavorka']
        if feature_filter is None:
            feature_filter = {'cell_id': '',
                              'animal_id': '',
                              'bank': '',
                              'sessions': True,
                              'feature': ''}
        self.ratemap_mat_dir = ratemap_mat_dir
        self.feature_filter = feature_filter
        self.animals = animals
        self.session_type_labels = session_type_labels
        self.session_num_labels = session_num_labels
        self.save_fig = save_fig
        self.fig_format = fig_format
        self.save_dir = save_dir

    def make_clean_plots(self, **kwargs):
        """
        Description
        ----------
        This method enables plotting of clean ratemaps.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        use_smoothed_occ (bool)
            Use smoothed occupancies to make ratemaps; defaults to False.
        min_acceptable_occ (float)
            The minimum acceptable occupancy; defaults to 0.4 (ms).
        use_smoothed_rm (bool)
            Use smoothed firing rates to make ratemaps; defaults to False.
        """

        use_smoothed_occ = 6 if 'use_smoothed_occ' in kwargs.keys() and kwargs['use_smoothed_occ'] is True else 2
        min_acceptable_occ = kwargs['min_acceptable_occ'] if 'min_acceptable_occ' in kwargs.keys() and type(kwargs['min_acceptable_occ']) == float else 0.4
        use_smoothed_rm = 3 if 'use_smoothed_rm' in kwargs.keys() and kwargs['use_smoothed_rm'] is True else 1

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
                    good_ranges[file_idx] = [idx for idx, occ in enumerate(chosen_file_mat[feature_key][use_smoothed_occ, :]) if occ > min_acceptable_occ]
                    break

        # find overlap of good bins between sessions
        if len(good_ranges.keys()) == 1:
            indices_intersection = sorted(good_ranges[0])
        elif len(good_ranges.keys()) == 2:
            indices_intersection = sorted(list(set(good_ranges[0]) & set(good_ranges[1])), key=int)
        elif len(good_ranges.keys()) == 3:
            indices_intersection = sorted(list(set(good_ranges[0]) & set(good_ranges[1]) & set(good_ranges[2])), key=int)
        elif len(good_ranges.keys()) == 4:
            indices_intersection = sorted(list(set(good_ranges[0]) & set(good_ranges[1]) & set(good_ranges[2]) & set(good_ranges[3])), key=int)

        # get ratemap and shuffled low/high values for plotting
        rm_to_plot = {}
        for file_idx, chosen_file in enumerate(sorted(file_names)):
            # get file id in order
            file_session_label = [label for label in self.session_type_labels if label in chosen_file][0]
            file_session_num = [num for num in self.session_num_labels if num in chosen_file][0]

            plot_id = f'{file_session_label}_{file_session_num}'

            rm_to_plot[plot_id] = {'x': np.array([]), 'rm': np.array([]), 'shuffled': {'up': np.array([]), 'down': np.array([])}}

            # load the data
            chosen_file_mat = sio.loadmat(f'{self.ratemap_mat_dir}{os.sep}{chosen_file}')
            for feature_key in chosen_file_mat.keys():
                if self.feature_filter['feature'] in feature_key and 'data' in feature_key:
                    rm_to_plot[plot_id]['x'] = chosen_file_mat[feature_key][0, :].take(indices=indices_intersection)
                    rm_to_plot[plot_id]['rm'] = chosen_file_mat[feature_key][use_smoothed_rm, :].take(indices=indices_intersection)

                    shuffled_key = feature_key.replace('data', 'rawacc_shuffles')
                    rm_to_plot[plot_id]['shuffled']['down'] = np.percentile(chosen_file_mat[shuffled_key].take(indices=indices_intersection, axis=0), q=.5, axis=1)
                    rm_to_plot[plot_id]['shuffled']['up'] = np.percentile(chosen_file_mat[shuffled_key].take(indices=indices_intersection, axis=0), q=99.5, axis=1)
                    break

        # plot
        cl_name = self.feature_filter['cell_id']
        feature_name = self.feature_filter['feature']
        file_animal = [animal for animal in self.animals if animal in file_names[0]][0]
        file_bank = [bank for bank in ['distal', 'intermediate'] if bank in file_names[0]][0]
        get_date_idx = [date.start() for date in re.finditer('20', file_names[0])][-1]
        file_date = file_names[0][get_date_idx-4:get_date_idx+2]

        # determine ratemap color
        for one_feature in self.feature_colors.keys():
            if one_feature in self.feature_filter['feature']:
                designated_color = self.feature_colors[one_feature]
                break

        col_num = len(rm_to_plot.keys())
        fig, ax = plt.subplots(nrows=1, ncols=col_num, figsize=(4.3*col_num, 3.3))
        for data_idx, data in enumerate(rm_to_plot.keys()):
            ax = plt.subplot(1, col_num, data_idx+1)
            ax.plot(rm_to_plot[data]['x'], rm_to_plot[data]['rm'], ls='-', color=designated_color, lw=3)
            ax.fill_between(rm_to_plot[data]['x'], rm_to_plot[data]['shuffled']['down'], rm_to_plot[data]['shuffled']['up'],
                            where=rm_to_plot[data]['shuffled']['up'] >= rm_to_plot[data]['shuffled']['down'], facecolor='#D3D3D3', interpolate=True)
            ax.set_yticks(np.arange(0, max(rm_to_plot[data]['rm']) + 2, max(rm_to_plot[data]['rm']) + 1, dtype='int'))
            ax.yaxis.set_ticks_position('none')
            ax.set_xticks([min(rm_to_plot[data]['x']), max(rm_to_plot[data]['x'])])
            ax.set_title(data)
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig.savefig(f'{self.save_dir}{os.sep}{file_animal}_{file_bank}_{file_date}_{cl_name}_{feature_name}.{self.fig_format}', dpi=300)
            else:
                print("Specified save directory doesn't exist. Try again.")
                sys.exit()
        plt.show()
