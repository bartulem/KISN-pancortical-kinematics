# -*- coding: utf-8 -*-

"""

@author: bartulem

Make comparisons of behavioral occupancies between medicated and non-medicated animals.

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.stats import sem
from sessions2load import Session
from make_ratemaps import Ratemap


@njit(parallel=False)
def get_bins(feature_arr, min_val, max_val, num_bins_1d, camera_framerate):
    bin_edges = np.linspace(min_val, max_val, num_bins_1d + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    occupancy = np.zeros(np.shape(bin_edges)[0] - 1)
    for i in range(1, np.shape(bin_edges)[0], 1):
        occupancy[i - 1] = np.sum((feature_arr > bin_edges[i - 1]) * (feature_arr <= bin_edges[i])) / camera_framerate
    rel_occupancy = occupancy.copy() / occupancy.copy().sum()
    return bin_edges, bin_centers, occupancy, rel_occupancy


class Behavior:

    variable_bounds = {'neck_elevation': np.array([0, 0.36]),
                       'body_direction': np.array([-180, 180]),
                       'allo_head_ang': np.array([[-180, -180, -180], [180, 180, 180]]),
                       'ego3_head_ang': np.array([[-180, -180, -180], [180, 180, 180]]),
                       'ego2_head_ang': np.array([[-180, -180, -180], [180, 180, 180]]),
                       'back_ang': np.array([[-60, -60], [60, 60]]),
                       'opt_back_ang': np.array([[-60, -60], [60, 60]]),
                       'speeds': np.array([[0, 0, 0, 0], [120, 120, 120, 120]]),
                       'speeds_1st_der': np.array([[-150, -150, -150, -150], [150, 150, 150, 150]]),
                       'neck_1st_der': np.array([-0.1, 0.1 ]),
                       'neck_2nd_der': np.array([-0.8, 0.8]),
                       'allo_head_1st_der': np.array([[-400, -300, -400], [400, 300, 400]]),
                       'allo_head_2nd_der': np.array([[-4000, -3000, -4000], [4000, 3000, 4000]]),
                       'bodydir_1st_der': np.array([-300, 300]),
                       'bodydir_2nd_der': np.array([-1000, 1000]),
                       'ego3_head_1st_der': np.array([[-400, -300, -400], [400, 300, 400]]),
                       'ego3_head_2nd_der': np.array([[-4000, -3000, -4000], [4000, 3000, 4000]]),
                       'ego2_head_1st_der': np.array([[-400, -300, -400], [400, 300, 400]]),
                       'ego2_head_2nd_der': np.array([[-4000, -3000, -4000], [4000, 3000, 4000]]),
                       'back_1st_der': np.array([-100, 100]),
                       'back_2nd_der': np.array([-1000, 1000]),
                       'opt_back_1st_der': np.array([-100, 100]),
                       'opt_back_2nd_der': np.array([-1000, 1000]),
                       'imu_allo_head_ang': np.array([[-180, -180, -180], [180, 180, 180]]),
                       'imu_ego3_head_ang': np.array([[-180, -180, -180], [180, 180, 180]]),
                       'imu_ego2_head_ang': np.array([[-180, -180, -180], [180, 180, 180]]),
                       'imu_allo_head_1st_der': np.array([[-400, -300, -400], [400, 300, 400]]),
                       'imu_allo_head_2nd_der': np.array([[-4000, -3000, -4000], [4000, 3000, 4000]]),
                       'imu_ego3_head_1st_der': np.array([[-400, -300, -400], [400, 300, 400]]),
                       'imu_ego3_head_2nd_der': np.array([[-4000, -3000, -4000], [4000, 3000, 4000]]),
                       'imu_ego2_head_1st_der': np.array([[-400, -300, -400], [400, 300, 400]]),
                       'imu_ego2_head_2nd_der': np.array([[-4000, -3000, -4000], [4000, 3000, 4000]])}

    def __init__(self, variable_list=[], beh_plot_sessions={},
                 save_dir='', save_fig=False, fig_format='png',):
        self.variable_list = variable_list
        self.beh_plot_sessions = beh_plot_sessions
        self.save_dir=save_dir
        self.save_fig=save_fig
        self.fig_format=fig_format

    def pull_occ_histograms(self, **kwargs):
        """
        Description
        ----------
        This method gets the bin information to make occupancy histograms.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        data_file (str)
            Path to the data file of interest; defaults to None.
        num_bins_1d (int)
            The total number of bins for 1D features; defaults to 36.
        speed_idx (int)
            The speed definition to use; defaults to 3.
        ----------

        Returns
        ----------
        occ_dict (dict)
            A dictionary with all desired features and their 'bin_edges', 'bin_centers' and 'occupancy'.
        ----------
        """

        data_file = kwargs['data_file'] if 'data_file' in kwargs.keys() and type(kwargs['data_file']) == str else None
        num_bins_1d = kwargs['num_bins_1d'] if 'num_bins_1d' in kwargs.keys() and type(kwargs['num_bins_1d']) == int else 36
        speed_idx = kwargs['speed_idx'] if 'speed_idx' in kwargs.keys() and type(kwargs['speed_idx']) == int else 3

        extract_variables = ['framerate', 'sorted_point_data']
        for variable in self.variable_list:
            if variable != 'neck_elevation':
                extract_variables.append(variable)

        file_name, data = Session(session=data_file).data_loader(extract_variables=extract_variables)

        if 'sorted_point_data' in self.variable_list:
            self.variable_list.remove('sorted_point_data')

        occ_dict = {file_name: {}}
        for variable in self.variable_list + ['neck_elevation']:
            occ_dict[file_name][variable] = {}
            if variable != 'neck_elevation' and self.variable_bounds[variable].ndim == 1:
                occ_dict[file_name][variable]['bin_edges'], \
                occ_dict[file_name][variable]['bin_centers'], \
                occ_dict[file_name][variable]['occupancy'], \
                occ_dict[file_name][variable]['rel_occupancy'] = get_bins(feature_arr=data[variable],
                                                                          min_val=self.variable_bounds[variable][0],
                                                                          max_val=self.variable_bounds[variable][1],
                                                                          num_bins_1d=num_bins_1d,
                                                                          camera_framerate=data['framerate'])
            elif variable != 'neck_elevation' and self.variable_bounds[variable].ndim > 1 and 'speeds' not in variable:
                for eu_idx, euler_feature in enumerate(['roll', 'pitch', 'azimuth']):
                    occ_dict[file_name][variable][euler_feature] = {}
                    occ_dict[file_name][variable][euler_feature]['bin_edges'], \
                    occ_dict[file_name][variable][euler_feature]['bin_centers'], \
                    occ_dict[file_name][variable][euler_feature]['occupancy'], \
                    occ_dict[file_name][variable][euler_feature]['rel_occupancy'] = get_bins(feature_arr=data[variable][:, eu_idx],
                                                                                             min_val=self.variable_bounds[variable][0, eu_idx],
                                                                                             max_val=self.variable_bounds[variable][1, eu_idx],
                                                                                             num_bins_1d=num_bins_1d,
                                                                                             camera_framerate=data['framerate'])
            elif variable != 'neck_elevation' and self.variable_bounds[variable].ndim > 1 and 'speeds' in variable:
                occ_dict[file_name][variable]['bin_edges'], \
                occ_dict[file_name][variable]['bin_centers'], \
                occ_dict[file_name][variable]['occupancy'], \
                occ_dict[file_name][variable]['rel_occupancy'] = get_bins(feature_arr=data[variable][:, speed_idx],
                                                                          min_val=self.variable_bounds[variable][0, speed_idx],
                                                                          max_val=self.variable_bounds[variable][1, speed_idx],
                                                                          num_bins_1d=num_bins_1d,
                                                                          camera_framerate=data['framerate'])
            else:
                occ_dict[file_name][variable]['bin_edges'], \
                occ_dict[file_name][variable]['bin_centers'], \
                occ_dict[file_name][variable]['occupancy'], \
                occ_dict[file_name][variable]['rel_occupancy'] = get_bins(feature_arr=data['sorted_point_data'][:, 4, 2],
                                                                          min_val=self.variable_bounds[variable][0],
                                                                          max_val=self.variable_bounds[variable][1],
                                                                          num_bins_1d=num_bins_1d,
                                                                          camera_framerate=data['framerate'])


        return occ_dict

    def plot_behavior_differences(self, **kwargs):
        """
        Description
        ----------
        This method plots behavioral differences between e.g. weight/no-weight
        sessions (across animals).
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        example_session_baseline (str)
            Baseline session to take examples from; defaults to 'jacopo_150620_s4_intermediate_light_notreheaded_XYZeuler_notricks'.
        example_session_test (str)
            Test session to take examples from; defaults to 'jacopo_150620_s2_intermediate_weight_reheaded_XYZeuler_notricks'.
        min_acceptable_occ (float)
            The minimum acceptable occupancy; defaults to 0.4 (ms).
        ----------

        Returns
        ----------
        beh_differences (plot)
            A plot with examples and statistics of behavioral differences.
        ----------
        """

        example_session_baseline = kwargs['example_session_baseline'] if 'example_session_baseline' in kwargs.keys() \
                                                         and type(kwargs['example_session_baseline']) == str \
            else 'jacopo_150620_s4_intermediate_light_notreheaded_XYZeuler_notricks'
        example_session_test = kwargs['example_session_test'] if 'example_session_test' in kwargs.keys() \
                                                                         and type(kwargs['example_session_test']) == str \
            else 'jacopo_150620_s2_intermediate_weight_reheaded_XYZeuler_notricks'
        min_acceptable_occ = kwargs['min_acceptable_occ'] if 'min_acceptable_occ' in kwargs.keys() and type(kwargs['min_acceptable_occ']) == float else .4

        feature_dict = {'baseline': {}, 'test': {}}
        for session_type in self.beh_plot_sessions.keys():
            for file_loc in self.beh_plot_sessions[session_type]:
                extract_variables = ['sorted_point_data']
                for variable in self.variable_list:
                    if variable != 'neck_elevation':
                        extract_variables.append(variable)
                file_name, data = Session(session=file_loc).data_loader(extract_variables=extract_variables)
                feature_dict[session_type][file_name] = {}
                feature_dict[session_type][file_name]['examples'] = data
                feature_dict[session_type][file_name]['occ'] = self.pull_occ_histograms(data_file=file_loc)

        all_data_dict = {}
        stats_dict = {}
        for fn_idx, (fn_base, fn_test) in enumerate(zip(feature_dict['baseline'].keys(), feature_dict['test'].keys())):
            all_data_dict[fn_idx] = {'fn': {'baseline': 0, 'test': 0}}
            all_data_dict[fn_idx]['fn']['baseline'] = fn_base
            all_data_dict[fn_idx]['fn']['test'] = fn_test
            for variable in feature_dict['baseline'][fn_base]['occ'][fn_base]:
                if 'ego' not in variable:
                    all_data_dict[fn_idx][variable] = {}
                    if variable not in stats_dict.keys():
                        stats_dict[variable] = {}
                    good_ranges_base = [idx for idx, occ in enumerate(feature_dict['baseline'][fn_base]['occ'][fn_base][variable]['occupancy']) if occ > min_acceptable_occ]
                    good_ranges_test = [idx for idx, occ in enumerate(feature_dict['test'][fn_test]['occ'][fn_test][variable]['occupancy']) if occ > min_acceptable_occ]
                    indices_intersection = sorted(list(set(good_ranges_base) & set(good_ranges_test)), key=int)
                    bin_centers = feature_dict['test'][fn_test]['occ'][fn_test][variable]['bin_centers'].take(indices=indices_intersection)
                    all_data_dict[fn_idx][variable]['bin_centers'] = bin_centers
                    diffs = feature_dict['baseline'][fn_base]['occ'][fn_base][variable]['rel_occupancy'].take(indices=indices_intersection) \
                            - feature_dict['test'][fn_test]['occ'][fn_test][variable]['rel_occupancy'].take(indices=indices_intersection)
                    all_data_dict[fn_idx][variable]['diffs'] = diffs
                    for bc, diff in zip(bin_centers, diffs):
                        if bc not in stats_dict[variable].keys():
                            stats_dict[variable][bc] = []
                        stats_dict[variable][bc].append(diff)
                else:
                    for ego_f in ['roll', 'pitch', 'azimuth']:
                        all_data_dict[fn_idx][ego_f] = {}
                        if ego_f not in stats_dict.keys():
                            stats_dict[ego_f] = {}
                        good_ranges_base = [idx for idx, occ in enumerate(feature_dict['baseline'][fn_base]['occ'][fn_base][variable][ego_f]['occupancy']) if occ > min_acceptable_occ]
                        good_ranges_test = [idx for idx, occ in enumerate(feature_dict['test'][fn_test]['occ'][fn_test][variable][ego_f]['occupancy']) if occ > min_acceptable_occ]
                        indices_intersection = sorted(list(set(good_ranges_base) & set(good_ranges_test)), key=int)
                        bin_centers = feature_dict['test'][fn_test]['occ'][fn_test][variable][ego_f]['bin_centers'].take(indices=indices_intersection)
                        all_data_dict[fn_idx][ego_f]['bin_centers'] = bin_centers
                        diffs = feature_dict['baseline'][fn_base]['occ'][fn_base][variable][ego_f]['rel_occupancy'].take(indices=indices_intersection) \
                                - feature_dict['test'][fn_test]['occ'][fn_test][variable][ego_f]['rel_occupancy'].take(indices=indices_intersection)
                        all_data_dict[fn_idx][ego_f]['diffs'] = diffs
                        for bc, diff in zip(bin_centers, diffs):
                            if bc not in stats_dict[ego_f].keys():
                                stats_dict[ego_f][bc] = []
                            stats_dict[ego_f][bc].append(diff)

        plot_dict = {}
        for variable in stats_dict.keys():
            plot_dict[variable] = {'x': list(stats_dict[variable].keys()),
                                   'y': [np.mean(stats_dict[variable][f_bin]) for f_bin in stats_dict[variable].keys()],
                                   'yerr': [sem(stats_dict[variable][f_bin])*2.58 for f_bin in stats_dict[variable].keys()]}

        # print(plot_dict)


        fig = plt.figure(figsize=(7, 10))
        gs1 = fig.add_gridspec(nrows=5, ncols=12, left=.075,
                               right=.98, wspace=6.5, hspace=.5)
        for f_idx, feature in enumerate(['roll', 'pitch', 'azimuth', 'neck_elevation', 'speeds']):
            f_color = [val for key, val in Ratemap.feature_colors.items() if feature in key][0]
            ax = fig.add_subplot(gs1[f_idx, :3])
            ax2 = fig.add_subplot(gs1[f_idx, 3:])
            if f_idx == 0:
                ax2.set_ylim(-.1, .1)
            elif f_idx == 2:
                ax2.set_ylim(-.05, .05)
            elif f_idx == 1 or f_idx == 4:
                ax2.set_ylim(-.2, .2)
            else:
                ax2.set_ylim(-.5, .5)
            ax2.axhline(y=0, ls='-.', lw=.5, color='#000000')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.errorbar(x=plot_dict[feature]['x'],
                         y=plot_dict[feature]['y'],
                         yerr = plot_dict[feature]['yerr'],
                         ls = 'None',
                         color=f_color)
            ax2.scatter(x=plot_dict[feature]['x'],
                        y=plot_dict[feature]['y'],
                        s=20,
                        color=f_color)
            if f_idx < 3:
                ax.plot(range(1500), feature_dict['baseline'][example_session_baseline]['examples']['ego3_head_ang'][5000:6500, f_idx],
                        color=f_color, ls='--')
                ax.plot(range(1500), feature_dict['test'][example_session_test]['examples']['ego3_head_ang'][5000:6500, f_idx],
                        color=f_color)
            elif f_idx == 3:
                ax.plot(range(1500), feature_dict['baseline'][example_session_baseline]['examples']['sorted_point_data'][5000:6500, 4, 2]*100,
                        color=f_color, ls='--')
                ax.plot(range(1500), feature_dict['test'][example_session_test]['examples']['sorted_point_data'][5000:6500, 4, 2]*100,
                        color=f_color)
            else:
                ax.plot(range(1500), feature_dict['baseline'][example_session_baseline]['examples']['speeds'][5000:6500, 3],
                        color=f_color, ls='--')
                ax.plot(range(1500), feature_dict['test'][example_session_test]['examples']['speeds'][5000:6500, 3],
                        color=f_color)
            if f_idx == 0:
                ax.axhline(y=-50, xmin=.5, xmax=.58, ls='-', lw=1, color='#000000')
            ax.tick_params(labelsize=6)
            if f_idx < 2:
                ax.set_ylim(-75, 75)
            elif f_idx == 2:
                ax.set_ylim(-100, 100)
            elif f_idx == 3:
                ax.set_ylim(0, 25)
            elif f_idx == 4:
                ax.set_ylim(0, 25)
            ax.set_xlim(xmin=0)
            if f_idx < 3:
                ax.set_ylabel(f'{feature} ($^\circ$)', fontsize=6, labelpad=.5)
                ax.axhline(y=0, ls='-.', lw=.2, color='#000000')
            elif f_idx == 3:
                ax.set_ylabel(f'{feature} (cm)', fontsize=6, labelpad=.5)
            else:
                ax.set_ylabel(f'{feature} (cm/s)', fontsize=6, labelpad=.5)
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig.savefig(f'{self.save_dir}{os.sep}beh_differences.{self.fig_format}', dpi=300)
            else:
                print("Specified save directory doesn't exist. Try again.")
                sys.exit()
        plt.show()