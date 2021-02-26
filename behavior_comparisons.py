# -*- coding: utf-8 -*-

"""

@author: bartulem

Make comparisons of behavioral occupancies between medicated and non-medicated animals.

"""

import numpy as np
from numba import njit
from sessions2load import Session


@njit(parallel=False)
def get_bins(feature_arr, min_val, max_val, num_bins_1d, camera_framerate):
    bin_edges = np.linspace(min_val, max_val, num_bins_1d + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    occupancy = np.zeros(np.shape(bin_edges)[0] - 1)
    for i in range(1, np.shape(bin_edges)[0], 1):
        occupancy[i - 1] = np.sum((feature_arr > bin_edges[i - 1]) * (feature_arr <= bin_edges[i])) / camera_framerate
    return bin_edges, bin_centers, occupancy


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

    def __init__(self, data_file, variable_list):
        self.data_file = data_file
        self.variable_list = variable_list

    def pull_occ_histograms(self, **kwargs):
        """
        Description
        ----------
        This method gets the bin information to make occupancy histograms.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
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

        num_bins_1d = kwargs['num_bins_1d'] if 'num_bins_1d' in kwargs.keys() and type(kwargs['num_bins_1d']) == int else 36
        speed_idx = kwargs['speed_idx'] if 'speed_idx' in kwargs.keys() and type(kwargs['speed_idx']) == int else 3

        extract_variables = ['framerate', 'sorted_point_data']
        for variable in self.variable_list:
            if variable != 'neck_elevation':
                extract_variables.append(variable)

        file_name, data = Session(session=self.data_file).data_loader(extract_variables=extract_variables)

        occ_dict = {file_name: {}}
        for variable in self.variable_list:
            occ_dict[file_name][variable] = {}
            if variable != 'neck_elevation' and self.variable_bounds[variable].ndim == 1:
                occ_dict[file_name][variable]['bin_edges'], \
                occ_dict[file_name][variable]['bin_centers'], \
                occ_dict[file_name][variable]['occupancy'] = get_bins(feature_arr=data[variable],
                                                                      min_val=self.variable_bounds[variable][0],
                                                                      max_val=self.variable_bounds[variable][1],
                                                                      num_bins_1d=num_bins_1d,
                                                                      camera_framerate=data['framerate'])
            elif variable != 'neck_elevation' and self.variable_bounds[variable].ndim > 1 and 'speeds' not in variable:
                for eu_idx, euler_feature in enumerate(['roll', 'pitch', 'azimuth']):
                    occ_dict[file_name][variable][euler_feature] = {}
                    occ_dict[file_name][variable][euler_feature]['bin_edges'], \
                    occ_dict[file_name][variable][euler_feature]['bin_centers'], \
                    occ_dict[file_name][variable][euler_feature]['occupancy'] = get_bins(feature_arr=data[variable][:, eu_idx],
                                                                                         min_val=self.variable_bounds[variable][0, eu_idx],
                                                                                         max_val=self.variable_bounds[variable][1, eu_idx],
                                                                                         num_bins_1d=num_bins_1d,
                                                                                         camera_framerate=data['framerate'])
            elif variable != 'neck_elevation' and self.variable_bounds[variable].ndim > 1 and 'speeds' in variable:
                occ_dict[file_name][variable]['bin_edges'], \
                occ_dict[file_name][variable]['bin_centers'], \
                occ_dict[file_name][variable]['occupancy'] = get_bins(feature_arr=data[variable][:, speed_idx],
                                                                      min_val=self.variable_bounds[variable][0, speed_idx],
                                                                      max_val=self.variable_bounds[variable][1, speed_idx],
                                                                      num_bins_1d=num_bins_1d,
                                                                      camera_framerate=data['framerate'])
            else:
                occ_dict[file_name][variable]['bin_edges'], \
                occ_dict[file_name][variable]['bin_centers'], \
                occ_dict[file_name][variable]['occupancy'] = get_bins(feature_arr=data['sorted_point_data'][:, 4, 2],
                                                                      min_val=self.variable_bounds[variable][0],
                                                                      max_val=self.variable_bounds[variable][1],
                                                                      num_bins_1d=num_bins_1d,
                                                                      camera_framerate=data['framerate'])


        return occ_dict