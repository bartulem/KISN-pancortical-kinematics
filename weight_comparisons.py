# -*- coding: utf-8 -*-

"""

@author: bartulem

Compare tuning-curve rate differences in weight/no-weight sessions.

"""

import numpy as np
import os
import scipy.io

# data[0,:] = xvals
# data[1,:] = raw rate map
# data[2,:] = occupancy
# data[3,:] = smoothed rate map
# data[4,:] = shuffled mean
# data[5,:] = shuffled std
# data[6,:] = smoothed occupancy


class WeightComparer:

    def __init__(self, light_one, weight_one, light_two, variable_list):
        self.light_one = light_one
        self.weight_one = weight_one
        self.light_two = light_two
        self.variable_list = variable_list

    def plot_comparisons(self, **kwargs):

        # locate the designated files
        files_dict = {'light_one': [], 'weight_one': [], 'light_two': []}
        for session_list, key_type in zip([self.light_one, self.weight_one, self.light_two], files_dict.keys()):
            for one_dict in session_list:
                for one_file in sorted(os.listdir(one_dict['directory'])):
                    if one_dict['session_date'] in one_file and one_dict['session_id'] in one_file and one_dict['session_type'] in one_file:
                        files_dict['{}'.format(key_type)].append('{}{}{}'.format(one_dict['directory'], os.sep, one_file))

        for file_index, one_file in enumerate(files_dict['light_one']):
            mat_file = scipy.io.loadmat(one_file)
            for one_key in mat_file.keys():
                if 'data' in one_key and one_key.split('-')[1] in self.variable_list:
                    print(one_key)


light_one = [{'directory': '/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020', 'session_date': '040520', 'session_id': 's2', 'session_type': 'light'}]
weight_one = [{'directory': '/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020', 'session_date': '040520', 'session_id': 's1', 'session_type': 'weight'}]
light_two = [{'directory': '/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020', 'session_date': '050520', 'session_id': 's4', 'session_type': 'light'}]
variable_list = ['Allo_head_azimuth', 'Allo_head_pitch', 'Allo_head_roll', 'Neck_elevation', 'Back_pitch', 'Back_azimuth', 'Speeds']

classWeight = WeightComparer(light_one, weight_one, light_two, variable_list)
classWeight.plot_comparisons()
