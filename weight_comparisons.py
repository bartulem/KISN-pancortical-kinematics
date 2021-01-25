# -*- coding: utf-8 -*-

"""

@author: bartulem

Compare tuning-curve rate differences in weight/no-weight sessions.

"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from make_ratemaps import Ratemap

def extract_json_data(json_file, weight=True, feature='Speeds'):
    with open(json_file) as j_file:
        json_data = json.load(j_file)

    if weight:
        weight_dict = {feature: {'light1-light2': [], 'light1-weight': []},
                       f'{feature}_1st_der': {'light1-light2': [], 'light1-weight': []}}

    for cl_num in json_data.keys():
        if weight:
            for key in json_data[cl_num].keys():
                if key == feature or key == f'{feature}_1st_der':
                    weight_dict[key]['light1-light2'].append(json_data[cl_num][key]['light1-light2'])
                    weight_dict[key]['light1-weight'].append(json_data[cl_num][key]['light1-weight'])

    if weight:
        return weight_dict



class WeightComparer:

    def __init__(self, weight_json_file, chosen_feature):
        self.weight_json_file = weight_json_file
        self.chosen_feature = chosen_feature

    def plot_weight_results(self):
        weight_dict = extract_json_data(json_file=self.weight_json_file,
                                        feature=self.chosen_feature)
        min_joint = int(np.floor(np.min(weight_dict[self.chosen_feature]['light1-light2'] + weight_dict[self.chosen_feature]['light1-weight']) / 10.0)) * 10
        max_joint = int(np.ceil(np.max(weight_dict[self.chosen_feature]['light1-light2'] + weight_dict[self.chosen_feature]['light1-weight']) / 10.0)) * 10
        hist_color = [val for key, val in Ratemap.feature_colors.items() if key in self.chosen_feature][0]
        hist_bins = np.arange(-20, 20 + 5, 1)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax = plt.subplot(111)
        ax.hist(weight_dict[self.chosen_feature]['light1-light2'], bins=hist_bins, color=hist_color, edgecolor='#000000', alpha=.2, label='light1-light2')
        ax.hist(weight_dict[self.chosen_feature]['light1-weight'], bins=hist_bins, color=hist_color, edgecolor='#000000', alpha=.5, label='light1-weight')
        ax.set_xlabel('Firing rate difference (spikes/s)')
        ax.set_ylabel('Number of units')
        ax.legend(loc='best')
        plt.show()

        print(np.median(weight_dict[self.chosen_feature]['light1-light2']), np.median(weight_dict[self.chosen_feature]['light1-weight']))

