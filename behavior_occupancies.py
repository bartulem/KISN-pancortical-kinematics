# -*- coding: utf-8 -*-

"""

@author: jettucet

Make comparisons of behavioral occupancies between medicated and non-medicated animals.

"""

import numpy as np
from numba import njit


@njit(parallel=False)
def get_bins(feature_arr, min_val, max_val, num_bins_1d, camera_framerate):
    bin_edges = np.linspace(min_val, max_val, num_bins_1d + 1)
    occupancy = np.zeros(len(bin_edges) - 1)
    for i in range(1, len(bin_edges), 1):
        occupancy[i - 1] = np.sum((feature_arr > bin_edges[i - 1]) * (feature_arr <= bin_edges[i])) / camera_framerate
    return bin_edges, occupancy


"""class Behavior:

    variable_bounds = {'speeds': [0, 90], 'body_direction': [-180, 180], 'neck_elevation': [0, .35]}"""

file_name, data = Session(session=ls1[0]).data_loader(extract_variables=['body_direction', 'framerate'])

bin_edges, occ = behavior_occupancies.get_bins(data['body_direction'], -180, 180, 36, data['framerate'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x=range(len(bin_edges)-1), height=occ)
plt.show()