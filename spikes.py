# -*- coding: utf-8 -*-

"""

@author: bartulem

Load spike data, bin and smooth.

"""

import numpy as np
from sessions2load import Session
from numba import njit

@njit(parallel=False)
def shuffle_spike_trains(cell_spikes,
                         number_of_shuffles,
                         shuffle_range,
                         total_frames,
                         camera_framerate):

    """
    Parameters
    ----------
    cell_spikes : np.ndarray
        Spike train np.ndarray in tracking frames.
    number_of_shuffles : int
        How many times to shuffle.
    shuffle_range : tuple
        Minimum and maximum number of seconds to shift the spike train.
    total_frames : int
        Total number of tracking frames in the recording.
    camera_framerate : float
        The sampling frequency of the tracking system.
    ----------

    Returns
    ----------
    shuffled_data : np.ndarray
        The shuffled spike train (shuffles x frames).
    ----------
    """

    # initialize shuffled_data as zeroed np.ndarray
    shuffled_data = np.zeros((number_of_shuffles, total_frames))

    # create a seed & seed the random number generator
    seed_value = np.random.randint(0, 2**32 - 1)
    np.random.seed(seed_value)

    # get frame shifts for every shuffle
    random_time_shifts = np.random.uniform(shuffle_range[0], shuffle_range[1], size=(number_of_shuffles, ))
    random_frame_shifts = random_time_shifts*camera_framerate
    random_frame_shifts = np.round_(random_frame_shifts, 0, np.empty_like(random_frame_shifts))

    # conduct the shuffling
    for shuffle in range(number_of_shuffles):
        shuffled_data[shuffle, random_frame_shifts[shuffle]:] = cell_spikes[:total_frames-random_frame_shifts[shuffle]]

    return shuffled_data


class Activity():

    def __init__(self):
        pass
