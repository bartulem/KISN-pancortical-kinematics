# -*- coding: utf-8 -*-

"""

@author: bartulem

Load spike data, bin and smooth.

"""

import numpy as np
from sessions2load import Session
from numba import njit
from numba import types
from numba.typed import Dict


@njit(parallel=False)
def get_shuffling_shifts(number_of_shuffles=1000, shuffle_range=(20, 60)):
    """
    Parameters
    ----------
    number_of_shuffles : int
        How many times to shuffle; defaults to 1000.
    shuffle_range : tuple
        Minimum and maximum number of seconds to shift the spike train; defaults to (20, 60).
    ----------

    Returns
    ----------
    seed_value : int64
        The specific seed for generating this set of random numbers.
    shuffle_shifts : np.ndarray
        The pseudorandom shifts for generating shuffled spike trains.
    ----------
    """

    # create a seed & seed the random number generator
    seed_value = np.random.randint(0, 2 ** 32 - 1)
    np.random.seed(seed_value)

    # get time shifts for every shuffle
    shuffle_shifts = np.random.uniform(shuffle_range[0], shuffle_range[1], size=(number_of_shuffles,))

    return seed_value, shuffle_shifts


@njit(parallel=False)
def purge_spikes_beyond_tracking(spike_train, tracking_ts):
    """
    Parameters
    ----------
    spike_train : np.ndarray
        Spike times in seconds.
    tracking_ts : np.ndarray (2, )
        The start and end of tracking relative to sessions start.
    ----------

    Returns
    ----------
    purged_spike_train : np.ndarray
        The spike train without spikes that precede or succeed tracking, relative to tracking start.
    ----------
    """

    # re-calculate spike times relative to tracking start
    purged_spike_train = spike_train - tracking_ts[0]

    # remove spikes that precede or succeed tracking
    purged_spike_train = purged_spike_train[(purged_spike_train >= 0) & (purged_spike_train < tracking_ts[1] - tracking_ts[0])]

    return purged_spike_train


@njit(parallel=False)
def convert_spikes_to_frame_events(purged_spike_train, frames_total, camera_framerate=120.):
    """
    Parameters
    ----------
    purged_spike_train : np.ndarray
        Spike times in seconds (relative to tracking start).
    frames_total : np.ndarray (2, )
        The start and end of tracking relative to sessions start.
    camera_framerate : np.float64
        The sampling frequency of the tracking system; defaults to 120.
    ----------

    Returns
    ----------
    spikes_frames : np.ndarray (frames_total, )
        How many spikes happened in each frame of tracking.
    ----------
    """

    # initialize an array of zeros with the size of the number of frames
    spikes_frames = np.zeros(frames_total)

    # covert spike times to frames when they happened
    spikes_tracking = purged_spike_train * camera_framerate
    spikes_tracking = np.floor(spikes_tracking, np.empty_like(spikes_tracking))

    # categorize spikes
    for frame in spikes_tracking:
        spikes_frames[frame] += 1

    return spikes_frames


class Spikes:

    def __init__(self, input_files):
        self.input_files = input_files

    def convert_activity_to_frames(self, **kwargs):

        """
        Parameters
        ----------
        **kwargs: dictionary
        get_clusters : str/int/list
            Cluster IDs to extract (if int, takes first n clusters; if 'all', takes all); defaults to 'all'.
        ----------

        Returns
        ----------
        spike_data : np.ndarray (frames_total, )
            How .
        ----------
        """

        get_clusters = kwargs['get_clusters'] if 'get_clusters' in kwargs.keys() \
                                                 and (kwargs['get_clusters'] == 'all' or type(kwargs['get_clusters']) == int or type(kwargs['get_clusters']) == list) else 'all'

        # get spike data in seconds and tracking start and end time
        extracted_data = Session(session_list=self.input_files).data_loader(extract_clusters=get_clusters, extract_variables=['tracking_ts'])

        #
        for file_id in extracted_data.keys():
            track_ts = extracted_data[file_id]['tracking_ts']
            extracted_activity = extracted_data[file_id]['cell_spikes']

            # initialize and fill in numba-type dictionary with purged spike trains
            numba_extracted_activity = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])

            for cell_id, spikes in extracted_activity.items():
                numba_extracted_activity[cell_id] = purge_spikes_beyond_tracking(spike_train=spikes, tracking_ts=track_ts)


























