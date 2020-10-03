# -*- coding: utf-8 -*-

"""

@author: bartulem

Load spike data, bin and smooth.

"""

import numpy as np
from sessions2load import Session
from numba import njit


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
def purge_spikes_beyond_tracking(spike_train, tracking_ts, full_purge=True):
    """
    Parameters
    ----------
    spike_train : np.ndarray
        Spike times in seconds.
    tracking_ts : np.ndarray (2, )
        The start and end of tracking relative to sessions start.
    full_purge : bool
        Remove spikes before and after tracking; defaults to True.
    ----------

    Returns
    ----------
    purged_spike_train : np.ndarray
        The spike train without spikes that precede or succeed tracking, relative to tracking start.
    ----------
    """

    if full_purge:
        # re-calculate spike times relative to tracking start
        purged_spike_train = spike_train - tracking_ts[0]

        # remove spikes that precede or succeed tracking
        purged_spike_train = purged_spike_train[(purged_spike_train >= 0) & (purged_spike_train < tracking_ts[1] - tracking_ts[0])]
    else:
        # remove spikes that succeed tracking
        purged_spike_train = spike_train[spike_train < tracking_ts[1] - tracking_ts[0]]

    return purged_spike_train


@njit(parallel=False)
def convert_spikes_to_frame_events(purged_spike_train, frames_total, camera_framerate=120.):
    """
    Parameters
    ----------
    purged_spike_train : np.ndarray
        Spike times in seconds (relative to tracking start).
    frames_total : int
        The total number of tracking frames in the recording.
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
        spikes_frames[int(frame)] += 1

    return spikes_frames


@njit(parallel=False)
def shuffle_spike_train(spike_train, random_shifts):
    """
    Parameters
    ----------
    spike_train : np.ndarray (number_of_spikes, )
        Spike times in seconds (relative to tracking start).
    random_shifts : np.ndarray (number_of_shuffles, )
        The pseudorandom shifts for generating shuffled spike trains.
    ----------

    Returns
    ----------
    shuffled_spike_train : (number_of_shuffles, number_of_spikes)
        The shuffled spike trains without spikes that precede or succeed tracking, relative to tracking start.
    ----------
    """

    # create array of zeroed values to store shuffled spikes in
    shuffled_spike_train_sec = np.zeros((random_shifts.shape[0], spike_train.shape[0]))

    # get shuffled spike time values
    for shuffle_idx in range(random_shifts.shape[0]):
        shuffled_spike_train_sec[shuffle_idx, :] = spike_train + random_shifts[shuffle_idx]

    return shuffled_spike_train_sec


class Spikes:

    # get shuffling shifts
    shuffle_seed, shuffle_shifts = get_shuffling_shifts()
    print(f"The pseudorandom number generator was seeded at {shuffle_seed}.")

    def __init__(self, input_files):
        self.input_files = input_files

    def convert_activity_to_frames_with_shuffles(self, **kwargs):

        """
        Parameters
        ----------
        **kwargs: dictionary
        get_clusters : str/int/list
            Cluster IDs to extract (if int, takes first n clusters; if 'all', takes all); defaults to 'all'.
        ----------

        Returns
        ----------
        activity_dictionary : dict
            A dictionary with frame-converted cluster activity and shuffled data.
        ----------
        """

        get_clusters = kwargs['get_clusters'] if 'get_clusters' in kwargs.keys() \
                                                 and (kwargs['get_clusters'] == 'all' or type(kwargs['get_clusters']) == int or type(kwargs['get_clusters']) == list) else 'all'

        # get spike data in seconds and tracking start and end time
        extracted_data = Session(session_list=self.input_files).data_loader(extract_clusters=get_clusters, extract_variables=['tracking_ts', 'framerate', 'total_frame_num'])

        # convert spike arrays to frame arrays
        activity_dictionary = {}
        for file_id in extracted_data.keys():
            activity_dictionary[file_id] = {}
            track_ts = extracted_data[file_id]['tracking_ts']
            extracted_activity = extracted_data[file_id]['cluster_spikes']
            empirical_camera_fr = extracted_data[file_id]['framerate']
            total_frame_num = extracted_data[file_id]['total_frame_num']

            for cell_id, spikes in extracted_activity.items():
                activity_dictionary[file_id][cell_id] = {}

                # eliminate spikes that happen prior to and post tracking
                purged_spikes_sec = purge_spikes_beyond_tracking(spike_train=spikes, tracking_ts=track_ts)

                # covert spikes to frame arrays
                activity_dictionary[file_id][cell_id]['activity'] = convert_spikes_to_frame_events(purged_spike_train=purged_spikes_sec,
                                                                                                   frames_total=total_frame_num,
                                                                                                   camera_framerate=empirical_camera_fr)
                activity_dictionary[file_id][cell_id]['shuffled'] = np.zeros((Spikes.shuffle_shifts.shape[0], total_frame_num))

                # shuffle the purged spike train N times
                shuffled_spikes_sec = shuffle_spike_train(purged_spikes_sec, Spikes.shuffle_shifts)

                # convert shuffles to frame arrays
                for shuffle_idx in range(shuffled_spikes_sec.shape[0]):
                    purged_shuffle = purge_spikes_beyond_tracking(spike_train=shuffled_spikes_sec[shuffle_idx, :], tracking_ts=track_ts, full_purge=False)
                    activity_dictionary[file_id][cell_id]['shuffled'][shuffle_idx, :] = convert_spikes_to_frame_events(purged_spike_train=purged_shuffle,
                                                                                                                       frames_total=total_frame_num,
                                                                                                                       camera_framerate=empirical_camera_fr)

        return activity_dictionary
