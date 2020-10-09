# -*- coding: utf-8 -*-

"""

@author: bartulem

Load spike data, bin and smooth.

"""

import numpy as np
from sessions2load import Session
from numba import njit
from scipy.ndimage.filters import gaussian_filter1d


def gaussian_smoothing(array, sigma=1, axis=1):
    """
    Parameters
    ----------
    array : np.ndarray
        The input array to be smoothed.
    sigma : int
        The SD of the smoothing window; defaults to 1.
    axis : int
        The filter smooths in 1D, so you choose the axis; defaults to 1.
    ----------

    Returns
    ----------
    smoothed_array : np.ndarray
        The 1D smoothed input array.
    ----------
    """

    return gaussian_filter1d(input=array, sigma=sigma, axis=axis)


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
    shuffled_spike_train : np.ndarray (number_of_shuffles, number_of_spikes)
        The shuffled spike trains without spikes that precede or succeed tracking, relative to tracking start.
    ----------
    """

    # create array of zeroed values to store shuffled spikes in
    shuffled_spike_train_sec = np.zeros((random_shifts.shape[0], spike_train.shape[0]))

    # get shuffled spike time values
    for shuffle_idx in range(random_shifts.shape[0]):
        shuffled_spike_train_sec[shuffle_idx, :] = spike_train + random_shifts[shuffle_idx]

    return shuffled_spike_train_sec


@njit(parallel=False)
def find_event_starts(event_array, return_all=True,
                      camera_framerate=120.,
                      expected_event_duration=5.,
                      min_inter_event_interval=10.):
    """
    Parameters
    ----------
    event_array : np.ndarray (frames_total, )
        The array of events (should be binary, i.e. 0/1).
    return_all : bool
        Return all event starts, irrespective of duration; defaults to True.
    camera_framerate : np.float64
        The sampling frequency of the tracking system; defaults to 120.
    expected_event_duration : int/float
        The expected duration of the designated event; defaults to 5 (seconds).
    min_inter_event_interval : int/float
        The minimum interval between any two adjacent events; defaults to 10 (seconds).
    ----------

    Returns
    ----------
    event_start_frames: np.ndarray
        Every frame ON (1) start in the input array.
    ----------
    """

    event_change_points = np.where(np.roll(event_array, 1) != event_array)[0]
    event_start_frames = event_change_points[::2]

    if not return_all:
        # this returns only events that satisfy: expected_event_duration - .1 < duration < expected_event_duration + .1
        event_end_frames = event_change_points[1::2]
        event_durations = (event_end_frames - event_start_frames) / camera_framerate
        inter_event_intervals = np.concatenate((np.array([min_inter_event_interval + .1]),
                                                (event_start_frames[1:] - event_start_frames[:-1]) / camera_framerate))
        event_start_frames = event_start_frames[(event_durations > (expected_event_duration - .1))
                                                & (event_durations < (expected_event_duration + .1))
                                                & (inter_event_intervals > min_inter_event_interval)]

    return event_start_frames


@njit(parallel=False)
def calculate_peth(activity, event_start_frames,
                   bin_size_ms=50, window_size=10,
                   camera_framerate=120.):
    """
    Parameters
    ----------
    activity : np.ndarray
        Arrays with spikes allocated to tracking frames.
    event_start_frames : np.ndarray
        Every frame ON (1) start in the session.
    bin_size_ms : int
        The bin size of the PETH; defaults to 50 (ms).
    window_size : int
        The unilateral window size; defaults to 10 (seconds).
    camera_framerate : np.float64
        The sampling frequency of the tracking system; defaults to 120.
    ----------

    Returns
    ----------
    peth_array : np.ndarray (epoch_num, total_window)
        Peri-event time histogram.
    ----------
    """

    # convert bin size to seconds
    bin_size = bin_size_ms / 1e3

    # get bin step (number of frames in each bin)
    bin_step = int(round(camera_framerate * bin_size))

    # get total window
    window_one_side = int(round((window_size / bin_size)))
    total_window = 2 * window_one_side

    # calculate PETH
    peth_array = np.zeros((event_start_frames.shape[0], total_window))
    for epoch in range(event_start_frames.shape[0]):
        window_start_bin = int(round(event_start_frames[epoch] - (bin_step * window_one_side)))
        for one_bin in range(total_window):
            peth_array[epoch, one_bin] = np.sum(activity[window_start_bin:window_start_bin + bin_step]) / bin_size
            window_start_bin += bin_step

    return peth_array


class Spikes:
    # get shuffling shifts
    shuffle_seed, shuffle_shifts = get_shuffling_shifts()
    print(f"The pseudorandom number generator was seeded at {shuffle_seed}.")

    def __init__(self, input_file=0):
        self.input_file = input_file

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
        file_id, extracted_data = Session(session=self.input_file).data_loader(extract_clusters=get_clusters, extract_variables=['tracking_ts', 'framerate', 'total_frame_num'])

        # convert spike arrays to frame arrays
        activity_dictionary = {}
        track_ts = extracted_data['tracking_ts']
        extracted_activity = extracted_data['cluster_spikes']
        empirical_camera_fr = extracted_data['framerate']
        total_frame_num = extracted_data['total_frame_num']

        for cell_id, spikes in extracted_activity.items():
            activity_dictionary[cell_id] = {}

            # eliminate spikes that happen prior to and post tracking
            purged_spikes_sec = purge_spikes_beyond_tracking(spike_train=spikes, tracking_ts=track_ts)

            # covert spikes to frame arrays
            activity_dictionary[cell_id]['activity'] = convert_spikes_to_frame_events(purged_spike_train=purged_spikes_sec,
                                                                                      frames_total=total_frame_num,
                                                                                      camera_framerate=empirical_camera_fr)
            activity_dictionary[cell_id]['shuffled'] = np.zeros((Spikes.shuffle_shifts.shape[0], total_frame_num))

            # shuffle the purged spike train N times
            shuffled_spikes_sec = shuffle_spike_train(purged_spikes_sec, Spikes.shuffle_shifts)

            # convert shuffles to frame arrays
            for shuffle_idx in range(shuffled_spikes_sec.shape[0]):
                purged_shuffle = purge_spikes_beyond_tracking(spike_train=shuffled_spikes_sec[shuffle_idx, :], tracking_ts=track_ts, full_purge=False)
                activity_dictionary[cell_id]['shuffled'][shuffle_idx, :] = convert_spikes_to_frame_events(purged_spike_train=purged_shuffle,
                                                                                                          frames_total=total_frame_num,
                                                                                                          camera_framerate=empirical_camera_fr)

        return file_id, activity_dictionary

    def get_peths(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs: dictionary
        get_clusters : str/int/list
            Cluster IDs to extract (if int, takes first n clusters; if 'all', takes all); defaults to 'all'.
        bin_size_ms : int
            The bin size of the PETH; defaults to 50 (ms).
        window_size : int
            The unilateral window size; defaults to 10 (seconds).
        return_all : bool
            Return all event starts, irrespective of duration; defaults to True.
        expected_event_duration : int/float
            The expected duration of the designated event; defaults to 5 (seconds).
        min_inter_event_interval : int/float
            The minimum interval between any two adjacent events; defaults to 10 (seconds).
        smooth : bool
            Smooth PETHs; defaults to False.
        smooth_sd : int
            The SD of the smoothing window; defaults to 1 (bin).
        smooth_axis : int
            The smoothing axis in a 2D array; defaults to 1 (smooths rows).
        ----------

        Returns
        ----------
        peth_dictionary : dict
            Peri-event time histogram for all clusters (np.ndarray (epoch_num, total_window)).
        ----------
        """

        get_clusters = kwargs['get_clusters'] if 'get_clusters' in kwargs.keys() \
                                                 and (kwargs['get_clusters'] == 'all' or type(kwargs['get_clusters']) == int or type(kwargs['get_clusters']) == list) else 'all'

        bin_size_ms = kwargs['bin_size_ms'] if 'bin_size_ms' in kwargs.keys() and type(kwargs['bin_size_ms']) == int else 50
        window_size = kwargs['window_size'] if 'window_size' in kwargs.keys() and type(kwargs['window_size']) == int else 10
        return_all = kwargs['return_all'] if 'return_all' in kwargs.keys() and type(kwargs['return_all']) == bool else True
        expected_event_duration = kwargs['expected_event_duration'] if 'expected_event_duration' in kwargs.keys() \
                                                                       and (type(kwargs['expected_event_duration']) == int or type(kwargs['expected_event_duration']) == float) else 5
        min_inter_event_interval = kwargs['min_inter_event_interval'] if 'min_inter_event_interval' in kwargs.keys() \
                                                                         and (type(kwargs['min_inter_event_interval']) == int or type(kwargs['min_inter_event_interval']) == float) else 10
        smooth = kwargs['smooth'] if 'smooth' in kwargs.keys() and type(kwargs['smooth']) == bool else False
        smooth_sd = kwargs['smooth_sd'] if 'smooth_sd' in kwargs.keys() and type(kwargs['smooth_sd']) == int else 1
        smooth_axis = kwargs['smooth_axis'] if 'smooth_axis' in kwargs.keys() and type(kwargs['smooth_axis']) == int else 1

        ses_name, session_vars = Session(session=self.input_file).data_loader(extract_variables=['imu_sound', 'framerate'])

        file_id, activity_dictionary = self.convert_activity_to_frames_with_shuffles(get_clusters=get_clusters)

        event_start_frames = find_event_starts(session_vars['imu_sound'],
                                               return_all=return_all,
                                               camera_framerate=session_vars['framerate'],
                                               expected_event_duration=expected_event_duration,
                                               min_inter_event_interval=min_inter_event_interval)

        peth_dictionary = {}
        for cell_id in activity_dictionary.keys():
            peth_dictionary[cell_id] = {}
            peth_array = calculate_peth(activity_dictionary[cell_id]['activity'],
                                        event_start_frames,
                                        bin_size_ms=bin_size_ms,
                                        window_size=window_size,
                                        camera_framerate=session_vars['framerate'])
            if smooth:
                peth_dictionary[cell_id]['peth'] = gaussian_smoothing(array=peth_array, sigma=smooth_sd, axis=smooth_axis)
            else:
                peth_dictionary[cell_id]['peth'] = peth_array

        return peth_dictionary
