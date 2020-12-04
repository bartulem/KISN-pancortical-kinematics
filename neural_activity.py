# -*- coding: utf-8 -*-

"""

@author: bartulem

Load spike data, bin and smooth.

"""

import sys
import sparse
import warnings
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from numba import njit
from sessions2load import Session
import decode_events

warnings.simplefilter('ignore')


def gaussian_smoothing(array, sigma=1, axis=1):
    """
    Parameters
    ----------
    array : np.ndarray
        The input array to be smoothed.
    sigma : int
        The SD of the smoothing window; defaults to 1 (bin).
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

    # convert spike times to frames when they happened
    spikes_tracking = purged_spike_train * camera_framerate
    spikes_tracking = np.floor(spikes_tracking, np.empty_like(spikes_tracking))

    # categorize spikes
    for frame in spikes_tracking:
        spikes_frames[int(frame)] += 1

    return spikes_frames


@njit(parallel=False)
def condense_frame_arrays(frame_array, camera_framerate=120.,
                          bin_size_ms=100, arr_type=True):

    """
    Parameters
    ----------
    frame_array : np.ndarray (frames_total, )
        The input frame array.
    bin_size_ms : int
        The bin size of the PETH; defaults to 100 (ms).
    camera_framerate : np.float64
        The sampling frequency of the tracking system; defaults to 120.
    arr_type : bool
        True if it's a spike array, False if it's the sound array; defaults to True.
    ----------

    Returns
    ----------
    new_arr : np.ndarray
        The frame array with the reduced shape.
    ----------
    """

    total_frames = frame_array.shape[0]

    # calculate size of new frame
    step = int(camera_framerate * (bin_size_ms / 1000))
    new_shape = total_frames // step
    new_arr = np.zeros(new_shape)

    # fill it in
    ls_iter = list(range(0, new_shape*step, step))
    for idx, one_bin in enumerate(ls_iter):
        array_excerpt = frame_array[one_bin:one_bin+step]
        if arr_type:
            new_arr[idx] = array_excerpt.sum()
        else:
            new_arr[idx] = 1 if array_excerpt.sum() >= (step / 2) else 0

    return new_arr


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
def calculate_peth(input_array, event_start_frames,
                   bin_size_ms=50, window_size=10,
                   camera_framerate=120.,
                   behavior_input=False):
    """
    Parameters
    ----------
    input_array : np.ndarray
        Arrays with spikes/behavior allocated to tracking frames.
    event_start_frames : np.ndarray
        Every frame ON (1) start in the session.
    bin_size_ms : int
        The bin size of the PETH; defaults to 50 (ms).
    window_size : int
        The unilateral window size; defaults to 10 (seconds).
    camera_framerate : np.float64
        The sampling frequency of the tracking system; defaults to 120.
    behavior_input : bool
        Whether or not the input array is behavioral; defaults to False.
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
            if behavior_input:
                if window_start_bin < 0:
                    peth_array[epoch, one_bin] = np.nan
                else:
                    peth_array[epoch, one_bin] = np.nanmean(input_array[window_start_bin:window_start_bin + bin_step])
            else:
                if window_start_bin < 0:
                    peth_array[epoch, one_bin] = np.nan
                else:
                    peth_array[epoch, one_bin] = np.sum(input_array[window_start_bin:window_start_bin + bin_step]) / bin_size
            window_start_bin += bin_step

    return peth_array


@njit(parallel=False)
def calculate_discontinuous_peth(input_array_lst, esf, event_number,
                                 bin_size_ms=50, window_size=6,
                                 camera_framerate=120.):
    """
    Parameters
    ----------
    input_array_lst : list
        List of session arrays with spikes allocated to tracking frames.
    esf : list
        List of session behavior arrays.
    event_number : int
        Number of events to consider.
    bin_size_ms : int
        The bin size of the PETH; defaults to 50 (ms).
    window_size : int
        The complete window size; defaults to 6 (seconds).
    camera_framerate : np.float64
        The sampling frequency of the tracking system; defaults to 120.
    ----------

    Returns
    ----------
    peth_array : np.ndarray (event_number, total_window)
        Peri-event time histogram.
    ----------
    """

    # convert bin size to seconds
    bin_size = bin_size_ms / 1e3

    # get bin step (number of frames in each bin)
    bin_step = int(round(camera_framerate * bin_size))

    # get total window
    total_window = int(round((window_size / bin_size)))
    switch_points = np.arange(0, total_window, total_window // 3)

    # calculate PETH
    peth_array = np.zeros((event_number, total_window))
    for arr_idx, arr in enumerate(input_array_lst):
        for epoch in range(event_number):
            window_start_bin = int(round(esf[arr_idx][epoch]))
            for one_bin in range(total_window // 3):
                real_bin = one_bin + switch_points[arr_idx]
                peth_array[epoch, real_bin] = np.sum(arr[window_start_bin:window_start_bin + bin_step]) / bin_size
                window_start_bin += bin_step

    return peth_array


@njit(parallel=False)
def raster_preparation(purged_spike_train, event_start_frames,
                       camera_framerate=120., window_size=10):
    """
    Parameters
    ----------
    purged_spike_train : np.ndarray
        The spike train without spikes that precede or succeed tracking, relative to tracking start.
    event_start_frames : np.ndarray
        Every frame ON (1) start in the session.
    camera_framerate : np.float64
        The sampling frequency of the tracking system; defaults to 120.
    window_size : int
        The unilateral window size; defaults to 10 (seconds).
    ----------

    Returns
    ----------
    raster_list : list
        List of raster events (np.ndarrays) for that spike train.
    ----------
    """

    raster_list = []

    for event in event_start_frames:
        window_start_seconds = (event / camera_framerate) - window_size
        window_centered_spikes = purged_spike_train[(purged_spike_train >= window_start_seconds)
                                                    & (purged_spike_train < window_start_seconds+(window_size*2))] - window_start_seconds
        raster_list.append(window_centered_spikes[window_centered_spikes > 0])

    return raster_list


def discontinuous_raster_preparation(purged_spike_arr, event_start_arr, event_number,
                                     camera_framerate_arr, window_size=2):
    """
    Parameters
    ----------
    purged_spike_arr : np.ndarray
        An array of spike trains without spikes that precede or succeed tracking, relative to tracking start.
    event_start_arr : np.ndarray
        An array of every start frame of speed within a specified range.
    event_number : int
        Number of events to consider.
    camera_framerate_arr : np.ndarray
        An array with camera sampling frequencies for all sessions.
    window_size : int
        The unilateral window size; defaults to 2 (seconds).
    ----------

    Returns
    ----------
    raster_list : list
        List of raster events (np.ndarrays) for that spike train.
    ----------
    """

    raster_list = []

    for event_idx in range(event_number):
        temp_raster_list = []
        for session_idx, session in enumerate(event_start_arr):
            if len(purged_spike_arr[session_idx]) > 0:
                purged_spike_train = purged_spike_arr[session_idx]
                window_start_seconds = (session[event_idx] / camera_framerate_arr[session_idx])
                window_centered_spikes = purged_spike_train[(purged_spike_train >= window_start_seconds)
                                                            & (purged_spike_train < window_start_seconds+window_size)] - window_start_seconds + (session_idx*2)
                for spike in window_centered_spikes:
                    temp_raster_list.append(spike)
        raster_list.append(np.array(temp_raster_list))

    return raster_list


@njit(parallel=False)
def find_variable_sequences(variable, threshold_low=0., threshold_high=5.,
                            min_seq_duration=2, camera_framerate=120.):

    """
    Parameters
    ----------
    variable : np.ndarray
        The spike train without spikes that precede or succeed tracking, relative to tracking start.
    threshold_low : int/float
        Value above which variable should be considered; defaults to 0.
    threshold_high : int/float
        Value above which variable should not be considered; defaults to 5.
    min_seq_duration : int/float
        The minimum duration for chosen sequences; defaults to 2 (seconds).
    camera_framerate : np.float64
        The sampling frequency of the tracking system; defaults to 120.
    ----------

    Returns
    ----------
    seq_starts : np.ndarray
        An array of sequence starts for the designated variable.
    ----------
    """

    # transform sequence duration to bins
    min_seq_duration = int(round(min_seq_duration*camera_framerate))

    indices_above_threshold = np.where((threshold_low <= variable) & (variable <= threshold_high))[0]
    seq_starts = []
    for idx, item in enumerate(indices_above_threshold):
        # both idx and item need to be below array length minus min_seq_duration
        idx_truth = idx <= indices_above_threshold.shape[0]-min_seq_duration
        item_truth = item <= variable.shape[0]-min_seq_duration
        if idx_truth and item_truth \
                and (np.arange(item, item+min_seq_duration, 1) == indices_above_threshold[idx:idx+min_seq_duration]).all():
            if len(seq_starts) == 0:
                seq_starts.append(item)
            else:
                if item > seq_starts[-1]+(min_seq_duration*2):
                    seq_starts.append(item)

    return np.array(seq_starts).astype(np.int32)


class Spikes:
    # get shuffling shifts
    shuffle_seed, shuffle_shifts = get_shuffling_shifts()
    print(f"The pseudorandom number generator was seeded at {shuffle_seed}.")

    def __init__(self, input_file='', purged_spikes_dictionary='', input_012=['', '', ''],
                 cluster_groups_dir='/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020/cluster_groups_info',
                 sp_profiles_csv='/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020/spiking_profiles/spiking_profiles.csv'):
        self.input_file = input_file
        self.purged_spikes_dictionary = purged_spikes_dictionary
        self.input_012 = input_012
        self.cluster_groups_dir = cluster_groups_dir
        self.sp_profiles_csv = sp_profiles_csv

    def convert_activity_to_frames_with_shuffles(self, **kwargs):
        """
        Description
        ----------
        This method converts cluster spiking activity into trains that match the tracking
        resolution, as spikes are allocated to appropriate frames. It returns such spike
        trains both for true and shuffled data.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        get_clusters (str / int / list)
            Cluster IDs to extract (if int, takes first n clusters; if 'all', takes all); defaults to 'all'.
        to_shuffle (bool)
            Yey or ney on shuffling; defaults to False.
        condense_arr (bool)
            Yey or ney on the condensing (reducing the number of bins); defaults to False.
        ----------

        Returns
        ----------
        file_info (str)
            The shortened version of the file name.
        activity_dictionary (dict)
            A dictionary with frame-converted cluster activity and shuffled data.
        ----------
        """

        get_clusters = kwargs['get_clusters'] if 'get_clusters' in kwargs.keys() \
                                                 and (kwargs['get_clusters'] == 'all' or type(kwargs['get_clusters']) == int or type(kwargs['get_clusters']) == list) else 'all'
        to_shuffle = kwargs['to_shuffle'] if 'to_shuffle' in kwargs.keys() and type(kwargs['to_shuffle']) == bool else False
        condense_arr = kwargs['condense_arr'] if 'condense_arr' in kwargs.keys() and type(kwargs['condense_arr']) == bool else False

        # get spike data in seconds and tracking start and end time
        file_id, extracted_data = Session(session=self.input_file).data_loader(extract_clusters=get_clusters, extract_variables=['tracking_ts', 'framerate', 'total_frame_num'])

        # convert spike arrays to frame arrays
        activity_dictionary = {}
        purged_spikes_dictionary = {}
        track_ts = extracted_data['tracking_ts']
        extracted_activity = extracted_data['cluster_spikes']
        empirical_camera_fr = extracted_data['framerate']
        total_frame_num = extracted_data['total_frame_num']

        for cell_id, spikes in extracted_activity.items():
            activity_dictionary[cell_id] = {}

            # eliminate spikes that happen prior to and post tracking
            purged_spikes_sec = purge_spikes_beyond_tracking(spike_train=spikes, tracking_ts=track_ts)
            purged_spikes_dictionary[cell_id] = purged_spikes_sec

            # covert spikes to frame arrays
            cell_id_activity = convert_spikes_to_frame_events(purged_spike_train=purged_spikes_sec,
                                                              frames_total=total_frame_num,
                                                              camera_framerate=empirical_camera_fr)

            if not condense_arr:
                activity_dictionary[cell_id]['activity'] = sparse.COO(cell_id_activity).astype(np.int16)
            else:
                activity_dictionary[cell_id]['activity'] = sparse.COO(condense_frame_arrays(frame_array=cell_id_activity)).astype(np.int16)

            if to_shuffle:
                activity_dictionary[cell_id]['shuffled'] = {}

                # shuffle the purged spike train N times
                shuffled_spikes_sec = shuffle_spike_train(purged_spikes_sec, Spikes.shuffle_shifts)

                # convert shuffles to frame arrays
                for shuffle_idx in range(shuffled_spikes_sec.shape[0]):
                    purged_shuffle = purge_spikes_beyond_tracking(spike_train=shuffled_spikes_sec[shuffle_idx, :], tracking_ts=track_ts, full_purge=False)
                    shuffle_cell_id = convert_spikes_to_frame_events(purged_spike_train=purged_shuffle,
                                                                     frames_total=total_frame_num,
                                                                     camera_framerate=empirical_camera_fr)
                    if not condense_arr:
                        activity_dictionary[cell_id]['shuffled'][shuffle_idx] = sparse.COO(shuffle_cell_id).astype(np.int16)
                    else:
                        activity_dictionary[cell_id]['shuffled'][shuffle_idx] = sparse.COO(condense_frame_arrays(frame_array=shuffle_cell_id)).astype(np.int16)

        return file_id, activity_dictionary, purged_spikes_dictionary

    def get_peths(self, **kwargs):
        """
        Description
        ----------
        This method converts cluster spiking activity into peri-event time histograms (PETHs),
        where you have the option to define bin and window size. NB: As of yet, it is NOT set
        to do the same for shuffled spike data (but it's a simple fix).

        Details: Each spike train is zeroed to tracking start and purged of spikes that exceed
        those boundaries. The spike train is then binned to match the tracking resolution, and
        spike counts are allocated to the appropriate frames. These spike counts are further
        binned (50 ms) to encompass a window (10 s) before and after every event onset (the
        start of the white noise stimulation). Rates are calculated and smoothed with a 3 bin
        Gaussian kernel. Raster arrays are prepared by zeroing spike times to each start of the
        trial window. Behavioral peths bin and compute the status of any given behavioral feature
        around relevant events (NB: works only for speed as of yet).
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        get_clusters (str / int / list)
            Cluster IDs to extract (if int, takes first n clusters; if 'all', takes all); defaults to 'all'.
        bin_size_ms (int)
            The bin size of the PETH; defaults to 50 (ms).
        window_size (int / float)
            The unilateral window size; defaults to 10 (seconds).
        return_all (bool)
            Return all event starts, irrespective of duration; defaults to True.
        expected_event_duration (int / float)
            The expected duration of the designated event; defaults to 5 (seconds).
        min_inter_event_interval (int / float)
            The minimum interval between any two adjacent events; defaults to 10 (seconds).
        smooth (bool)
            Smooth PETHs; defaults to False.
        smooth_sd (int)
            The SD of the smoothing window; defaults to 1 (bin).
        smooth_axis (int)
            The smoothing axis in a 2D array; defaults to 1 (smooths within rows).
        raster (bool)
            Prepare arrays from making raster plots; defaults to False.
        beh_raster (str / bool)
            Prepare behavior arrays from making raster plots; defaults to False.
        ----------

        Returns
        ----------
        peth_dictionary (dict)
            Peri-event time histogram for all clusters (np.ndarray (epoch_num, total_window)).
        raster_dictionary (dict)
            Raster arrays for all clusters zeroed to window start.
        peth_beh (np.ndarray)
            Peri-event time histogram for the designated behavioral feature (np.ndarray (epoch_num, total_window)).
        ----------
        """

        get_clusters = kwargs['get_clusters'] if 'get_clusters' in kwargs.keys() \
                                                 and (kwargs['get_clusters'] == 'all' or type(kwargs['get_clusters']) == int or type(kwargs['get_clusters']) == list) else 'all'
        bin_size_ms = kwargs['bin_size_ms'] if 'bin_size_ms' in kwargs.keys() and type(kwargs['bin_size_ms']) == int else 50
        window_size = kwargs['window_size'] if 'window_size' in kwargs.keys() and (type(kwargs['window_size']) == int or type(kwargs['window_size']) == float) else 10
        return_all = kwargs['return_all'] if 'return_all' in kwargs.keys() and type(kwargs['return_all']) == bool else True
        expected_event_duration = kwargs['expected_event_duration'] if 'expected_event_duration' in kwargs.keys() \
                                                                       and (type(kwargs['expected_event_duration']) == int or type(kwargs['expected_event_duration']) == float) else 5
        min_inter_event_interval = kwargs['min_inter_event_interval'] if 'min_inter_event_interval' in kwargs.keys() \
                                                                         and (type(kwargs['min_inter_event_interval']) == int or type(kwargs['min_inter_event_interval']) == float) else 10
        smooth = kwargs['smooth'] if 'smooth' in kwargs.keys() and type(kwargs['smooth']) == bool else False
        smooth_sd = kwargs['smooth_sd'] if 'smooth_sd' in kwargs.keys() and type(kwargs['smooth_sd']) == int else 1
        smooth_axis = kwargs['smooth_axis'] if 'smooth_axis' in kwargs.keys() and type(kwargs['smooth_axis']) == int else 1
        raster = kwargs['raster'] if 'raster' in kwargs.keys() and type(kwargs['raster']) == bool else False
        beh_raster = kwargs['beh_raster'] if 'beh_raster' in kwargs.keys() and ( type(kwargs['beh_raster']) == str or type(kwargs['beh_raster']) == bool) else False

        # extract relevant variables / clusters from session data
        get_variables = ['imu_sound', 'framerate']
        if type(beh_raster) == str:
            get_variables.append(beh_raster)
        ses_name, session_vars = Session(session=self.input_file).data_loader(extract_variables=get_variables)

        # get activity converted to frames
        file_id, activity_dictionary, purged_spikes_dictionary = self.convert_activity_to_frames_with_shuffles(get_clusters=get_clusters)

        # get event start frames
        event_start_frames = find_event_starts(session_vars['imu_sound'],
                                               return_all=return_all,
                                               camera_framerate=session_vars['framerate'],
                                               expected_event_duration=expected_event_duration,
                                               min_inter_event_interval=min_inter_event_interval)

        # get raster plot
        if raster:
            raster_dictionary = {}
            for cell_id, purged_spikes in purged_spikes_dictionary.items():
                if cell_id in get_clusters:
                    raster_dictionary[cell_id] = raster_preparation(purged_spike_train=purged_spikes,
                                                                    event_start_frames=event_start_frames,
                                                                    camera_framerate=session_vars['framerate'],
                                                                    window_size=window_size)

        # get PETHs for each cluster and smooth if necessary
        peth_dictionary = {}
        for cell_id in activity_dictionary.keys():
            peth_dictionary[cell_id] = {}
            peth_array = calculate_peth(input_array=activity_dictionary[cell_id]['activity'].todense().astype(np.float32),
                                        event_start_frames=event_start_frames,
                                        bin_size_ms=bin_size_ms,
                                        window_size=window_size,
                                        camera_framerate=session_vars['framerate'])
            if smooth:
                peth_dictionary[cell_id]['peth'] = gaussian_smoothing(array=peth_array, sigma=smooth_sd, axis=smooth_axis)
            else:
                peth_dictionary[cell_id]['peth'] = peth_array

        # get behavior for raster (nb: currently only works for speed)
        if type(beh_raster) == str:
            peth_beh = calculate_peth(input_array=session_vars[beh_raster][:, 3],
                                      event_start_frames=event_start_frames,
                                      bin_size_ms=bin_size_ms,
                                      window_size=window_size,
                                      camera_framerate=session_vars['framerate'],
                                      behavior_input=True)
            if smooth:
                peth_beh = gaussian_smoothing(array=peth_beh, sigma=smooth_sd, axis=smooth_axis)

        if raster and beh_raster is not False:
            return ses_name, peth_dictionary, raster_dictionary, peth_beh
        elif raster and beh_raster is False:
            return ses_name, peth_dictionary, raster_dictionary
        else:
            return ses_name, peth_dictionary

    def get_discontinuous_peths(self, **kwargs):
        """
        Description
        ----------
        This method converts cluster spiking activity into peri-event time histograms (PETHs),
        where you have the option to define bin and window size. It should be used to construct
        PETHs whose trial parts come from different sessions.

        Details: Each session spike train is zeroed to tracking start and purged of spikes that exceed
        those boundaries. The spike train is then binned to match the tracking resolution, and
        spike counts are allocated to the appropriate frames. These spike counts are further
        binned (50 ms) to encompass a window (2 s) after every event onset (NB: which for our purpose
        is a 2s window where the speed of the animal was < 5 cm/s) Rates are calculated and smoothed
        with a 3 bin Gaussian kernel for each session segment separately. Raster arrays are prepared
        by zeroing spike times to each start of the trial window.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        get_clusters (str / int / list)
            Cluster IDs to extract (if int, takes first n clusters; if 'all', takes all); defaults to 'all'.
        decode_what (str)
            What are you decoding; defaults to 'luminance'.
        cluster_areas (list)
            Cluster area(s) of choice; defaults to ['A'].
        cluster_type (str)
            Cluster type of choice; defaults to True.
        speed_threshold_low (int/float)
            Value above which variable should be considered; defaults to 0.
        speed_threshold_high (int/float)
            Value below which variable should not be considered; defaults to 5.
        speed_min_seq_duration (int/float)
            The minimum duration for chosen sequences; defaults to 2 (seconds).
        discontinuous_raster (bool)
            Prepare arrays from making raster plots; defaults to False.
        bin_size_ms (int)
            The bin size of the PETH; defaults to 50 (ms).
        window_size (int / float)
            The unilateral window size; defaults to 10 (seconds).
        smooth (bool)
            Smooth PETHs; defaults to False.
        smooth_sd (int)
            The SD of the smoothing window; defaults to 1 (bin).
        ----------

        Returns
        ----------
        peth_dictionary (dict)
            Peri-event time histogram for all clusters (np.ndarray (epoch_num, total_window)).
        raster_dictionary (dict)
            Raster arrays for all clusters zeroed to window start.
        ----------
        """

        get_clusters = kwargs['get_clusters'] if 'get_clusters' in kwargs.keys() \
                                                 and (kwargs['get_clusters'] == 'all' or type(kwargs['get_clusters']) == int or type(kwargs['get_clusters']) == list) else 'all'
        decode_what = kwargs['decode_what'] if 'decode_what' in kwargs.keys() and type(kwargs['decode_what']) == str else 'luminance'
        cluster_areas = kwargs['cluster_areas'] if 'cluster_areas' in kwargs.keys() and type(kwargs['cluster_areas']) == list else ['A']
        cluster_type = kwargs['cluster_type'] if 'cluster_type' in kwargs.keys() and type(kwargs['cluster_type']) == str else True
        speed_threshold_high = kwargs['speed_threshold_high'] if 'speed_threshold_high' in kwargs.keys() and (type(kwargs['speed_threshold_high']) == int or type(kwargs['speed_threshold_high']) == float) else 5.
        speed_threshold_low = kwargs['speed_threshold_low'] if 'speed_threshold_low' in kwargs.keys() and (type(kwargs['speed_threshold_low']) == int or type(kwargs['speed_threshold_low']) == float) else 0.
        speed_min_seq_duration = kwargs['speed_min_seq_duration'] if 'speed_min_seq_duration' in kwargs.keys() \
                                                                     and (type(kwargs['speed_min_seq_duration']) == int or type(kwargs['speed_min_seq_duration']) == float) else 2.
        discontinuous_raster = kwargs['discontinuous_raster'] if 'discontinuous_raster' in kwargs.keys() and type(kwargs['discontinuous_raster']) == bool else False
        bin_size_ms = kwargs['bin_size_ms'] if 'bin_size_ms' in kwargs.keys() and type(kwargs['bin_size_ms']) == int else 50
        window_size = kwargs['window_size'] if 'window_size' in kwargs.keys() and (type(kwargs['window_size']) == int or type(kwargs['window_size']) == float) else 6
        to_smooth = kwargs['to_smooth'] if 'to_smooth' in kwargs.keys() and type(kwargs['to_smooth']) == bool else False
        smooth_sd = kwargs['smooth_sd'] if 'smooth_sd' in kwargs.keys() and type(kwargs['smooth_sd']) == int else 1

        # choose clusters for PETHs
        all_clusters, chosen_clusters, extra_chosen_clusters, cluster_dict = decode_events.choose_012_clusters(the_input_012=self.input_012,
                                                                                                               cl_gr_dir=self.cluster_groups_dir,
                                                                                                               sp_prof_csv=self.sp_profiles_csv,
                                                                                                               cl_areas=cluster_areas,
                                                                                                               cl_type=cluster_type,
                                                                                                               dec_type=decode_what)
        # check if cluster(s) exist in the input sessions
        for cluster in get_clusters:
            if cluster not in all_clusters:
                print(f"Sorry, cluster {cluster} not in the input files!")
                sys.exit()

        # get activity dictionary
        zero_first_second_activity = {0: {}, 1: {}, 2: {}}
        zero_first_second_purged_spikes = {0: {}, 1: {}, 2: {}}
        for cluster in get_clusters:
            for file_idx, one_file in enumerate(self.input_012):
                if cluster in cluster_dict[file_idx]:
                    file_id, activity_dictionary, purged_spikes_dictionary = Spikes(input_file=one_file).convert_activity_to_frames_with_shuffles(get_clusters=cluster,
                                                                                                                                                  to_shuffle=False)
                    zero_first_second_activity[file_idx][cluster] = activity_dictionary[cluster]
                    zero_first_second_purged_spikes[file_idx][cluster] = purged_spikes_dictionary[cluster]

        # get behavior onsets
        session_variables = {0: {}, 1: {}, 2: {}}
        zero_first_second_behavior = {0: [], 1: [], 2: []}
        for file_idx, one_file in enumerate(self.input_012):
            ses_name, session_vars = Session(session=one_file).data_loader(extract_variables=['speeds', 'framerate'])
            session_variables[file_idx] = session_vars
            zero_first_second_behavior[file_idx] = find_variable_sequences(variable=session_vars['speeds'][:, 3],
                                                                           threshold_low=speed_threshold_low,
                                                                           threshold_high=speed_threshold_high,
                                                                           min_seq_duration=speed_min_seq_duration,
                                                                           camera_framerate=session_variables[file_idx]['framerate'])

        # find session with least events and get that number
        max_event_num_all_sessions = min([len(list(value)) for value in zero_first_second_behavior.values()])

        # get raster plot
        if discontinuous_raster:
            raster_dictionary = {}
            for cluster in get_clusters:

                pu_sp_tr_lst = []
                es_lst = []
                cam_fr = []
                for session_idx in range(len(self.input_012)):
                    cam_fr.append(session_variables[session_idx]['framerate'])
                    es_lst.append(zero_first_second_behavior[session_idx])
                    if cluster in zero_first_second_purged_spikes[session_idx].keys():
                        pu_sp_tr_lst.append(zero_first_second_purged_spikes[session_idx][cluster])
                    else:
                        pu_sp_tr_lst.append(np.empty(1))

                raster_dictionary[cluster] = discontinuous_raster_preparation(purged_spike_arr=np.array(pu_sp_tr_lst),
                                                                              event_start_arr=np.array(es_lst),
                                                                              event_number=max_event_num_all_sessions,
                                                                              camera_framerate_arr=np.array(cam_fr),
                                                                              window_size=speed_min_seq_duration)

        # get PETHs for each cluster and smooth if necessary
        peth_dictionary = {}
        for cluster in get_clusters:
            peth_dictionary[cluster] = {}
            input_arr_ls = []
            esf_lst = []
            for session in zero_first_second_activity.keys():
                esf_lst.append(zero_first_second_behavior[session])
                if cluster in zero_first_second_activity[session].keys():
                    input_arr_ls.append(zero_first_second_activity[session][cluster]['activity'].todense().astype(np.float32))
                else:
                    input_arr_ls.append(np.zeros(session_variables[session]['total_frame_num']).astype(np.float32))

            peth_array = calculate_discontinuous_peth(input_array_lst=input_arr_ls,
                                                      esf=esf_lst,
                                                      event_number=max_event_num_all_sessions,
                                                      bin_size_ms=bin_size_ms,
                                                      window_size=window_size)
            # smooth every sequence separately
            if to_smooth:
                total_window = int(round((window_size / (bin_size_ms / 1e3))))
                switch_points = np.arange(0, total_window, total_window // 3)
                for epoch in range(max_event_num_all_sessions):
                    for sp_idx, sp in enumerate(switch_points):
                        peth_array[epoch, sp:sp+(total_window // 3)] = gaussian_smoothing(array=peth_array[epoch, sp:sp+(total_window // 3)], sigma=smooth_sd, axis=0)

            peth_dictionary[cluster]['discontinuous_peth'] = peth_array

        if discontinuous_raster:
            return peth_dictionary, raster_dictionary
        else:
            return peth_dictionary
