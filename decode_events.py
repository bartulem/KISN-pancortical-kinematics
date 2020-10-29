# -*- coding: utf-8 -*-

"""

@author: bartulem

Decode events like sound stimulation or luminance.

"""

import os
import time
import numpy as np
import neural_activity
from select_clusters import ClusterFinder
from sessions2load import Session
from numba import njit
from tqdm import tqdm


@njit(parallel=False)
def correlate_quickly(big_x, y):
    big_x_mean = np.zeros(big_x.shape[0])
    for idx in range(big_x.shape[0]):
        big_x_mean[idx] = big_x[idx, :].mean()
    big_x_mean = np.reshape(big_x_mean, (big_x.shape[0], 1))
    y_mean = y.mean()
    r_num = ((big_x-big_x_mean)*(y-y_mean)).sum(axis=1)
    r_den = np.sqrt(((big_x-big_x_mean)**2).sum(axis=1)*((y-y_mean)**2).sum())
    r = r_num/r_den
    return r


def predict_events(total_frame_num, fold_num, train_folds, test_folds,
                   activity_arr, sound_arr, se_p):
    pred_sound_events = np.zeros(total_frame_num)
    for fold_idx in range(fold_num):
        training_frames = train_folds[fold_idx]
        test_frames = test_folds[fold_idx]
        train_arr = activity_arr.take(indices=training_frames, axis=0)
        test_arr = activity_arr.take(indices=test_frames, axis=0)
        for test_frame in test_frames:
            # check whether all array elements are identical (if so, their variance is 0)
            if not (test_arr[test_frame, :] == test_arr[test_frame, 0]).all():
                corr_arr = correlate_quickly(big_x=train_arr, y=test_arr[test_frame, :])
                max_corr_train_frame_raw = np.nanargmax(corr_arr)
                actual_train_frame = training_frames[max_corr_train_frame_raw]
                pred_sound_events[test_frame] = sound_arr[actual_train_frame]
            else:
                pred_sound_events[test_frame] = np.random.choice(a=2, p=se_p)
    return pred_sound_events


class Decoder:

    def __init__(self, input_file='', save_results_dir=''):
        self.input_file = input_file
        self.save_results_dir = save_results_dir

    def decode_sound_stim(self, **kwargs):
        """
        Description
        ----------
        This method uses a simple nearest neighbor decoder to classify sound stimulation
        events. More specifically, we bin the spike train to match the tracking resolution
        and smooth it with a 3 bin Gaussian kernel. Since our three animals have a varying
        number of auditory single units (K=288, JJ=132, F=198), for each one we choose a
        distinct combination of units (either 5, 10, 20, 50 or 100) to decode the presence
        of the sound stimulus in each run (10 runs in total). In each run, we divided the data
        in three folds where 1/3 of the data was the test set and 2/3 were the training set.
        For each test set population vector we computed Pearson correlations to every population
        vector in the training set and obtained a predicted sound stimulus value by assigning the
        the sound stimulus value of the most correlated training set population vector to the
        test set predictions. Decoding accuracy was defined as the proportion of correctly matched
        stimulus states across the entire recording session. We also shuffled the spikes of these
        units 1000 times in the first run to obtain the null-distribution of decoded accuracy.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        cluster_groups_dir (str)
            The directory with the cluster_groups .json file; defaults to '/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020/cluster_groups_info'.
        sp_profiles_csv (str)
            The sp_profiles .csv file; defaults to '/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020/spiking_profiles/spiking_profiles.csv'.
        cluster_areas (list)
            Cluster area(s) of choice; defaults to ['A'].
        cluster_type (str)
            Cluster type of choice; defaults to True.
        number_of_decoding_per_run (int)
            The number of times to decode event per run; defaults to 10.
        decoding_cell_number_array (np.ndarray):
            An array of numbers of cells to decode with; defaults to np.array([5, 10, 20, 50, 100]).
        shuffle_num (int)
            Number of shuffles; defaults to 1000.
        to_smooth (bool)
            Smooth PETHs; defaults to False.
        smooth_sd (int)
            The SD of the smoothing window; defaults to 1 (bin).
        smooth_axis (int)
            The smoothing axis in a 2D array; defaults to 0.
        fold_n (int)
            The number of folds for decoding; defaults to 3.
        ----------

        Returns
        ----------
        file_info (str)
            The shortened version of the file name.
        ----------
        """

        cluster_groups_dir = kwargs['cluster_groups_dir'] if 'cluster_groups_dir' in kwargs.keys() and type(kwargs['cluster_groups_dir']) == str \
            else '/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020/cluster_groups_info'
        sp_profiles_csv = kwargs['sp_profiles_csv'] if 'sp_profiles_csv' in kwargs.keys() and type(kwargs['sp_profiles_csv']) == str \
            else '/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020/spiking_profiles/spiking_profiles.csv'
        cluster_areas = kwargs['cluster_areas'] if 'cluster_areas' in kwargs.keys() and type(kwargs['cluster_areas']) == list else ['A']
        cluster_type = kwargs['cluster_type'] if 'cluster_type' in kwargs.keys() and kwargs['cluster_type'] in ['good', 'mua'] else True
        number_of_decoding_per_run = kwargs['number_of_decoding_per_run'] if 'number_of_decoding_per_run' in kwargs.keys() \
                                                                             and type(kwargs['number_of_decoding_per_run']) == int else 10
        decoding_cell_number_array = kwargs['decoding_cell_number_array'] if 'decoding_cell_number_array' in kwargs.keys() \
                                                                             and type(kwargs['decoding_cell_number_array']) == np.array else np.array([5, 10, 20, 50, 100])
        shuffle_num = kwargs['shuffle_num'] if 'shuffle_num' in kwargs.keys() and type(kwargs['shuffle_num']) == int else 1000
        to_smooth = kwargs['to_smooth'] if 'to_smooth' in kwargs.keys() and type(kwargs['to_smooth']) == bool else False
        smooth_sd = kwargs['smooth_sd'] if 'smooth_sd' in kwargs.keys() and type(kwargs['smooth_sd']) == int else 1
        smooth_axis = kwargs['smooth_axis'] if 'smooth_axis' in kwargs.keys() and type(kwargs['smooth_axis']) == int else 0
        fold_n = kwargs['fold_n'] if 'fold_n' in kwargs.keys() and type(kwargs['fold_n']) == int else 3

        # choose clusters you'd like to decode with
        chosen_clusters = ClusterFinder(session=self.input_file,
                                        cluster_groups_dir=cluster_groups_dir,
                                        sp_profiles_csv=sp_profiles_csv).get_desired_clusters(filter_by_area=cluster_areas,
                                                                                              filter_by_cluster_type=cluster_type)
        # get framerate and total frame count
        file_name, extracted_frame_info = Session(session=self.input_file).data_loader(extract_variables=['framerate', 'total_frame_num', 'imu_sound'])

        # get sound event empirical probabilities
        sound_event_empirical_probabilities = [(extracted_frame_info['imu_sound'] == 0).sum() / extracted_frame_info['imu_sound'].shape[0],
                                               (extracted_frame_info['imu_sound'] == 1).sum() / extracted_frame_info['imu_sound'].shape[0]]

        # get activity dictionary
        file_id, activity_dictionary = neural_activity.Spikes(input_file=self.input_file).convert_activity_to_frames_with_shuffles(get_clusters=chosen_clusters,
                                                                                                                                   to_shuffle=True)
        # get all cluster IDs
        all_cluster_names = list(activity_dictionary.keys())

        # get fold edges
        fold_edges = np.floor(np.linspace(0, extracted_frame_info['total_frame_num'], fold_n+1)).astype(np.int64)

        # get train / test indices for each fold
        train_indices_for_folds = []
        test_indices_for_folds = []
        for fold in range(fold_n):
            all_frames = np.arange(0, extracted_frame_info['total_frame_num'])
            ond_fold_arr = np.arange(fold_edges[fold], fold_edges[fold+1])
            test_indices_for_folds.append(ond_fold_arr.astype(np.int64))
            train_indices_for_folds.append(np.setdiff1d(all_frames, ond_fold_arr).astype(np.int64))

        # keep time
        start_time = time.time()

        # conduct decoding
        decoding_accuracy = np.zeros((decoding_cell_number_array.shape[0], number_of_decoding_per_run))
        shuffled_decoding_accuracy = np.zeros((decoding_cell_number_array.shape[0], shuffle_num))
        for decode_num in tqdm(range(number_of_decoding_per_run)):
            for ca_idx, cell_amount in enumerate(decoding_cell_number_array):
                cells_array = np.zeros((extracted_frame_info['total_frame_num'], cell_amount))

                if decode_num == 0:
                    shuffled_cells_array = np.zeros((shuffle_num, extracted_frame_info['total_frame_num'], cell_amount))

                # shuffle cluster names
                np.random.shuffle(all_cluster_names)

                # select clusters
                selected_clusters = all_cluster_names[:cell_amount]

                # get all cell / shuffled data in their respective arrays
                for sc_idx, selected_cluster in enumerate(selected_clusters):
                    cells_array[:, sc_idx] = activity_dictionary[selected_cluster]['activity']
                    if decode_num == 0:
                        for shuffle_idx in range(shuffle_num):
                            shuffled_cells_array[shuffle_idx, :, sc_idx] = activity_dictionary[selected_cluster]['shuffled'][shuffle_idx, :]

                # smooth spike trains if desired
                if to_smooth:
                    cells_array = neural_activity.gaussian_smoothing(array=cells_array, sigma=smooth_sd, axis=smooth_axis)
                    if decode_num == 0:
                        shuffled_cells_array = neural_activity.gaussian_smoothing(array=shuffled_cells_array, sigma=smooth_sd, axis=smooth_axis+1)

                # go through folds and predict sound
                predicted_sound_events = predict_events(total_frame_num=extracted_frame_info['total_frame_num'], fold_num=fold_n, train_folds=train_indices_for_folds,
                                                        test_folds=test_indices_for_folds, activity_arr=cells_array, sound_arr=extracted_frame_info['imu_sound'],
                                                        se_p=sound_event_empirical_probabilities)
                if decode_num == 0:
                    shuffle_predicted_sound_events = np.zeros((extracted_frame_info['total_frame_num'], shuffle_num))
                    for sh in range(shuffle_num):
                        shuffle_predicted_sound_events[:, sh] = predict_events(total_frame_num=extracted_frame_info['total_frame_num'], fold_num=fold_n, train_folds=train_indices_for_folds,
                                                                               test_folds=test_indices_for_folds, activity_arr=shuffled_cells_array[sh], sound_arr=extracted_frame_info['imu_sound'],
                                                                               se_p=sound_event_empirical_probabilities)

                # calculate accuracy and fill in the array
                decoding_accuracy[ca_idx, decode_num] = ((predicted_sound_events-extracted_frame_info['imu_sound']) == 0).sum() / predicted_sound_events.shape[0]
                if decode_num == 0:
                    for sh_idx in range(shuffle_num):
                        shuffled_decoding_accuracy[ca_idx, sh_idx] = ((shuffle_predicted_sound_events[:, sh_idx]-extracted_frame_info['imu_sound']) == 0).sum() \
                                                                     / shuffle_predicted_sound_events.shape[0]

        # save results as .npy files
        np.save(f'{self.save_results_dir}{os.sep}sound_decoding_accuracy_{cluster_areas[0]}clusters', decoding_accuracy)
        np.save(f'{self.save_results_dir}{os.sep}sound_shuffled_decoding_accuracy_{cluster_areas[0]}clusters', shuffled_decoding_accuracy)

        print("Decoding complete! It took {:.2f} minutes.".format((time.time() - start_time) / 60))
