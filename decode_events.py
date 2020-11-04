# -*- coding: utf-8 -*-

"""

@author: bartulem

Decode events like sound stimulation or luminance.

"""

import os
import time
import numpy as np
import gc
import neural_activity
from select_clusters import ClusterFinder
from sessions2load import Session
from numba import njit


@njit(parallel=False)
def correlate_quickly(big_x, big_x_mean, big_y, big_y_mean):
    big_x_shape_0 = big_x.shape[0]
    big_y_shape_0 = big_y.shape[0]
    big_y_shape_1 = big_y.shape[1]
    big_x_mean = np.reshape(big_x_mean, (big_x_shape_0, 1))
    big_y_mean = np.reshape(big_y_mean, (big_y_shape_0, 1))
    x_ds = big_x-big_x_mean
    y_ds = big_y-big_y_mean
    r_num = (x_ds.reshape((big_x_shape_0, 1, big_y_shape_1))*y_ds).sum(axis=2)
    r_den = np.sqrt(((x_ds**2).sum(axis=1).reshape((big_x_shape_0, 1)))*((y_ds**2).sum(axis=1).reshape((1, big_y_shape_0))))
    r = r_num / r_den
    nan_positions = np.isnan(r)
    return r, nan_positions


def predict_events(total_frame_num, fold_num, train_folds, test_folds,
                   activity_arr, event_arr, fe, luminance=False, hd=[0, 0]):
    if not luminance:
        pred_events = np.zeros(total_frame_num).astype(np.float32)
    else:
        pred_events = np.concatenate((np.zeros(hd[0]), np.ones(hd[1]))).astype(np.float32)
    for fold_idx in range(fold_num):
        training_frames = train_folds[fold_idx]
        test_frames = test_folds[fold_idx]
        train_arr = activity_arr.take(indices=training_frames, axis=0)
        test_arr = activity_arr.take(indices=test_frames, axis=0)
        corr_arr, nan_pos = correlate_quickly(big_x=train_arr,
                                              big_x_mean=train_arr.mean(axis=1),
                                              big_y=test_arr,
                                              big_y_mean=test_arr.mean(axis=1))
        # exchange nans for lowest value
        corr_arr[nan_pos] = -1

        # find best congruent train frame
        max_corr_train_frames_raw = np.nanargmax(corr_arr, axis=0)
        actual_train_frames = training_frames.take(max_corr_train_frames_raw)

        # get event values
        if not luminance:
            pred_events[fe[fold_idx]:fe[fold_idx+1]] = event_arr.take(actual_train_frames)
        else:
            all_pred_events = event_arr.take(actual_train_frames)
            first_half_end = fe[fold_idx+1]-fe[fold_idx]
            pred_events[fe[fold_idx]:fe[fold_idx+1]] = all_pred_events[:first_half_end]
            pred_events[fe[fold_idx+3]:fe[fold_idx+3+1]] = all_pred_events[first_half_end:]

    return pred_events


class Decoder:

    def __init__(self, input_file='', save_results_dir='', input_ldl=['', '', ''],
                 cluster_groups_dir='/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020/cluster_groups_info',
                 sp_profiles_csv='/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020/spiking_profiles/spiking_profiles.csv',
                 number_of_decoding_per_run=10, decoding_cell_number_array=np.array([5, 10, 20, 50, 100]), fold_n=3, shuffle_num=1000,
                 to_smooth=False, smooth_sd=1, smooth_axis=0, condense=True,
                 cluster_areas=['A'], cluster_type=True, animal_names=['kavorka', 'frank', 'johnjohn']):
        self.input_file = input_file
        self.save_results_dir = save_results_dir
        self.input_ldl = input_ldl
        self.cluster_groups_dir = cluster_groups_dir
        self.sp_profiles_csv = sp_profiles_csv
        self.number_of_decoding_per_run = number_of_decoding_per_run
        self.decoding_cell_number_array = decoding_cell_number_array
        self.fold_n = fold_n
        self.shuffle_num = shuffle_num
        self.to_smooth = to_smooth
        self.smooth_sd = smooth_sd
        self.smooth_axis = smooth_axis
        self.condense = condense
        self.cluster_areas = cluster_areas
        self.cluster_type = cluster_type
        self.animal_names = animal_names

    def decode_sound_stim(self, **kwargs):
        """
        Description
        ----------
        This method uses a simple nearest neighbor decoder to predict sound stimulation
        events. More specifically, we bin the spike train of every cluster in 100 ms bins
        and smooth it with a 3 bin Gaussian kernel. Since our three animals have a varying
        number of auditory single units (K=288, JJ=132, F=198), for each one we choose a
        distinct combination of units (either 5, 10, 20, 50 or 100) to decode the presence
        of the sound stimulus in each run (10 runs in total). In each run, we divided the data
        in three folds where 1/3 of the data was the test set and 2/3 were the training set.
        For each test set population vector we computed Pearson correlations to every population
        vector in the training set. We obtained a predicted sound stimulus value for each test
        frame by assigning it the sound stimulus value of the most correlated training set population
        vector. Decoding accuracy was defined as the proportion of correctly matched stimulus states
        across the entire recording session. We also shuffled the spikes of these units 1000 times
        in the first run to obtain the null-distribution of decoded accuracy.
        ----------

        Parameters
        ----------
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
            Smooth spike trains; defaults to False.
        smooth_sd (int)
            The SD of the smoothing window; defaults to 1 (bin).
        smooth_axis (int)
            The smoothing axis in a 2D array; defaults to 0.
        fold_n (int)
            The number of folds for decoding; defaults to 3.
        condense (bool)
            Yey or ney on the spike array condensing; defaults to True.
        ----------

        Returns
        ----------
        sound_decoding_accuracy (.npy file)
            The decoding accuracy data for a given input file.
        sound_shuffled_decoding_accuracy (.npy file)
            The shuffled data decoding accuracy for a given input file.
        ----------
        """

        # keep time
        start_time = time.time()

        # get animal name
        animal_name = [name for name in self.animal_names if name in self.input_file][0]

        # choose clusters you'd like to decode with
        chosen_clusters = ClusterFinder(session=self.input_file,
                                        cluster_groups_dir=self.cluster_groups_dir,
                                        sp_profiles_csv=self.sp_profiles_csv).get_desired_clusters(filter_by_area=self.cluster_areas,
                                                                                                   filter_by_cluster_type=self.cluster_type)
        # get total frame count and sound array
        file_name, extracted_frame_info = Session(session=self.input_file).data_loader(extract_variables=['total_frame_num', 'imu_sound'])

        # get activity dictionary
        file_id, activity_dictionary = neural_activity.Spikes(input_file=self.input_file).convert_activity_to_frames_with_shuffles(get_clusters=chosen_clusters,
                                                                                                                                   to_shuffle=True,
                                                                                                                                   condense_arr=self.condense)
        if self.condense:
            sound_array = neural_activity.condense_frame_arrays(frame_array=extracted_frame_info['imu_sound'], arr_type=False)
            total_frame_num = sound_array.shape[0]
        else:
            sound_array = extracted_frame_info['imu_sound']
            total_frame_num = extracted_frame_info['total_frame_num']

        # get fold edges
        fold_edges = np.floor(np.linspace(0, total_frame_num, self.fold_n+1)).astype(np.int32)

        # get train / test indices for each fold
        train_indices_for_folds = []
        test_indices_for_folds = []
        for fold in range(self.fold_n):
            all_frames = np.arange(0, total_frame_num)
            one_fold_arr = np.arange(fold_edges[fold], fold_edges[fold+1])
            test_indices_for_folds.append(one_fold_arr.astype(np.int32))
            train_indices_for_folds.append(np.setdiff1d(all_frames, one_fold_arr).astype(np.int32))

        # conduct decoding
        decoding_accuracy = np.zeros((self.decoding_cell_number_array.shape[0], self.number_of_decoding_per_run))
        shuffled_decoding_accuracy = np.zeros((self.decoding_cell_number_array.shape[0], self.shuffle_num))
        for decode_num in range(self.number_of_decoding_per_run):
            for ca_idx, cell_amount in enumerate(self.decoding_cell_number_array):
                clusters_array = np.zeros((total_frame_num, cell_amount)).astype(np.float32)

                if decode_num == 0:
                    shuffled_clusters_array = np.zeros((self.shuffle_num, total_frame_num, cell_amount))

                # shuffle cluster names
                np.random.shuffle(chosen_clusters)

                # select clusters
                selected_clusters = chosen_clusters[:cell_amount]

                # get all cell / shuffled data in their respective arrays
                for sc_idx, selected_cluster in enumerate(selected_clusters):
                    clusters_array[:, sc_idx] = activity_dictionary[selected_cluster]['activity'].todense().astype(np.float32)
                    if decode_num == 0:
                        for shuffle_idx in range(self.shuffle_num):
                            shuffled_clusters_array[shuffle_idx, :, sc_idx] = activity_dictionary[selected_cluster]['shuffled'][shuffle_idx].todense().astype(np.float32)

                # smooth spike trains if desired
                if self.to_smooth:
                    clusters_array = neural_activity.gaussian_smoothing(array=clusters_array, sigma=self.smooth_sd, axis=self.smooth_axis).astype(np.float32)
                    if decode_num == 0:
                        shuffled_clusters_array = neural_activity.gaussian_smoothing(array=shuffled_clusters_array, sigma=self.smooth_sd, axis=self.smooth_axis+1).astype(np.float32)

                # go through folds and predict sound
                predicted_sound_events = predict_events(total_frame_num=total_frame_num, fold_num=self.fold_n, train_folds=train_indices_for_folds,
                                                        test_folds=test_indices_for_folds, activity_arr=clusters_array, event_arr=sound_array,
                                                        fe=fold_edges)
                if decode_num == 0:
                    shuffle_predicted_sound_events = np.zeros((total_frame_num, self.shuffle_num))
                    for sh in range(self.shuffle_num):
                        shuffle_predicted_sound_events[:, sh] = predict_events(total_frame_num=total_frame_num, fold_num=self.fold_n, train_folds=train_indices_for_folds,
                                                                               test_folds=test_indices_for_folds, activity_arr=shuffled_clusters_array[sh], event_arr=sound_array,
                                                                               fe=fold_edges)

                # calculate accuracy and fill in the array
                decoding_accuracy[ca_idx, decode_num] = ((predicted_sound_events-sound_array) == 0).sum() / predicted_sound_events.shape[0]
                if decode_num == 0:
                    for sh_idx in range(self.shuffle_num):
                        shuffled_decoding_accuracy[ca_idx, sh_idx] = ((shuffle_predicted_sound_events[:, sh_idx]-sound_array) == 0).sum() \
                                                                     / shuffle_predicted_sound_events.shape[0]
                # free memory
                gc.collect()

        # save results as .npy files
        np.save(f'{self.save_results_dir}{os.sep}{animal_name}_sound_decoding_accuracy_{self.cluster_areas[0]}_clusters', decoding_accuracy)
        np.save(f'{self.save_results_dir}{os.sep}{animal_name}_sound_shuffled_decoding_accuracy_{self.cluster_areas[0]}_clusters', shuffled_decoding_accuracy)

        print("Sound decoding complete! It took {:.2f} hours.".format((time.time() - start_time) / 3600))

    def decode_luminance_condition(self, **kwargs):
        """
        Description
        ----------
        This method uses a nearest neighbor decoder to predict the luminance condition
        of individual timepoints over two recording sessions (light / dark). We take half
        of each recording and bin the spike train of every cluster common to both sessions
        in 100 ms bins and smooth them separately with a 3 bin Gaussian kernel. Since our
        three animals have a varying number of visual single units (K=410, JJ=404, F=193),
        for each one we choose a distinct combination of units (either 5, 10, 20, 50 or 100)
        to decode the  luminance condition in each run (10 runs in total). In each run, we
        divided the data in three folds where 1/3 of the data was the test set and 2/3 were
        the training set. Each fold had the same amount of light and dark timepoints.
        For each test set population vector we computed Pearson correlations to every population
        vector in the training set. We obtained a predicted sound stimulus value for each test
        frame by assigning it the luminance status of the most correlated training set population
        vector. Decoding accuracy was defined as the proportion of correctly matched stimulus states
        across the entire recording session. We also shuffled the spikes of these units 1000 times
        in the first run to obtain the null-distribution of decoded accuracy.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        condensed_bin_size (int)
            Condensed bin size ; defaults to 100 (ms).
        ----------

        Returns
        ----------
        luminance_decoding_accuracy (.npy file)
            The decoding accuracy data for a given input file pair.
        luminance_shuffled_decoding_accuracy (.npy file)
            The shuffled data decoding accuracy for a given input file pair.
        ----------
        """

        condensed_bin_size = kwargs['condensed_bin_size'] if 'condensed_bin_size' in kwargs.keys() and type(kwargs['condensed_bin_size']) == int else 100

        # keep time
        luminance_decoding_start_time = time.time()

        # get animal name
        animal_name = [name for name in self.animal_names if name in self.input_ldl[0]][0]

        # choose clusters for decoding
        chosen_clusters = []
        extra_chosen_clusters = {0: [], 1: []}
        cluster_dict = {0: [], 1: [], 2: []}
        for session_idx, one_session in enumerate(self.input_ldl):
            cluster_dict[session_idx] = ClusterFinder(session=one_session,
                                                      cluster_groups_dir=self.cluster_groups_dir,
                                                      sp_profiles_csv=self.sp_profiles_csv).get_desired_clusters(filter_by_area=self.cluster_areas,
                                                                                                                 filter_by_cluster_type=self.cluster_type)

        # find clusters present in all 3 sessions
        if 'V' in self.cluster_areas:
            for one_cl in list(set(cluster_dict[0]).intersection(cluster_dict[1], cluster_dict[2])):
                if one_cl not in chosen_clusters:
                    chosen_clusters.append(one_cl)
        else:
            for one_cl in list(set(cluster_dict[0]).intersection(cluster_dict[1])):
                if one_cl not in chosen_clusters:
                    chosen_clusters.append(one_cl)

        # find cells present only in the dark session
        if 'V' in self.cluster_areas:
            for d_cl in cluster_dict[1]:
                if d_cl not in chosen_clusters:
                    extra_chosen_clusters[1].append(d_cl)

        # find clusters present in two light session but not in the dark
        if 'V' in self.cluster_areas:
            for l_cl in cluster_dict[0]:
                if l_cl in cluster_dict[2] and l_cl not in chosen_clusters:
                    extra_chosen_clusters[0].append(l_cl)

        # all chosen clusters
        all_clusters = chosen_clusters + extra_chosen_clusters[0] + extra_chosen_clusters[1]

        # get total frame count in each session
        l_name, l_extracted_frame_info = Session(session=self.input_ldl[0]).data_loader(extract_variables=['total_frame_num'])
        d_name, d_extracted_frame_info = Session(session=self.input_ldl[1]).data_loader(extract_variables=['total_frame_num'])

        # get total frame number in future array
        if self.condense:
            total_frame_num = np.array([l_extracted_frame_info['total_frame_num'], d_extracted_frame_info['total_frame_num']]).min() // int(120. * (condensed_bin_size / 1e3))
            change_point = total_frame_num // 2
            half_durations = [change_point, total_frame_num-change_point]
            luminance_array = np.concatenate((np.ones(half_durations[0]), np.zeros(half_durations[1]))).astype(np.float32)
        else:
            total_frame_num = np.array([l_extracted_frame_info['total_frame_num'], d_extracted_frame_info['total_frame_num']]).min()
            change_point = total_frame_num // 2
            half_durations = [change_point, total_frame_num-change_point]
            luminance_array = np.concatenate((np.ones(half_durations[0]), np.zeros(half_durations[1]))).astype(np.float32)

        # get activity dictionary
        light_dark_activity = {0: {}, 1: {}}
        for file_idx, one_file in enumerate(self.input_ldl[:2]):
            if 'V' in self.cluster_areas:
                file_id, activity_dictionary = neural_activity.Spikes(input_file=one_file).convert_activity_to_frames_with_shuffles(get_clusters=chosen_clusters+extra_chosen_clusters[file_idx],
                                                                                                                                    to_shuffle=True,
                                                                                                                                    condense_arr=self.condense)
            else:
                file_id, activity_dictionary = neural_activity.Spikes(input_file=one_file).convert_activity_to_frames_with_shuffles(get_clusters=chosen_clusters,
                                                                                                                                    to_shuffle=True,
                                                                                                                                    condense_arr=self.condense)
            light_dark_activity[file_idx] = activity_dictionary

        # get fold edges
        fold_edges = np.floor(np.linspace(0, total_frame_num, (self.fold_n*2)+1)).astype(np.int32)

        # get train / test indices for each fold
        train_indices_for_folds = []
        test_indices_for_folds = []
        for fold in range(self.fold_n):
            all_frames = np.arange(0, total_frame_num)
            one_fold_arr = np.concatenate((np.arange(fold_edges[fold], fold_edges[fold+1]), np.arange(fold_edges[fold+3], fold_edges[fold+3+1])))
            test_indices_for_folds.append(one_fold_arr.astype(np.int32))
            train_indices_for_folds.append(np.setdiff1d(all_frames, one_fold_arr).astype(np.int32))

        # conduct decoding
        decoding_accuracy = np.zeros((self.decoding_cell_number_array.shape[0], self.number_of_decoding_per_run))
        shuffled_decoding_accuracy = np.zeros((self.decoding_cell_number_array.shape[0], self.shuffle_num))
        for decode_num in range(self.number_of_decoding_per_run):
            for ca_idx, cell_amount in enumerate(self.decoding_cell_number_array):
                clusters_array = np.zeros((total_frame_num, cell_amount)).astype(np.float32)

                if decode_num == 0:
                    shuffled_clusters_array = np.zeros((self.shuffle_num, total_frame_num, cell_amount))

                # shuffle cluster names
                np.random.shuffle(all_clusters)

                # select clusters
                selected_clusters = all_clusters[:cell_amount]

                # get all cell / shuffled data in their respective arrays
                for sc_idx, selected_cluster in enumerate(selected_clusters):
                    for luminance_type in light_dark_activity.keys():
                        seq_len = half_durations[luminance_type]
                        if selected_cluster in light_dark_activity[luminance_type].keys():
                            temp_cl_arr = light_dark_activity[luminance_type][selected_cluster]['activity'][:seq_len].todense().astype(np.float32)
                            if self.to_smooth:
                                temp_cl_arr = neural_activity.gaussian_smoothing(array=temp_cl_arr, sigma=self.smooth_sd, axis=self.smooth_axis).astype(np.float32)
                            if luminance_type == 0:
                                clusters_array[:change_point, sc_idx] = temp_cl_arr
                            else:
                                clusters_array[change_point:, sc_idx] = temp_cl_arr
                        else:
                            if luminance_type == 0:
                                clusters_array[:change_point, sc_idx] = np.zeros(seq_len).astype(np.float32)
                            else:
                                clusters_array[change_point:, sc_idx] = np.zeros(seq_len).astype(np.float32)

                        if decode_num == 0:
                            for shuffle_idx in range(self.shuffle_num):
                                if selected_cluster in light_dark_activity[luminance_type].keys():
                                    temp_shuffled_cl_arr = light_dark_activity[luminance_type][selected_cluster]['shuffled'][shuffle_idx][:seq_len].todense().astype(np.float32)
                                else:
                                    lum_type = abs(luminance_type-1)
                                    temp_shuffled_cl_arr = light_dark_activity[lum_type][selected_cluster]['shuffled'][shuffle_idx][seq_len:].todense().astype(np.float32)
                                if self.to_smooth:
                                    temp_shuffled_cl_arr = neural_activity.gaussian_smoothing(array=temp_shuffled_cl_arr, sigma=self.smooth_sd, axis=self.smooth_axis).astype(np.float32)
                                if luminance_type == 0:
                                    shuffled_clusters_array[shuffle_idx, :change_point, sc_idx] = temp_shuffled_cl_arr
                                else:
                                    shuffled_clusters_array[shuffle_idx, change_point:, sc_idx] = temp_shuffled_cl_arr

                # go through folds and predict sound
                predicted_luminance_events = predict_events(total_frame_num=total_frame_num, fold_num=self.fold_n, train_folds=train_indices_for_folds,
                                                            test_folds=test_indices_for_folds, activity_arr=clusters_array, event_arr=luminance_array,
                                                            fe=fold_edges, luminance=True, hd=half_durations)
                if decode_num == 0:
                    shuffle_predicted_luminance_events = np.zeros((total_frame_num, self.shuffle_num))
                    for sh in range(self.shuffle_num):
                        shuffle_predicted_luminance_events[:, sh] = predict_events(total_frame_num=total_frame_num, fold_num=self.fold_n, train_folds=train_indices_for_folds,
                                                                                   test_folds=test_indices_for_folds, activity_arr=shuffled_clusters_array[sh], event_arr=luminance_array,
                                                                                   fe=fold_edges, luminance=True, hd=half_durations)

                # calculate accuracy and fill in the array
                decoding_accuracy[ca_idx, decode_num] = ((predicted_luminance_events-luminance_array) == 0).sum() / predicted_luminance_events.shape[0]
                if decode_num == 0:
                    for sh_idx in range(self.shuffle_num):
                        shuffled_decoding_accuracy[ca_idx, sh_idx] = ((shuffle_predicted_luminance_events[:, sh_idx]-luminance_array) == 0).sum() \
                                                                     / shuffle_predicted_luminance_events.shape[0]
                # free memory
                gc.collect()

        # save results as .npy files
        np.save(f'{self.save_results_dir}{os.sep}{animal_name}_luminance_decoding_accuracy_{self.cluster_areas[0]}_clusters', decoding_accuracy)
        np.save(f'{self.save_results_dir}{os.sep}{animal_name}_luminance_shuffled_decoding_accuracy_{self.cluster_areas[0]}_clusters', shuffled_decoding_accuracy)

        print("Luminance decoding complete! It took {:.2f} hours.".format((time.time() - luminance_decoding_start_time) / 3600))

