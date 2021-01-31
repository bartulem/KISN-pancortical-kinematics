# -*- coding: utf-8 -*-

"""

@author: bartulem

Decode events like sound stimulation or luminance/weight presence.

"""

import os
import gc
import time
import numpy as np
import neural_activity
from select_clusters import ClusterFinder
from sessions2load import Session
from sklearn import svm
from tqdm import tqdm
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


def predict_events(pred_arr_len, fold_num, train_folds, test_folds,
                   activity_arr, event_arr, fe, three_sessions=False):

    pred_events = np.zeros(pred_arr_len).astype(np.float32)
    for fold_idx in range(fold_num):
        training_frames = train_folds[fold_idx]
        test_frames = test_folds[fold_idx]
        train_arr = activity_arr.take(indices=training_frames, axis=0).astype(np.float32)
        test_arr = activity_arr.take(indices=test_frames, axis=0).astype(np.float32)
        corr_arr, nan_pos = correlate_quickly(big_x=train_arr,
                                              big_x_mean=train_arr.mean(axis=1),
                                              big_y=test_arr,
                                              big_y_mean=test_arr.mean(axis=1))
        # exchange nans for lowest value
        corr_arr[nan_pos] = -1

        # find best congruent train frame
        # since argmax would always take the first frame if multiple frames had the same max value,
        # it's necessary to make a random selection from all the frames that have the max value
        max_corr_each_frame = np.nanmax(corr_arr, axis=0).astype(np.float32)
        max_corr_train_frames_raw = np.zeros(max_corr_each_frame.shape[0]).astype(np.int32)
        for test_fr in range(max_corr_each_frame.shape[0]):
            max_corr_train_frames_raw[test_fr] = np.random.choice(np.where(corr_arr[:, test_fr] == max_corr_each_frame[test_fr])[0]).astype(np.int32)

        actual_train_frames = training_frames.take(max_corr_train_frames_raw)

        # get event values
        if not three_sessions:
            pred_events[fe[fold_idx]:fe[fold_idx+1]] = event_arr.take(actual_train_frames)
        else:
            pred_events = event_arr.take(actual_train_frames)

    return pred_events


def choose_012_clusters(the_input_012, cl_gr_dir, sp_prof_csv, cl_areas, cl_type, dec_type, desired_profiles):
    chosen_clusters = []
    extra_chosen_clusters = {0: [], 1: []}
    cluster_dict = {0: [], 1: [], 2: []}
    for session_idx, one_session in enumerate(the_input_012):
        cluster_dict[session_idx] = ClusterFinder(session=one_session,
                                                  cluster_groups_dir=cl_gr_dir,
                                                  sp_profiles_csv=sp_prof_csv).get_desired_clusters(filter_by_area=cl_areas,
                                                                                                    filter_by_cluster_type=cl_type,
                                                                                                    filter_by_spiking_profile=desired_profiles)

    # find clusters present in all 3 sessions
    for one_cl in list(set(cluster_dict[0]).intersection(cluster_dict[1], cluster_dict[2])):
        chosen_clusters.append(one_cl)

    if dec_type == 'luminance':
        # find cells present only in the dark session
        if 'V' in cl_areas:
            for d_cl in cluster_dict[1]:
                if d_cl not in cluster_dict[0] and d_cl not in cluster_dict[2]:
                    extra_chosen_clusters[1].append(d_cl)

        # find clusters present in two light session but not in the dark
        if 'V' in cl_areas:
            for l_cl in cluster_dict[0]:
                if l_cl in cluster_dict[2] and l_cl not in cluster_dict[1]:
                    extra_chosen_clusters[0].append(l_cl)

    # all chosen clusters
    all_clusters = chosen_clusters + extra_chosen_clusters[0] + extra_chosen_clusters[1]

    return all_clusters, chosen_clusters, extra_chosen_clusters, cluster_dict


class Decoder:

    def __init__(self, input_file='', save_results_dir='', input_012=None,
                 cluster_groups_dir='/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020/cluster_groups_info',
                 sp_profiles_csv='/home/bartulm/Insync/mimica.bartul@gmail.com/OneDrive/Work/data/posture_2020/spiking_profiles/spiking_profiles.csv',
                 number_of_decoding_per_run=10, decoding_cell_number_array=np.array([5, 10, 20, 50, 100]), fold_n=3, shuffle_num=1000,
                 to_smooth=False, smooth_sd=1, smooth_axis=0, condense=True,
                 cluster_areas=None, cluster_type=True, animal_names=None,
                 profiles_to_include=True):
        if cluster_areas is None:
            cluster_areas = ['A']
        if input_012 is None:
            input_012 = ['', '', '']
        if animal_names is None:
            animal_names = ['kavorka', 'frank', 'johnjohn', 'roy', 'jacopo', 'bruno', 'crazyjoe']
        self.input_file = input_file
        self.save_results_dir = save_results_dir
        self.input_012 = input_012
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
        self.profiles_to_include = profiles_to_include

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
        file_id, activity_dictionary, purged_spikes_dict = neural_activity.Spikes(input_file=self.input_file).convert_activity_to_frames_with_shuffles(get_clusters=chosen_clusters,
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
                    clusters_array[:, sc_idx] = activity_dictionary[selected_cluster]['activity'].todense().astype(np.float32).copy()
                    if decode_num == 0:
                        for shuffle_idx in range(self.shuffle_num):
                            shuffled_clusters_array[shuffle_idx, :, sc_idx] = activity_dictionary[selected_cluster]['shuffled'][shuffle_idx].todense().astype(np.float32).copy()

                # smooth spike trains if desired
                if self.to_smooth:
                    clusters_array = neural_activity.gaussian_smoothing(array=clusters_array, sigma=self.smooth_sd, axis=self.smooth_axis).astype(np.float32)
                    if decode_num == 0:
                        shuffled_clusters_array = neural_activity.gaussian_smoothing(array=shuffled_clusters_array, sigma=self.smooth_sd, axis=self.smooth_axis+1).astype(np.float32)

                # go through folds and predict sound
                predicted_sound_events = predict_events(pred_arr_len=total_frame_num, fold_num=self.fold_n, train_folds=train_indices_for_folds,
                                                        test_folds=test_indices_for_folds, activity_arr=clusters_array, event_arr=sound_array,
                                                        fe=fold_edges)
                if decode_num == 0:
                    shuffle_predicted_sound_events = np.zeros((total_frame_num, self.shuffle_num))
                    for sh in range(self.shuffle_num):
                        shuffle_predicted_sound_events[:, sh] = predict_events(pred_arr_len=total_frame_num, fold_num=self.fold_n, train_folds=train_indices_for_folds,
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

    def decode_session_type(self, **kwargs):
        """
        Description
        ----------
        This method uses a SVM/nearest neighbor decoder to predict the luminance / weight condition
        of individual timepoints in a separate recording session (i.e. the second light session).
        We take the last quarter of recording of condition 1 (e.g. light), the first quarter of
        recording of condition 2 (e.g. dark) and the second half of a separate recording of condition
        1 (e.g. second light session). The spike trains of every cluster common to all three sessions
        are binned in 100 ms bins (and smoothed with a 3 bin Gaussian). Since our three animals have
        a varying number of visual and  A1 single units, we choose a distinct combination of units
        (either 5, 10, 20, 35 or 50) to decode the condition in each run (100 runs in total).
        For each test set population vector (e.g. second half of the second light session), we computed
        Pearson correlations to every population vector in the training set (light 1 + dark). We obtained
        a predicted condition status for each test frame by assigning it the condition status of the most
        correlated training set population vector. Decoding accuracy was defined as the proportion of
        correctly matched stimulus states across the test session. Since the luminance condition didn't
        change within a session, shuffling spike trains would not make sense because even time shifted
        activity, if it's overall lower/higher relative to the other session, would still enable accurate
        decoding. Instead, we randomly permuted the joint unit activity (quarter light / quarter dark) at each
        training time point a 1000 times in the first run to obtain the null-distribution of decoded accuracy.

        NB: To run this we used the following parameters:
            cluster_areas = ['V'] or ['A1']
            cluster_type = 'good'
            to_smooth = True
            smooth_sd = 3
            condense = True
            decoding_cell_number_array = np.array([5, 10, 20, 35, 50])
            number_of_decoding_per_run = 100
            pop_vector_decoding = True
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        condensed_bin_size (int)
            Condensed bin size; defaults to 100 (ms).
        decode_what (str)
            What are you decoding; defaults to 'luminance'.
        pop_vector_decoding (str)
            What decoding to use; defaults to True.
        ----------

        Returns
        ----------
        decoding_accuracy (.npy file)
            The decoding accuracy data for a given input file pair.
        shuffled_decoding_accuracy (.npy file)
            The shuffled data decoding accuracy for a given input file pair.
        ----------
        """

        condensed_bin_size = kwargs['condensed_bin_size'] if 'condensed_bin_size' in kwargs.keys() and type(kwargs['condensed_bin_size']) == int else 100
        decode_what = kwargs['decode_what'] if 'decode_what' in kwargs.keys() and type(kwargs['decode_what']) == str else 'luminance'
        pop_vector_decoding = kwargs['pop_vector_decoding'] if 'pop_vector_decoding' in kwargs.keys() and type(kwargs['pop_vector_decoding']) == bool else True

        # keep time
        session_type_decoding_start_time = time.time()

        # get animal name
        animal_name = [name for name in self.animal_names if name in self.input_012[0]][0]

        # choose clusters for decoding
        all_clusters, chosen_clusters, extra_chosen_clusters, cluster_dict = choose_012_clusters(the_input_012=self.input_012,
                                                                                                 cl_gr_dir=self.cluster_groups_dir,
                                                                                                 sp_prof_csv=self.sp_profiles_csv,
                                                                                                 cl_areas=self.cluster_areas,
                                                                                                 cl_type=self.cluster_type,
                                                                                                 dec_type=decode_what,
                                                                                                 desired_profiles=self.profiles_to_include)

        # get total frame count in each session
        zero_ses_name, zero_extracted_frame_info = Session(session=self.input_012[0]).data_loader(extract_variables=['total_frame_num'])
        first_ses_name, first_extracted_frame_info = Session(session=self.input_012[1]).data_loader(extract_variables=['total_frame_num'])
        second_ses_name, second_extracted_frame_info = Session(session=self.input_012[2]).data_loader(extract_variables=['total_frame_num'])

        # get total frame number in future array
        if self.condense:
            total_frame_num = np.array([zero_extracted_frame_info['total_frame_num'],
                                        first_extracted_frame_info['total_frame_num'],
                                        second_extracted_frame_info['total_frame_num']]).min() // int(120. * (condensed_bin_size / 1e3))
        else:
            total_frame_num = np.array([zero_extracted_frame_info['total_frame_num'],
                                        first_extracted_frame_info['total_frame_num'],
                                        second_extracted_frame_info['total_frame_num']]).min()

        middle_change_point = total_frame_num // 2
        fold_edges = np.array([0, middle_change_point // 2, middle_change_point, total_frame_num])
        third_durations = np.diff(fold_edges)
        decoding_event_array = np.concatenate((np.ones(third_durations[0]), np.zeros(third_durations[1]), np.ones(third_durations[2]))).astype(np.float32)

        # get activity dictionary
        zero_first_second_activity = {0: {}, 1: {}, 2: {}}
        for file_idx, one_file in enumerate(self.input_012):
            if 'V' in self.cluster_areas and decode_what == 'luminance':
                if file_idx == 0 or file_idx == 2:
                    chosen_idx = 0
                else:
                    chosen_idx = 1
                file_id, activity_dictionary, purged_spikes_dict = neural_activity.Spikes(input_file=one_file).convert_activity_to_frames_with_shuffles(get_clusters=chosen_clusters+extra_chosen_clusters[chosen_idx],
                                                                                                                                                        to_shuffle=False,
                                                                                                                                                        condense_arr=self.condense)
            else:
                file_id, activity_dictionary, purged_spikes_dict = neural_activity.Spikes(input_file=one_file).convert_activity_to_frames_with_shuffles(get_clusters=chosen_clusters,
                                                                                                                                                        to_shuffle=False,
                                                                                                                                                        condense_arr=self.condense)
            zero_first_second_activity[file_idx] = activity_dictionary

        # get train / test indices
        train_indices_for_folds = [np.arange(0, middle_change_point).astype(np.int32)]
        test_indices_for_folds = [np.arange(middle_change_point, total_frame_num).astype(np.int32)]

        # conduct decoding
        decoding_accuracy = np.zeros((self.decoding_cell_number_array.shape[0], self.number_of_decoding_per_run))
        shuffled_decoding_accuracy = np.zeros((self.decoding_cell_number_array.shape[0], self.shuffle_num))
        for decode_num in tqdm(range(self.number_of_decoding_per_run)):
            for ca_idx, cell_amount in enumerate(self.decoding_cell_number_array):
                clusters_array = np.zeros((total_frame_num, cell_amount)).astype(np.float32)

                if decode_num == 0:
                    shuffled_clusters_array = np.zeros((self.shuffle_num, total_frame_num, cell_amount)).astype(np.float32)

                # shuffle cluster names
                np.random.shuffle(all_clusters)

                # select clusters
                selected_clusters = all_clusters[:cell_amount]

                # get all cell / shuffled data in their respective arrays
                for sc_idx, selected_cluster in enumerate(selected_clusters):
                    for condition_type in range(len(zero_first_second_activity.keys())):
                        seq_len = third_durations[condition_type]
                        if selected_cluster in zero_first_second_activity[condition_type].keys():
                            if condition_type == 1:
                                temp_cl_arr = zero_first_second_activity[condition_type][selected_cluster]['activity'][:seq_len].todense().astype(np.float32).copy()
                            else:
                                temp_cl_arr = zero_first_second_activity[condition_type][selected_cluster]['activity'][-seq_len:].todense().astype(np.float32).copy()
                            if self.to_smooth:
                                temp_cl_arr = neural_activity.gaussian_smoothing(array=temp_cl_arr, sigma=self.smooth_sd, axis=self.smooth_axis).astype(np.float32)
                            clusters_array[fold_edges[condition_type]: fold_edges[condition_type+1], sc_idx] = temp_cl_arr

                # prepare array for shuffling
                if decode_num == 0:
                    copy_clu_train_arr = clusters_array[:middle_change_point, :].copy()
                    copy_clu_test_arr = clusters_array[middle_change_point:, :].copy()
                    for shuffle_idx in range(self.shuffle_num):
                        np.random.shuffle(copy_clu_train_arr)
                        shuffled_clusters_array[shuffle_idx, :, :] = np.concatenate((copy_clu_train_arr, copy_clu_test_arr))

                if pop_vector_decoding:
                    # go through folds and predict event
                    predicted_condition_events = predict_events(pred_arr_len=total_frame_num-third_durations[2], fold_num=1, train_folds=train_indices_for_folds,
                                                                test_folds=test_indices_for_folds, activity_arr=clusters_array, event_arr=decoding_event_array.copy(),
                                                                fe=fold_edges, three_sessions=True)

                    if decode_num == 0:
                        shuffle_predicted_condition_events = np.zeros((total_frame_num-middle_change_point, self.shuffle_num))
                        for sh in range(self.shuffle_num):
                            shuffle_predicted_condition_events[:, sh] = predict_events(pred_arr_len=total_frame_num-third_durations[2], fold_num=1, train_folds=train_indices_for_folds,
                                                                                       test_folds=test_indices_for_folds, activity_arr=shuffled_clusters_array[sh], event_arr=decoding_event_array.copy(),
                                                                                       fe=fold_edges, three_sessions=True)

                    # calculate accuracy and fill in the array
                    decoding_accuracy[ca_idx, decode_num] = ((predicted_condition_events-np.ones(total_frame_num-middle_change_point)) == 0).sum() / predicted_condition_events.shape[0]

                    if decode_num == 0:
                        for sh_idx in range(self.shuffle_num):
                            shuffled_decoding_accuracy[ca_idx, sh_idx] = ((shuffle_predicted_condition_events[:, sh_idx]-np.ones(total_frame_num-middle_change_point)) == 0).sum() \
                                                                         / shuffle_predicted_condition_events[:, sh_idx].shape[0]

                    # free memory
                    gc.collect()

                else:
                    # this utilizes the SVM classifier for decoding
                    test_arr = clusters_array[middle_change_point:, :].copy()
                    train_arr = clusters_array[:middle_change_point, :].copy()
                    decoding_event_array_reduced = decoding_event_array[:middle_change_point].copy()

                    # train model and generate predictions
                    condition_model = svm.SVC()
                    condition_model.fit(X=train_arr, y=decoding_event_array_reduced)
                    generated_predictions = condition_model.predict(test_arr)

                    # calculate accuracy
                    decoding_accuracy[ca_idx, decode_num] = ((generated_predictions-np.ones(total_frame_num-middle_change_point)) == 0).sum() / generated_predictions.shape[0]

                    # same for shuffling
                    if decode_num == 0:
                        for sh_idx in range(self.shuffle_num):
                            shuffled_test_arr = shuffled_clusters_array[sh_idx, middle_change_point:, :].copy()
                            shuffled_train_arr = shuffled_clusters_array[sh_idx, :middle_change_point, :].copy()
                            shuffled_decoding_event_array_reduced = decoding_event_array[:middle_change_point].copy()

                            shuffled_condition_model = svm.SVC()
                            shuffled_condition_model.fit(X=shuffled_train_arr, y=shuffled_decoding_event_array_reduced)
                            shuffled_generated_predictions = shuffled_condition_model.predict(shuffled_test_arr)
                            dea_randomized = np.ones(shuffled_generated_predictions.shape[0])

                            shuffled_decoding_accuracy[ca_idx, sh_idx] = ((shuffled_generated_predictions-dea_randomized) == 0).sum() / shuffled_generated_predictions.shape[0]

        # save results as .npy files
        np.save(f'{self.save_results_dir}{os.sep}{animal_name}_{decode_what}_decoding_accuracy_{self.cluster_areas[0]}_clusters', decoding_accuracy)
        np.save(f'{self.save_results_dir}{os.sep}{animal_name}_{decode_what}_shuffled_decoding_accuracy_{self.cluster_areas[0]}_clusters', shuffled_decoding_accuracy)

        print("Decoding complete! It took {:.2f} hours.".format((time.time() - session_type_decoding_start_time) / 3600))
