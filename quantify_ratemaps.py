# -*- coding: utf-8 -*-

"""

@author: bartulem

Gets (1) tuning peak locations, (2) occupancies, (3) computes inter-session stability

"""

import io
import os
import re
import sys
import json
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from numba import njit
from scipy.stats import spearmanr
from select_clusters import ClusterFinder
from neural_activity import Spikes

# data[0, :] = xvals (bin centers)
# data[1, :] = raw rate map (ratemap / no smoothing)
# data[2, :] = occupancy (occupancy / no smoothing)
# data[3, :] = smoothed rate map
# data[4, :] = shuffled mean
# data[5, :] = shuffled std
# data[6, :] = smoothed occupancy
# data[7, :] = rawrm_p1 (even minutes ratemap / no smoothing)
# data[8, :] = smrm_p1 (even minutes ratemap / smoothed)
# data[9, :] = occ_p1 (even minutes occupancy / no smoothing)
# data[10, :] = smocc_p1 (even minutes occupancy / smoothed)
# data[11, :] = rawrm_p2 (odd minutes ratemap / no smoothing)
# data[12, :] = smrm_p2 (odd minutes ratemap / smoothed)
# data[13, :] = occ_p2 (odd minutes occupancy / no smoothing)
# data[14, :] = smocc_p2 (odd minutes occupancy / smoothed)


def uncover_file_specifics(file_name):
    animal_name = [name for name in ClusterFinder.probe_site_areas.keys() if name in file_name][0]
    get_date_idx = [date.start() for date in re.finditer('20_s', file_name)][-1]
    recording_date = file_name[get_date_idx-4:get_date_idx+2]
    if animal_name == 'bruno':
        recording_bank = 'distal'
    else:
        recording_bank = [bank for bank in ['distal', 'intermediate'] if bank in file_name][0]
    return f'{animal_name}_{recording_date}_{recording_bank}'


def get_shuffled_stability(n_times=10, shuffled_data_1=np.zeros((36, 1000)),
                           shuffled_data_2=np.zeros((36, 1000)), valid_sh_ind=None):
    if valid_sh_ind is None:
        valid_sh_ind = list(range(36))

    """
    Parameters
    ----------
    n_times : int
        The number of times to compute shuffled stability; defaults to 10.
    shuffled_data_1 : np.ndarray
        The shuffled data from the first chosen session.
    shuffled_data_2 : np.ndarray
        The shuffled data from the second chosen session.
    valid_sh_ind : (bool / int / float)
        The default indices to compute the shuffled stability over.
    ----------

    Returns
    ----------
    shuffled_stability : np.ndarray
        An array of length n_times with stability values for shuffled curves.
    ----------
    """

    shuffled_stability = np.zeros(n_times)

    for n in range(n_times):
        ran_num_1 = np.random.choice(a=1000)
        ran_num_2 = np.random.choice(a=1000)

        shuffled_first = shuffled_data_1[:, ran_num_1].take(indices=valid_sh_ind)
        shuffled_second = shuffled_data_2[:, ran_num_2].take(indices=valid_sh_ind)

        shuffled_stability[n] = spearmanr(shuffled_first, shuffled_second)[0]

    return shuffled_stability



@njit(parallel=False)
def find_valid_rm_range(rm_occ, min_acceptable_occ):
    return np.array([idx for idx, occ in enumerate(rm_occ) if occ > min_acceptable_occ])


@njit(parallel=False)
def check_curve_exceeds_shuffled(curve1d, shuffled_mean, shuffled_std,  min_acc_rate=True, bin_radius_to_check=1):
    """
    Parameters
    ----------
    curve1d : np.ndarray
        The 1-D tuning curve of choice.
    shuffled_mean : np.ndarray
        The mean of the shuffled distribution for the 1-D tuning curve of choice.
    shuffled_std : np.ndarray
        The std of the shuffled distribution for the 1-D tuning curve of choice.
    min_acc_rate : (bool / int / float)
        The minimum acceptable peak rate; defaults to True.
    bin_radius_to_check : int
        The radius of bins around peak to check whether they exceed shuffled data.
    ----------

    Returns
    ----------
    truth_value : bool
        Whether the 1-D tuning curve exceeds the shuffled distribution.
    ----------
    """
    if (min_acc_rate is True or curve1d.max() > min_acc_rate) and curve1d.max() \
            > shuffled_mean[np.argmax(curve1d)] + 3*shuffled_std[np.argmax(curve1d)]:
        peak_position = np.argmax(curve1d)
        if peak_position == 0:
            if (curve1d[peak_position:bin_radius_to_check*3] > shuffled_mean[peak_position:bin_radius_to_check*3] + 3*shuffled_std[peak_position:bin_radius_to_check*3]).all():
                return True
            else:
                return False
        elif peak_position == curve1d.shape[0]-1:
            if (curve1d[peak_position-(bin_radius_to_check*2):] > shuffled_mean[peak_position-(bin_radius_to_check*2):] + 3*shuffled_std[peak_position-(bin_radius_to_check*2):]).all():
                return True
            else:
                return False
        else:
            if (curve1d[peak_position-bin_radius_to_check:peak_position] > shuffled_mean[peak_position-bin_radius_to_check:peak_position] + 3*shuffled_std[peak_position-bin_radius_to_check:peak_position]).all() \
                    and (curve1d[peak_position:peak_position+bin_radius_to_check] > shuffled_mean[peak_position:peak_position+bin_radius_to_check] + 3*shuffled_std[peak_position:peak_position+bin_radius_to_check]).all():
                return True
            else:
                return False
    else:
        return False


class RatemapCharacteristics:

    areas_to_animals = {'CaPu': {'bruno': ['distal']},
                        'WhMa': {'bruno': ['distal']},
                        'S': {'bruno': ['distal'], 'roy': ['intermediate'], 'jacopo': ['intermediate'], 'crazyjoe': ['intermediate']},
                        'S1HL': {'bruno': ['distal'], 'roy': ['intermediate'], 'jacopo': ['intermediate'], 'crazyjoe': ['intermediate']},
                        'S1Tr': {'bruno': ['distal'], 'jacopo': ['intermediate'], 'crazyjoe': ['intermediate']},
                        'M': {'bruno': ['distal'], 'roy': ['distal', 'intermediate'], 'jacopo': ['distal', 'intermediate'], 'crazyjoe': ['distal', 'intermediate']},
                        'M1': {'bruno': ['distal'], 'roy': ['distal', 'intermediate'], 'jacopo': ['distal', 'intermediate'], 'crazyjoe': ['distal', 'intermediate']},
                        'M2': {'jacopo': ['distal'], 'crazyjoe': ['distal']},
                        'PPC': {'bruno': ['distal'], 'jacopo': ['intermediate']},
                        'A': {'frank': ['distal'], 'johnjohn': ['distal'], 'kavorka': ['distal']},
                        'A1': {'frank': ['distal'], 'johnjohn': ['distal'], 'kavorka': ['distal']},
                        'A2D': {'frank': ['distal'], 'johnjohn': ['distal'], 'kavorka': ['distal']},
                        'V': {'frank': ['distal', 'intermediate'], 'johnjohn': ['distal', 'intermediate'], 'kavorka': ['distal', 'intermediate']},
                        'V1': {'frank': ['distal', 'intermediate'], 'johnjohn': ['distal', 'intermediate'], 'kavorka': ['distal', 'intermediate']},
                        'V1d': {'frank': ['distal'], 'johnjohn': ['distal'], 'kavorka': ['distal', 'intermediate']},
                        'V1s': {'frank': ['distal', 'intermediate'], 'johnjohn': ['distal', 'intermediate'], 'kavorka': ['intermediate']},
                        'V2M': {'frank': ['intermediate'], 'johnjohn': ['intermediate'], 'kavorka': ['intermediate']},
                        'V2L': {'frank': ['distal'], 'johnjohn': ['distal'], 'kavorka': ['distal']}}

    def __init__(self, ratemap_mat_dir='', pkl_sessions_dir='', save_dir='', area_filter='M', animal_filter=True, profile_filter=True,
                 session_id_filter='s1', session_non_filter=True, session_type_filter=True, cluster_type_filter=True,
                 cluster_groups_dir='', sp_profiles_csv='', specific_date=None):
        if specific_date is None:
            specific_date = {'bruno': ['020520', '030520'],
                             'roy': True,
                             'jacopo': True,
                             'crazyjoe': True,
                             'frank': True,
                             'johnjohn': ['210520', '220520'],
                             'kavorka': True}
        self.ratemap_mat_dir = ratemap_mat_dir
        self.pkl_sessions_dir = pkl_sessions_dir
        self.save_dir = save_dir
        self.area_filter = area_filter
        self.cluster_type_filter = cluster_type_filter
        self.profile_filter = profile_filter
        self.animal_filter = animal_filter
        self.session_id_filter = session_id_filter
        self.session_non_filter = session_non_filter
        self.session_type_filter = session_type_filter
        self.specific_date = specific_date
        self.cluster_groups_dir = cluster_groups_dir
        self.sp_profiles_csv = sp_profiles_csv

    def file_finder(self, **kwargs):
        """
        Description
        ----------
        This method finds files of interest in the ratemap .mat file directory. There are
        two options: you can look for all files with certain characteristics (i.e. lights
        & s1) in only one recording session (seek_stability=False), or if you are further
        interested in computing stability measures, you look for files across two specified
        recording sessions (seek_stability=True).
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        area_filter (str)
            Area of interest; defaults to 'M'.
        animal_filter (bool / list)
            Animals to consider: defaults to True (considers all).
        cluster_type_filter (bool / str)
            Cluster type to be included: 'good' or 'mua'; defaults to True.
        profile_filter (str / bool)
            Profile to be included: 'RS' or 'FS'; defaults to True.
        session_id_filter (bool / str)
            The session number of interest; defaults to 's1'.
        session_type_filter (bool / str)
            The session type of interest; defaults to True.
        specific_date (dict)
            The date of interest (for animals that had recordings across days); defaults to True for most animals.
        seek_stability (bool)
            If True, looks for ratemap files across two desired sessions; defaults to False.
        session_2_type (str)
            The type of the session you want to measure stability for; defaults to 'light'.
        return_clusters (bool)
            If True returns clusters for each animal, if False, returns .mat files; defaults to False.
        seek_third_session (bool)
            If True, looks for ratemap files across three desired sessions; defaults to False.
        session_3_type (str)
            The type of session you want to get the third session data for; defaults to 'light'.
        ----------

        Returns
        ----------
        cluster_dict / essential_files (dict)
            A dictionary with all the clusters from a certain brain area organized by animal/bank, or
            a dictionary of all relevant .mat ratemap files (for max two relevant session types) for further analyses.
        ----------
        """

        seek_stability = kwargs['seek_stability'] if 'seek_stability' in kwargs.keys() and type(kwargs['seek_stability']) == bool else False
        session_2_type = kwargs['session_2_type'] if 'session_2_type' in kwargs.keys() and type(kwargs['session_2_type']) == str else 'light'
        return_clusters = kwargs['return_clusters'] if 'return_clusters' in kwargs.keys() and type(kwargs['return_clusters']) == bool else False
        seek_third_session = kwargs['seek_third_session'] if 'seek_third_session' in kwargs.keys() and type(kwargs['seek_third_session']) == bool else False
        session_3_type = kwargs['session_3_type'] if 'session_3_type' in kwargs.keys() and type(kwargs['session_3_type']) == str else 'light'

        # get clusters of interest
        cluster_dict = {}
        total_clusters = 0
        for animal in self.areas_to_animals[self.area_filter].keys():
            cluster_dict[animal] = {}
            for bank in self.areas_to_animals[self.area_filter][animal]:
                for pkl_file in os.listdir(self.pkl_sessions_dir):
                    if animal in pkl_file and bank in pkl_file and (self.session_id_filter is True or self.session_id_filter in pkl_file) \
                            and (self.session_non_filter is True or self.session_non_filter not in pkl_file) \
                            and (self.session_type_filter is True or self.session_type_filter in pkl_file) \
                            and (self.specific_date[animal] is True or any(one_date in pkl_file for one_date in self.specific_date[animal])):
                        cluster_list = ClusterFinder(session=f'{self.pkl_sessions_dir}{os.sep}{pkl_file}',
                                                     cluster_groups_dir=self.cluster_groups_dir,
                                                     sp_profiles_csv=self.sp_profiles_csv).get_desired_clusters(filter_by_area=[self.area_filter],
                                                                                                                filter_by_cluster_type=self.cluster_type_filter,
                                                                                                                filter_by_spiking_profile=self.profile_filter)
                        cluster_dict[animal][bank] = cluster_list
                        total_clusters += len(cluster_list)
                        break

        print(f"Cluster search complete. Found {total_clusters} valid cluster(s) in area {self.area_filter}.")

        if return_clusters:
            return cluster_dict
        else:
            # collect relevant file names in appropriate lists
            essential_files = {'chosen_session_1': [], 'chosen_session_2': [], 'chosen_session_3': [], 'unique': []}
            if os.path.exists(self.ratemap_mat_dir):
                for file_name in tqdm(os.listdir(self.ratemap_mat_dir), desc='Checking all ratemap files'):
                    if (self.animal_filter is True or any(one_animal in file_name for one_animal in self.animal_filter)) \
                            and (self.session_id_filter is True or self.session_id_filter in file_name) \
                            and (self.session_non_filter is True or self.session_non_filter not in file_name) \
                            and (self.session_type_filter is True or self.session_type_filter in file_name):
                        animal_id = [name for name in ClusterFinder.probe_site_areas.keys() if name in file_name][0]
                        if animal_id == 'bruno':
                            bank_id = 'distal'
                        else:
                            bank_id = [bank for bank in ['distal', 'intermediate'] if bank in file_name][0]
                        cluster_id = file_name[file_name.find('imec'):file_name.find('imec') + 18]
                        if (self.specific_date[animal_id] is True or any(one_date in file_name for one_date in self.specific_date[animal_id])) \
                                and animal_id in cluster_dict.keys() and bank_id in cluster_dict[animal_id].keys():
                            if cluster_id in cluster_dict[animal_id][bank_id]:
                                for unique_file in essential_files['unique']:
                                    if animal_id in unique_file and bank_id in unique_file:
                                        break
                                else:
                                    essential_files['unique'].append(file_name)
                                if not seek_stability:
                                    essential_files['chosen_session_1'].append(file_name)
                                else:
                                    for file_name_2 in os.listdir(self.ratemap_mat_dir):
                                        first_for_loop = False
                                        if file_name != file_name_2 and animal_id in file_name_2 and bank_id in file_name_2 and cluster_id in file_name_2 \
                                                and (self.specific_date[animal_id] is True or any(one_date in file_name_2 for one_date in self.specific_date[animal_id])) \
                                                and session_2_type in file_name_2:
                                            if not seek_third_session:
                                                essential_files['chosen_session_1'].append(file_name)
                                                essential_files['chosen_session_2'].append(file_name_2)
                                            else:
                                                for file_name_3 in os.listdir(self.ratemap_mat_dir):
                                                    if file_name != file_name_3 and file_name_2 != file_name_3 and animal_id in file_name_3 and bank_id in file_name_3 and cluster_id in file_name_3 \
                                                            and (self.specific_date[animal_id] is True or any(one_date in file_name_3 for one_date in self.specific_date[animal_id])) \
                                                            and session_3_type in file_name_3:
                                                        essential_files['chosen_session_1'].append(file_name)
                                                        essential_files['chosen_session_2'].append(file_name_2)
                                                        essential_files['chosen_session_3'].append(file_name_3)
                                                        first_for_loop = True
                                                        break
                                        if first_for_loop:
                                            break
            else:
                print(f"Invalid location for ratemap directory {self.ratemap_mat_dir}. Please try again.")
                sys.exit()

            print(f"File search complete. Found {len(essential_files['chosen_session_1'])} .mat file(s) for area {self.area_filter}.")

            return essential_files

    def tuning_peaks_stability(self, **kwargs):
        """
        Description
        ----------
        This method finds bin centers where the peak 1D tuning curve firing rate resides,
        or calculates inter-session stability for all 1D variables.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        use_smoothed_occ (bool)
            Use smoothed occupancies to make ratemaps; defaults to False.
        min_acceptable_occ (float)
            The minimum acceptable occupancy; defaults to 0.4 (s).
        use_smoothed_rm (bool)
            Use smoothed firing rates to make ratemaps; defaults to False.
        min_acc_rate (bool / int / float)
            The minimum acceptable peak rate; defaults to True.
        bin_radius_to_check (int)
            The radius of bins around peak to check whether they exceed shuffled data.
        get_stability (bool)
            Get inter-session stability.
        session_2_type (str)
            The type of the session you want to measure stability for; defaults to 'light'.
        ----------

        Returns
        ----------
        tuning_peak_locations (.json file)
            A file with significant (>3 bins) tuning peak positions for all 1-D variables.
        ----------
        """

        use_smoothed_occ = 6 if 'use_smoothed_occ' in kwargs.keys() and kwargs['use_smoothed_occ'] is True else 2
        min_acceptable_occ = kwargs['min_acceptable_occ'] if 'min_acceptable_occ' in kwargs.keys() and type(kwargs['min_acceptable_occ']) == float else 0.4
        use_smoothed_rm = 3 if 'use_smoothed_rm' in kwargs.keys() and kwargs['use_smoothed_rm'] is True else 1
        min_acc_rate = kwargs['min_acc_rate'] if 'min_acc_rate' in kwargs.keys() and type(kwargs['min_acc_rate']) == float else True
        bin_radius_to_check = kwargs['bin_radius_to_check'] if 'bin_radius_to_check' in kwargs.keys() and type(kwargs['bin_radius_to_check']) == int else 1
        get_stability = kwargs['get_stability'] if 'get_stability' in kwargs.keys() and type(kwargs['get_stability']) == bool else False
        session_2_type = kwargs['session_2_type'] if 'session_2_type' in kwargs.keys() and type(kwargs['session_2_type']) == str else 'light'

        if not get_stability:
            essential_files = self.file_finder()
        else:
            essential_files = self.file_finder(seek_stability=True,
                                               session_2_type=session_2_type)

        # get tuning peak locations / stability
        tuning_peak_locations = {}
        stability = {}
        cl_num = 0
        for file_idx, file in enumerate(tqdm(essential_files['chosen_session_1'], desc='Checking ratemap files for tuning peaks / stability')):
            mat = sio.loadmat(f'{self.ratemap_mat_dir}{os.sep}{file}')
            if get_stability:
                file2 = essential_files['chosen_session_2'][file_idx]
                mat2 = sio.loadmat(f'{self.ratemap_mat_dir}{os.sep}{file2}')

            # get animal name, bank id and date of session
            session_id = uncover_file_specifics(file)
            start_cl_idx = essential_files['chosen_session_1'][file_idx].find('imec')
            cl_id = essential_files['chosen_session_1'][file_idx][start_cl_idx: start_cl_idx+18]

            for key in mat.keys():
                if 'imec' in key and 'data' in key:

                    # find feature ID
                    reduced_key = key[19:]
                    feature_id = reduced_key[:reduced_key.index('-')]

                    # find feature range with acceptable occupancies
                    valid_rm_range = find_valid_rm_range(rm_occ=mat[key][use_smoothed_occ, :],
                                                         min_acceptable_occ=min_acceptable_occ)

                    # conduct the checking
                    valid_rm = mat[key][use_smoothed_rm, :].take(indices=valid_rm_range)
                    if check_curve_exceeds_shuffled(curve1d=valid_rm,
                                                    shuffled_mean=mat[key][4, :].take(indices=valid_rm_range),
                                                    shuffled_std=mat[key][5, :].take(indices=valid_rm_range),
                                                    min_acc_rate=min_acc_rate,
                                                    bin_radius_to_check=bin_radius_to_check):
                        if not get_stability:
                            if cl_num not in tuning_peak_locations.keys():
                                tuning_peak_locations[cl_num] = {}

                            if session_id not in tuning_peak_locations[cl_num].keys():
                                tuning_peak_locations[cl_num]['session_id'] = session_id

                            if cl_id not in tuning_peak_locations[cl_num].keys():
                                tuning_peak_locations[cl_num]['cl_id'] = cl_id

                            if 'features' not in tuning_peak_locations[cl_num].keys():
                                tuning_peak_locations[cl_num]['features'] = {}
                            tuning_peak_locations[cl_num]['features'][feature_id] = mat[key][0, :].take(indices=valid_rm_range)[np.argmax(valid_rm)]
                        else:
                            if cl_num not in stability.keys():
                                stability[cl_num] = {}

                            if session_id not in stability[cl_num].keys():
                                stability[cl_num]['session_id'] = session_id

                            if cl_id not in stability[cl_num].keys():
                                stability[cl_num]['cl_id'] = cl_id

                            if 'features' not in stability[cl_num].keys():
                                stability[cl_num]['features'] = {}

                            if 'shuffled' not in stability.keys():
                                stability['shuffled'] = {}

                            if feature_id not in stability['shuffled'].keys():
                                stability['shuffled'][feature_id] = []

                            # find feature range with acceptable occupancies for second session
                            valid_rm2_range = find_valid_rm_range(rm_occ=mat2[key][use_smoothed_occ, :],
                                                                  min_acceptable_occ=min_acceptable_occ)

                            # get valid indices intersection
                            indices_intersection = sorted(list(set(valid_rm_range) & set(valid_rm2_range)), key=int)

                            # calculate stability
                            valid_rm_revised = mat[key][use_smoothed_rm, :].take(indices=indices_intersection)
                            valid_rm2_revised = mat2[key][use_smoothed_rm, :].take(indices=indices_intersection)
                            stability[cl_num]['features'][feature_id] = spearmanr(valid_rm_revised, valid_rm2_revised)[0]

                            # get shuffled stability
                            shuffled_key = f'{key[:19]}{feature_id}-rawacc_shuffles'
                            sh_stability = get_shuffled_stability(shuffled_data_1=mat[shuffled_key],
                                                                  shuffled_data_2=mat2[shuffled_key],
                                                                  valid_sh_ind=indices_intersection)
                            for sh in sh_stability:
                                stability['shuffled'][feature_id].append(sh)
            cl_num += 1

        # save results to file
        if self.session_type_filter is True:
            session_type_file_name = 'light'
        else:
            session_type_file_name = self.session_type_filter
        if not get_stability:
            with io.open(f'{self.save_dir}{os.sep}tuning_peak_locations_{self.area_filter}_{session_type_file_name}.json', 'w', encoding='utf-8') as to_save_file:
                to_save_file.write(json.dumps(tuning_peak_locations, ensure_ascii=False, indent=4))
        else:
            with io.open(f'{self.save_dir}{os.sep}stability_{self.area_filter}_{session_type_file_name}_{session_2_type}.json', 'w', encoding='utf-8') as to_save_file:
                to_save_file.write(json.dumps(stability, ensure_ascii=False, indent=4))

    def occupancies(self, **kwargs):
        """
        Description
        ----------
        This method finds occupancies for all 1D variables.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        use_smoothed_occ (bool)
            Use smoothed occupancies to make ratemaps; defaults to False.
        ----------

        Returns
        ----------
        occupancies (.json file)
            A file with accumulated occupancies for all 1-D variables.
        ----------
        """

        use_smoothed_occ = 6 if 'use_smoothed_occ' in kwargs.keys() and kwargs['use_smoothed_occ'] is True else 2

        essential_files = self.file_finder()

        # get occupancies
        occupancies = {}
        for file in tqdm(essential_files['unique'], desc='Gathering occupanices'):
            mat = sio.loadmat(f'{self.ratemap_mat_dir}{os.sep}{file}')
            for key in mat.keys():
                if 'imec' in key and 'data' in key:

                    # find feature ID
                    reduced_key = key[19:]
                    feature_id = reduced_key[:reduced_key.index('-')]
                    if feature_id not in occupancies.keys():
                        occupancies[feature_id] = {'xvals': np.zeros(mat[key].shape[1]), 'occ': np.zeros(mat[key].shape[1])}

                    if occupancies[feature_id]['xvals'].sum() == 0:
                        occupancies[feature_id]['xvals'] = mat[key][0, :]

                    occupancies[feature_id]['occ'] += mat[key][use_smoothed_occ, :]

        # convert to lists
        for feature in occupancies.keys():
            for key in occupancies[feature].keys():
                occupancies[feature][key] = occupancies[feature][key].tolist()

        # save results to file
        with io.open(f'{self.save_dir}{os.sep}occupancies_{self.area_filter}.json', 'w', encoding='utf-8') as to_save_file:
            to_save_file.write(json.dumps(occupancies, ensure_ascii=False, indent=4))

    def make_weight_comparisons(self, **kwargs):
        """
        Description
        ----------
        This method finds tuning-curve peak rate differences in weight/no-weight sessions.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        session_2_type (str)
            The type of the session you want to compare against; defaults to 'weight'.
        session_3_type (str)
            The type of the session you want to compare against; defaults to 'light'.
        use_smoothed_occ (bool)
            Use smoothed occupancies to make ratemaps; defaults to False.
        min_acceptable_occ (float)
            The minimum acceptable occupancy; defaults to 0.4 (s).
        use_smoothed_rm (bool)
            Use smoothed firing rates to make ratemaps; defaults to False.
        min_acc_rate (bool / int / float)
            The minimum acceptable peak rate; defaults to True.
        bin_radius_to_check (int)
            The radius of bins around peak to check whether they exceed shuffled data.
        ----------

        Returns
        ----------
        weight_comparisons (.json file)
            A file with weight comparisons for all 1-D variables.
        ----------
        """

        session_2_type = kwargs['session_2_type'] if 'session_2_type' in kwargs.keys() and type(kwargs['session_2_type']) == str else 'weight'
        session_3_type = kwargs['session_3_type'] if 'session_3_type' in kwargs.keys() and type(kwargs['session_3_type']) == str else 'light'
        use_smoothed_occ = 6 if 'use_smoothed_occ' in kwargs.keys() and kwargs['use_smoothed_occ'] is True else 2
        min_acceptable_occ = kwargs['min_acceptable_occ'] if 'min_acceptable_occ' in kwargs.keys() and type(kwargs['min_acceptable_occ']) == float else 0.4
        use_smoothed_rm = 3 if 'use_smoothed_rm' in kwargs.keys() and kwargs['use_smoothed_rm'] is True else 1
        min_acc_rate = kwargs['min_acc_rate'] if 'min_acc_rate' in kwargs.keys() and type(kwargs['min_acc_rate']) == float else True
        bin_radius_to_check = kwargs['bin_radius_to_check'] if 'bin_radius_to_check' in kwargs.keys() and type(kwargs['bin_radius_to_check']) == int else 1

        essential_files = self.file_finder(seek_stability=True,
                                           session_2_type=session_2_type,
                                           seek_third_session=True,
                                           session_3_type=session_3_type)

        # make weight comparisons
        weight_comparison = {}
        cl_num = 0
        for file_idx, file in enumerate(tqdm(essential_files['chosen_session_1'], desc='Making weight comparisons')):
            mat = sio.loadmat(f'{self.ratemap_mat_dir}{os.sep}{file}')
            file2 = essential_files['chosen_session_2'][file_idx]
            mat2 = sio.loadmat(f'{self.ratemap_mat_dir}{os.sep}{file2}')
            file3 = essential_files['chosen_session_3'][file_idx]
            mat3 = sio.loadmat(f'{self.ratemap_mat_dir}{os.sep}{file3}')

            # get animal name, bank id and date of session
            session_id = uncover_file_specifics(file)
            start_cl_idx = essential_files['chosen_session_1'][file_idx].find('imec')
            cl_id = essential_files['chosen_session_1'][file_idx][start_cl_idx: start_cl_idx+18]

            for key in mat.keys():
                if 'imec' in key and 'data' in key:

                    # find feature ID
                    reduced_key = key[19:]
                    feature_id = reduced_key[:reduced_key.index('-')]

                    # find feature range with acceptable occupancies
                    valid_rm_range = find_valid_rm_range(rm_occ=mat[key][use_smoothed_occ, :],
                                                         min_acceptable_occ=min_acceptable_occ)

                    valid_rm2_range = find_valid_rm_range(rm_occ=mat2[key][use_smoothed_occ, :],
                                                          min_acceptable_occ=min_acceptable_occ)

                    valid_rm3_range = find_valid_rm_range(rm_occ=mat3[key][use_smoothed_occ, :],
                                                          min_acceptable_occ=min_acceptable_occ)

                    # get valid indices intersection
                    indices_intersection = sorted(list(set(valid_rm_range) & set(valid_rm2_range) & set(valid_rm3_range)), key=int)

                    # get valid range in all three sessions
                    valid_rm_revised = mat[key][use_smoothed_rm, :].take(indices=indices_intersection)
                    valid_rm2_revised = mat2[key][use_smoothed_rm, :].take(indices=indices_intersection)
                    valid_rm3_revised = mat3[key][use_smoothed_rm, :].take(indices=indices_intersection)

                    # conduct the checking
                    if check_curve_exceeds_shuffled(curve1d=valid_rm_revised,
                                                    shuffled_mean=mat[key][4, :].take(indices=indices_intersection),
                                                    shuffled_std=mat[key][5, :].take(indices=indices_intersection),
                                                    min_acc_rate=min_acc_rate,
                                                    bin_radius_to_check=bin_radius_to_check) \
                            and check_curve_exceeds_shuffled(curve1d=valid_rm3_revised,
                                                             shuffled_mean=mat3[key][4, :].take(indices=indices_intersection),
                                                             shuffled_std=mat3[key][5, :].take(indices=indices_intersection),
                                                             min_acc_rate=min_acc_rate,
                                                             bin_radius_to_check=bin_radius_to_check):

                        if cl_num not in weight_comparison.keys():
                            weight_comparison[cl_num] = {}

                        if session_id not in weight_comparison[cl_num].keys():
                            weight_comparison[cl_num]['session_id'] = session_id

                        if cl_id not in weight_comparison[cl_num].keys():
                            weight_comparison[cl_num]['cl_id'] = cl_id

                        if 'features' not in weight_comparison[cl_num].keys():
                            weight_comparison[cl_num]['features'] = {}

                        if feature_id not in weight_comparison[cl_num]['features'].keys():
                            weight_comparison[cl_num]['features'][feature_id] = {}

                        if 'baseline_firing_rates' not in weight_comparison[cl_num].keys():
                            if self.session_non_filter is True:
                                first_session_id = 's1'
                            else:
                                first_session_id = self.session_non_filter
                            weight_comparison[cl_num]['baseline_firing_rates'] = {}
                            for pkl_file in os.listdir(self.pkl_sessions_dir):
                                if all(one_item in pkl_file for one_item in session_id.split('_')) and 'light' in pkl_file and first_session_id in pkl_file:
                                    file_id, baseline_activity_dictionary = Spikes(input_file=f'{self.pkl_sessions_dir}{os.sep}{pkl_file}').get_baseline_firing_rates(get_clusters=[cl_id])
                                    weight_comparison[cl_num]['baseline_firing_rates']['light1'] = baseline_activity_dictionary[cl_id]
                                elif all(one_item in pkl_file for one_item in session_id.split('_')) and session_2_type in pkl_file:
                                    file_id_2, baseline_activity_dictionary_2 = Spikes(input_file=f'{self.pkl_sessions_dir}{os.sep}{pkl_file}').get_baseline_firing_rates(get_clusters=[cl_id])
                                    weight_comparison[cl_num]['baseline_firing_rates'][session_2_type] = baseline_activity_dictionary_2[cl_id]
                                elif all(one_item in pkl_file for one_item in session_id.split('_')) and 'light' in pkl_file and first_session_id not in pkl_file:
                                    file_id_3, baseline_activity_dictionary_3 = Spikes(input_file=f'{self.pkl_sessions_dir}{os.sep}{pkl_file}').get_baseline_firing_rates(get_clusters=[cl_id])
                                    weight_comparison[cl_num]['baseline_firing_rates']['light2'] = baseline_activity_dictionary_3[cl_id]

                        weight_comparison[cl_num]['features'][feature_id]['light1'] = list(valid_rm_revised)
                        weight_comparison[cl_num]['features'][feature_id][session_2_type] = list(valid_rm2_revised)
                        weight_comparison[cl_num]['features'][feature_id]['light2'] = list(valid_rm3_revised)
                        weight_comparison[cl_num]['features'][feature_id]['ICr-light1'] = mat[key.replace('data', 'ICr')].ravel()[0]
                        weight_comparison[cl_num]['features'][feature_id][f'ICr-{session_2_type}'] = mat2[key.replace('data', 'ICr')].ravel()[0]
                        weight_comparison[cl_num]['features'][feature_id]['ICr-light2'] = mat3[key.replace('data', 'ICr')].ravel()[0]

            cl_num += 1

        # save results to file
        with io.open(f'{self.save_dir}{os.sep}weight_comparison_{self.area_filter}.json', 'w', encoding='utf-8') as to_save_file:
            to_save_file.write(json.dumps(weight_comparison, ensure_ascii=False, indent=4))
