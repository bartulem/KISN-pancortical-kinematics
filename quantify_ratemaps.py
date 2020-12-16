# -*- coding: utf-8 -*-

"""

@author: bartulem

Gets (1) tuning peak locations, (2) occupancies, (3) computes inter-session stability

"""

import os
import sys
import numpy as np
from select_clusters import ClusterFinder


class RatemapCharacteristics:

    areas_to_animals = {'Capu': {'bruno': ['distal']},
                        'WhMa': ['bruno'],
                        'S': ['bruno', 'roy', 'jacopo', 'crazyjoe'],
                        'S1HL': ['bruno', 'roy', 'jacopo', 'crazyjoe'],
                        'S1Tr': ['bruno', 'jacopo', 'crazyjoe'],
                        'M': ['bruno', 'roy', 'jacopo', 'crazyjoe'],
                        'M1': ['bruno', 'roy', 'jacopo', 'crazyjoe'],
                        'M2': ['jacopo', 'crazyjoe'],
                        'PPC': ['bruno', 'jacopo'],
                        'A': ['frank', 'johnjohn', 'kavorka'],
                        'A1': ['frank', 'johnjohn', 'kavorka'],
                        'A2D': ['frank', 'johnjohn', 'kavorka'],
                        'V': ['frank', 'johnjohn', 'kavorka'],
                        'V1': ['frank', 'johnjohn', 'kavorka'],
                        'V2M': ['frank', 'johnjohn', 'kavorka'],
                        'V2L': ['frank', 'johnjohn', 'kavorka']}

    def __init__(self, ratemap_mat_dir='', pkl_sessions_dir='', area_filter='M', animal_filter=True, profile_filter=True,
                 session_id_filter='s1', session_type_filter=True, specific_date=True, cluster_type_filter=True,
                 cluster_groups_dir='', sp_profiles_csv=''):
        self.ratemap_mat_dir = ratemap_mat_dir
        self.pkl_sessions_dir = pkl_sessions_dir
        self.area_filter = area_filter
        self.cluster_type_filter = cluster_type_filter
        self.profile_filter = profile_filter
        self.animal_filter = animal_filter
        self.session_id_filter = session_id_filter
        self.session_type_filter = session_type_filter
        self.specific_date = specific_date
        self.cluster_groups_dir = cluster_groups_dir
        self.sp_profiles_csv = sp_profiles_csv

    def tuning_peak_locations(self, **kwargs):
        """
        Description
        ----------
        This method finds bin centers where the peak 1D tuning curve firing rate resides.
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
        session_id_filter (str)
            The session number of interest; defaults to 's1'.
        session_type_filter (bool / str)
            The session type of interest; defaults to True.
        specific_date (bool / str)
            The date of interest (for animals that had recordings across days); defaults to True.
        ----------

        Returns
        ----------
        file_info (str)
            The shortened version of the file name.
        ----------
        """

        # get clusters of interest
        cluster_dict = {}
        for animal in self.areas_to_animals[self.area_filter].keys():
            cluster_dict[animal] = {}
            for bank in self.areas_to_animals[self.area_filter][animal]:
                for pkl_file in os.listdir(self.pkl_sessions_dir):
                    if animal in pkl_file and bank in pkl_file and self.session_id_filter in pkl_file \
                            and (self.specific_date is True or self.specific_date in pkl_file):
                        cluster_dict[animal][bank] = ClusterFinder(session=f'{self.pkl_sessions_dir}{os.sep}{pkl_file}',
                                                                   cluster_groups_dir=self.cluster_groups_dir,
                                                                   sp_profiles_csv=self.sp_profiles_csv).get_desired_clusters(filter_by_area=self.area_filter,
                                                                                                                              filter_by_cluster_type=self.cluster_type_filter,
                                                                                                                              filter_by_spiking_profile=self.profile_filter)
                        break

        # collect relevant file names in a list
        essential_files = []
        if os.path.exists(self.ratemap_mat_dir):
            for file_name in os.listdir(self.ratemap_mat_dir):
                if (self.animal_filter is True or any(one_animal in file_name for one_animal in self.animal_filter)) \
                        and (self.session_id_filter in file_name) \
                        and (self.session_type_filter is True or self.session_type_filter in file_name):
                    essential_files.append(file_name)
        else:
            print(f"Invalid location for ratemap directory {self.ratemap_mat_dir}. Please try again.")
            sys.exit()

