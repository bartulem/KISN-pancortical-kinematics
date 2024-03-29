"""
Select appropriate clusters.
@author: bartulem
"""

import os
import re
import sys
import pickle
import json
import pandas as pd


class ClusterFinder:

    probe_site_areas = {'bruno': {'distal': {'CaPu': [(0, 20)],
                                             'WhMa': [(20, 49)],
                                             'S1HL': [(49, 277)],
                                             'S1Tr': [(277, 348)],
                                             'M1': [(348, 362)],
                                             'PPC': [(362, 384)]}},
                        'roy': {'distal': {'M1': [(0, 384)]},
                                'intermediate': {'M1': [(0, 17), (217, 230)],
                                                 'S1HL': [(17, 147)],
                                                 'S1Tr': [(147, 217)],
                                                 'PPC': [(230, 243)]}},
                        'jacopo': {'distal': {'M1': [(0, 314)],
                                              'S1HL': [(314, 384)]},
                                   'intermediate': {'S1HL': [(0, 182)],
                                                    'S1Tr': [(182, 248)],
                                                    'M1': [(248, 274)],
                                                    'PPC': [(274, 300)]}},
                        'crazyjoe': {'distal': {'M1': [(0, 369)],
                                                'S1HL': [(369, 384)]},
                                     'intermediate': {'S1HL': [(0, 273)],
                                                      'S1Tr': [(273, 338)]}},
                        'frank': {'distal': {'A1': [(0, 47)],
                                             'A2D': [(47, 97)],
                                             'V2L': [(97, 235)],
                                             'V1d': [(235, 384)]},
                                  'intermediate': {'V1d': [(0, 52)],
                                                   'V1s': [(52, 90)],
                                                   'V2M': [(90, 195)]}},
                        'johnjohn': {'distal': {'A1': [(0, 44)],
                                                'A2D': [(44, 93)],
                                                'V2L': [(93, 233)],
                                                'V1d': [(233, 361)],
                                                'V1s': [(361, 384)]},
                                     'intermediate': {'V1s': [(0, 126)],
                                                      'V2M': [(126, 188)]}},
                        'kavorka': {'distal': {'A1': [(0, 110)],
                                               'A2D': [(110, 166)],
                                               'V2L': [(166, 334)],
                                               'V1d': [(334, 384)]},
                                    'intermediate': {'V1d': [(0, 147)],
                                                     'V1s': [(147, 221)],
                                                     'V2M': [(221, 291)]}}}

    def __init__(self, session='', cluster_groups_dir='', sp_profiles_csv=''):
        self.session = session
        self.cluster_groups_dir = cluster_groups_dir
        self.sp_profiles_csv = sp_profiles_csv

    def get_desired_clusters(self, **kwargs):
        """
        Description
        ----------
        This method enables one to filter in/out desired clusters based on a variety of
        properties: (1) animal name, (2) brain area of interest, (3) cluster type,
        (4) recording session type, (5) recording bank on the probe (6) session
        number, and (7) spiking profile, (8) LMI and SMI.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        filter_by_animal (list / bool)
            Animals to be included; defaults to True.
        filter_by_area (list / bool)
            Areas to be included, you can pick specific areas or
            general (A - auditory, M - motor, P - parietal, S - somatosensory, V - visual); defaults to True.
        filter_by_cluster_type (str / bool)
            Cluster type to be included: 'good' or 'mua'; defaults to True.
        filter_by_session_type (list / bool)
            Sessions to be included: 'light', 'dark', 'weight', 'sound'; defaults to True.
        filter_by_bank (str / bool)
            Bank to be included: 'distal' or 'intermediate'; defaults to True.
        filter_by_session_num (list / bool)
            Sessions to be included: 's1', 's2', etc.; defaults to True.
        filter_by_spiking_profile (str / bool)
            Profile to be included: 'RS' or 'FS'; defaults to True.
        filter_by_smi (str / bool)
            Select clusters that have a significant SMI; defaults to True.
        smi_criterion (float)
            The p-value for SMI significance; defaults to .001.
        filter_by_lmi (str / bool)
            Select clusters that have a significant LMI; defaults to True.
        lmi_criterion (float)
            The p-value for LMI significance; defaults to .005.
        sort_ch_num (bool)
            If True, sorts clusters by channel number; defaults to False.
        ----------

        Returns
        ----------
        cluster_list (list)
            A list with the name-sorted filtered clusters.
        ----------
        """

        filter_by_animal = kwargs['filter_by_animal'] if 'filter_by_animal' in kwargs.keys() and type(kwargs['filter_by_animal']) == list else True
        filter_by_area = kwargs['filter_by_area'] if 'filter_by_area' in kwargs.keys() and type(kwargs['filter_by_area']) == list else True
        filter_by_cluster_type = kwargs['filter_by_cluster_type'] if 'filter_by_cluster_type' in kwargs.keys() and type(kwargs['filter_by_cluster_type']) == str else True
        filter_by_session_type = kwargs['filter_by_session_type'] if 'filter_by_session_type' in kwargs.keys() and type(kwargs['filter_by_session_type']) == list else True
        filter_by_bank = kwargs['filter_by_bank'] if 'filter_by_bank' in kwargs.keys() and type(kwargs['filter_by_bank']) == str else True
        filter_by_session_num = kwargs['filter_by_session_num'] if 'filter_by_session_num' in kwargs.keys() and type(kwargs['filter_by_session_num']) == list else True
        filter_by_spiking_profile = kwargs['filter_by_spiking_profile'] if 'filter_by_spiking_profile' in kwargs.keys() and type(kwargs['filter_by_spiking_profile']) == str else True
        filter_by_smi = kwargs['filter_by_smi'] if 'filter_by_smi' in kwargs.keys() and type(kwargs['filter_by_smi']) == str else True
        smi_criterion = kwargs['smi_criterion'] if 'smi_criterion' in kwargs.keys() and type(kwargs['smi_criterion']) == float else .01
        filter_by_lmi = kwargs['filter_by_lmi'] if 'filter_by_lmi' in kwargs.keys() and type(kwargs['filter_by_lmi']) == str else True
        lmi_criterion = kwargs['lmi_criterion'] if 'lmi_criterion' in kwargs.keys() and type(kwargs['lmi_criterion']) == float else .05
        sort_ch_num = kwargs['sort_ch_num'] if 'sort_ch_num' in kwargs.keys() and type(kwargs['sort_ch_num']) == bool else False

        cluster_list = []
        if self.session != 0:
            if os.path.exists(self.session):
                if (filter_by_animal is True or any(animal in self.session for animal in filter_by_animal)) \
                        and (filter_by_bank is True or filter_by_bank in self.session) \
                        and (filter_by_session_type is True or any(s_type in self.session for s_type in filter_by_session_type)) \
                        and (filter_by_session_num is True or any(s_num in self.session for s_num in filter_by_session_num)):

                    # load specific pickle file segments
                    with open(self.session, 'rb') as session_file:
                        loaded_session = pickle.load(session_file)

                    file_info = loaded_session['file_info']
                    clusters = loaded_session['cell_names']

                    # get animal name, bank id and date of session
                    file_animal = [name for name in ClusterFinder.probe_site_areas.keys() if name in file_info][0]
                    if file_animal == 'bruno':
                        file_bank = 'distal'
                    else:
                        file_bank = [bank for bank in ['distal', 'intermediate'] if bank in file_info][0]
                    get_date_idx = [date.start() for date in re.finditer('20_s', file_info)][-1]
                    file_date = file_info[get_date_idx-4:get_date_idx+2]
                    if file_animal == 'bruno' and file_date == '030520':
                        file_date_cg = '020520'
                        file_date = '020520'
                    else:
                        file_date_cg = file_date

                    for cluster in clusters:
                        if filter_by_area is True and filter_by_cluster_type is True and filter_by_spiking_profile is True \
                                and filter_by_smi is True and filter_by_lmi is True:
                            cluster_list.append(cluster)
                        else:
                            if type(filter_by_cluster_type) == str:
                                # get cluster category ('good' or 'mua')
                                if not os.path.exists(self.cluster_groups_dir):
                                    print(f"Invalid location for directory {self.cluster_groups_dir}. Please try again.")
                                    sys.exit()
                                cluster_groups_json = f'{self.cluster_groups_dir}{os.sep}{file_animal}_{file_date_cg}_{file_bank}.json'
                                with open(cluster_groups_json) as json_file:
                                    cg_json = json.load(json_file)['imec0']
                            if filter_by_cluster_type is True or cluster in cg_json[filter_by_cluster_type]:
                                # get cluster area
                                cluster_peak_ch = int(cluster[15:])
                                for key, value in ClusterFinder.probe_site_areas[file_animal][file_bank].items():
                                    for site_range in value:
                                        if site_range[0] <= cluster_peak_ch < site_range[1]:
                                            cluster_area = key
                                            break
                                    else:
                                        continue
                                    break

                                if filter_by_area is True or any(area in cluster_area for area in filter_by_area):
                                    if type(filter_by_spiking_profile) == str or type(filter_by_smi) == str or type(filter_by_lmi) == str:
                                        # load profile data
                                        if not os.path.exists(self.sp_profiles_csv):
                                            print(f"Invalid location for file {self.sp_profiles_csv}. Please try again.")
                                            sys.exit()
                                        else:
                                            profile_data = pd.read_csv(self.sp_profiles_csv)

                                            # find cluster profile
                                            for idx, row in profile_data.iterrows():
                                                if row[0] == f'{file_animal}_{file_date}_{file_bank}' and row[1] == cluster:
                                                    cl_profile = row[7]
                                                    if row[9] < smi_criterion and row[8] < 0:
                                                        cl_smi = ['sign', 'neg']
                                                    elif row[9] < smi_criterion and row[8] > 0:
                                                        cl_smi = ['sign', 'pos']
                                                    else:
                                                        cl_smi = ['ns']

                                                    if row[11] < lmi_criterion < row[12] and row[10] < 0:
                                                        cl_lmi = ['sign', 'neg']
                                                    elif row[11] < lmi_criterion < row[12] and row[10] > 0:
                                                        cl_lmi = ['sign', 'pos']
                                                    else:
                                                        cl_lmi = ['ns']
                                                    break

                                    if (filter_by_spiking_profile is True or filter_by_spiking_profile == cl_profile) \
                                            and (filter_by_smi is True or filter_by_smi in cl_smi) \
                                            and (filter_by_lmi is True or filter_by_lmi in cl_lmi):
                                        cluster_list.append(cluster)

            else:
                print(f"Invalid location for file {self.session}. Please try again.")
                sys.exit()
        else:
            print("No session provided.")
            sys.exit()

        if sort_ch_num:
            cluster_list = sorted(cluster_list, key = lambda x: x.split('ch')[1])
        else:
            cluster_list.sort()
        return cluster_list
