# -*- coding: utf-8 -*-

"""

@author: bartulem

Select appropriate clusters.

"""

import os
import sys
import pickle
import json


class ClusterFinder:

    probe_site_areas = {'bruno': {'distal': {'CaPu': [(0, 15)],
                                             'WhMa': [(15, 80)],
                                             'S1HL': [(80, 225)],
                                             'S1Tr': [(225, 276)],
                                             'M1': [(276, 313)],
                                             'PPC': [(313, 384)]}},
                        'roy': {'distal': {'M1': [(0, 384)]},
                                'intermediate': {'M1': [(0, 32), (102, 243)],
                                                 'S1HL': [(32, 102)]}},
                        'jacopo': {'distal': {'M2': [(0, 124)],
                                              'M1': [(124, 384)]},
                                   'intermediate': {'M1': [(0, 5), (233, 256)],
                                                    'S1HL': [(5, 192)],
                                                    'S1Tr': [(192, 233)],
                                                    'PPC': [(256, 300)]}},
                        'crazyjoe': {'distal': {'M2': [(0, 73)],
                                                'M1': [(73, 384)]},
                                     'intermediate': {'M1': [(0, 55)],
                                                      'S1HL': [(55, 296)],
                                                      'S1Tr': [(296, 338)]}},
                        'frank': {'distal': {'A1': [(0, 61)],
                                             'A2D': [(61, 138)],
                                             'V2L': [(138, 205)],
                                             'V1': [(205, 384)]},
                                  'intermediate': {'V1': [(0, 67)],
                                                   'V2M': [(67, 195)]}},
                        'johnjohn': {'distal': {'A1': [(0, 44)],
                                                'A2D': [(44, 100)],
                                                'V2L': [(100, 160)],
                                                'V1': [(160, 384)]},
                                     'intermediate': {'V1': [(0, 67)],
                                                      'V2M': [(67, 188)]}},
                        'kavorka': {'distal': {'A1': [(0, 75)],
                                               'A2D': [(75, 179)],
                                               'V2L': [(179, 281)],
                                               'V1': [(281, 384)]},
                                    'intermediate': {'V1': [(0, 230)],
                                                     'V2M': [(230, 291)]}}}

    def __init__(self, session=0, cluster_groups_dir=0):
        self.session = session
        self.cluster_groups_dir = cluster_groups_dir

    def get_desired_clusters(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs: dictionary
        filter_by_animal : list/bool
            Animals to be included; defaults to True.
        filter_by_area : list/bool
            Areas to be included, you can pick specific areas or
            general (A - auditory, M - motor, P - parietal, S - somatosensory, V - visual); defaults to True.
        filter_by_cluster_type : str/bool
            Cluster type to be included: 'good' or 'mua'; defaults to True.
        filter_by_session_type : list/bool
            Sessions to be included: 'light', 'dark', 'weight', 'sound'; defaults to True.
        filer_by_bank : str/bool
            Bank to be included: 'distal' or 'intermediate'; defaults to True.
        filter_by_session_num : list/bool
            Sessions to be included: 's1', 's2', etc.; defaults to True.
        ----------

        Returns
        ----------
        cluster_dictionary : dict
            A dictionary with file names as keys and list of filtered clusters as values.
        ----------
        """

        filter_by_animal = kwargs['filter_by_animal'] if 'filter_by_animal' in kwargs.keys() and type(kwargs['filter_by_animal']) == list else True
        filter_by_area = kwargs['filter_by_area'] if 'filter_by_area' in kwargs.keys() and type(kwargs['filter_by_area']) == list else True
        filter_by_cluster_type = kwargs['filter_by_cluster_type'] if 'filter_by_cluster_type' in kwargs.keys() and type(kwargs['filter_by_cluster_type']) == str else True
        filter_by_session_type = kwargs['filter_by_session_type'] if 'filter_by_session_type' in kwargs.keys() and type(kwargs['filter_by_session_type']) == list else True
        filer_by_bank = kwargs['filer_by_bank'] if 'filer_by_bank' in kwargs.keys() and type(kwargs['filer_by_bank']) == str else True
        filter_by_session_num = kwargs['filter_by_session_num'] if 'filter_by_session_num' in kwargs.keys() and type(kwargs['filter_by_session_num']) == list else True

        cluster_list = []
        if self.session != 0:
            if os.path.exists(self.session):
                if (filter_by_animal is True or any(animal in self.session for animal in filter_by_animal)) \
                        and (filer_by_bank is True or filer_by_bank in self.session) \
                        and (filter_by_session_type is True or any(s_type in self.session for s_type in filter_by_session_type)) \
                        and (filter_by_session_num is True or any(s_num in self.session for s_num in filter_by_session_num)):

                    # load specific pickle file segments
                    with open(self.session, 'rb') as session_file:
                        loaded_session = pickle.load(session_file)

                    file_info = loaded_session['file_info']
                    clusters = loaded_session['cell_names']

                    # get animal name, bank id and date of session
                    file_animal = [name for name in ClusterFinder.probe_site_areas.keys() if name in file_info][0]
                    file_bank = [bank for bank in ['distal', 'intermediate'] if bank in file_info][0]
                    file_date = file_info[file_info.find('20')-4:file_info.find('20')+2]

                    for cluster in clusters:
                        if filter_by_area is True and filter_by_cluster_type is True:
                            cluster_list.append(cluster)
                        else:
                            # get cluster category ('good' or 'mua')
                            if not os.path.exists(self.cluster_groups_dir):
                                print(f"Invalid location for directory {self.cluster_groups_dir}. Please try again.")
                                sys.exit()
                            cluster_groups_json = f'{self.cluster_groups_dir}{os.sep}{file_animal}_{file_date}_{file_bank}.json'
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
                                    cluster_list.append(cluster)
            else:
                print(f"Invalid location for file {self.session}. Please try again.")
                sys.exit()
        else:
            print("No session provided.")
            sys.exit()

        cluster_list.sort()
        return cluster_list
