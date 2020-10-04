# -*- coding: utf-8 -*-

"""

@author: bartulem

Select appropriate clusters.

"""

import os
import sys
import pickle


class ClusterFinder:

    probe_site_areas = {'bruno': {'distal': {'N/A': [(0, 16)],
                                             'white_m': [(16, 81)],
                                             'S1HL': [(81, 226)],
                                             'S1Tr': [(226, 277)],
                                             'N/A2': [(277, 314)],
                                             'PPC': [(314, 384)]}},
                        'roy': {'distal': {'M1': [(0, 384)]},
                                'intermediate': {'M1': [(0, 32), (102, 243)],
                                                 'S1HL': [(32, 102)]}},
                        'jacopo': {'distal': {'M2': [(0, 124)],
                                              'M1': [(124, 384)]},
                                   'intermediate': {'M1': [(0, 5)],
                                                    'S1HL': [(5, 192)],
                                                    'S1Tr': [(192, 233)],
                                                    'N/A2': [(233, 256)],
                                                    'PPC': [(256, 300)]}},
                        'crazyjoe': {'distal': {'M2': [(0, 73)],
                                                'M1': [(73, 384)]},
                                     'intermediate': {'M1': [(0, 55)],
                                                      'S1HL': [(55, 296)],
                                                      'S1Tr': [(296, 339)]}},
                        'frank': {'distal': {'A1': [(0, 63)],
                                             'A2D': [(63, 141)],
                                             'V2L': [(141, 209)],
                                             'V1': [(209, 384)]},
                                  'intermediate': {'V1': [(0, 75)],
                                                   'V2M': [(75, 205)]}},
                        'johnjohn': {'distal': {'A1': [(0, 64)],
                                                'A2D': [(64, 120)],
                                                'V2L': [(120, 181)],
                                                'V1': [(181, 384)]},
                                     'intermediate': {'V1': [(0, 87)],
                                                      'V2M': [(87, 208)]}},
                        'kavorka': {'distal': {'A1': [(0, 76)],
                                               'A2D': [(76, 180)],
                                               'V2L': [(180, 282)],
                                               'V1': [(282, 384)]},
                                    'intermediate': {'V1': [(0, 231)],
                                                     'V2M': [(231, 292)]}}}

    def __init__(self, session_list=0):
        self.session_list = session_list

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
        ----------

        Returns
        ----------
        cluster_dictionary : dict
            A dictionary with file names as keys and list of filtered clusters as values.
        ----------
        """

        filter_by_animal = kwargs['filter_by_animal'] if 'filter_by_animal' in kwargs.keys() and kwargs['filter_by_animal'] == list else True
        filter_by_area = kwargs['filter_by_area'] if 'filter_by_area' in kwargs.keys() and kwargs['filter_by_area'] == list else True
        filter_by_cluster_type = kwargs['filter_by_cluster_type'] if 'filter_by_cluster_type' in kwargs.keys() and kwargs['filter_by_cluster_type'] == str else True
        filter_by_session_type = kwargs['filter_by_session_type'] if 'filter_by_session_type' in kwargs.keys() and kwargs['filter_by_session_type'] == list else True
        filer_by_bank = kwargs['filer_by_bank'] if 'filer_by_bank' in kwargs.keys() and kwargs['filer_by_bank'] == str else True

        cluster_dictionary = {}
        if self.session_list != 0:
            for session in self.session_list:
                if os.path.exists(session):
                    if (filter_by_animal is True or any(animal in session for animal in filter_by_animal)) \
                            and (filer_by_bank is True or filer_by_bank in session) \
                            and (filter_by_session_type is True or any(s_type in session for s_type in filter_by_session_type)):

                        # load specific pickle file segments
                        with open(session, 'rb') as session_file:
                            file_info = pickle.load(session_file)['file_info']
                            clusters = pickle.load(session_file)['cell_names']

                        # get animal name and bank id
                        file_animal = [name for name in ClusterFinder.probe_site_areas.keys() if name in file_info][0]
                        file_bank = [bank for bank in ['distal', 'intermediate'] if bank in file_info][0]

                        # create dictionary entry with file name
                        cluster_dictionary[file_info] = []

                        for cluster in clusters:
                            if filter_by_area is True and filter_by_cluster_type is True:
                                cluster_dictionary[file_info].append(cluster)
                            else:
                                cluster_peak_ch = int(cluster[15:])
                                ClusterFinder.probe_site_areas[file_animal][file_bank] = 0

                else:
                    print(f"Location invalid for file {session}. Please try again.")
                    sys.exit()
        else:
            print("No sessions provided.")
            sys.exit()