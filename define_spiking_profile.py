# -*- coding: utf-8 -*-

"""

@author: bartulem

Define spiking profile: regular (RS) or fast (FS) spiking.

"""

import os
import sys
import json
import pandas as pd


class SpikingProfile:

    def __init__(self, cluster_quality_dir=0, cluster_groups_dir=0):
        self.cluster_quality_dir = cluster_quality_dir
        self.cluster_groups_dir = cluster_groups_dir

    def collect_measures_info(self, **kwargs):
        """
        Description
        ----------
        This method goes through all the "cluster quality measures" files and, for each
        cluster, picks out the "waveform duration", "FWHM" and "peak-to-through ratio",
        which help to define the spiking profile.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        categorize_mua (bool)
            Categorize spiking profile of MUA; defaults to False.
        ----------

        Returns
        ----------
        measures_info_df (pd.DataFrame)
            A DataFrame containing cluster measures info for all sessions.
        ----------
        """

        categorize_mua = kwargs['categorize_mua'] if 'categorize_mua' in kwargs.keys() and type(kwargs['categorize_mua']) == bool else False

        measures_info_df = pd.DataFrame(columns=['session_id', 'cluster_id', 'cluster_type',
                                                 'waveform_duration', 'fwhm', 'pt_ratio', 'spiking_profile'])

        if os.path.exists(self.cluster_groups_dir):
            cluster_id_count = 0
            for file in os.listdir(self.cluster_groups_dir):
                print(file)
                # get cluster names
                with open(f'{self.cluster_groups_dir}{os.sep}{file}') as json_file:
                    cg_json = json.load(json_file)['imec0']

                # get file name base
                file_name_base = file[:-5]

                # select clusters of interest
                if categorize_mua:
                    clusters = cg_json['good'] + cg_json['mua']
                else:
                    clusters = cg_json['good']

                # load appropriate cluster measures file
                if os.path.exists(self.cluster_quality_dir):
                    quality_file_name = f'cqm_{file_name_base}'

                    # load cluster quality measures file
                    with open(f'{self.cluster_quality_dir}{os.sep}{quality_file_name}.json') as cqm_json_file:
                        cqm_json = json.load(cqm_json_file)
                else:
                    print(f"Invalid location for directory {self.cluster_quality_dir}. Please try again.")
                    sys.exit()

                # find relevant information
                for cluster in clusters:
                    measures_info_df.loc[cluster_id_count, 'session_id'] = file_name_base
                    measures_info_df.loc[cluster_id_count, 'cluster_id'] = cluster
                    measures_info_df.loc[cluster_id_count, 'cluster_type'] = [key for key, value in cg_json.items() if cluster in value][0]

                    # get shortened cluster name
                    cl_name_short = str(int(cluster[cluster.index('cl')+2:cluster.index('cl')+6]))

                    # get cluster quality variables
                    measures_info_df.loc[cluster_id_count, 'waveform_duration'] = cqm_json[cl_name_short]['waveform_metrics'][1]
                    measures_info_df.loc[cluster_id_count, 'fwhm'] = cqm_json[cl_name_short]['waveform_metrics'][2]
                    measures_info_df.loc[cluster_id_count, 'pt_ratio'] = cqm_json[cl_name_short]['waveform_metrics'][3]

                    cluster_id_count += 1

        else:
            print(f"Invalid location for directory {self.cluster_groups_dir}. Please try again.")
            sys.exit()

        return measures_info_df
