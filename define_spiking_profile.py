# -*- coding: utf-8 -*-

"""

@author: bartulem

Define spiking profile: regular (RS) or fast (FS) spiking.

"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import operator
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm


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
        which could aid in characterizing the spiking profile.
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
                                                 'waveform_duration', 'fwhm', 'pt_ratio'])

        if os.path.exists(self.cluster_groups_dir):
            cluster_id_count = 0
            for file in os.listdir(self.cluster_groups_dir):
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

    def get_spiking_profiles(self, **kwargs):
        """
        Description
        ----------
        This method computes the spiking profile for every input cluster. Based on the
        distribution of putative single-unit clusters on 3 relevant variables ('waveform_duration',
        'fwhm', and 'pt_ratio'), we assume there are three types of spiking profiles: 'RS' (regular
        spiking), 'FS' (fast spiking) and 'BS' (broad spiking), whose centroids are further uncovered
        by KMeans clustering. Based on the distances to those centroids, we classify every cluster
        in one of those three categories.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        relevant_variables (list)
            Variables to perform KMeans on; defaults to ['waveform_duration', 'fwhm', 'pt_ratio'].
        num_of_clusters (int)
            Number of clusters to form (centroids to generate); defaults to 3.
        to_plot (bool)
            Yey or ney on the KMeans scatter; defaults to False.
        profile_colors (dict)
            What colors to use for each profile in the scatter plot; defaults to {'RS': '#7F7F7F', 'FS': '#000000', 'BS': '#B0B0B0'}.
        save_fig (bool)
            Save the figure or not; defaults to False.
        fig_format (str)
            The format of the figure; defaults to 'png'.
        save_fig_dir (bool)
            Directory to save the figure in; defaults to '/home/bartulm/Downloads'.
        save_df (bool)
            Save DataFrame as .csv file; defaults to False.
        save_df_dir (bool)
            Directory to save the DataFrame in; defaults to '/home/bartulm/Downloads'.
        ----------

        Returns
        ----------
        measures_info_df (pd.DataFrame)
            A DataFrame containing cluster measures with spiking profiles for all sessions.
        ----------
        """

        relevant_variables = kwargs['relevant_variables'] if 'relevant_variables' in kwargs.keys() and type(kwargs['relevant_variables']) == list else ['waveform_duration', 'fwhm', 'pt_ratio']
        num_of_clusters = kwargs['num_of_clusters'] if 'num_of_clusters' in kwargs.keys() and type(kwargs['num_of_clusters']) == int else 3
        to_plot = kwargs['to_plot'] if 'to_plot' in kwargs.keys() and type(kwargs['to_plot']) == bool else False
        profile_colors = kwargs['profile_colors'] if 'profile_colors' in kwargs.keys() and type(kwargs['profile_colors']) == dict else {'RS': '#7F7F7F', 'FS': '#000000', 'BS': '#B0B0B0'}
        save_fig = kwargs['save_fig'] if 'save_fig' in kwargs.keys() and type(kwargs['save_fig']) == bool else False
        fig_format = kwargs['fig_format'] if 'fig_format' in kwargs.keys() and type(kwargs['fig_format']) == str else 'png'
        save_fig_dir = kwargs['save_fig_dir'] if 'save_fig_dir' in kwargs.keys() and type(kwargs['save_fig_dir']) == str else '/home/bartulm/Downloads'
        save_df = kwargs['save_df'] if 'save_df' in kwargs.keys() and type(kwargs['save_df']) == bool else False
        save_df_dir = kwargs['save_df_dir'] if 'save_df_dir' in kwargs.keys() and type(kwargs['save_df_dir']) == str else '/home/bartulm/Downloads'

        # extract measures info dataframe
        measures_info_df = self.collect_measures_info()

        # the dataframe has nans in the 'fwhm' column, and we need to know those rows that don't
        rows_without_nan = [index for index, row in measures_info_df.iterrows() if ~row.isnull().any()]

        # prepare the nan-purged variables for KMeans (by peak-normalizing)
        analysis_arr = np.zeros((len(rows_without_nan), len(relevant_variables)))
        normalizing_peaks = [measures_info_df.loc[rows_without_nan, variable].max() for variable in relevant_variables]
        for idx, variable in enumerate(relevant_variables):
            analysis_arr[:, idx] = measures_info_df.loc[rows_without_nan, variable] / normalizing_peaks[idx]

        # perform KMeans
        Kmean = KMeans(n_clusters=num_of_clusters).fit(analysis_arr)

        # if there are three clusters, the cluster labels are 1, 2, 3 (which mean very little)
        # but we also know that there are more RS than FS clusters, and more FS than BS ones, so we can get them by ordering
        value_counts_dict = sorted(dict(Counter(Kmean.labels_)).items(), key=operator.itemgetter(1), reverse=True)

        # create a character array (that can take unicode) and fill it with profile IDs
        # again, this rests on the assumption: count(RS) > count(FS) > count(BS)
        char_arr = np.chararray(Kmean.labels_.shape[0], itemsize=2, unicode=True)
        char_arr[:] = 'AA'
        char_arr[Kmean.labels_ == value_counts_dict[0][0]] = 'RS'
        char_arr[Kmean.labels_ == value_counts_dict[1][0]] = 'FS'
        if len(profile_colors.keys()) == 3:
            char_arr[Kmean.labels_ == value_counts_dict[2][0]] = 'BS'

        # place these values into a new column of the dataframe
        measures_info_df.loc[rows_without_nan, 'spiking_profile'] = char_arr

        # plot the KMeans results if desired
        if to_plot:
            fig = plt.figure(dpi=300)
            ax1 = fig.add_subplot(111, projection='3d')
            zm = np.ma.masked_greater_equal(np.array(measures_info_df.loc[rows_without_nan, 'pt_ratio']), 15, copy=True)
            xm = np.ma.MaskedArray(np.array(measures_info_df.loc[rows_without_nan, 'waveform_duration']), mask=zm.mask)
            ym = np.ma.MaskedArray(np.array(measures_info_df.loc[rows_without_nan, 'fwhm']), mask=zm.mask)
            cm = np.ma.MaskedArray(np.array([profile_colors[profile] for profile in char_arr]), mask=zm.mask)
            ax1.scatter(xm, ym, zm, c=cm)
            ax1.set_zlim(0, 15)
            ax1.set_xlabel('spike duration')
            ax1.set_ylabel('FWHM')
            ax1.set_zlabel('peak-to-through ratio')
            if len(profile_colors.keys()) == 3:
                circle_1 = Line2D([0], [0], linestyle='none', marker='o', alpha=1., markersize=8, markeredgecolor='none', markerfacecolor=profile_colors['FS'])
                circle_2 = Line2D([0], [0], linestyle='none', marker='o', alpha=1., markersize=8, markeredgecolor='none', markerfacecolor=profile_colors['RS'])
                circle_3 = Line2D([0], [0], linestyle='none', marker='o', alpha=1., markersize=8, markeredgecolor='none', markerfacecolor=profile_colors['BS'])
                ax1.legend((circle_1, circle_2, circle_3), ['FS', 'RS', 'BS'])
            if save_fig:
                fig.savefig(f'{save_fig_dir}{os.sep}spiking_profiles_kmeans.{fig_format}')
            plt.show()

        # categorize clusters that had nans in the 'fwhm' column
        for row in tqdm(measures_info_df.index):
            if ~measures_info_df.loc[row, :].isnull().any():
                continue
            else:
                # normalize the individual values by variable peaks
                xz_arr = np.array([measures_info_df.loc[row, 'waveform_duration'] / normalizing_peaks[0], measures_info_df.loc[row, 'pt_ratio'] / normalizing_peaks[2]])

                # re-order profiles based on KMeans ID
                profiles = list(profile_colors.keys())
                xz_distances_dict = {profiles[item[0]]: 0 for item in value_counts_dict}

                # calculate distances to all three centroids
                xz_distances_dict = {key: np.abs(np.linalg.norm(xz_arr-np.array([Kmean.cluster_centers_[idx, 0], Kmean.cluster_centers_[idx, 2]])))
                                     for idx, (key, value) in enumerate(xz_distances_dict.items())}

                # select the profile with the smallest distance from centroid to point (idx 0 is kye, idx 1 is item)
                measures_info_df.loc[row, 'spiking_profile'] = min(xz_distances_dict.items(), key=operator.itemgetter(1))[0]

        # save dataframe for posterity
        if save_df:
            measures_info_df.to_csv(path_or_buf=f'{save_df_dir}{os.sep}spiking_profiles.csv', sep=';', index=False)

        return measures_info_df
