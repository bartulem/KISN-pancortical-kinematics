"""
Performs dimensionality reduction on neural data.
@author: bartulem
"""

import io
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap
from scipy.stats import zscore
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import decode_events
import neural_activity
import sessions2load
import select_clusters


def get_condensed_features(behavioral_data, bin_size_ms=100):
    condensed = {}
    for var in behavioral_data.keys():
        if var == 'speeds':
            condensed[var] = neural_activity.condense_frame_arrays(frame_array=behavioral_data[var][:, 3],
                                                                   camera_framerate=120.,
                                                                   bin_size_ms=bin_size_ms,
                                                                   arr_type=False,
                                                                   sound=False)
        if var == 'sorted_point_data':
            condensed['neck_elevation'] = neural_activity.condense_frame_arrays(frame_array=behavioral_data[var][:, 4, 2],
                                                                                camera_framerate=120.,
                                                                                bin_size_ms=bin_size_ms,
                                                                                arr_type=False,
                                                                                sound=False)
        elif var == 'neck_1st_der' or var == 'body_direction':
            condensed[var] = neural_activity.condense_frame_arrays(frame_array=behavioral_data[var],
                                                                   camera_framerate=120.,
                                                                   bin_size_ms=bin_size_ms,
                                                                   arr_type=False,
                                                                   sound=False)
        elif var != 'total_frame_num' and var != 'speeds' and var != 'sorted_point_data' \
                and var != 'neck_1st_der' and var != 'body_direction':
            condensed[var] = {}
            for n_col in range(behavioral_data[var].shape[1]):
                condensed[var][n_col] = neural_activity.condense_frame_arrays(frame_array=behavioral_data[var][:, n_col],
                                                                              camera_framerate=120., bin_size_ms=bin_size_ms,
                                                                              arr_type=False,
                                                                              sound=False)
    return condensed


class LatentSpace:

    def __init__(self, input_dict={}, cluster_groups_dir='', sp_profiles_csv='',
                 save_fig=False, fig_format='png', save_dir='', cch_summary_file=''):
        self.input_dict = input_dict
        self.cluster_groups_dir = cluster_groups_dir
        self.sp_profiles_csv = sp_profiles_csv
        self.save_fig = save_fig
        self.fig_format = fig_format
        self.save_dir = save_dir
        self.cch_summary_file = cch_summary_file

    def activity_pca(self, **kwargs):
        """
        Description
        ----------
        This method does PCA on neural activity concatenated across three different
        sessions and correlates the latent variables with behavioral features, and
        also low-dimensionality embedding of activity PCs with UMAP.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        area_filter (list / bool)
            Areas to be included, you can pick specific areas or
            general (A - auditory, M - motor, P - parietal, S - somatosensory, V - visual); defaults to True.
        cluster_type_filter (str / bool)
            Cluster type to be included: 'good' or 'mua'; defaults to True.
        profile_filter (str / bool)
            Profile to be included: 'RS' or 'FS'; defaults to True.
        condition (str)
            If this is 'luminance' and there is 'V' in areas,
            it selects clusters which are only present in light/dark; defaults to 'luminance'.
        feature_list (list)
            Features of interest; defaults to ['speeds', 'ego3_head_ang', 'opt_back_ang',
                                               'ego3_head_1st_der', 'sorted_point_data',
                                               'neck_1st_der', 'allo_head_ang', 'body_direction']
        condense_bin_ms (int)
            The size of bin for spikes; defaults to 100 (ms).
        num_components (int)
            How many components to plot; defaults to True (all).

        Returns
        ----------
        pc_corrs (figure)
            A figure depicting correlation of behavioral features with PCs across three sessions.
        ----------
        """

        cluster_areas = kwargs['cluster_areas'] if 'cluster_areas' in kwargs.keys() and type(kwargs['cluster_areas']) == list else True
        cluster_type = kwargs['cluster_type'] if 'cluster_type' in kwargs.keys() and type(kwargs['cluster_type']) == str else True
        cluster_profile = kwargs['cluster_profile'] if 'cluster_profile' in kwargs.keys() and type(kwargs['cluster_profile']) == str else True
        condition = kwargs['condition'] if 'condition' in kwargs.keys() and type(kwargs['condition']) == str else 'luminance'
        feature_list = kwargs['feature_list'] if 'feature_list' in kwargs.keys() and type(kwargs['feature_list']) == list else ['speeds', 'ego3_head_ang', 'opt_back_ang',
                                                                                                                                'ego3_head_1st_der', 'sorted_point_data',
                                                                                                                                'neck_1st_der', 'allo_head_ang', 'body_direction']
        condense_bin_ms = kwargs['condense_bin_ms'] if 'condense_bin_ms' in kwargs.keys() and type(kwargs['condense_bin_ms']) == int else 100
        num_components = kwargs['num_components'] if 'num_components' in kwargs.keys() and type(kwargs['num_components']) == int else True

        # find session info
        session_id = self.input_dict['light1'][88:]
        session_animal = [animal for animal in select_clusters.ClusterFinder.probe_site_areas.keys() if animal in session_id][0]
        session_bank = [bank for bank in ['distal', 'intermediate'] if bank in session_id][0]
        session_date = session_id[session_id.find('20') - 4:session_id.find('20') + 2]

        # choose clusters present in all sessions
        all_clusters, \
        chosen_clusters, \
        extra_chosen_clusters, \
        cluster_dict = decode_events.choose_012_clusters(the_input_012=list(self.input_dict.values()),
                                                         cl_gr_dir=self.cluster_groups_dir,
                                                         sp_prof_csv=self.sp_profiles_csv,
                                                         cl_areas=cluster_areas,
                                                         cl_type=cluster_type,
                                                         desired_profiles=cluster_profile,
                                                         dec_type=condition)

        # extract/condense beh. feature data
        behavioral_data = {}
        new_shapes = {}
        for session in self.input_dict.keys():
            file_id, feature_data = sessions2load.Session(session=self.input_dict[session]).data_loader(extract_variables=feature_list)
            behavioral_data[session] = get_condensed_features(feature_data)
            new_shapes[session] = feature_data['total_frame_num'] // int(120. * (condense_bin_ms / 1000))

        # condense neural data
        neural_data = {}
        for s_idx, (session, full_session_path) in enumerate(self.input_dict.items()):
            if s_idx < 2:
                which_extra = s_idx
            else:
                which_extra = 0
            file_id, \
            activity_dictionary, \
            purged_spikes_dictionary = neural_activity.Spikes(input_file=full_session_path).convert_activity_to_frames_with_shuffles(get_clusters=chosen_clusters + extra_chosen_clusters[which_extra],
                                                                                                                                     to_shuffle=False,
                                                                                                                                     condense_arr=True,
                                                                                                                                     condense_bin_ms=condense_bin_ms)
            neural_data_binned = np.zeros((len(all_clusters), new_shapes[session]))
            for cl_idx, cl in enumerate(all_clusters):
                if cl in activity_dictionary.keys():
                    neural_data_binned[cl_idx, :] = activity_dictionary[cl]['activity'].todense()

            neural_data[session] = neural_data_binned

        # concatenate sessions/standardize data and conduct PCA
        neural_data_all = np.concatenate((neural_data[list(self.input_dict.keys())[0]],
                                          neural_data[list(self.input_dict.keys())[1]],
                                          neural_data[list(self.input_dict.keys())[2]]), axis=1)

        neural_data_all = zscore(neural_data_all, axis=1)
        shape_1 = neural_data[list(self.input_dict.keys())[0]].shape[1]
        shape_2 = neural_data[list(self.input_dict.keys())[1]].shape[1]
        # shape_3 = neural_data[list(self.input_dict.keys())[2]].shape[1]

        # rearrange array to shape (n_samples, n_features)
        input_arr = neural_data_all.T

        # do PCA
        pca = PCA(whiten=True)
        pc_embedding = pca.fit_transform(input_arr)

        # find the knee/elbow in scree plot and keep only pcs before
        pc_var_explained = pca.explained_variance_ratio_
        kn = KneeLocator(list(range(len(pc_var_explained))), pc_var_explained, curve='convex', direction='decreasing')
        pcs_to_keep = pc_embedding[:, :kn.knee]

        # plot the knee position
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(pc_var_explained, color='#999999')
        # ax.axvline(x=kn.knee, ls='--', color='#000000')
        # ax.set_xlabel('PC')
        # ax.set_xticks(list(range(0, len(pc_var_explained), 100)) + [kn.knee])
        # ax.set_ylabel('% variance explained')
        # ax.set_yticks([.005, .015, .025, .035])
        # ax.set_yticklabels([.5, 1.5, 2.5, 3.5])
        # plt.show()

        # do UMAP on reduced PC space
        reducer = umap.UMAP(n_components=2, n_neighbors=20)
        mapper = reducer.fit_transform(pcs_to_keep)

        # plot results
        fig2, ax2 = plt.subplots(1, 1)
        ax2.scatter(mapper[:shape_1, 0], mapper[:shape_1, 1], s=.01, c='#EEC900', alpha=1, label='light1')
        ax2.scatter(mapper[shape_1:shape_1 + shape_2, 0], mapper[shape_1:shape_1 + shape_2, 1], s=.01, c='#00008B', alpha=1, label='dark')
        ax2.scatter(mapper[shape_1 + shape_2:, 0], mapper[shape_1 + shape_2:, 1], s=.01, c='#CD950C', alpha=1, label='light2')
        ax2.set_xlim(-5, 7.5)
        ax2.set_xlabel('UMAP1 (A.U.)')
        ax2.set_ylabel('UMAP2 (A.U.)')
        ax2.legend(markerscale=100)
        ax2.set_title(f'{session_animal}_{session_date}_{session_bank}: {cluster_areas[0]}')
        if self.save_fig:
            if os.path.exists(self.save_dir):
                fig2.savefig(f'{self.save_dir}{os.sep}UMAP_{session_animal}_{session_bank}_{cluster_areas[0]}.{self.fig_format}')
            else:
                print("Specified save directory doesn't exist. Try again.")
                sys.exit()
        plt.show()

    def functional_subspaces_cch(self, **kwargs):
        """
        Description
        ----------
        This method uses GLM results and sensory tuning information to create
        functional subspaces through PCA and low-dimensionality embedding (UMAP).
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        df_pca_columns (list / bool)
            Columns of the spiking profile csv file to be used for dimensionality reduction;
            defaults to ['SMI', 'pSMI', 'LMI', 'pLMI', 'pLMIcheck',
                         'B Speeds', 'C Body_direction', 'C Body_direction_1st_der',
                         'D Allo_head_direction', 'D Allo_head_direction_1st_der',
                         'G Neck_elevation', 'G Neck_elevation_1st_der', 'K Ego3_Head_roll',
                         'K Ego3_Head_roll_1st_der', 'L Ego3_Head_pitch', 'L Ego3_Head_pitch_1st_der',
                         'M Ego3_Head_azimuth',	'M Ego3_Head_azimuth_1st_der', 'N Back_pitch',
                         'N Back_pitch_1st_der', 'O Back_azimuth', 'O Back_azimuth_1st_der',
                         'P Ego2_head_roll', 'P Ego2_head_roll_1st_der', 'Q Ego2_head_pitch',
                         'Q Ego2_head_pitch_1st_der', 'Z Position', 'Z Self_motion'].
        pc_selection (str)
            Has three options: 'knee' for inflexion in scree plot, 'all' for everything
            and 'proportion' for variance explained proportion; defaults to 'knee'
        var_explained_proportion (float)
            Works only if above is 'proportion'; defaults to .9.
        umap_parameters (dict)
            User defined parameters; defaults to {'n_components'=2, 'n_neighbors'=5
                                                  'min_dist': .1, 'metric':'euclidean'}
        plot_scree (bool)
            To show or not to show scree plot; defaults to False.
        save_md (bool)
            To save or not to save multi-dimensional functional distances; defaults to False.

        Returns
        ----------
        functional_umap (.npy file)
            An array with the UMAP 2-dimensional embedding of functional features.
        ----------
        """

        df_pca_columns = kwargs['df_pca_columns'] if 'df_pca_columns' in kwargs.keys() and \
                                                     type(kwargs['df_pca_columns']) == list else ['SMI', 'pSMI', 'LMI', 'pLMI', 'pLMIcheck',
                                                                                                  'B Speeds', 'C Body_direction', 'C Body_direction_1st_der',
                                                                                                  'D Allo_head_direction', 'D Allo_head_direction_1st_der',
                                                                                                  'G Neck_elevation', 'G Neck_elevation_1st_der', 'K Ego3_Head_roll',
                                                                                                  'K Ego3_Head_roll_1st_der', 'L Ego3_Head_pitch', 'L Ego3_Head_pitch_1st_der',
                                                                                                  'M Ego3_Head_azimuth', 'M Ego3_Head_azimuth_1st_der', 'N Back_pitch',
                                                                                                  'N Back_pitch_1st_der', 'O Back_azimuth', 'O Back_azimuth_1st_der',
                                                                                                  'P Ego2_head_roll', 'P Ego2_head_roll_1st_der', 'Q Ego2_head_pitch',
                                                                                                  'Q Ego2_head_pitch_1st_der', 'Z Position', 'Z Self_motion']
        pc_selection = kwargs['pc_selection'] if 'pc_selection' in kwargs.keys() and kwargs['pc_selection'] in ['knee', 'all', 'proportion'] else 'knee'
        var_explained_proportion = kwargs['var_explained_proportion'] if 'var_explained_proportion' in kwargs.keys() and type(kwargs['var_explained_proportion']) == float else .9
        umap_parameters = kwargs['umap_parameters'] if 'umap_parameters' in kwargs.keys() and type(kwargs['umap_parameters']) == dict \
            else {'n_components': 2, 'n_neighbors': 5, 'min_dist': .1, 'metric': 'euclidean'}
        plot_scree = kwargs['plot_scree'] if 'plot_scree' in kwargs.keys() and type(kwargs['plot_scree']) == bool else False
        save_md = kwargs['save_md'] if 'save_md' in kwargs.keys() and type(kwargs['save_md']) == bool else False

        spc = pd.read_csv(self.sp_profiles_csv)

        with open(self.cch_summary_file, 'r') as summary_file:
            synaptic_data = json.load(summary_file)

        # screen for first covariate nan values, so they can be excluded
        non_nan_idx_list = spc.loc[~pd.isnull(spc.loc[:, 'first_covariate'])].index.tolist()

        # select important columns, standardize and replace NANs
        spc_filtered = spc.loc[non_nan_idx_list, df_pca_columns].values
        spc_filtered = StandardScaler().fit_transform(spc_filtered)
        np.nan_to_num(x=spc_filtered, copy=False, nan=0)

        # before dim reduction, test there is a md distance between areas
        if save_md:
            pl_dict = {'VV': {'md_distance': []}, 'AA': {'md_distance': []},
                       'MM': {'md_distance': []}, 'SS': {'md_distance': []}}
            for area in synaptic_data.keys():
                if area in pl_dict.keys():
                    if area == 'VV' or area == 'AA':
                        animal_list = ['kavorka', 'johnjohn', 'frank']
                    else:
                        animal_list = ['jacopo', 'crazyjoe', 'roy']
                    for animal in animal_list:
                        for animal_session in synaptic_data[area][animal].keys():
                            for pair_idx, pair in enumerate(synaptic_data[area][animal][animal_session]['pairs']):
                                cl1, cl2 = pair.split('-')
                                spc_pos1 = spc[(spc['cluster_id'] == cl1) & (spc['session_id'] == animal_session)].index.tolist()[0]
                                spc_pos2 = spc[(spc['cluster_id'] == cl2) & (spc['session_id'] == animal_session)].index.tolist()[0]
                                if spc_pos1 in non_nan_idx_list and spc_pos2 in non_nan_idx_list:
                                    pos1 = non_nan_idx_list.index(spc_pos1)
                                    pos2 = non_nan_idx_list.index(spc_pos2)
                                    pl_dict[area]['md_distance'].append(np.abs(np.linalg.norm(spc_filtered[pos1, :] - spc_filtered[pos2, :])))
            with io.open(f'{self.save_dir}{os.sep}md_distances.json', 'w', encoding='utf-8') as to_save_file:
                to_save_file.write(json.dumps(pl_dict, ensure_ascii=False, indent=4))

        # do PCA
        pca = PCA(whiten=True)
        pc_embedding = pca.fit_transform(spc_filtered)
        pc_var_explained = pca.explained_variance_ratio_

        # select the relevant PC loadings
        if pc_selection == 'all':
            pcs_to_keep = pc_embedding[:, :]
            pc_cutoff = pc_embedding.shape[1] - 1
        elif pc_selection == 'knee':
            kn = KneeLocator(list(range(len(pc_var_explained))), pc_var_explained, curve='convex', direction='decreasing')
            pcs_to_keep = pc_embedding[:, :kn.knee + 1]
            pc_cutoff = kn.knee
        else:
            cumulative_var = 0
            var_idx = 0
            while cumulative_var < var_explained_proportion:
                cumulative_var += pc_var_explained[var_idx]
                var_idx += 1
            pcs_to_keep = pc_embedding[:, :var_idx + 1]
            pc_cutoff = var_idx + 1

        # plot the selected PCs with variance explained
        if plot_scree:
            fig, ax = plt.subplots(1, 1)
            ax.plot(pc_var_explained, color='#999999')
            ax.axvline(x=pc_cutoff, ls='--', color='#000000')
            ax.set_xlabel('PC')
            ax.set_xticks(list(range(0, len(pc_var_explained), 100)) + [pc_cutoff])
            ax.set_ylabel('% variance explained')
            ax.set_yticks([.0, .10, .20, .30, .40])
            ax.set_yticklabels([0, 10, 20, 30, 40])
            plt.show()

        # do UMAP on reduced PC space
        umap_embedding = umap.UMAP(n_components=umap_parameters['n_components'], n_neighbors=umap_parameters['n_neighbors'],
                                   min_dist=umap_parameters['min_dist'], metric=umap_parameters['metric'])
        mapper = umap_embedding.fit_transform(pcs_to_keep)
        np.save(f"{self.save_dir}{os.sep}UMAP_embedding_cluster_function_{pc_selection}_n{umap_parameters['n_neighbors']}_{umap_parameters['metric']}.npy", mapper)
