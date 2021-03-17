# -*- coding: utf-8 -*-

"""

@author: bartulem

Dimensionality reduction on neural data.

"""

import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
import matplotlib.animation as animation
from scipy.stats import pearsonr
from scipy.stats import zscore
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE, SpectralEmbedding
from decode_events import choose_012_clusters
from neural_activity import condense_frame_arrays
from neural_activity import Spikes
from sessions2load import Session

def get_condensed_features(behavioral_data, bin_size_ms=100):
    condensed = {}
    for var in behavioral_data.keys():
        if var == 'speeds':
            condensed[var] = condense_frame_arrays(frame_array=behavioral_data[var][:, 3],
                                                   camera_framerate=120.,
                                                   bin_size_ms=bin_size_ms,
                                                   arr_type=False,
                                                   sound=False)
        if var == 'sorted_point_data':
            condensed['neck_elevation'] = condense_frame_arrays(frame_array=behavioral_data[var][:, 4, 2],
                                                                camera_framerate=120.,
                                                                bin_size_ms=bin_size_ms,
                                                                arr_type=False,
                                                                sound=False)
        elif var == 'neck_1st_der' or var == 'body_direction':
            condensed[var] = condense_frame_arrays(frame_array=behavioral_data[var],
                                                   camera_framerate=120.,
                                                   bin_size_ms=bin_size_ms,
                                                   arr_type=False,
                                                   sound=False)
        elif var != 'total_frame_num' and var != 'speeds' and var != 'sorted_point_data' \
                and var != 'neck_1st_der' and var != 'body_direction':
            condensed[var] = {}
            for n_col in range(behavioral_data[var].shape[1]):
                condensed[var][n_col] = condense_frame_arrays(frame_array=behavioral_data[var][:, n_col],
                                                              camera_framerate=120., bin_size_ms=bin_size_ms,
                                                              arr_type=False,
                                                              sound=False)
    return condensed

class LatentSpace:

    def __init__(self, input_dict, cluster_groups_dir, sp_profiles_csv):
        self.input_dict = input_dict
        self.cluster_groups_dir = cluster_groups_dir
        self.sp_profiles_csv = sp_profiles_csv

    def activity_pca(self, **kwargs):
        """
        Description
        ----------
        This method does PCA on neural activity concatenated across three different
        sessions and correlates the latent variables with behavioral features.
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
            A figure depicting correlation of beh.features with PCs across three sessions.
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

        # choose clusters present in all sessions
        all_clusters, \
        chosen_clusters, \
        extra_chosen_clusters, \
        cluster_dict = choose_012_clusters(the_input_012=list(self.input_dict.values()),
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
            file_id, feature_data = Session(session= self.input_dict[session]).data_loader(extract_variables=feature_list)
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
            purged_spikes_dictionary = Spikes(input_file=full_session_path).convert_activity_to_frames_with_shuffles(get_clusters=chosen_clusters+extra_chosen_clusters[which_extra],
                                                                                                                     to_shuffle=False,
                                                                                                                     condense_arr=True,
                                                                                                                     condense_bin_ms=condense_bin_ms)
            neural_data_binned =  np.zeros((len(all_clusters), new_shapes[session]))
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
        shape_3 = neural_data[list(self.input_dict.keys())[2]].shape[1]

        # rearrange array to shape (n_samples, n_features)
        input_arr = neural_data_all.T

        frames = np.arange(93, 181, .5)
        frames = frames[(frames < 149) | (frames > 153)]

        # do PCA
        pca = PCA(whiten=True)
        embedding = pca.fit_transform(input_arr)

        comp_1 = 0
        comp_2 = 8
        comp_3 = 34
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(embedding[:shape_1, comp_1], embedding[:shape_1, comp_2], embedding[:shape_1, comp_3], c='#EEC900', alpha=.5)
        ax.scatter(embedding[shape_1:shape_1+shape_2, comp_1], embedding[shape_1:shape_1+shape_2, comp_2], embedding[shape_1:shape_1+shape_2, comp_3], c='#00008B', alpha=.5)
        ax.scatter(embedding[shape_1+shape_2:, comp_1], embedding[shape_1+shape_2:, comp_2], embedding[shape_1+shape_2:, comp_3], c='#CD950C', alpha=.5)
        ax.view_init(elev=10, azim=92)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC9')
        ax.set_zlabel('PC35')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)
        plt.show()

        # def animate(i):
        #     ax.view_init(elev=10, azim=frames[i])
        #
        # ani = animation.FuncAnimation(fig=fig, func=animate, frames=frames.shape[0], interval=50, blit=False)
        # ani.save('/home/bartulm/Downloads/yeyo.mp4', dpi=300)


        # split data back to individual sessions
        # if num_components is True:
        #     num_components = x_new.shape[1]
        # split_data = {}
        # split_counts = [0] + list(np.cumsum(list(new_shapes.values())))
        # for s_idx, session in enumerate(new_shapes.keys()):
        #     split_data[session] = x_new[split_counts[s_idx]:split_counts[s_idx+1], :num_components]
        #
        # # plot correlations of PCs with beh. features
        # plot_dict = {}
        # for session in split_data.keys():
        #     pc_corr = np.zeros((15, num_components))
        #     variables = []
        #     row = 0
        #     for var in behavioral_data[session].keys():
        #         if var == 'speeds' or var == 'neck_elevation' or var == 'body_direction' or var == 'neck_1st_der':
        #             nas = np.isnan(behavioral_data[session][var])
        #             for nc in range(num_components):
        #                 x, y = split_data[session][:, nc], behavioral_data[session][var]
        #                 pc_corr[row, nc] = pearsonr(x[~nas], y[~nas])[0]
        #             variables.append(var)
        #             row += 1
        #         else:
        #             for sub_var in behavioral_data[session][var].keys():
        #                 nas = np.isnan(behavioral_data[session][var][sub_var])
        #                 for nc in range(num_components):
        #                     x, y = split_data[session][:, nc], behavioral_data[session][var][sub_var]
        #                     pc_corr[row, nc] = pearsonr(x[~nas], y[~nas])[0]
        #                 variables.append(f'{var}_{sub_var}')
        #                 row += 1
        #     plot_dict[session] = pc_corr
        #
        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))
        # for s_idx, session in enumerate(plot_dict.keys()):
        #     ax = plt.subplot(1, 3, s_idx+1)
        #     ax.imshow(plot_dict[session], aspect='auto', cmap='seismic', vmin=-1, vmax=1)
        #     ax.set_title(session)
        #     if s_idx == 0:
        #         ax.set_yticks(list(range(0, 15)))
        #         ax.set_yticklabels(variables)
        #     else:
        #         ax.set_yticks(list(range(0, 15)))
        #         ax.set_yticklabels([])
        # plt.tight_layout(pad=1.08)
        # plt.show()


