# -*- coding: utf-8 -*-

"""

@author: bartulem

Load sessions with tracking data/spikes.

The method data_loader allows one to extract any combination of cluster activity
and variables possible; the special method __str__ returns the type and description
of each queried variable, respectively.

"""

import os
import sys
import pickle


class Session:

    # class object attribute
    data_info = {'file_info': {'type': 'str', 'description': 'Name of origin file.'},
                 'total_frame_num': {'type': 'int', 'description': 'Total number of frames in origin file.'},
                 'framerate': {'type': 'np.float64', 'description': 'The camera framerate for the session.'},
                 'point_data_dimensions': {'type': 'np.ndarray (1, 5, total frame number)', 'description': 'The second number refers to X, Y, Z, labels, nans; last is number of frames.'},
                 'tracking_ts': {'type': 'np.ndarray (2, )', 'description': 'The start and stop of tracking in seconds.'},
                 'session_ts': {'type': 'np.ndarray (2, )', 'description': 'The session start and stop in seconds.'},
                 'ratcam_ts': {'type': 'np.ndarray (2, )', 'description': 'The video camera start and stop in seconds.'},
                 'bbtrans_xy': {'type': 'np.ndarray (2, )', 'description': 'The XY bounding box translation.'},
                 'bbscale_xy': {'type': 'np.ndarray (2, )', 'description': 'The XY bounding box scale.'},
                 'bbrot': {'type': 'np.ndarray (1, )', 'description': 'The bounding box rotation.'},
                 'head_origin': {'type': 'np.ndarray (total frame number, 3)', 'description': 'The XYZ of the head root point.'},
                 'head_x': {'type': 'np.ndarray (total frame number, 3)', 'description': 'First column/row of the original rotation matrix.'},
                 'head_z': {'type': 'np.ndarray (total frame number, 3)', 'description': 'Last column/row of the original rotation matrix.'},
                 'sorted_point_data': {'type': 'np.ndarray (total frame number, number of points, 3)', 'description': 'The XYZ of all points for all frames.'},
                 'cell_names': {'type': 'list', 'description': 'List of all cell names.'},
                 'cell_activities': {'type': 'list', 'description': 'List of cluster activity np.ndarrays.'},
                 'global_head_rot_mat': {'type': 'np.ndarray (total frame number, 3, 3)', 'description': 'The rotation matrix for converting global coordinates to the head coordinates.'},
                 'r_root_inv': {'type': 'np.ndarray (total frame number, 3, 3)', 'description': 'The root rotation matrix.'},
                 'r_root_inv_oriented': {'type': 'np.ndarray (total frame number, 3, 3)', 'description': 'The oriented root rotation matrix.'},
                 'allo_head_ang': {'type': 'np.ndarray (total frame number, 3)', 'description': 'Allocentric head direction: roll (X), pitch (Y) and azimuth (Z).'},
                 'body_direction': {'type': 'np.ndarray (total frame number, )', 'description': 'Allocentric body direction in the XY coordinate system.'},
                 'ego3_head_ang': {'type': 'np.ndarray (total frame number, 3)', 'description': 'Egocentric head angles (head relative to body in 3D): roll (X), pitch (Y) and azimuth (Z).'},
                 'ego2_head_ang': {'type': 'np.ndarray (total frame number, 3)', 'description': 'Egocentric head angles (head relative to body in XY plane): roll (X), pitch (Y) and azimuth (Z).'},
                 'back_ang': {'type': 'np.ndarray (total frame number, 2)', 'description': 'Back angles: pitch (Y) and azimuth (Z).'},
                 'opt_back_ang': {'type': 'np.ndarray (total frame number, 2)', 'description': 'Optimized back angles: pitch (Y) and azimuth (Z) / this one was used in the paper.'},
                 'speeds': {'type': 'np.ndarray (total frame number, 2)', 'description': 'Speed of the animal in the XY plane / there are 2 because of 2 ways of calculating speed - speeds[:, 1] is the one we use.'},
                 'selfmotion': {'type': 'np.ndarray (total frame number, 4)', 'description': 'Self-motion of the animal in the XY plane / there are 4 (dx0, dy0, dx1, dy1) because of 2 ways of calculating speed.'},
                 'speeds_1st_der': {'type': 'np.ndarray (total frame number, 2)', 'description': 'First derivative of speed / there are 2 because of 2 ways of calculating speed - speeds[:, 1] is the one we use.'},
                 'neck_1st_der': {'type': 'np.ndarray (total frame number, )', 'description': 'First derivative of neck elevation.'},
                 'neck_2nd_der': {'type': 'np.ndarray (total frame number, )', 'description': 'Second derivative of neck elevation.'},
                 'allo_head_1st_der': {'type': 'np.ndarray (total frame number, 3)', 'description': 'First derivative of allocentric head angles: roll (X), pitch (Y) and azimuth (Z).'},
                 'allo_head_2nd_der': {'type': 'np.ndarray (total frame number, 3)', 'description': 'Second derivative of allocentric head angles: roll (X), pitch (Y) and azimuth (Z).'},
                 'bodydir_1sr_der': {'type': 'np.ndarray (total frame number, )', 'description': 'Allocentric body direction first derivative.'},
                 'bodydir_2nd_der': {'type': 'np.ndarray (total frame number, )', 'description': 'Allocentric body direction second derivative.'},
                 'ego3_head_1st_der': {'type': 'np.ndarray (total frame number, 3)', 'description': 'First derivative of 3D egocentric head angles: roll (X), pitch (Y) and azimuth (Z).'},
                 'ego3_head_2nd_der': {'type': 'np.ndarray (total frame number, 3)', 'description': 'Second derivative of 3D egocentric head angles: roll (X), pitch (Y) and azimuth (Z).'},
                 'ego2_head_1st_der': {'type': 'np.ndarray (total frame number, 3)', 'description': 'First derivative of XY egocentric head angles: roll (X), pitch (Y) and azimuth (Z).'},
                 'ego2_head_2nd_der': {'type': 'np.ndarray (total frame number, 3)', 'description': 'Second derivative of XY egocentric head angles: roll (X), pitch (Y) and azimuth (Z).'},
                 'back_1st_der': {'type': 'np.ndarray (total frame number, 2)', 'description': 'First derivative of back angles: pitch (Y) and azimuth (Z).'},
                 'back_2nd_der': {'type': 'np.ndarray (total frame number, 2)', 'description': 'Second derivative of back angles: pitch (Y) and azimuth (Z).'},
                 'opt_back_1st_der': {'type': 'np.ndarray (total frame number, 2)', 'description': 'First derivative of opt_back angles: pitch (Y) and azimuth (Z).'},
                 'opt_back_2nd_der': {'type': 'np.ndarray (total frame number, 2)', 'description': 'Second derivative of opt_back angles: pitch (Y) and azimuth (Z).'},
                 'imu_allo_head_ang': {'type': 'np.ndarray (total frame number, 3)', 'description': 'IMU allocentric head direction: roll (X), pitch (Y) and azimuth (Z).'},
                 'imu_ego3_head_ang': {'type': 'np.ndarray (total frame number, 3)', 'description': 'IMU egocentric head angles (head relative to body in 3D): roll (X), pitch (Y) and azimuth (Z).'},
                 'imu_ego2_head_ang': {'type': 'np.ndarray (total frame number, 3)', 'description': 'IMU egocentric head angles (head relative to body in XY plane): roll (X), pitch (Y) and azimuth (Z).'},
                 'imu_allo_head_1st_der': {'type': 'np.ndarray (total frame number, 3)', 'description': 'First derivative of IMU allocentric head angles: roll (X), pitch (Y) and azimuth (Z).'},
                 'imu_allo_head_2nd_der': {'type': 'np.ndarray (total frame number, 3)', 'description': 'Second derivative of IMU allocentric head angles: roll (X), pitch (Y) and azimuth (Z).'},
                 'imu_ego3_head_1st_der': {'type': 'np.ndarray (total frame number, 3)', 'description': 'First derivative of IMU 3D egocentric head angles: roll (X), pitch (Y) and azimuth (Z).'},
                 'imu_ego3_head_2nd_der': {'type': 'np.ndarray (total frame number, 3)', 'description': 'Second derivative of IMU 3D egocentric head angles: roll (X), pitch (Y) and azimuth (Z).'},
                 'imu_ego2_head_1st_der': {'type': 'np.ndarray (total frame number, 3)', 'description': 'First derivative of IMU XY egocentric head angles: roll (X), pitch (Y) and azimuth (Z).'},
                 'imu_ego2_head_2nd_der': {'type': 'np.ndarray (total frame number, 3)', 'description': 'Second derivative of IMU XY egocentric head angles: roll (X), pitch (Y) and azimuth (Z).'},
                 'imu_sound': {'type': 'np.ndarray (total frame number, )', 'description': 'Was the white noise sound ON (1) or OFF (0).'}}

    def __init__(self, session_list=0, describe='file_info'):
        self.session_list = session_list
        self.describe = describe

    def data_loader(self, **kwargs):

        """
        Parameters
        ----------
        **kwargs: dictionary
        extract_clusters : str/int/list
            Cluster IDs to extract (if int, takes first n clusters; if 'all', takes all); defaults to 'None'.
        extract_variables : str/list
            Variables to extract (if 'all', take all); defaults to 'None'.
        ----------

        Returns
        ----------
        data : dictionary
            The data dictionary tree; outer dictionary - file names are keys, inner dictionary is the value;
            inner dictionary - describe names are keys, describe arrays are values.
        ----------
        """

        extract_clusters = kwargs['extract_clusters'] if 'extract_clusters' in kwargs.keys() \
                                                         and (kwargs['extract_clusters'] == 'all' or type(kwargs['extract_clusters']) == int or type(kwargs['extract_clusters']) == list) else 'None'

        extract_variables = kwargs['extract_variables'] if 'extract_variables' in kwargs.keys() \
                                                           and (kwargs['extract_variables'] == 'all' or type(kwargs['extract_variables']) == list) else 'None'

        # load sessions
        if extract_clusters != 'None' or extract_variables != 'None':
            data = {}
            for session in self.session_list:
                if os.path.exists(session):

                    # load pickle file
                    with open(session, 'rb') as session_file:
                        loaded = pickle.load(session_file)

                    # create data entry with file name
                    data[loaded['file_info']] = {}

                    for key, value in loaded.items():
                        if key == 'cell_activities' or key == 'file_info':
                            continue
                        elif key == 'cell_names':
                            if extract_clusters != 'None':
                                if 'cluster_spikes' not in data:
                                    data[loaded['file_info']]['cluster_spikes'] = {}
                                if extract_clusters == 'all':
                                    for name_idx, name in enumerate(loaded['cell_names']):
                                        data[loaded['file_info']]['cluster_spikes'][name] = loaded['cell_activities'][name_idx].ravel()
                                elif type(extract_clusters) == list:
                                    for name in extract_clusters:
                                        name_idx = loaded['cell_names'].index(name)
                                        data[loaded['file_info']]['cluster_spikes'][name] = loaded['cell_activities'][name_idx].ravel()
                                else:
                                    for name_idx in range(extract_clusters):
                                        data[loaded['file_info']]['cluster_spikes'][loaded['cell_names'][name_idx]] = loaded['cell_activities'][name_idx].ravel()
                        else:
                            if extract_variables != 'None':
                                data[loaded['file_info']]['total_frame_num'] = loaded['file_info']['head_origin'].shape[0]
                                if extract_variables == 'all' or key in extract_variables:
                                    data[loaded['file_info']][key] = value
                else:
                    print(f"Location invalid for file {session}. Please try again.")
                    sys.exit()

            return data

    # returns type & description of desired variable
    def __str__(self):

        """
        Returns
        ----------
        type : str
            Variable type & description.
        ----------
        """

        return f"Type: {self.data_info[self.describe]['type']}. \n" \
               f"Description: {self.data_info[self.describe]['description']}"


if __name__ == '__main__':
    # sys.argv[0] is name of script
    print(Session(describe=str(sys.argv[1])))
