# -*- coding: utf-8 -*-

"""

@author: bartulem

Locate single-units in 3D anatomical space (relative to bregma coordinates).

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ClusterLocator:

    locator_dict = {'LH': {'bruno': {'alpha': 57.5,
                                     'beta': 4.7,
                                     'XYZ_start': (-3.3, -2.1, 0),
                                     'inserted_probe_len': 3.84},
                           'roy': {'alpha': 64,
                                   'beta': 3,
                                   'XYZ_start': (-3.24, -1.9, 0),
                                   'inserted_probe_len': 6.27},
                           'jacopo': {'alpha': 65,
                                      'beta': 3,
                                      'XYZ_start': (-3, -2.3, 0),
                                      'inserted_probe_len': 6.84},
                           'crazyjoe': {'alpha': 66.5,
                                        'beta': 2.7,
                                        'XYZ_start': (-3.48, -2.7, 0),
                                        'inserted_probe_len': 7.22}},
                    'RH': {'johnjohn': {'alpha': 48,
                                        'beta': 2.9,
                                        'XYZ_start': (-6.5, 2.1, 0),
                                        'inserted_probe_len': 5.72},
                           'frank': {'alpha': 46.5,
                                     'beta': 2.9,
                                     'XYZ_start': (-5.52, 2.07, 0),
                                     'inserted_probe_len': 5.79},
                           'kavorka': {'alpha': 45,
                                       'beta': 0,
                                       'XYZ_start': (-5.88, 2.4, 0),
                                       'inserted_probe_len': 6.75}}}

    def __init__(self, probe_site_configuration, template_shapes_dir='', sp_profiles_csv='',
                       channel_maps_dir='', new_sp_profiles_csv=''):
        self.probe_site_configuration = probe_site_configuration
        self.template_shapes_dir = template_shapes_dir
        self.channel_maps_dir = channel_maps_dir
        self.sp_profiles_csv = sp_profiles_csv
        self.new_sp_profiles_csv = new_sp_profiles_csv

    def find_cluster_location(self, **kwargs):
        """
        Description
        ----------
        This method finds waveform templates for each single unit cluster and then computes
        the center of mass in probe XY coordinates for each putative cell.
        ----------

        Parameters
        ----------
        **kwargs (dictionary)
        channel_window (int)
            Max number of channels around either side of peak channel; defaults to 20.
        peak_fraction (float)
            The acceptable fraction of peak template amplitude; defaults to 0.15.
        ----------

        Returns
        ----------
        sp_profiles_csv (pd.DataFrame)
            A DataFrame containing cluster anatomical location info.
        ----------
        """

        channel_window = kwargs['channel_window'] if 'channel_window' in kwargs.keys() and type(kwargs['channel_window']) == int else 20
        peak_fraction = kwargs['peak_fraction'] if 'peak_fraction' in kwargs.keys() and type(kwargs['peak_fraction']) == float else .15

        # load probe site layout
        probe_coords = np.load(self.probe_site_configuration)

        # load waveform template files
        template_shapes_dict = {}
        for template_file in os.listdir(self.template_shapes_dir):
            template_shapes_dict[template_file[:-4]] = np.load(f'{self.template_shapes_dir}{os.sep}{template_file}')

        # load channel map files
        channel_map_dict = {}
        for channel_file in os.listdir(self.channel_maps_dir):
            channel_map_dict[channel_file[:-4]] = np.load(f'{self.channel_maps_dir}{os.sep}{channel_file}')

        # load sp_profiles_csv
        cluster_data = pd.read_csv(self.sp_profiles_csv)

        # determine anatomical location for clusters
        for cl_idx in range(cluster_data.shape[0]):
            file_id = cluster_data.iloc[cl_idx, 0]
            animal_id = [animal for animal in list(self.locator_dict['LH'].keys())+list(self.locator_dict['RH'].keys())
                         if animal in file_id][0]
            cl_name = cluster_data.iloc[cl_idx, 1]
            cl_id = int(cl_name[8:12])
            cl_peak_channel = int(cluster_data.iloc[cl_idx, 1][-3:])
            if cl_id < template_shapes_dict[file_id].shape[0]:
                ch_map_file = channel_map_dict[file_id].ravel()

                # find peak channel in channel map
                peak_in_ch_map = np.where(ch_map_file==cl_peak_channel)[0][0]

                # find putative borders of the triangulation window.
                c_min = np.max([0, peak_in_ch_map-channel_window])
                c_max = np.min([peak_in_ch_map+channel_window, ch_map_file.shape[0]])

                ts = template_shapes_dict[file_id][cl_id, :, peak_in_ch_map]
                # find actual borders of triangulation window
                peak_bin_data = np.zeros(c_max-c_min)
                for ch_idx, channel in enumerate(range(c_min, c_max)):
                    peak_bin_data[ch_idx] = np.max(np.abs(template_shapes_dict[file_id][cl_id, :, channel]))
                min_max_arr = np.arange(c_min, c_max, 1)[peak_bin_data > np.max(np.abs(ts))*peak_fraction]
                peak_bin_data = peak_bin_data[peak_bin_data > np.max(np.abs(ts))*peak_fraction]

                # get probe coordinates
                ch_data = np.zeros((min_max_arr.shape[0], 3))
                chs_in_ch_map = ch_map_file[min_max_arr]
                if 'distal' in file_id:
                    ch_data[:, 0:2] = probe_coords[chs_in_ch_map, :]
                else:
                    ch_data[:, 0:2] = (probe_coords+np.array([0, 3800.]))[chs_in_ch_map, :]
                ch_data[:, 2] = peak_bin_data

                # calculate center of mass
                com = np.average(ch_data[:, :2], axis=0, weights=ch_data[:, 2])
            else:
                if 'distal' in file_id:
                    com = probe_coords[cl_peak_channel]
                else:
                    com = (probe_coords+np.array([0, 3800.]))[cl_peak_channel]

            # compute DV, ML and AP(RC) for each cluster
            if animal_id in self.locator_dict['LH'].keys():
                ### here 0 is surface (result in mm)
                dv = self.locator_dict['LH'][animal_id]['XYZ_start'][2] \
                     - np.cos(np.radians(self.locator_dict['LH'][animal_id]['alpha'])) \
                     *(self.locator_dict['LH'][animal_id]['inserted_probe_len']-(com[1]/1000))
                ### ML and AP need to be corrected for the X component in the probe
                first_step = 35-com[0]
                extract_sign = 1 if first_step >= 0 else -1
                second_step = ((2*first_step)**2)/2
                final_correction = (extract_sign*np.sqrt(second_step))/1000
                ml = self.locator_dict['LH'][animal_id]['XYZ_start'][1] \
                     - np.sin(np.radians(self.locator_dict['LH'][animal_id]['alpha'])) \
                     * np.sin(np.radians(self.locator_dict['LH'][animal_id]['beta'])) \
                     *(self.locator_dict['LH'][animal_id]['inserted_probe_len']-(com[1]/1000)) \
                     + final_correction
                ap = self.locator_dict['LH'][animal_id]['XYZ_start'][0] \
                     + np.cos(np.radians(self.locator_dict['LH'][animal_id]['beta'])) \
                     * np.sin(np.radians(self.locator_dict['LH'][animal_id]['alpha'])) \
                     *(self.locator_dict['LH'][animal_id]['inserted_probe_len']-(com[1]/1000)) \
                     + final_correction
            else:
                dv = self.locator_dict['RH'][animal_id]['XYZ_start'][2] \
                     - np.cos(np.radians(self.locator_dict['RH'][animal_id]['alpha'])) \
                     *(self.locator_dict['RH'][animal_id]['inserted_probe_len']-(com[1]/1000))
                ### ML and AP need to be corrected for the X component in the probe
                first_step = com[0]-35
                extract_sign = 1 if first_step >= 0 else -1
                second_step = ((2*first_step)**2)/2
                final_correction = (extract_sign*np.sqrt(second_step))/1000
                ml = self.locator_dict['RH'][animal_id]['XYZ_start'][1] \
                     + np.cos(np.radians(self.locator_dict['RH'][animal_id]['beta'])) \
                     * np.sin(np.radians(self.locator_dict['RH'][animal_id]['alpha'])) \
                     *(self.locator_dict['RH'][animal_id]['inserted_probe_len']-(com[1]/1000)) \
                     + final_correction
                ap = self.locator_dict['RH'][animal_id]['XYZ_start'][0] \
                     - np.sin(np.radians(self.locator_dict['RH'][animal_id]['alpha'])) \
                     * np.sin(np.radians(self.locator_dict['RH'][animal_id]['beta'])) \
                     *(self.locator_dict['RH'][animal_id]['inserted_probe_len']-(com[1]/1000)) \
                     + final_correction
            row = cluster_data[(cluster_data['session_id'] == file_id) & (cluster_data['cluster_id'] == cl_name)].index
            cluster_data.loc[row, 'AP'] = ap
            cluster_data.loc[row, 'ML'] = ml
            cluster_data.loc[row, 'DV'] = dv

        # save df in csv file
        cluster_data.to_csv(path_or_buf=self.new_sp_profiles_csv, sep='\t', index=False)