#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:56:09 2022

@author: claudia

Test file for behavioural clustering and clasification

"""

# Load module for behavioural decomposition
from behavioral_decomposition import *

# creat class
cl_b=BehaviouralDecomposition()


## Training classifier of behaviour

# Set input list of training raw datafiles

cl_b.training_data_files=['quat_data_bruno_020520_s1_light_reheaded_XYZeuler_notricks.pkl']

# Perform time frequency analysis on each dataset from rawdatafile list

cl_b.time_frequency_analysis(plot_inputvariables_dist=True,
                                plot_detrendedvariables=True,
                                plot_inversetransform=True,
                                plot_power_hist=True,
                                plot_time_frequency=True, training=True)

# Perform PCA

cl_b.PCA()

# TSNE embedding on training data subsampled in time

cl_b.TSNE_embedding(plot_TSNE_embeddings=True)

# Watershed segmentation

cl_b.wathershed_segmentation()



## Classifying test dataset

cl_b.testing_data_file='quat_data_bruno_020520_s2_dark_reheaded_XYZeuler_notricks.pkl'

# Perform time frequency analysis on on test data

cl_b.time_frequency_analysis(plot_inputvariables_dist=True,
                                plot_detrendedvariables=True,
                                plot_inversetransform=True,
                                plot_power_hist=True,
                                plot_time_frequency=True, training=False)

# Classify timepoints and get behavioural labels

labels=cl_b.classification()