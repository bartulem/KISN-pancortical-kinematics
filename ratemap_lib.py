"""
Calculates ratemaps.
@author: SolVind
"""

import matplotlib
from scipy import *
import scipy.linalg as linalg
import scipy.ndimage.filters
import scipy.io
import numpy as np
import math
import copy as copycopy

import pickle
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv
import os
import sys
import matplotlib.image
import time
from scipy import interpolate
import scipy.stats
from random import shuffle
from .optitracking_lib import *


def get_tcount_time(tracking_times, cell_start_times, cell_end_times, debug_mode=False):
    # tracking_times in s
    post = tracking_times

    if debug_mode:
        print(('Tracking start time according to neuralynx', tracking_times[0]))
        print(('First interpolated points', post[:10]))
        print(('Last interpolated points', post[-10:]))
        print(('Tracking end time according to neuralynx', tracking_times[-1]))
        print(
            'Those last ones are used for going from time bins to movement bins... consider using smaller bins but then you have to interpolate or record at higher temporal resolution')

    tcount = (post >= cell_start_times[0]) * (post <= cell_end_times[0])
    for i in np.arange(1, len(cell_start_times), 1):
        tcount += (post >= cell_start_times[i]) * (post <= cell_end_times[i])
    tcount[tcount > 1] = 1
    return tcount


def get_tcount(nframes, frame_rate, tracking_ts, cell_start_times, cell_end_times, debug_mode=False):
    post = np.arange(nframes) / float(frame_rate) + tracking_ts[0]

    if debug_mode:
        print(('Tracking start time according to neuralynx', tracking_ts[0]))
        print(('First interpolated points', post[:10]))
        print(('Last interpolated points', post[-10:]))
        print(('Tracking end time according to neuralynx', tracking_ts[1]))
        print('Those last ones are used for going from time bins to movement bins... consider using smaller bins but then you have to interpolate or record at higher temporal resolution')

    tcount = (post >= cell_start_times[0]) * (post <= cell_end_times[0])
    for i in np.arange(1, len(cell_start_times), 1):
        tcount += (post >= cell_start_times[i]) * (post <= cell_end_times[i])
    tcount[tcount > 1] = 1
    return tcount


# cell_data, time_delay, frame_rate, tracking_ts, cell_start_times, cell_end_times = da_cell_activity, toff, captureframerate, tracking_ts, startaltbins[1::2], endaltbins[1::2]
def get_cell_occ(cell_data, time_delay, frame_rate, tracking_ts, cell_start_times, cell_end_times, full_tracking_ts=None):
    whiches = (cell_data <= cell_end_times[0]) * (cell_data >= cell_start_times[0])
    for i in np.arange(1, len(cell_start_times), 1):
        whiches += (cell_data >= cell_start_times[i]) * (cell_data <= cell_end_times[i])
        # print(whiches)
    # whiches[whiches > 1] = 1
    if np.sum(whiches) < 1:
        return []
    cell_data = cell_data[whiches]
    if len(np.ravel(frame_rate)) == 1:
        celldata_indofoccupancy = (np.round((cell_data - tracking_ts[0] + time_delay) * frame_rate)).astype(int)  # go back to frame where cell fires
    else:
        celldata_indofoccupancy = np.ones(len(cell_data), 'i') * -1
        for i in range(len(frame_rate)):
            if i == 0:
                inds = np.where(cell_data <= full_tracking_ts[i + 1])[0]
                if len(inds) > 0:
                    celldata_indofoccupancy[inds] = (
                        np.round((cell_data[inds] - full_tracking_ts[0] + time_delay) * frame_rate[i])).astype(int)
            elif i == len(frame_rate) - 1:
                inds = np.where(cell_data > full_tracking_ts[i])[0]
                if len(inds) > 0:
                    celldata_indofoccupancy[inds] = (
                        np.round((cell_data[inds] - full_tracking_ts[0] + time_delay) * frame_rate[i])).astype(int)
            else:
                inds = np.where((cell_data > full_tracking_ts[i]) * (cell_data <= full_tracking_ts[i + 1]))[0]
                if len(inds) > 0:
                    celldata_indofoccupancy[inds] = (
                        np.round((cell_data[inds] - full_tracking_ts[0] + time_delay) * frame_rate[i])).astype(int)
    return celldata_indofoccupancy


def get_2d_ratemap(factor1, factor2, bins1, bins2, tcount, cellocc_ind, frame_rate,
                   occupancy_thresh, smoothing_par, session_indicator=None, is_periodic=False, debug_mode=False):
    """
    Purpose
    -------------
    Get the rate map data for later use.

    Inputs
    -------------


    """
    if isinstance(cellocc_ind, list):
        if not cellocc_ind:
            return []

    if is_periodic:
        print('SMOOTHING AROUND EDGES OF GENERIC PERIODIC 2D GUYS IS NOT YET IMPLEMENTED!!!!')

    cA = 0.5 * (bins1[1:] + bins1[:(-1)])
    cB = 0.5 * (bins2[1:] + bins2[:(-1)])

    binnedAs = np.ones(len(factor1)) * -10
    binnedBs = np.ones(len(factor2)) * -10
    for i in np.arange(len(cA)):
        whiches = (factor1 > bins1[i]) * (factor1 <= bins1[i + 1])
        binnedAs[whiches] = i
    for i in np.arange(len(cB)):
        whiches = (factor2 > bins2[i]) * (factor2 <= bins2[i + 1])
        binnedBs[whiches] = i

    binnedocc = np.zeros((len(cA), len(cB)))
    for i in np.arange(len(cA)):
        icount = (binnedAs == i) * tcount
        for j in np.arange(len(cB)):
            if len(np.ravel(frame_rate)) == 1:
                binnedocc[i, j] = float(np.sum(icount * (binnedBs == j))) / float(frame_rate)  # in seconds
            else:
                total_mat = []
                for k in range(len(frame_rate)):
                    scount = (session_indicator == k)
                    part_mat = float(np.sum(icount * (binnedBs == j) * scount)) / float(frame_rate[k])
                    total_mat.append(part_mat)
                total_num = np.sum(total_mat)
                binnedocc[i, j] = np.sum(total_num)

    celldata_indofoccupancy = cellocc_ind

    binnedacc = np.zeros((len(cA), len(cB)))
    numinemptyspot = 0
    numoutsidetracking = 0
    for i in np.arange(len(celldata_indofoccupancy)):
        ii = celldata_indofoccupancy[i]

        if ii > len(factor1) - 1 or ii < 0:
            numoutsidetracking += 1
            continue

        if binnedAs[ii] < 0 or binnedAs[ii] > len(cA) - 1 or binnedBs[ii] < 0 or binnedBs[ii] > len(cB) - 1:
            numinemptyspot += 1
            continue

        binnedacc[int(round(binnedAs[ii])), int(round(binnedBs[ii]))] += 1

    if debug_mode:
        if numinemptyspot > 0:
            print(('Spikes skipped due to empty spots in tracking or outside of range', numinemptyspot))
        if numoutsidetracking > 0:
            print(('Spikes skipped due to occurring before or after tracking', numoutsidetracking))

    # convert to firing rates
    binnedacc[binnedocc > 0] = binnedacc[binnedocc > 0] / binnedocc[binnedocc > 0]
    binnedacc[binnedocc < occupancy_thresh] = 0.
    rawbinnedacc = binnedacc + 0.

    # don't forget that angles wrap around!!!
    # add some gaussian smoothing (currently the same as for self motion maps!!!
    gaussian_smoothing_width = smoothing_par[0]
    gaussian_smoothing_height = smoothing_par[1]
    xvar = gaussian_smoothing_width ** 2
    yvar = gaussian_smoothing_height ** 2
    for x in np.arange(
            len(cA)):  # note this is a very slow and inefficient way of writing this code but it doesn't really matter
        for y in np.arange(len(cB)):
            if binnedocc[x, y] > float(occupancy_thresh):
                denom = 0.
                numer = 0.
                for xx in np.arange(len(cA)):
                    for yy in np.arange(len(cB)):
                        if binnedocc[xx, yy] > float(occupancy_thresh):
                            ddyy = yy - y  # do this because it is periodic!
                            # if(abs(ddyy)>0.5*Ngeneric2Dbins):
                            # ddyy = Ngeneric2Dbins-abs(ddyy)
                            dist = math.exp(-0.5 * (xx - x) ** 2 / xvar - 0.5 * ddyy ** 2 / yvar) + 0.
                            numer += dist * rawbinnedacc[xx, yy]
                            denom += dist
                binnedacc[x, y] = numer / denom + 0.

    return rawbinnedacc, binnedacc, binnedocc


def get_1d_binnocc(factors, bins, tcount, frame_rate, session_indicator=None, gaussian_smoothing=1, periodic=False):
    """
        Purpose
        -------------
        This binned Occ is particular for shuffling.

        Inputs
        -------------
        factors : all the 1D factors are interested
        bins : all the corresponding 1D bins

        Outputs
        -------------


        """
    all_binnocc = {}
    all_smoothed_occ = {}
    all_keys = list(factors.keys())
    for k in range(len(all_keys)):
        da_key = all_keys[k]
        da_factor = factors[da_key]
        da_bin = bins[da_key]

        number_of_bins = len(da_bin) - 1

        index_ok = np.where(~np.isnan(da_factor))[0]
        factor_ok = da_factor[index_ok]
        tcount_ok = tcount[index_ok]
        if session_indicator is not None:
            session_indok = session_indicator[index_ok]
        else:
            session_indok = np.zeros(len(tcount_ok), 'i')

        # binnocc = np.ravel([np.sum(tcount_ok * (factor_ok > da_bin[i - 1]) * (factor_ok <= da_bin[i])) for i in range(1, len(da_bin))]) / frame_rate

        binnocc = np.zeros(number_of_bins)
        for i in range(1, len(da_bin), 1):
            if len(np.ravel(frame_rate)) == 1:
                binnocc[i - 1] = float(np.sum(tcount_ok * (factor_ok > da_bin[i - 1]) * (factor_ok <= da_bin[i]))) / float(frame_rate)  # in seconds
            else:
                total_mat = []
                for k in range(len(frame_rate)):
                    scount = (session_indok == k)
                    part_mat = float(np.sum(tcount_ok * (factor_ok > da_bin[i - 1]) * (factor_ok <= da_bin[i]) * scount)) / float(frame_rate[k])
                    total_mat.append(part_mat)
                total_num = np.sum(total_mat)
                binnocc[i - 1] = np.sum(total_num)

        smoothed_occ = np.zeros(number_of_bins)
        if not periodic:
            xvar = gaussian_smoothing ** 2
            for x in np.arange(number_of_bins):
                dist = np.exp(-0.5 * (np.arange(number_of_bins) - x) ** 2 / xvar) + 0.
                numer = np.sum(dist * binnocc)
                denom = np.sum(dist)
                smoothed_occ[x] = numer / denom + 0.

        all_binnocc[da_key] = binnocc
        all_smoothed_occ[da_key] = smoothed_occ
    return all_binnocc, all_smoothed_occ


def get_1d_ratemap(factor, bins, binnocc, cellocc_ind, gaussian_smoothing=1, debug_mode=False, periodic=False):
    """
    Purpose
    -------------


    Inputs
    -------------
    avec :

    Outputs
    -------------


    """
    # print 'Fraction of NaNs in the movement data = fraction of bins that are crap', sum(ixes<-1)/float(len(ixes)),
    # this makes a True / False vector of all time points inside the start/end times that we want to analyse
    if isinstance(cellocc_ind, list):
        if not cellocc_ind:
            return []
    numberofbins = len(bins) - 1
    celldata_indofoccupancy = cellocc_ind

    is_outside_tracking = np.logical_or(celldata_indofoccupancy > len(factor) - 1, celldata_indofoccupancy < 0)
    celldata_inside_tracking = celldata_indofoccupancy[~is_outside_tracking]
    is_nan_tracking = np.isnan(factor[celldata_inside_tracking])
    cell_ok = celldata_inside_tracking[~is_nan_tracking]
    num_cell_ok = len(cell_ok)

    binnacc_mat = np.zeros((num_cell_ok, numberofbins))
    for i in range(num_cell_ok):
        ifac = factor[cell_ok[i]]
        binnacc_mat[i] = np.logical_and(bins[0:numberofbins] < ifac, ifac <= bins[1:len(bins)]) * 1
        if np.sum(binnacc_mat[i]) > 1:
            raise Exception('crap, we have an error!!!')

    binnacc = np.sum(binnacc_mat, 0)

    if debug_mode:
        num_outside_tracking = np.sum(is_outside_tracking)
        num_in_empty_spot = np.sum(is_nan_tracking)
        num_outside_bound = np.sum((np.sum(binnacc_mat, 1) == 0))
        if num_in_empty_spot > 0:
            print(('Spikes skipped due to empty spots in tracking', num_in_empty_spot))
        if num_outside_tracking > 0:
            print(('Spikes skipped due to occurring before or after tracking', num_outside_tracking))
        if num_outside_bound > 0:
            print(('Spikes skipped due to being outside bounds of ratemap', num_outside_bound))

    # convert to firing rates
    binnacc[binnocc > 0] = binnacc[binnocc > 0] / binnocc[binnocc > 0]
    # binnedacc[binnedocc<occupancy_thresh] = 0.
    rawbinnedacc = binnacc + 0.

    # add some gaussian smoothing

    if not periodic:
        xvar = gaussian_smoothing ** 2
        for x in np.arange(numberofbins):
            dist = np.exp(-0.5 * (np.arange(numberofbins) - x) ** 2 / xvar) + 0.
            numer = np.sum(dist * rawbinnedacc)
            denom = np.sum(dist)
            binnacc[x] = numer / denom + 0.

    return 0.5 * (bins[1:] + bins[:(-1)]), rawbinnedacc, binnacc


def get_selfmotion_map(ixes, jyes, tcount, cellocc_ind, frame_rate, occupancy_thresh, smoothing_par,
                       nbin_max_xval, nbin_min_yval, nbin_max_yval, session_indicator=None, debug_mode=False):
    """
    Purpose
    -------------
    make self motion maps.

    Inputs
    -------------
    ixes :

    jyes :



    Outputs
    -------------
    rawbinnedacc :

    """
    if isinstance(cellocc_ind, list):
        if not cellocc_ind:
            return []

    binnedocc = np.zeros((2 * nbin_max_xval, nbin_min_yval + nbin_max_yval))
    for i in np.arange(2 * nbin_max_xval):
        icount = (ixes == i) * tcount
        for j in np.arange(nbin_min_yval + nbin_max_yval):
            if len(np.ravel(frame_rate)) == 1:
                binnedocc[i, j] = float(np.sum(icount * (jyes == j))) / float(frame_rate)  # in seconds
            else:
                total_mat = []
                for k in range(len(frame_rate)):
                    scount = (session_indicator == k)
                    part_mat = float(np.sum(icount * (jyes == j) * scount)) / float(frame_rate[k])
                    total_mat.append(part_mat)
                total_num = np.sum(total_mat)
                binnedocc[i, j] = np.sum(total_num)

    celldata_indofoccupancy = cellocc_ind

    binnedacc = np.zeros((2 * nbin_max_xval, nbin_min_yval + nbin_max_yval))
    numinemptyspot = 0
    numoutsidetracking = 0
    numoutsideofbounds = 0
    for i in np.arange(len(celldata_indofoccupancy)):
        ii = celldata_indofoccupancy[i]
        if ii > len(ixes) - 1 or ii < 0:
            numoutsidetracking += 1
            continue
        if ixes[ii] < 0 or jyes[ii] < 0:
            numinemptyspot += 1
            continue

        if ixes[ii] < 2 * nbin_max_xval and jyes[ii] < nbin_min_yval + nbin_max_yval:
            binnedacc[ixes[ii], jyes[ii]] += 1
        else:
            numoutsideofbounds += 1
    if debug_mode:
        if numinemptyspot > 0:
            print(('Spikes skipped due to empty spots in tracking', numinemptyspot))
        if numoutsidetracking > 0:
            print(('Spikes skipped due to occurring before or after tracking', numoutsidetracking))
        if numoutsideofbounds > 0:
            print(('Spikes skipped due to being outside bounds of ratemap', numoutsideofbounds))

    # convert to firing rates
    binnedacc[binnedocc > 0] = binnedacc[binnedocc > 0] / binnedocc[binnedocc > 0]
    binnedacc[binnedocc < occupancy_thresh] = 0.
    rawbinnedacc = binnedacc + 0.

    # add some gaussian smoothing
    gaussian_smoothing_width = smoothing_par[0]
    gaussian_smoothing_height = smoothing_par[1]
    xvar = gaussian_smoothing_width ** 2
    yvar = gaussian_smoothing_height ** 2
    for y in np.arange((
            nbin_min_yval + nbin_max_yval)):  # note this is a very slow and inefficient way of writing this code but it doesn't really matter
        for x in np.arange(2 * nbin_max_xval):
            if binnedocc[x, y] > float(occupancy_thresh):
                denom = 0.
                numer = 0.
                for yy in range((nbin_min_yval + nbin_max_yval)):
                    for xx in range(2 * nbin_max_xval):
                        if binnedocc[xx, yy] > float(occupancy_thresh):
                            dist = math.exp(-0.5 * (xx - x) ** 2 / xvar - 0.5 * (yy - y) ** 2 / yvar) + 0.
                            numer += dist * rawbinnedacc[xx, yy]
                            denom += dist
                binnedacc[x, y] = numer / denom + 0.

    return rawbinnedacc, binnedacc, binnedocc


def get_spatial_tuning(spatial_values, nframes, tcount, cellocc_ind, frame_rate,
                       nbins, occupancy_thresh, smoothing_par, session_indicator=None, debug_mode=False):
    """
    Purpose
    -------------
    make velocity maps

    Inputs
    -------------
    avec :

    Outputs
    -------------


    """
    if isinstance(cellocc_ind, list):
        if not cellocc_ind:
            return []
    Xmin = np.nanmin(spatial_values[:, 0])
    Xmax = np.nanmax(spatial_values[:, 0])
    dX = (Xmax - Xmin) / float(nbins - 1)
    Ymin = np.nanmin(spatial_values[:, 1])
    Ymax = np.nanmax(spatial_values[:, 1])
    dY = (Ymax - Ymin) / float(nbins - 1)
    binnedXs = np.ones(nframes) * -10
    binnedYs = np.ones(nframes) * -10
    for i in np.arange(nbins):
        whiches = (spatial_values[:, 0] >= i * dX + Xmin) * (spatial_values[:, 0] < (i + 1) * dX + Xmin)
        binnedXs[whiches] = i
    for i in np.arange(nbins):
        whiches = (spatial_values[:, 1] >= i * dY + Ymin) * (spatial_values[:, 1] < (i + 1) * dY + Ymin)
        binnedYs[whiches] = i

    binnedocc = np.zeros((nbins, nbins))
    for i in np.arange(nbins):
        icount = (binnedXs == i) * tcount
        for j in np.arange(nbins):
            if len(np.ravel(frame_rate)) == 1:
                binnedocc[i, j] = float(np.sum(icount * (binnedYs == j))) / float(frame_rate)  # in seconds
            else:
                total_mat = []
                for k in range(len(frame_rate)):
                    scount = (session_indicator == k)
                    part_mat = float(np.sum(icount * (binnedYs == j) * scount)) / float(frame_rate[k])
                    total_mat.append(part_mat)
                total_num = np.sum(total_mat)
                binnedocc[i, j] = np.sum(total_num)

    celldata_indofoccupancy = cellocc_ind

    binnedacc = np.zeros((nbins, nbins))
    numinemptyspot = 0
    numoutsidetracking = 0
    for i in np.arange(len(celldata_indofoccupancy)):
        ii = celldata_indofoccupancy[i]

        if ii > nframes - 1 or ii < 0:
            numoutsidetracking += 1
            continue

        if np.isnan(spatial_values[ii, 1]):
            numinemptyspot += 1
            continue

        binnedacc[int(round(binnedXs[ii])), int(round(binnedYs[ii]))] += 1

    if debug_mode:
        if numinemptyspot > 0:
            print(('Spikes skipped due to empty spots in tracking or outside of speed range', numinemptyspot))
        if numoutsidetracking > 0:
            print(('Spikes skipped due to occurring before or after tracking', numoutsidetracking))

    # convert to firing rates
    binnedacc[binnedocc > 0] = binnedacc[binnedocc > 0] / binnedocc[binnedocc > 0]
    binnedacc[binnedocc < occupancy_thresh] = 0.
    rawbinnedacc = binnedacc + 0.

    # don't forget that angles wrap around!!!
    # add some gaussian smoothing (currently the same as for self motion maps!!!
    gaussian_smoothing_width = smoothing_par[0]
    gaussian_smoothing_height = smoothing_par[1]
    xvar = gaussian_smoothing_width ** 2
    yvar = gaussian_smoothing_height ** 2
    for x in np.arange(
            nbins):  # note this is a very slow and inefficient way of writing this code but it doesn't really matter
        for y in np.arange(nbins):
            if binnedocc[x, y] > float(occupancy_thresh):
                denom = 0.
                numer = 0.
                for xx in np.arange(nbins):
                    for yy in np.arange(nbins):
                        if binnedocc[xx, yy] > float(occupancy_thresh):
                            ddyy = yy - y  # do this because it is periodic!
                            if abs(ddyy) > 0.5 * nbins:
                                ddyy = nbins - abs(ddyy)
                            dist = math.exp(-0.5 * (xx - x) ** 2 / xvar - 0.5 * ddyy ** 2 / yvar) + 0.
                            numer += dist * rawbinnedacc[xx, yy]
                            denom += dist
                binnedacc[x, y] = numer / denom + 0.

    return rawbinnedacc, binnedacc, binnedocc


def get_velocity_tuning(move_angles, speeds, tcount, cellocc_ind, frame_rate,
                        velocity_max_speed, nbin_speed, nbin_angle,
                        occupancy_thresh, smoothing_par, session_indicator=None, debug_mode=False):
    """
    Purpose
    -------------
    This function makes velocity maps.

    Inputs
    -------------
    move_anlges :

    speeds :

    time_delay :

    cell_data :



    Outputs
    -------------


    """

    move_angles = move_angles + math.pi

    if np.nanmin(move_angles) < -0.0001 or np.nanmax(move_angles) > 2. * math.pi + 0.0001:
        raise Exception('Shit, move_angles is not between -pi and pi, what is going on?????')

    dspeed = float(velocity_max_speed) / float(nbin_speed - 1)
    dangle = float(2. * math.pi + 0.000001) / float(nbin_angle - 1)
    binnedspeeds = np.ones(len(move_angles)).astype(int) * -10
    binnedangles = np.ones(len(move_angles)).astype(int) * -10

    for i in np.arange(nbin_speed):
        whiches = (speeds >= i * dspeed) * (speeds < (i + 1) * dspeed)
        binnedspeeds[whiches] = i
    for i in np.arange(nbin_angle):
        whiches = (move_angles >= i * dangle) * (move_angles < (i + 1) * dangle)
        binnedangles[whiches] = i

    binnedocc = np.zeros((nbin_speed, nbin_angle))
    for i in np.arange(nbin_speed):
        icount = (binnedspeeds == i) * tcount
        for j in np.arange(nbin_angle):
            if len(np.ravel(frame_rate)) == 1:
                binnedocc[i, j] = float(np.sum(icount * (binnedangles == j))) / float(frame_rate)  # in seconds
            else:
                total_mat = []
                for k in range(len(frame_rate)):
                    scount = (session_indicator == k)
                    part_mat = float(np.sum(icount * (binnedangles == j) * scount)) / float(frame_rate[k])
                    total_mat.append(part_mat)
                total_num = np.sum(total_mat)
                binnedocc[i, j] = np.sum(total_num)

    celldata_indofoccupancy = cellocc_ind

    binnedacc = np.zeros((nbin_speed, nbin_angle))
    numinemptyspot = 0
    numoutsidetracking = 0
    for i in np.arange(len(celldata_indofoccupancy)):
        ii = celldata_indofoccupancy[i]

        if ii > len(binnedangles) - 1 or ii < 0:
            numoutsidetracking += 1
            continue

        if binnedangles[ii] < 0 or binnedspeeds[ii] < 0:
            numinemptyspot += 1
            continue

        binnedacc[binnedspeeds[ii], binnedangles[ii]] += 1

    if (debug_mode == True):
        if numinemptyspot > 0:
            print(('Spikes skipped due to empty spots in tracking or outside of speed range', numinemptyspot))
        if numoutsidetracking > 0:
            print(('Spikes skipped due to occurring before or after tracking', numoutsidetracking))

    # convert to firing rates
    binnedacc[binnedocc > 0] = binnedacc[binnedocc > 0] / binnedocc[binnedocc > 0]
    binnedacc[binnedocc < occupancy_thresh] = 0.
    rawbinnedacc = binnedacc + 0.

    # don't forget that angles wrap around!!!
    # add some gaussian smoothing (currently the same as for self motion maps!!!
    gaussian_smoothing_width = smoothing_par[0]
    gaussian_smoothing_height = smoothing_par[1]
    xvar = gaussian_smoothing_width ** 2
    yvar = gaussian_smoothing_height ** 2
    for x in np.arange(
            nbin_speed):  # note this is a very slow and inefficient way of writing this code but it doesn't really matter
        for y in np.arange(nbin_angle):
            if binnedocc[x, y] > float(occupancy_thresh):
                denom = 0.
                numer = 0.
                for xx in np.arange(nbin_speed):
                    for yy in np.arange(nbin_angle):
                        if binnedocc[xx, yy] > float(occupancy_thresh):
                            ddyy = yy - y  # do this because it is periodic!
                            if abs(ddyy) > 0.5 * nbin_angle:
                                ddyy = nbin_angle - abs(ddyy)
                            dist = math.exp(-0.5 * (xx - x) ** 2 / xvar - 0.5 * ddyy ** 2 / yvar) + 0.
                            numer += dist * rawbinnedacc[xx, yy]
                            denom += dist
                binnedacc[x, y] = numer / denom + 0.

    return rawbinnedacc, binnedacc, binnedocc


def plot_1d_values(xvals, yvals, y2vals, occs, means, stds, occupancy_thresh_1d, tit):
    """
        Purpose
        -------------
        Service function for ratemap_generator. Plot 1d values.

        Inputs
        -------------
        xvals:

        yvals:

        y2vals:

        occs:

        means:

        stds:

        tit:

        Outputs
        -------------
        Plots.

        """
    if len(xvals) != len(yvals):
        print(('should all be same', tit, len(xvals), len(yvals), len(y2vals), len(occs), len(means), len(stds)))
    if np.sum(~np.isnan(stds)) > 0:
        xx = xvals[~np.isnan(stds)]
        yyA = means[~np.isnan(stds)] + 2. * stds[np.isnan(stds) == False]
        yyB = means[~np.isnan(stds)] - 2. * stds[np.isnan(stds) == False]
        plt.plot(xx, yyA, '.', color='black')
        plt.plot(xx, yyB, '.', color='black')
        for i in range(len(xx)):
            plt.plot([xx[i], xx[i]], [yyA[i], yyB[i]], '-', color='black')
        plt.plot(xx, means[np.isnan(stds) == False], '.', color='black')
    plt.plot(xvals[occs > occupancy_thresh_1d], yvals[occs > occupancy_thresh_1d], 'o', color='blue')
    plt.plot(xvals[occs > occupancy_thresh_1d], y2vals[occs > occupancy_thresh_1d], '+', color='blue')
    plt.title(tit, fontsize=24)


def plot_ratemaps(ratemap_data, occupancy, occupancy_thresh, title, name):
    masked_array = np.ma.masked_where(occupancy < float(occupancy_thresh), ratemap_data)
    cmap = copycopy.copy(matplotlib.cm.get_cmap("jet"))
    # cmap = matplotlib.cm.jet
    cmap.set_bad('white')
    imshowobj = plt.imshow(np.transpose(masked_array), cmap=cmap, interpolation='nearest', origin='lower')
    plt.title(title, fontsize=24)
    clb = plt.colorbar()
    imshowobj.set_clim(0., max(np.ravel(ratemap_data)))
    clb.draw_all()
    plt.axis('off')
    if len(name) > 2:
        plt.savefig('%s.png' % name)


def calculate_skinfo_rate(darm, occ):
    occ = np.ravel(occ)
    darm = np.ravel(darm)
    AveFR = 0.
    Pocc = np.zeros(len(occ))
    for occi in range(len(occ)):
        if np.sum(occ) > 0.000001:
            Pocc[occi] = occ[occi] / np.sum(occ) + 0.
            AveFR += (Pocc[occi]) * darm[occi]
    ICr = 0
    for occi in range(len(occ)):
        if darm[occi] > 0.000001:
            ICr += ((Pocc[occi] * darm[occi]) / AveFR) * math.log2(darm[occi] / AveFR)
    return ICr


def re_calc_der(factors, bounds, xaxis, framerate, first_derivative_bins, second_derivative_bins, session_indicator=None):
    all_keys = list(factors.keys())
    nkeys = len(all_keys)
    for j in range(nkeys):
        tkey = all_keys[j]
        vals = factors[tkey]
        nvals = len(vals)
        first_der = np.zeros(nvals)
        isangle = False
        if 'direction' in tkey or 'roll' in tkey or 'pitch' in tkey or 'azimuth' in tkey:
            isangle = True

        for t in range(nvals):
            ts = t - first_derivative_bins
            te = t + first_derivative_bins
            if ts < 0 or te > nvals - 1 or np.isnan(vals[ts]) or np.isnan(vals[te]):
                first_der[t] = np.nan
                continue
            first_der[t] = vals[te] - vals[ts]
            if isangle:
                if first_der[t] > 180: first_der[t] -= 360.
                if first_der[t] < -180: first_der[t] += 360.
            if len(np.ravel(framerate)) == 1:
                first_der[t] /= (2. * first_derivative_bins / framerate)
            else:
                first_der[t] /= (2. * first_derivative_bins / framerate[session_indicator[t]])

        factors['%s_1st_der' % tkey] = first_der + 0.
        bounds['%s_1st_der' % tkey] = [0, 0]
        if isangle:
            xaxis['%s_1st_der' % tkey] = 'degrees per second'
        else:
            xaxis['%s_1st_der' % tkey] = 'cm per second'

        if 'Speeds' in tkey:
            continue

        second_der = np.zeros(nvals)
        for t in range(nvals):
            ts = t - second_derivative_bins
            te = t + second_derivative_bins
            if ts < 0 or te > nvals - 1 or np.isnan(first_der[ts]) or np.isnan(first_der[te]):
                second_der[t] = np.nan
                continue
            second_der[t] = first_der[te] - first_der[ts]
            if len(np.ravel(framerate)) == 1:
                second_der[t] /= (2. * first_derivative_bins / framerate)
            else:
                second_der[t] /= (2. * first_derivative_bins / framerate[session_indicator[t]])
        factors['%s_2nd_der' % tkey] = second_der + 0.
        bounds['%s_2nd_der' % tkey] = [0, 0]
        if isangle:
            xaxis['%s_2nd_der' % tkey] = 'degrees per second squared'
        else:
            xaxis['%s_2nd_der' % tkey] = 'cm per second squared'
    return factors, bounds, xaxis


def get_time_split(data, use_even_odd_minutes=True):
    session_ts = data['session_ts']
    if use_even_odd_minutes:
        print('calculating even odd minutes ....')
        otherbins = np.arange(session_ts[0], session_ts[1] + 60., 60)
        otherbins[otherbins > session_ts[1]] = session_ts[1]
        if abs(otherbins[-1] - otherbins[-2]) < 1:
            otherbins = otherbins[:(-1)]
        if abs(otherbins[-1] - otherbins[-2]) < 59.99999999999999999:
            print(('FYI: one of the time chunks that should be one minute long is really just',
                   abs(otherbins[-1] - otherbins[-2]), 'seconds and is currently included in the analysis'))
        startaltbins = otherbins[:(-1)]
        endaltbins = otherbins[1:]
    else:
        startaltbins = np.ravel(np.array([session_ts[0], 0.5 * (session_ts[0] + session_ts[1])]))
        endaltbins = np.ravel(np.array([0.5 * (session_ts[0] + session_ts[1]), session_ts[1]]))

    return startaltbins, endaltbins


def prepare4ratemap(data, boundary=None, speed_type='jump',
                    derivatives_param=None, speed_filter_ind=250, speed_filter_var=None,
                    use_even_odd_minutes=True, spatial_filter_diameter=None,
                    split_data_along_an_axis='', split_values=(0, 1),
                    save_data=True):
    """
    Purpose
    -------------
    generate the data for later use (upload to clusters) for making rate maps.

    Inputs
    -------------
    data : see from data_generator() or merge_comparing_data().
    
    boundary : if any, see tutorial for example.
    
    speed_type : 'jump' (default) or 'cum'. Methods used to calculate the speed of the animal.
    
    derivatives_param : parameters used to calculate derivatives of factors, e.g (10,10).
                        If given, the derivatives will be re-calculated.
    
    speed_filter_ind : 250 (default). The time interval (ms) we used to calculate speed in self motion.
                       The default selfmotion parameters are (150, 250). So here we can choose 150 or 250.
                       If the selfmotion paramters are changed, here need to change to the corresponding values.
                       
    speed_filter_var : (0, 40).

    spatial_filter_diameter : 1 (default) cm.
    
    use_even_odd_minutes : True (default),  else first half / second half.


    Outputs
    -------------


    """

    # setup constant
    occupancy_thresh = 400. / 1000. + 0.001  # Minimum bin occupancy (seconds), Parameters for self motion maps!!
    smoothing_par = [1.15, 1.15]  # Width and Height of Gaussian smoothing (bins), Parameters for self motion maps!!

    # Parameters for velocity map thingies!!
    velocity_max_speed = 60
    nbin_speed = 18
    nbin_angle = 20

    # Parameters for spatial maps!!
    nbin_spatial = 30

    # Parameters for 1D curves
    num_bins_1d = 36
    occupancy_thresh_1d = 400. / 1000. + 0.001  # Minimum bin occupancy (seconds)

    # read input arguments
    start_split_value = split_values[0]
    end_split_value = split_values[1]

    startaltbins, endaltbins = get_time_split(data, use_even_odd_minutes)

    # check data
    dt_keys = (list(data.keys()))
    if 'file_info' not in dt_keys:
        raise Exception('file_info is not in the data !!! Please see data_generator().')
    filename = data['file_info']
    print(('File to be working', filename))

    settings = data['settings'].copy()
    settings['speed_filter_ind'] = speed_filter_ind
    settings['speed_filter_var'] = speed_filter_var
    settings['spatial_filter_diameter'] = spatial_filter_diameter
    settings['use_even_odd_minutes'] = use_even_odd_minutes
    settings['split_data_along_an_axis'] = split_data_along_an_axis
    settings['split_values'] = split_values

    # frame rate
    captureframerate = data['framerate']
    if len(np.ravel(captureframerate)) == 1:
        session_indicator = None
        frame_times = None
    else:
        session_indicator = data['session_indicator']
        frame_times = data['frame_times']
    session_ts = data['session_ts']
    tracking_ts = data['tracking_ts']

    # update settings
    include_speed_filter = False
    include_spatial_filter = False
    if speed_filter_var is not None:
        include_speed_filter = True
        minspeedfilt = speed_filter_var[0]
        maxspeedfilt = speed_filter_var[1]
    if spatial_filter_diameter is not None:
        include_spatial_filter = True

    if derivatives_param is not None:
        settings['bins_der'] = derivatives_param

    # construct

    der_param = settings['bins_der']
    num_par = len(settings['selfmotion_window_size'])

    if speed_type == 'jump':
        sfmat_ind = np.arange(2 * num_par)
        speed_ind = np.arange(num_par)
    elif speed_type == 'cum':
        sfmat_ind = np.arange(2 * num_par, 4 * num_par)
        speed_ind = np.arange(num_par, 2 * num_par)
    else:
        raise Exception('Not defined speed definitation !!!')

    # check data
    comparing = False
    if 'cf_allo_head_ang' in dt_keys:
        print('comparison data included.')
        comparing = True

    # read data
    bbscale_xy = data['bbscale_xy'].copy()
    bbtrans_xy = data['bbtrans_xy'].copy()

    sorted_point_data = data['sorted_point_data'].copy()
    animal_loc = sorted_point_data[:, 4, :]  # neck point

    # construct processing data
    temp_data = {}
    temp_data['allo_head_rotm'] = data['global_head_rot_mat'].copy()
    temp_data['r_root_inv'] = data['r_root_inv'].copy()
    temp_data['r_root_inv_oriented'] = data['r_root_inv_oriented'].copy()
    temp_data['body_direction'] = data['body_direction'].copy()
    temp_data['back_ang'] = data['back_ang'].copy()
    temp_data['opt_back_ang'] = data['opt_back_ang'].copy()
    temp_data['allo_head_ang'] = data['allo_head_ang'].copy()
    temp_data['ego3_head_ang'] = data['ego3_head_ang'].copy()
    temp_data['ego2_head_ang'] = data['ego2_head_ang'].copy()
    temp_data['speeds'] = data['speeds'].copy()
    temp_data['selfmotion'] = data['selfmotion'].copy()

    # get useful selfmotion vals
    print('getting self motion ....')
    selfmotiion_mat = temp_data['selfmotion'][:, sfmat_ind]
    speeds_mat = temp_data['speeds'][:, speed_ind]
    dxs = selfmotiion_mat[:, 0: 2 * num_par: 2]
    dys = selfmotiion_mat[:, 1: 2 * num_par: 2]
    speeds = speeds_mat

    main_key = temp_data.keys()
    n_main_key = len(main_key)
    main_key_ind = np.arange(n_main_key)

    if comparing:
        cf_bbscale_xy = data['bbscale_xy'].copy()
        cf_bbtrans_xy = data['bbtrans_xy'].copy()

        cf_sorted_point_data = data['sorted_point_data'].copy()
        cf_animal_loc = cf_sorted_point_data[:, 4, :]  # neck point

        temp_data['cf_allo_head_rotm'] = data['imu_global_head_rot_mat'].copy()
        temp_data['cf_r_root_inv'] = data['r_root_inv'].copy()
        temp_data['cf_r_root_inv_oriented'] = data['r_root_inv_oriented'].copy()
        temp_data['cf_body_direction'] = data['body_direction'].copy()
        temp_data['cf_back_ang'] = data['back_ang'].copy()
        temp_data['cf_opt_back_ang'] = data['opt_back_ang'].copy()
        temp_data['cf_allo_head_ang'] = data['imu_allo_head_ang'].copy()
        temp_data['cf_ego2_head_ang'] = data['imu_ego2_head_ang'].copy()
        temp_data['cf_ego3_head_ang'] = data['imu_ego3_head_ang'].copy()
        temp_data['cf_speeds'] = data['speeds'].copy()
        temp_data['cf_selfmotion'] = data['selfmotion'].copy()

        # get useful selfmotion vals
        print('getting comparing self motion ....')
        cf_selfmotiion_mat = temp_data['cf_selfmotion'][:, sfmat_ind]
        cf_speeds_mat = temp_data['cf_speeds'][:, speed_ind]
        cf_dxs = cf_selfmotiion_mat[:, 0: 2 * num_par: 2]
        cf_dys = cf_selfmotiion_mat[:, 1: 2 * num_par: 2]
        cf_speeds = cf_speeds_mat

    all_key_list = list(temp_data.keys())
    n_cf_key = len(all_key_list) - n_main_key
    cf_key_ind = np.arange(n_cf_key) + n_main_key
    cf_key = [all_key_list[ind] for ind in cf_key_ind]

    for i in range(num_par):
        print((
            'To get all the values of the movement vectors for window size %d, you would need a min/max horizontal value of %f %f and a min vertical value of %f and a max of %f' % (
                data['settings']['selfmotion_window_size'][i], np.nanmin(dxs[:, i]), np.nanmax(dxs[:, i]),
                np.nanmin(dys[:, i]), np.nanmax(dys[:, i])), 'for the second'))
    # dxs, dys, speeds = get_sf_var(temp_data, speed_type)

    if isinstance(settings['selfmotion_window_size'], int):
        if speed_filter_ind != settings['selfmotion_window_size']:
            print('Possible value for speed_filter_ind is', settings['selfmotion_window_size'], ', set to',
                  settings['selfmotion_window_size'], '.')
        speeds2 = speeds
        if comparing:
            cf_speeds2 = cf_speeds
    else:
        if speed_filter_ind not in settings['selfmotion_window_size']:
            print('Possible values for speed_filter_ind are', settings['selfmotion_window_size'], ', set to',
                  settings['selfmotion_window_size'][-1], '.')
            speed_filter_ind = settings['selfmotion_window_size'][-1]
        ind4speed = [ind for ind in range(len(settings['selfmotion_window_size'])) if
                     speed_filter_ind == settings['selfmotion_window_size'][ind]][0]
        speeds2 = speeds[:, ind4speed]
        if comparing:
            cf_speeds2 = cf_speeds[:, ind4speed]

    # start spatial filtering
    if include_spatial_filter:
        print('processing spatial fitering ....')
        rad_squared = (spatial_filter_diameter / 2.) ** 2
        dd = (animal_loc[:, 0] - bbtrans_xy[0]) ** 2 + (animal_loc[:, 1] - bbtrans_xy[1]) ** 2
        beyond_radius = dd > rad_squared
        numbef = np.sum(~np.isnan(animal_loc[:, 0]))
        if np.sum(beyond_radius) > 0:
            whiches = beyond_radius
            animal_loc[whiches] = np.nan
            dxs[whiches] = np.nan
            dys[whiches] = np.nan
            speeds[whiches] = np.nan
            speeds2[whiches] = np.nan
            for da_key in main_key:
                temp_data[da_key][whiches] = np.nan

        if comparing:
            cf_dd = (cf_animal_loc[:, 0] - cf_bbtrans_xy[0]) ** 2 + (cf_animal_loc[:, 1] - cf_bbtrans_xy[1]) ** 2
            cf_beyond_radius = cf_dd > rad_squared
            cf_numbef = np.sum(~np.isnan(cf_animal_loc[:, 0]))
            if np.sum(cf_beyond_radius) > 0:
                whiches = cf_beyond_radius
                cf_animal_loc[whiches] = np.nan
                cf_dxs[whiches] = np.nan
                cf_dys[whiches] = np.nan
                cf_speeds[whiches] = np.nan
                cf_speeds2[whiches] = np.nan
                for da_key in cf_key:
                    temp_data[da_key][whiches] = np.nan

        numnow = np.sum(~np.isnan(animal_loc[:, 0]))
        print(('After filtering, the remaining tracked data is (for the neck point)', numnow, 'which is',
               100. * numnow / float(len(animal_loc[:, 0])), 'percent of the total data length'))

        numnow = np.sum(~np.isnan(cf_animal_loc[:, 0]))
        print(('After filtering, the remaining tracked data is (for the neck point)', numnow, 'which is',
               100. * numnow / float(len(cf_animal_loc[:, 0])), 'percent of the total data length'))
    # start speed filtering
    if include_speed_filter:
        print('processing speed filter ....')
        below_thresh = speeds2 < minspeedfilt
        above_thresh = speeds2 > maxspeedfilt
        if np.sum(below_thresh) > 0:
            whiches = below_thresh
            animal_loc[whiches] = np.nan
            dxs[whiches] = np.nan
            dys[whiches] = np.nan
            speeds[whiches] = np.nan
            speeds2[whiches] = np.nan
            for da_key in main_key:
                temp_data[da_key][whiches] = np.nan

        if np.sum(above_thresh) > 0:
            whiches = above_thresh
            animal_loc[whiches] = np.nan
            dxs[whiches] = np.nan
            dys[whiches] = np.nan
            speeds[whiches] = np.nan
            speeds2[whiches] = np.nan
            for da_key in main_key:
                temp_data[da_key][whiches] = np.nan

        if comparing:
            print('processing comparing speed filter ....')
            below_thresh = cf_speeds2 < minspeedfilt
            above_thresh = cf_speeds2 > maxspeedfilt
            if np.sum(below_thresh) > 0:
                whiches = below_thresh
                cf_animal_loc[whiches] = np.nan
                cf_dxs[whiches] = np.nan
                cf_dys[whiches] = np.nan
                cf_speeds[whiches] = np.nan
                cf_speeds2[whiches] = np.nan
                for da_key in cf_key:
                    temp_data[da_key][whiches] = np.nan

            if np.sum(above_thresh) > 0:
                whiches = above_thresh
                cf_animal_loc[whiches] = np.nan
                cf_dxs[whiches] = np.nan
                cf_dys[whiches] = np.nan
                cf_speeds[whiches] = np.nan
                cf_speeds2[whiches] = np.nan
                for da_key in cf_key:
                    temp_data[da_key][whiches] = np.nan

    factor1d = {}
    factor1d['B Speeds'] = speeds2
    factor1d['C Body_direction'] = temp_data['body_direction']
    factor1d['D Allo_head_direction'] = temp_data['allo_head_ang'][:, 2]
    # factor1d['E Allo_head_pitch'] = data['allo_head_ang'][:,1]
    # factor1d['F Allo_head_roll'] = data['allo_head_ang'][:,0]
    factor1d['G Neck_elevation'] = animal_loc[:, 2] * 100

    factor1d['K Ego3_Head_roll'] = temp_data['ego3_head_ang'][:, 0]
    factor1d['L Ego3_Head_pitch'] = temp_data['ego3_head_ang'][:, 1]
    factor1d['M Ego3_Head_azimuth'] = temp_data['ego3_head_ang'][:, 2]
    factor1d['N Back_pitch'] = temp_data['opt_back_ang'][:, 0]
    factor1d['O Back_azimuth'] = temp_data['opt_back_ang'][:, 1]
    factor1d['P Ego2_head_roll'] = temp_data['ego2_head_ang'][:, 0]
    factor1d['Q Ego2_head_pitch'] = temp_data['ego2_head_ang'][:, 1]
    factor1d['R Ego2_head_azimuth'] = temp_data['ego2_head_ang'][:, 2]

    # all the bounds
    bounds1d = {'B Speeds': [0, 0, False], 'C Body_direction': [-180, 180, True], 'D Allo_head_direction': [-180, 180, True], 'G Neck_elevation': [0, 0, True], 'K Ego3_Head_roll': [-180, 180, True],
                'L Ego3_Head_pitch': [-180, 180, True], 'M Ego3_Head_azimuth': [-180, 180, True], 'N Back_pitch': [-60, 60, True], 'O Back_azimuth': [-60, 60, True],
                'P Ego2_head_roll': [-180, 180, True], 'Q Ego2_head_pitch': [-180, 180, True],
                'R Ego2_head_azimuth': [-180, 180, True]}  # zeros mean optimize it for me!! So it is min, max, periodic (True or False)

    xaxis1d = {'B Speeds': 'cm per second', 'C Body_direction': 'angles', 'D Allo_head_direction': 'angles', 'G Neck_elevation': 'cm', 'K Ego3_Head_roll': 'ccw --- angles --- cw',
               'L Ego3_Head_pitch': 'down --- angles --- up', 'M Ego3_Head_azimuth': 'left --- angles --- right', 'N Back_pitch': 'down --- angles --- up',
               'O Back_azimuth': 'left --- angles --- right', 'P Ego2_head_roll': 'ccw --- angles --- cw', 'Q Ego2_head_pitch': 'down --- angles --- up',
               'R Ego2_head_azimuth': 'left --- angles --- right'}
    # xaxis1d['E Allo_head_pitch'] = 'angles'
    # xaxis1d['F Allo_head_roll'] = 'ccw --- angles --- cw'

    if comparing:
        factor1d_cf = factor1d.copy()
        factor1d_cf['B Speeds'] = cf_speeds2
        factor1d_cf['C Body_direction'] = temp_data['cf_body_direction']
        factor1d_cf['G Neck_elevation'] = cf_animal_loc[:, 2] * 100
        factor1d_cf['D Allo_head_direction'] = temp_data['cf_allo_head_ang'][:, 2]
        factor1d_cf['K Ego3_Head_roll'] = temp_data['cf_ego3_head_ang'][:, 0]
        factor1d_cf['L Ego3_Head_pitch'] = temp_data['cf_ego3_head_ang'][:, 1]
        factor1d_cf['M Ego3_Head_azimuth'] = temp_data['cf_ego3_head_ang'][:, 2]
        factor1d_cf['N Back_pitch'] = temp_data['cf_opt_back_ang'][:, 0]
        factor1d_cf['O Back_azimuth'] = temp_data['cf_opt_back_ang'][:, 1]
        factor1d_cf['P Ego2_head_roll'] = temp_data['cf_ego2_head_ang'][:, 0]
        factor1d_cf['Q Ego2_head_pitch'] = temp_data['cf_ego2_head_ang'][:, 1]
        factor1d_cf['R Ego2_head_azimuth'] = temp_data['cf_ego2_head_ang'][:, 2]

        bounds1d_cf = bounds1d.copy()
        xaxis1d_cf = xaxis1d.copy()

    print('calculating derivatives ...')
    factor1d, bounds1d, xaxis1d = re_calc_der(factor1d, bounds1d, xaxis1d, captureframerate, der_param[0], der_param[1], session_indicator)

    if comparing:
        print('calculating derivatives for comparing data ...')
        factor1d_cf, bounds1d_cf, xaxis1d_cf = re_calc_der(factor1d_cf, bounds1d_cf, xaxis1d_cf,
                                                           captureframerate, der_param[0], der_param[1], session_indicator)

    if boundary is not None:
        mkeys = boundary.keys()
        bkeys = bounds1d.keys()
        for dm_key in mkeys:
            if (dm_key in bkeys):
                bounds1d[dm_key] = boundary[dm_key]

        if comparing:
            bkeys = bounds1d_cf.keys()
            for dm_key in mkeys:
                if (dm_key in bkeys):
                    bounds1d_cf[dm_key] = boundary[dm_key]

    # start split axis
    if len(split_data_along_an_axis) > 1:
        print('processing split ...')

        def greaterthanignorenan(x, value):
            notnans = ~np.isnan(x)
            whiches = np.zeros(len(x), dtype=bool)
            whiches[notnans] = x[notnans] > value
            return whiches

        def lessthanignorenan(x, value):
            notnans = ~np.isnan(x)
            whiches = np.zeros(len(x), dtype=bool)
            whiches[notnans] = x[notnans] < value
            return whiches

        split_val = factor1d[split_data_along_an_axis]
        gt = greaterthanignorenan(split_val, end_split_value)
        lt = lessthanignorenan(split_val, start_split_value)
        whiches = (gt + lt) > 0.5
        print('Num combined is', np.sum((split_val < start_split_value) + (split_val > end_split_value)))
        print('Num to remove then is', np.sum(whiches), 'since the total is', len(whiches))
        if np.sum(whiches) > 0:
            all_keys = list(factor1d.keys())
            for da_key in all_keys:
                vals = factor1d[da_key]
                sh = np.shape(vals)
                print(da_key, sh, len(sh), 'CHECK THIS YOU IDIOT!!!')
                if len(sh) == 1:
                    vals[whiches] = np.nan
                if len(sh) == 2:
                    vals[whiches, :] = np.nan
                if len(sh) == 3:
                    vals[whiches, :, :] = np.nan
                factor1d[da_key] = vals

        animal_loc[whiches] = np.nan
        dxs[whiches] = np.nan
        dys[whiches] = np.nan
        speeds[whiches] = np.nan
        for da_key in main_key:
            temp_data[da_key][whiches] = np.nan

        split_val = factor1d_cf[split_data_along_an_axis]
        gt = greaterthanignorenan(split_val, end_split_value)
        lt = lessthanignorenan(split_val, start_split_value)
        whiches = (gt + lt) > 0.5
        print('Num combined is', np.sum((split_val < start_split_value) + (split_val > end_split_value)))
        print('Num to remove then is', np.sum(whiches), 'since the total is', len(whiches))
        if np.sum(whiches) > 0:
            all_keys = list(factor1d_cf.keys())
            for da_key in all_keys:
                vals = factor1d_cf[da_key]
                sh = np.shape(vals)
                print(da_key, sh, len(sh), 'CHECK THIS YOU IDIOT!!!')
                if len(sh) == 1:
                    vals[whiches] = np.nan
                if len(sh) == 2:
                    vals[whiches, :] = np.nan
                if len(sh) == 3:
                    vals[whiches, :, :] = np.nan
                factor1d_cf[da_key] = vals

        cf_animal_loc[whiches] = np.nan
        cf_dxs[whiches] = np.nan
        cf_dys[whiches] = np.nan
        cf_speeds[whiches] = np.nan
        for da_key in cf_key:
            temp_data[da_key][whiches] = np.nan

    # get super bounds!!!
    def getbinswithenoughineach(values, minval, maxval, num_bins_1d, frame_rate, session_ind):
        bins = np.linspace(minval, maxval, num_bins_1d + 1)
        occupancy = np.zeros(len(bins) - 1)
        invalid_ind = np.isnan(values)
        values = values[~invalid_ind]
        if session_ind is not None:
            valid_session_ind = session_ind[~invalid_ind]
        else:
            valid_session_ind = np.zeros(len(values), 'i')
        for i in range(1, len(bins), 1):
            if len(np.ravel(frame_rate)) == 1:
                occupancy[i - 1] = float(np.sum((values > bins[i - 1]) * (values <= bins[i]))) / float(frame_rate)  # in seconds
            else:
                total_mat = []
                for j in range(len(frame_rate)):
                    scount = (valid_session_ind == j)
                    part_mat = float(np.sum((values > bins[i - 1]) * (values <= bins[i]) * scount)) / float(frame_rate[j])  # in seconds
                    total_mat.append(part_mat)
                total_num = np.sum(total_mat)
                occupancy[i - 1] = total_num

        return bins, occupancy

    def get_super_bounds(factors, bounds, num_bins_1d, captureframerate, occupancy_thresh_1d, session_ind):
        all_da_bins = {}
        all_keys = list(factors.keys())
        for i in range(len(all_keys)):
            da_key = all_keys[i]
            values = factors[da_key]  # values
            ips = bounds[da_key]  # bound

            minval = ips[0]  # lower limit of the bound
            maxval = ips[1]  # upper limit of the bound
            if abs(minval) < 0.00001 and abs(maxval) < 0.00001:
                minval = np.nanmin(values)
                maxval = np.nanmax(values)
                for j in range(10000):
                    bins, occ = getbinswithenoughineach(values, minval, maxval, num_bins_1d, captureframerate, session_ind)
                    goodtogo = True
                    if occ[0] < occupancy_thresh_1d:
                        minval = minval + 0.05 * (bins[1] - bins[0])
                        goodtogo = False
                    if occ[-1] < occupancy_thresh_1d:
                        maxval = maxval - 0.05 * (bins[-1] - bins[-2])
                        goodtogo = False
                    if goodtogo == True:
                        break
                    if goodtogo == False and j > 9999:
                        for k in range(10):
                            print(('SHIT! Could not find good bounds for the variable!!', da_key))

            bins, occ = getbinswithenoughineach(values, minval, maxval, num_bins_1d, captureframerate, session_ind)
            all_da_bins[da_key] = bins + 0.
            bounds[da_key][0] = bins[0]
            bounds[da_key][1] = bins[-1]
        return all_da_bins, bounds

    print('getting super bounds ...')
    bins1d, bounds1d = get_super_bounds(factor1d, bounds1d, num_bins_1d, captureframerate, occupancy_thresh_1d, session_indicator)
    if comparing:
        print('getting super bounds for comparing data ...')
        bins1d_cf, bounds1d_cf = get_super_bounds(factor1d_cf, bounds1d_cf, num_bins_1d, captureframerate, occupancy_thresh_1d, session_indicator)

    # making output file name
    output_file_prefix = '%s_1D2D' % (filename)

    if use_even_odd_minutes:
        output_file_prefix = '%s_evenodd' % output_file_prefix
    else:
        output_file_prefix = '%s_firstsecondhalves' % output_file_prefix
    if include_speed_filter:
        output_file_prefix = '%s_speedfiltered_from_%05d_to_%05d' % (output_file_prefix, minspeedfilt, maxspeedfilt)
    if include_spatial_filter:
        output_file_prefix = '%s_spatfilt' % output_file_prefix
    if len(split_data_along_an_axis) > 1:
        output_file_prefix = '%s_%s_filtered_from_%09d_to_%09d' % (
            output_file_prefix, split_data_along_an_axis[2:], int(round(1000. * start_split_value)),
            int(round(1000. * end_split_value)))

    output_file_prefix = '%s_rotatedback' % output_file_prefix

    mat_data = {'full_tracking_ts': None, 'session_indicator': session_indicator, 'frame_times': frame_times, 'settings': settings, 'framerate': captureframerate, 'session_ts': session_ts,
                'tracking_ts': tracking_ts, 'output_file_prefix': output_file_prefix, 'cell_names': data['cell_names'], 'cell_activities': data['cell_activities'], 'startaltbins': startaltbins,
                'endaltbins': endaltbins, 'bbscale_xy': bbscale_xy, 'bbtrans_xy': bbtrans_xy, 'dxs': dxs, 'dys': dys, 'speeds': speeds, 'animal_location': animal_loc, 'possiblecovariates': factor1d,
                'possiblecovariatesnames': xaxis1d, 'possiblecovariatesbounds': bounds1d, 'possiblecovariatesbins': bins1d}

    if comparing:
        mat_data['cf_bbscale_xy'] = cf_bbscale_xy
        mat_data['cf_bbtrans_xy'] = cf_bbtrans_xy
        mat_data['cf_dxs'] = cf_dxs
        mat_data['cf_dys'] = cf_dys
        mat_data['cf_speeds'] = cf_speeds
        mat_data['cf_animal_location'] = cf_animal_loc
        mat_data['cf_possiblecovariates'] = factor1d_cf
        mat_data['cf_possiblecovariatesnames'] = xaxis1d_cf
        mat_data['cf_possiblecovariatesbounds'] = bounds1d_cf
        mat_data['cf_possiblecovariatesbins'] = bins1d_cf

    if save_data:
        # scipy.io.savemat('ok4rms_%s.mat' % output_file_prefix, mat_data)
        a_file = open('ok4rms_%s.pkl' % output_file_prefix, "wb")
        pickle.dump(mat_data, a_file)
        a_file.close()
    return mat_data


def get_2d_tasks(include_derivatives=True, printout=False):
    stuff2d = {'A Ego2_H_pitch_and_Ego3_H_pitch': ['Q Ego2_head_pitch', 'L Ego3_Head_pitch'], 'B Ego2_H_roll_and_Ego3_H_roll': ['P Ego2_head_roll', 'K Ego3_Head_roll'],
               'C Ego2_H_azimuth_and_Ego3_H_azimuth': ['R Ego2_head_azimuth', 'M Ego3_Head_azimuth'], 'D Ego2_H_pitch_and_allo_HD': ['Q Ego2_head_pitch', 'D Allo_head_direction'],
               'E Ego2_H_pitch_and_neck_elevation': ['Q Ego2_head_pitch', 'G Neck_elevation'], 'E Ego2_H_pitch_and_Ego2_H_azimuth': ['Q Ego2_head_pitch', 'R Ego2_head_azimuth'],
               'E Ego2_H_pitch_and_Ego2_H_roll': ['Q Ego2_head_pitch', 'P Ego2_head_roll'], 'F Ego2_H_azimuth_and_Ego2_H_roll': ['R Ego2_head_azimuth', 'P Ego2_head_roll'],
               'G Ego3_H_pitch_and_neck_elevation': ['L Ego3_Head_pitch', 'G Neck_elevation'], 'G Ego3_H_pitch_and_Ego3_H_roll': ['L Ego3_Head_pitch', 'K Ego3_Head_roll'],
               'G Ego3_H_pitch_and_Ego3_H_azimuth': ['L Ego3_Head_pitch', 'M Ego3_Head_azimuth'], 'H Ego3_H_azimuth_and_Ego3_H_roll': ['M Ego3_Head_azimuth', 'K Ego3_Head_roll'],
               'J Ego3_H_roll_and_B_azimuth': ['K Ego3_Head_roll', 'O Back_azimuth'], 'P Speeds_and_Ego2_H_pitch': ['B Speeds', 'Q Ego2_head_pitch'],
               'Q Speeds_and_Ego3_H_pitch': ['B Speeds', 'L Ego3_Head_pitch'], 'R Speeds_and_Ego3_H_azimuth': ['B Speeds', 'M Ego3_Head_azimuth'],
               'S Speeds_and_Ego3_H_roll': ['B Speeds', 'K Ego3_Head_roll'], 'U B_pitch_and_B_direction': ['N Back_pitch', 'C Body_direction'],
               'V B_pitch_and_Ego2_H_pitch': ['N Back_pitch', 'Q Ego2_head_pitch'], 'W B_pitch_and_Ego3_H_pitch': ['N Back_pitch', 'L Ego3_Head_pitch'],
               'X B_pitch_and_Ego3_H_azimuth': ['N Back_pitch', 'M Ego3_Head_azimuth'], 'Y B_pitch_and_Ego3_H_roll': ['N Back_pitch', 'K Ego3_Head_roll'],
               'Z B_pitch_and_B_azimuth': ['N Back_pitch', 'O Back_azimuth']}

    # allda2Dstuff['R Allo_H_roll_and_neck_elevation'] = ['F Allo_head_roll', 'G Neck_elevation']
    # allda2Dstuff['S Allo_H_roll_and_H_roll'] = ['F Allo_head_roll', 'K Head_roll']

    if include_derivatives:
        stuff2d['L Ego3_H_pitch_and_Allo_HD_1st_der'] = ['L Ego3_Head_pitch', 'D Allo_head_direction_1st_der']
        stuff2d['M Speeds_and_Allo_HD_1st_der'] = ['B Speeds', 'D Allo_head_direction_1st_der']
        stuff2d['N Speeds_and_BD_1st_der'] = ['B Speeds', 'C Body_direction_1st_der']
        stuff2d['T B_pitch_and_Allo_HD_1st_der'] = ['N Back_pitch', 'D Allo_head_direction_1st_der']

    if printout:
        for dakey in stuff2d.keys():
            print(dakey)

    return stuff2d


def ratemap_generator(data, comparing=False, cell_index=(10, 11), tempoffsets=0,
                      nshuffles=1000, shuffle_offset=(15, 60),
                      include_selfmotion=True, selfmotion_lims=(0, 40, -5, 80), selfmotion_bin_size=3,
                      include_derivatives=True, include_spatial_maps=True,
                      include_generic_2d_plots=True, include_velocity_plots=False,
                      comparing_task_2d=(''), comparing_task_1d=(''),
                      compare_selfmotion=True, compare_spatial_maps=True, compare_velocity=False,
                      pl_subplot=(14, 10), pl_size=(70, 70),
                      cf_subplot=(14, 10), cf_size=(70, 70),
                      save_1d_ratemaps=True):
    """
    Purpose
    -------------
    Doing shufflings and make rate maps.

    Inputs
    -------------
    data :  data that contains all the information that needed, see data_generator() and prepare4ratemaps() for example.
    
    comparing : False(default), if plot 2 groups of datasets side by side to compare.
    
    cell_index : index of the cells included in the data. integer or integer array.
    
    tempoffsets : integer or np.array([ -1500, -1000, -500, -250, -150, -100, -50, -25, -16, 0, 16, 25, 50, 100, 250, 500, 1000]).
    
    nhuffles : 1000 (default), number of shuffles. you need enough for it to be well approximated by a Gaussian, assuming it can be.
               Each rate map distributions for the scores are calculated using random shuffles between +/- [MIN,MAX] relative to zero.
               Note, it is IMPORTANT that the MIN/MAX values are outside the range for the temporal offsets!!!
    
    shuffle_offset : Min/Max value for the shuffling range.
    
    include_selfmotion : True (default). Whether or not to make selfmotion maps.
    
    selfmotion_lims : Min/Max value for the horizontal axis (cm/s) and Min/Max value for the vertical axis (cm/s).
    
    selfmotion_bin_size : size of bins for selfmotion.

    
    Outputs
    -------------
    Auto saved rate maps.


    """
    # check data
    matplotlib.use('Agg')

    dkeys = list(data.keys())

    settings = data['settings']
    captureframerate = data['framerate']
    if len(np.ravel(captureframerate)) == 1:
        session_indicator = None
        frame_times = None
        full_tracking_ts = None
    else:
        session_indicator = data['session_indicator']
        frame_times = data['frame_times']
        full_tracking_ts = data['full_tracking_ts']
    session_ts = data['session_ts']
    tracking_ts = data['tracking_ts']
    bbscale_xy = data['bbscale_xy']
    bbtrans_xy = data['bbtrans_xy']
    output_file_prefix = data['output_file_prefix']
    startaltbins = data['startaltbins']
    endaltbins = data['endaltbins']

    if (isinstance(cell_index, int)):
        cell_names = [data['cell_names'][cell_index]]
        cell_activities = [data['cell_activities'][cell_index]]
    else:
        cell_names = [data['cell_names'][i] for i in cell_index]
        cell_activities = [data['cell_activities'][i] for i in cell_index]

    n_cells = len(cell_names)
    print(n_cells)

    animal_loc = data['animal_location']
    bbscale_xy = data['bbscale_xy']
    bbtrans_xy = data['bbtrans_xy']
    dxs = data['dxs']
    dys = data['dys']
    speeds = data['speeds']
    factor1d = data['possiblecovariates']
    xaxis1d = data['possiblecovariatesnames']
    bounds1d = data['possiblecovariatesbounds']
    bins1d = data['possiblecovariatesbins']
    if (comparing):
        cf_animal_loc = data['cf_animal_location']
        cf_bbscale_xy = data['cf_bbscale_xy']
        cf_bbtrans_xy = data['cf_bbtrans_xy']
        cf_dxs = data['cf_dxs']
        cf_dys = data['cf_dys']
        cf_speeds = data['cf_speeds']
        cf_factor1d = data['cf_possiblecovariates']
        cf_xaxis1d = data['cf_possiblecovariatesnames']
        cf_bounds1d = data['cf_possiblecovariatesbounds']
        cf_bins1d = data['cf_possiblecovariatesbins']

    n_frames = len(animal_loc)
    output_file_prefix = '%s_shuffling_%05d_times' % (output_file_prefix, nshuffles)

    all_1d_keys = factor1d.keys()
    if (len(comparing_task_1d) == 0):
        comparing_task_1d = all_1d_keys

    # setup constant
    # parameters for self motion maps !!
    # selfmotion parameters
    min_xval = selfmotion_lims[0]
    max_xval = selfmotion_lims[1]
    min_yval = selfmotion_lims[2]
    max_yval = selfmotion_lims[3]
    sizeofbins = float(selfmotion_bin_size)
    nbin_min_xval = int(np.ceil((float(min_xval)) / sizeofbins))
    nbin_max_xval = int(np.ceil((float(max_xval)) / sizeofbins))
    nbin_min_yval = int(np.ceil((abs(float(min_yval))) / sizeofbins))
    nbin_max_yval = int(np.ceil((abs(float(max_yval))) / sizeofbins))

    occupancy_thresh = 400. / 1000. + 0.001  # Minimum bin occupancy (seconds), Parameters for self motion maps!!
    smoothing_par = [1.15, 1.15]  # Width and Height of Gaussian smoothing (bins), Parameters for self motion maps!!

    # Parameters for velocity map thingies!!
    velocity_max_speed = 60
    nbin_speed = 18
    nbin_angle = 20

    # Parameters for spatial maps!!
    nbin_spatial = 30

    # Parameters for 1D curves
    num_bins_1d = 36
    occupancy_thresh_1d = 400. / 1000. + 0.001  # Minimum bin occupancy (seconds)

    tempoffsets = np.ravel(tempoffsets) / 1000.

    # read input arguments
    shuffleoffsetMIN = shuffle_offset[0]  # in seconds!!
    shuffleoffsetMAX = shuffle_offset[1]  # in seconds!!

    if (include_generic_2d_plots):
        stuff2d = get_2d_tasks(include_derivatives)
        all_2d_keys = stuff2d.keys()

        if (comparing):
            if (len(comparing_task_2d) == 0):
                comparing_task_2d = all_2d_keys
            stuff2d_cf = {}
            for dakey in comparing_task_2d:
                stuff2d_cf[dakey] = stuff2d[dakey]

    ixess = nbin_max_xval + (np.floor(dxs / sizeofbins)).astype(int)
    jyess = nbin_min_yval + (np.floor(dys / sizeofbins)).astype(int)

    if (comparing):
        cf_ixess = nbin_max_xval + (np.floor(cf_dxs / sizeofbins)).astype(int)
        cf_jyess = nbin_min_yval + (np.floor(cf_dys / sizeofbins)).astype(int)

    if frame_times is None:
        all_tcount = get_tcount(n_frames, captureframerate, tracking_ts, [session_ts[0]], [session_ts[1]])
        p1_tcount = get_tcount(n_frames, captureframerate, tracking_ts, startaltbins[1::2], endaltbins[1::2])
        p2_tcount = get_tcount(n_frames, captureframerate, tracking_ts, startaltbins[0::2], endaltbins[0::2])
    else:
        all_tcount = get_tcount_time(frame_times, [session_ts[0]], [session_ts[1]])
        p1_tcount = get_tcount_time(frame_times, startaltbins[1::2], endaltbins[1::2])
        p2_tcount = get_tcount_time(frame_times, startaltbins[0::2], endaltbins[0::2])

    all_binnocc, all_smoothed_occ = get_1d_binnocc(factor1d, bins1d, all_tcount, captureframerate, session_indicator)
    p1_binnocc, p1_smoothed_occ = get_1d_binnocc(factor1d, bins1d, p1_tcount, captureframerate, session_indicator)
    p2_binnocc, p2_smoothed_occ = get_1d_binnocc(factor1d, bins1d, p2_tcount, captureframerate, session_indicator)
    if (comparing):
        cf_all_binnocc, cf_all_smoothed_occ = get_1d_binnocc(cf_factor1d, cf_bins1d, all_tcount, captureframerate, session_indicator)
        cf_p1_binnocc, cf_p1_smoothed_occ = get_1d_binnocc(cf_factor1d, cf_bins1d, p1_tcount, captureframerate, session_indicator)
        cf_p2_binnocc, cf_p2_smoothed_occ = get_1d_binnocc(cf_factor1d, cf_bins1d, p2_tcount, captureframerate, session_indicator)

    # shuffle
    def getgetget(factor, bins, occ, cellocc_ind):
        valvalval = np.zeros(num_bins_1d)
        xvals, rawrm, smrm = get_1d_ratemap(factor, bins, occ, cellocc_ind)
        valvalval[occ > occupancy_thresh_1d] = smrm[occ > occupancy_thresh_1d]
        valvalval[occ <= occupancy_thresh_1d] = np.nan
        return valvalval, occ, rawrm

    # factors, bins = factor1d, bins1d
    def get_shuffled_data(factors, bins, all_binnocc):
        ratemaps_1d_data = {}
        shuffled_values_1d_means = {}
        shuffled_values_1d_stds = {}
        all_keys = list(factors.keys())
        nkeys = len(all_keys)
        for j in range(nkeys):
            da_key = all_keys[j]
            shuffled_values_1d_means[da_key] = np.zeros((num_bins_1d, n_cells))
            shuffled_values_1d_stds[da_key] = np.zeros((num_bins_1d, n_cells))

        random_numbers = np.random.rand(nshuffles, n_cells)
        toff_mat = random_numbers * (shuffleoffsetMAX - shuffleoffsetMIN) + shuffleoffsetMIN

        for cellnum in np.arange(n_cells):
            da_cell_name = cell_names[cellnum]
            da_cell_activity = cell_activities[cellnum]
            vals = np.zeros((nkeys, num_bins_1d, nshuffles))
            occs = np.zeros((nkeys, num_bins_1d, nshuffles))
            rawrms = np.zeros((nkeys, num_bins_1d, nshuffles))
            print(('Shuffling', 100 * float(cellnum + 1) / float(n_cells), 'going ... '))
            for i in np.arange(nshuffles):
                toff = toff_mat[i, cellnum]
                cellocc_ind_shuffles = get_cell_occ(da_cell_activity, toff, captureframerate, tracking_ts, [session_ts[0]], [session_ts[1]], full_tracking_ts)
                for j in range(nkeys):
                    da_key = all_keys[j]
                    vals[j, :, i], occs[j, :, i], rawrms[j, :, i] = getgetget(factors[da_key], bins[da_key],
                                                                              all_binnocc[da_key],
                                                                              cellocc_ind_shuffles)
            for j in np.arange(nkeys):
                da_key = all_keys[j]
                for k in np.arange(num_bins_1d):
                    if np.sum(np.isnan(vals[j, k, :]) == False) > 0:
                        nmmm = np.nanmean(vals[j, k, :]) + 0.
                        nsss = np.nanstd(vals[j, k, :]) + 0.
                        (shuffled_values_1d_means[da_key])[k, cellnum] = nmmm
                        (shuffled_values_1d_stds[da_key])[k, cellnum] = nsss
                    else:
                        (shuffled_values_1d_means[da_key])[k, cellnum] = np.nan
                        (shuffled_values_1d_stds[da_key])[k, cellnum] = np.nan

                ratemaps_1d_data['%s-%s-rawacc_shuffles' % (da_cell_name, (da_key)[2:])] = rawrms[j, :, :]
                ratemaps_1d_data['%s-%s-rawocc_shuffles' % (da_cell_name, (da_key)[2:])] = occs[j, :, :]
        return ratemaps_1d_data, shuffled_values_1d_means, shuffled_values_1d_stds

    print('doing shuffling .....')
    # cute_seed = np.random.randint(100000)
    # np.random.seed(cute_seed)
    # np.random.seed(103)
    ratemaps_1d_data, shuffled_values_1d_means, shuffled_values_1d_stds = get_shuffled_data(factor1d, bins1d, all_binnocc)
    if comparing:
        print('doing shuffling for comparing .....')
        # np.random.seed(cute_seed)
        # np.random.seed(103)
        cf_ratemaps_1d_data, cf_shuffled_values_1d_means, cf_shuffled_values_1d_stds = \
            get_shuffled_data(cf_factor1d, cf_bins1d, cf_all_binnocc)

    # start plot
    fig_width, fig_height = np.ravel(pl_size)
    plrows, plcols = np.ravel(pl_subplot)
    if comparing:
        cf_plot_row, cf_plot_column = np.ravel(cf_subplot)
        cf_fig_width, cf_fig_height = np.ravel(cf_size)

    print(('available monkeys', np.sort(list(xaxis1d.keys()))))
    # T = len(rotatedDirBacks[:, 0])
    for cellnum in np.arange(n_cells):
        da_cell_name = cell_names[cellnum]
        if n_cells == 1:
            print_cell = cell_index
        else:
            print_cell = cell_index[cellnum]

        print('plotting cell %d ...' % print_cell)
        da_cell_data = cell_activities[cellnum]
        with PdfPages('%s_%04d_%s.pdf' % (output_file_prefix, print_cell, da_cell_name)) as pdf:
            for i in np.arange(len(tempoffsets)):
                cellocc_ind = get_cell_occ(da_cell_data, tempoffsets[i], captureframerate, tracking_ts,
                                           [session_ts[0]],
                                           [session_ts[1]], full_tracking_ts)
                cellocc_ind_p1 = get_cell_occ(da_cell_data, tempoffsets[i], captureframerate, tracking_ts,
                                              startaltbins[1::2], endaltbins[1::2], full_tracking_ts)
                cellocc_ind_p2 = get_cell_occ(da_cell_data, tempoffsets[i], captureframerate, tracking_ts,
                                              startaltbins[0::2], endaltbins[0::2], full_tracking_ts)

                fig = plt.figure(i + 1, figsize=(fig_width, fig_height))
                plt.clf()
                subplot_index = 1
                # plot 2d rate maps
                if include_generic_2d_plots:
                    print('plotting 2D rate maps ...')
                    keys2d = list(stuff2d.keys())
                    keys2d.sort()
                    for j in range(len(keys2d)):
                        da_2d_key = keys2d[j]
                        k2d1 = (stuff2d[da_2d_key])[0]
                        k2d2 = (stuff2d[da_2d_key])[1]
                        fac1 = factor1d[k2d1]
                        fac2 = factor1d[k2d2]
                        bins1 = bins1d[k2d1]
                        bins2 = bins1d[k2d2]
                        rawratemap, ratemap, occupancy = get_2d_ratemap(fac1, fac2, bins1, bins2, all_tcount,
                                                                        cellocc_ind,
                                                                        captureframerate, occupancy_thresh,
                                                                        smoothing_par, session_indicator)
                        plt.subplot(plrows, plcols, subplot_index)
                        subplot_index += 1
                        tit = '%s' % (((da_2d_key)[2:]).replace('_', ' '))
                        plot_ratemaps(rawratemap, occupancy, occupancy_thresh, tit, '')

                        plt.subplot(plrows, plcols, subplot_index)
                        subplot_index += 1
                        tit = '    Smoothed'
                        # Calculate "spatial information rate" alla Skaggs and McNaughton
                        ICr = calculate_skinfo_rate(ratemap, occupancy)
                        tit = '%s (%f)' % (tit, ICr)
                        plot_ratemaps(ratemap, occupancy, occupancy_thresh, tit, '')

                # plot sel motion maps
                if include_selfmotion:
                    print('plotting self motion maps ...')
                    for sind in range(len(data['settings']['selfmotion_window_size'])):
                        if len(data['settings']['selfmotion_window_size']) == 1:
                            ixess = ixess.reshape((len(ixess), 1))
                            jyess = jyess.reshape((len(jyess), 1))
                        f_raw_rate_map1, f_rate_map1, f_occupancy1 = get_selfmotion_map(ixess[:, sind], jyess[:, sind], all_tcount,
                                                                                        cellocc_ind, captureframerate,
                                                                                        occupancy_thresh, smoothing_par,
                                                                                        nbin_max_xval, nbin_min_yval,
                                                                                        nbin_max_yval, session_indicator)
                        plt.subplot(plrows, plcols, subplot_index)
                        subplot_index += 1
                        tit = 'Unsmoothed self motion %d' % data['settings']['selfmotion_window_size'][sind]
                        plot_ratemaps(f_raw_rate_map1, f_occupancy1, occupancy_thresh, tit, '')

                        plt.subplot(plrows, plcols, subplot_index)
                        subplot_index += 1
                        tit = 'Smoothed'
                        ICr = calculate_skinfo_rate(f_rate_map1, f_occupancy1)
                        tit = '%s (%f)' % (tit, ICr)
                        plot_ratemaps(f_rate_map1, f_occupancy1, occupancy_thresh, tit, '')

                # plot velocity tuning
                if include_velocity_plots:
                    print('plotting velocity maps ...')
                    movement_body = data['possiblecovariates']['C Body_direction'] * math.pi / 180
                    speeds2 = data['possiblecovariates']['B Speeds']
                    # v_raw_rate_map1, v_rate_map1, v_occupancy1 = get_velocity_tuning(movement_body, speeds1,
                    #                                                                  all_tcount, cellocc_ind, captureframerate,
                    #                                                                  velocity_max_speed, nbin_speed,
                    #                                                                  nbin_angle, occupancy_thresh,
                    #                                                                  smoothing_par)
                    v_raw_rate_map2, v_rate_map2, v_occupancy2 = get_velocity_tuning(movement_body, speeds2,
                                                                                     all_tcount, cellocc_ind,
                                                                                     captureframerate,
                                                                                     velocity_max_speed, nbin_speed,
                                                                                     nbin_angle, occupancy_thresh,
                                                                                     smoothing_par, session_indicator)
                    plt.subplot(plrows, plcols, subplot_index)
                    subplot_index += 1
                    tit = 'Velocity, unsmoothed'
                    plot_ratemaps(v_raw_rate_map2, v_occupancy2, occupancy_thresh, tit, '')

                    plt.subplot(plrows, plcols, subplot_index)
                    subplot_index += 1
                    tit = 'Smoothed'
                    ICr = calculate_skinfo_rate(v_rate_map2, v_occupancy2)
                    tit = '%s (%f)' % (tit, ICr)
                    plot_ratemaps(v_rate_map2, v_occupancy2, occupancy_thresh, tit, '')
                # plot spatial maps
                if include_spatial_maps:
                    print('plotting spatial maps ...')
                    p_raw_rate_map2, p_rate_map2, p_occupancy2 = get_spatial_tuning(animal_loc, n_frames, all_tcount,
                                                                                    cellocc_ind,
                                                                                    captureframerate, nbin_spatial,
                                                                                    occupancy_thresh, smoothing_par,
                                                                                    session_indicator)
                    plt.subplot(plrows, plcols, subplot_index)
                    subplot_index += 1
                    tit = 'Space, unsmoothed'
                    plot_ratemaps(p_raw_rate_map2, p_occupancy2, occupancy_thresh, tit, '')

                    plt.subplot(plrows, plcols, subplot_index)
                    subplot_index += 1
                    tit = 'Smoothed'
                    ICr = calculate_skinfo_rate(p_rate_map2, p_occupancy2)
                    tit = '%s (%f)' % (tit, ICr)
                    plot_ratemaps(p_rate_map2, p_occupancy2, occupancy_thresh, tit, '')

                # start 1d plots
                print('plotting 1D rate maps ...')
                keys = list(factor1d.keys())
                keys.sort()

                var1st = 0.0
                var2nd = 0.0
                var3rd = 0.0
                var4th = 0.0

                name1st = ''
                name2nd = ''
                name3rd = ''
                name4th = ''

                max_ylim = -10

                onedguys = []
                axis_list = []
                for j in range(len(keys)):
                    # print(j)
                    da_key = keys[j]
                    axis_list.append(plt.subplot(plrows, plcols, subplot_index))
                    onedguys.append(subplot_index)
                    subplot_index += 1
                    occ = all_binnocc[da_key]
                    smoothed_occ = all_smoothed_occ[da_key]
                    xvals, rawrm, smrm = get_1d_ratemap(factor1d[da_key], bins1d[da_key], occ, cellocc_ind)

                    ICr = calculate_skinfo_rate(rawrm, occ)

                    if ICr > var1st:
                        var4th = var3rd
                        name4th = name3rd
                        var3rd = var2nd
                        name3rd = name2nd
                        var2nd = var1st
                        name2nd = name1st
                        var1st = ICr
                        name1st = ((da_key)[2:]).replace('_', ' ')
                    elif ICr > var2nd:
                        var4th = var3rd
                        name4th = name3rd
                        var3rd = var2nd
                        name3rd = name2nd
                        var2nd = ICr
                        name2nd = ((da_key)[2:]).replace('_', ' ')
                    elif ICr > var3rd:
                        var4th = var3rd
                        name4th = name3rd
                        var3rd = ICr
                        name3rd = ((da_key)[2:]).replace('_', ' ')
                    elif ICr > var4th:
                        var4th = ICr
                        name4th = ((da_key)[2:]).replace('_', ' ')

                    tit = ((da_key)[2:]).replace('_', ' ')
                    tit = '%s (%f)' % (tit, ICr)
                    plot_1d_values(xvals, smrm, rawrm, occ, (shuffled_values_1d_means[da_key])[:, cellnum],
                                   (shuffled_values_1d_stds[da_key])[:, cellnum], occupancy_thresh_1d, tit)
                    # plot_1d_value_shalves(factor1d[da_key], captureframerate, tracking_ts, bins1d[da_key],
                    #                       tempoffsets[i], da_cell_data, startaltbins, endaltbins,
                    #                       occupancy_thresh_1d, num_bins_1d)
                    occ_p1 = p1_binnocc[da_key]
                    smocc_p1 = p1_smoothed_occ[da_key]
                    xvals, rawrm_p1, smrm_p1 = get_1d_ratemap(factor1d[da_key], bins1d[da_key], occ_p1, cellocc_ind_p1)
                    plt.plot(xvals[occ_p1 > occupancy_thresh_1d], smrm_p1[occ_p1 > occupancy_thresh_1d], 'o',
                             color='green')
                    plt.plot(xvals[occ_p1 > occupancy_thresh_1d], rawrm_p1[occ_p1 > occupancy_thresh_1d], '+',
                             color='green')
                    occ_p2 = p2_binnocc[da_key]
                    smocc_p2 = p2_smoothed_occ[da_key]
                    xvals, rawrm_p2, smrm_p2 = get_1d_ratemap(factor1d[da_key], bins1d[da_key], occ_p2, cellocc_ind_p2)
                    plt.plot(xvals[occ_p2 > occupancy_thresh_1d], smrm_p2[occ_p2 > occupancy_thresh_1d], 'o',
                             color='red')
                    plt.plot(xvals[occ_p2 > occupancy_thresh_1d], rawrm_p2[occ_p2 > occupancy_thresh_1d], '+',
                             color='red')

                    if max_ylim < (axis_list[j].get_ylim())[1]:
                        max_ylim = (axis_list[j].get_ylim())[1]

                    plt.xlabel('%s' % (xaxis1d[da_key]))
                    plt.subplot(plrows, plcols, subplot_index)
                    subplot_index += 1
                    vals = factor1d[da_key]
                    plt.hist(vals[~np.isnan(vals)], 40)

                    # save data
                    onethingy = np.zeros((15, len(xvals)))
                    onethingy[0, :] = xvals
                    onethingy[1, :] = rawrm
                    onethingy[2, :] = occ
                    onethingy[3, :] = smrm
                    onethingy[4, :] = (shuffled_values_1d_means[da_key])[:, cellnum]
                    onethingy[5, :] = (shuffled_values_1d_stds[da_key])[:, cellnum]
                    onethingy[6, :] = smoothed_occ
                    onethingy[7, :] = rawrm_p1
                    onethingy[8, :] = smrm_p1
                    onethingy[9, :] = occ_p1
                    onethingy[10, :] = smocc_p1
                    onethingy[11, :] = rawrm_p2
                    onethingy[12, :] = smrm_p2
                    onethingy[13, :] = occ_p2
                    onethingy[14, :] = smocc_p2
                    ratemaps_1d_data['%s-%s-data' % (cell_names[cellnum], (da_key)[2:])] = onethingy
                    ratemaps_1d_data['%s-%s-ICr' % (cell_names[cellnum], (da_key)[2:])] = ICr
                # set ylim
                for j in range(len(onedguys)):
                    axis_list[j].set_ylim([0, max_ylim])

                good_max_ylim = max_ylim
                # fig title
                tit = "%s : %s : %f\nTop 1D info rates from: %s, %s, %s, %s" % (
                    output_file_prefix, cell_names[cellnum], tempoffsets[i], name1st, name2nd, name3rd, name4th)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.gcf().suptitle(tit, fontsize=40)
                if abs(tempoffsets[i] - 0) < 0.001:
                    plt.savefig('%s_%04d_%s_at_zero.png' % (output_file_prefix, print_cell, cell_names[cellnum]))

            pdf.savefig()
            plt.close()

        if comparing:

            with PdfPages('%s_%04d_%s_comparing.pdf' % (output_file_prefix, print_cell, da_cell_name)) as pdf:
                for i in np.arange(len(tempoffsets)):
                    cellocc_ind = get_cell_occ(da_cell_data, tempoffsets[i], captureframerate, tracking_ts,
                                               [session_ts[0]], [session_ts[1]], full_tracking_ts)
                    cellocc_ind_p1 = get_cell_occ(da_cell_data, tempoffsets[i], captureframerate, tracking_ts,
                                                  startaltbins[1::2], endaltbins[1::2], full_tracking_ts)
                    cellocc_ind_p2 = get_cell_occ(da_cell_data, tempoffsets[i], captureframerate, tracking_ts,
                                                  startaltbins[0::2], endaltbins[0::2], full_tracking_ts)

                    fig = plt.figure(i + 1, figsize=(cf_fig_width, cf_fig_height))
                    plt.clf()
                    subplot_index = 1
                    # plot 2d rate maps
                    if include_generic_2d_plots:
                        print('plotting 2D rate maps for comparing ...')
                        cf_keys2d = list(stuff2d_cf.keys())
                        cf_keys2d.sort()
                        for j in range(len(cf_keys2d)):
                            dacf2dkey = cf_keys2d[j]
                            k2d1 = (stuff2d_cf[dacf2dkey])[0]
                            k2d2 = (stuff2d_cf[dacf2dkey])[1]
                            cf_fac1 = cf_factor1d[k2d1]
                            cf_fac2 = cf_factor1d[k2d2]
                            cf_bins1 = cf_bins1d[k2d1]
                            cf_bins2 = cf_bins1d[k2d2]

                            fac1 = factor1d[k2d1]
                            fac2 = factor1d[k2d2]
                            bins1 = bins1d[k2d1]
                            bins2 = bins1d[k2d2]

                            rawratemap, ratemap, occupancy = get_2d_ratemap(fac1, fac2, bins1, bins2, all_tcount,
                                                                            cellocc_ind,
                                                                            captureframerate, occupancy_thresh,
                                                                            smoothing_par, session_indicator)
                            cf_rawratemap, cf_ratemap, cf_occupancy = get_2d_ratemap(cf_fac1, cf_fac2, cf_bins1, cf_bins2,
                                                                                     all_tcount, cellocc_ind, captureframerate,
                                                                                     occupancy_thresh, smoothing_par, session_indicator)
                            plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                            subplot_index += 1
                            tit = '%s' % (((dacf2dkey)[2:]).replace('_', ' '))
                            plot_ratemaps(rawratemap, occupancy, occupancy_thresh, tit, '')

                            plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                            subplot_index += 1
                            tit = '    Smoothed'
                            # Calculate "spatial information rate" alla Skaggs and McNaughton
                            ICr = calculate_skinfo_rate(ratemap, occupancy)
                            tit = '%s (%f)' % (tit, ICr)
                            plot_ratemaps(ratemap, occupancy, occupancy_thresh, tit, '')

                            plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                            subplot_index += 1
                            tit = 'Cf. %s' % (((dacf2dkey)[2:]).replace('_', ' '))
                            plot_ratemaps(cf_rawratemap, cf_occupancy, occupancy_thresh, tit, '')

                            plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                            subplot_index += 1
                            tit = '    Smoothed'
                            # Calculate "spatial information rate" alla Skaggs and McNaughton
                            ICr = calculate_skinfo_rate(cf_ratemap, cf_occupancy)
                            tit = 'Cf. %s (%f)' % (tit, ICr)
                            plot_ratemaps(cf_ratemap, cf_occupancy, occupancy_thresh, tit, '')

                    # plot sel motion maps
                    if compare_selfmotion:
                        print('plotting self motion maps for comparing ...')
                        for sind in range(len(data['settings']['selfmotion_window_size'])):
                            if len(data['settings']['selfmotion_window_size']) == 1:
                                ixess = ixess.reshape((len(ixess), 1))
                                jyess = jyess.reshape((len(jyess), 1))
                                cf_ixess = cf_ixess.reshape((len(cf_ixess), 1))
                                cf_jyess = cf_jyess.reshape((len(cf_jyess), 1))
                            f_raw_rate_map1, f_rate_map1, f_occupancy1 = get_selfmotion_map(ixess[:, sind],
                                                                                            jyess[:, sind],
                                                                                            all_tcount,
                                                                                            cellocc_ind,
                                                                                            captureframerate,
                                                                                            occupancy_thresh,
                                                                                            smoothing_par,
                                                                                            nbin_max_xval,
                                                                                            nbin_min_yval,
                                                                                            nbin_max_yval,
                                                                                            session_indicator)
                            plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                            subplot_index += 1
                            tit = 'Unsmoothed self motion %d' % data['settings']['selfmotion_window_size'][sind]
                            plot_ratemaps(f_raw_rate_map1, f_occupancy1, occupancy_thresh, tit, '')

                            plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                            subplot_index += 1
                            tit = 'Smoothed'
                            ICr = calculate_skinfo_rate(f_rate_map1, f_occupancy1)
                            tit = '%s (%f)' % (tit, ICr)
                            plot_ratemaps(f_rate_map1, f_occupancy1, occupancy_thresh, tit, '')

                            f_raw_rate_map1, f_rate_map1, f_occupancy1 = get_selfmotion_map(cf_ixess[:, sind],
                                                                                            cf_jyess[:, sind],
                                                                                            all_tcount,
                                                                                            cellocc_ind,
                                                                                            captureframerate,
                                                                                            occupancy_thresh,
                                                                                            smoothing_par,
                                                                                            nbin_max_xval,
                                                                                            nbin_min_yval,
                                                                                            nbin_max_yval,
                                                                                            session_indicator)
                            plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                            subplot_index += 1
                            tit = 'Cf. Unsmoothed self motion %d' % data['settings']['selfmotion_window_size'][sind]
                            plot_ratemaps(f_raw_rate_map1, f_occupancy1, occupancy_thresh, tit, '')

                            plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                            subplot_index += 1
                            tit = 'Cf. Smoothed'
                            ICr = calculate_skinfo_rate(f_rate_map1, f_occupancy1)
                            tit = '%s (%f)' % (tit, ICr)
                            plot_ratemaps(f_rate_map1, f_occupancy1, occupancy_thresh, tit, '')

                    # plot velocity tuning
                    if compare_velocity:
                        print('plotting velocity maps for comparing ...')
                        movement_body = data['possiblecovariates']['C Body_direction'] * math.pi / 180
                        speeds2 = data['possiblecovariates']['B Speeds']
                        cf_movement_body = data['cf_possiblecovariates']['C Body_direction'] * math.pi / 180
                        cf_speeds2 = data['cf_possiblecovariates']['B Speeds']
                        # v_raw_rate_map1, v_rate_map1, v_occupancy1 = get_velocity_tuning(movement_body, speeds1,
                        #                                                                  all_tcount, cellocc_ind, captureframerate,
                        #                                                                  velocity_max_speed, nbin_speed,
                        #                                                                  nbin_angle, occupancy_thresh,
                        #                                                                  smoothing_par)
                        v_raw_rate_map2, v_rate_map2, v_occupancy2 = get_velocity_tuning(movement_body, speeds2,
                                                                                         all_tcount,
                                                                                         cellocc_ind,
                                                                                         captureframerate,
                                                                                         velocity_max_speed,
                                                                                         nbin_speed,
                                                                                         nbin_angle,
                                                                                         occupancy_thresh,
                                                                                         smoothing_par,
                                                                                         session_indicator)
                        plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                        subplot_index += 1
                        tit = 'Velocity, unsmoothed'
                        plot_ratemaps(v_raw_rate_map2, v_occupancy2, occupancy_thresh, tit, '')

                        plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                        subplot_index += 1
                        tit = 'Smoothed'
                        ICr = calculate_skinfo_rate(v_rate_map2, v_occupancy2)
                        tit = '%s (%f)' % (tit, ICr)
                        plot_ratemaps(v_rate_map2, v_occupancy2, occupancy_thresh, tit, '')

                        v_raw_rate_map2, v_rate_map2, v_occupancy2 = get_velocity_tuning(cf_movement_body, cf_speeds2,
                                                                                         all_tcount,
                                                                                         cellocc_ind,
                                                                                         captureframerate,
                                                                                         velocity_max_speed,
                                                                                         nbin_speed,
                                                                                         nbin_angle,
                                                                                         occupancy_thresh,
                                                                                         smoothing_par,
                                                                                         session_indicator)
                        plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                        subplot_index += 1
                        tit = 'Cf. Velocity, unsmoothed'
                        plot_ratemaps(v_raw_rate_map2, v_occupancy2, occupancy_thresh, tit, '')

                        plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                        subplot_index += 1
                        tit = 'Cf. Smoothed'
                        ICr = calculate_skinfo_rate(v_rate_map2, v_occupancy2)
                        tit = '%s (%f)' % (tit, ICr)
                        plot_ratemaps(v_rate_map2, v_occupancy2, occupancy_thresh, tit, '')
                    # plot spatial maps
                    if compare_spatial_maps:
                        print('plotting spatial maps for comparing ...')
                        p_raw_rate_map2, p_rate_map2, p_occupancy2 = get_spatial_tuning(animal_loc, n_frames,
                                                                                        all_tcount,
                                                                                        cellocc_ind,
                                                                                        captureframerate,
                                                                                        nbin_spatial,
                                                                                        occupancy_thresh,
                                                                                        smoothing_par,
                                                                                        session_indicator)
                        plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                        subplot_index += 1
                        tit = 'Space, unsmoothed'
                        plot_ratemaps(p_raw_rate_map2, p_occupancy2, occupancy_thresh, tit, '')

                        plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                        subplot_index += 1
                        tit = 'Smoothed'
                        ICr = calculate_skinfo_rate(p_rate_map2, p_occupancy2)
                        tit = '%s (%f)' % (tit, ICr)
                        plot_ratemaps(p_rate_map2, p_occupancy2, occupancy_thresh, tit, '')

                        p_raw_rate_map2, p_rate_map2, p_occupancy2 = get_spatial_tuning(cf_animal_loc, n_frames,
                                                                                        all_tcount,
                                                                                        cellocc_ind,
                                                                                        captureframerate,
                                                                                        nbin_spatial,
                                                                                        occupancy_thresh,
                                                                                        smoothing_par,
                                                                                        session_indicator)
                        plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                        subplot_index += 1
                        tit = 'Cf. Space, unsmoothed'
                        plot_ratemaps(p_raw_rate_map2, p_occupancy2, occupancy_thresh, tit, '')

                        plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                        subplot_index += 1
                        tit = 'Cf. Smoothed'
                        ICr = calculate_skinfo_rate(p_rate_map2, p_occupancy2)
                        tit = '%s (%f)' % (tit, ICr)
                        plot_ratemaps(p_rate_map2, p_occupancy2, occupancy_thresh, tit, '')
                    # start 1d plots
                    print('plotting 1D rate maps for comparing ...')
                    cf_keys = list(cf_factor1d.keys())
                    cf_keys.sort()

                    var1st = 0.0
                    var2nd = 0.0
                    var3rd = 0.0
                    var4th = 0.0

                    name1st = ''
                    name2nd = ''
                    name3rd = ''
                    name4th = ''

                    # max_ylim = -10
                    axis_list = []
                    count_1d = 0
                    for j in range(len(cf_keys)):
                        da_key = cf_keys[j]
                        if da_key not in comparing_task_1d:
                            continue
                        axis_list.append(plt.subplot(cf_plot_row, cf_plot_column, subplot_index))
                        subplot_index += 1

                        onethingy_from_previous = ratemaps_1d_data['%s-%s-data' % (cell_names[cellnum], (da_key)[2:])]

                        xvals = onethingy_from_previous[0, :]
                        rawrm = onethingy_from_previous[1, :]
                        occ = onethingy_from_previous[2, :]
                        smrm = onethingy_from_previous[3, :]
                        rawrm_p1 = onethingy_from_previous[7, :]
                        smrm_p1 = onethingy_from_previous[8, :]
                        occ_p1 = onethingy_from_previous[9, :]
                        rawrm_p2 = onethingy_from_previous[11, :]
                        smrm_p2 = onethingy_from_previous[12, :]
                        occ_p2 = onethingy_from_previous[13, :]

                        ICr = calculate_skinfo_rate(rawrm, occ)

                        if ICr > var1st:
                            var4th = var3rd
                            name4th = name3rd
                            var3rd = var2nd
                            name3rd = name2nd
                            var2nd = var1st
                            name2nd = name1st
                            var1st = ICr
                            name1st = ((da_key)[2:]).replace('_', ' ')
                        elif ICr > var2nd:
                            var4th = var3rd
                            name4th = name3rd
                            var3rd = var2nd
                            name3rd = name2nd
                            var2nd = ICr
                            name2nd = ((da_key)[2:]).replace('_', ' ')
                        elif ICr > var3rd:
                            var4th = var3rd
                            name4th = name3rd
                            var3rd = ICr
                            name3rd = ((da_key)[2:]).replace('_', ' ')
                        elif ICr > var4th:
                            var4th = ICr
                            name4th = ((da_key)[2:]).replace('_', ' ')

                        tit = ((da_key)[2:]).replace('_', ' ')
                        tit = '%s (%f)' % (tit, ICr)
                        plot_1d_values(xvals, smrm, rawrm, occ, (shuffled_values_1d_means[da_key])[:, cellnum],
                                       (shuffled_values_1d_stds[da_key])[:, cellnum], occupancy_thresh_1d, tit)
                        plt.plot(xvals[occ_p1 > occupancy_thresh_1d], smrm_p1[occ_p1 > occupancy_thresh_1d], 'o',
                                 color='green')
                        plt.plot(xvals[occ_p1 > occupancy_thresh_1d], rawrm_p1[occ_p1 > occupancy_thresh_1d], '+',
                                 color='green')
                        plt.plot(xvals[occ_p2 > occupancy_thresh_1d], smrm_p2[occ_p2 > occupancy_thresh_1d], 'o',
                                 color='red')
                        plt.plot(xvals[occ_p2 > occupancy_thresh_1d], rawrm_p2[occ_p2 > occupancy_thresh_1d], '+',
                                 color='red')

                        # if (max_ylim < (axis_list[count_1d].get_ylim())[1]):
                        #     max_ylim = (axis_list[count_1d].get_ylim())[1]

                        count_1d += 1
                        plt.xlabel('%s' % (xaxis1d[da_key]))
                        plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                        subplot_index += 1
                        vals = factor1d[da_key]
                        plt.hist(vals[~np.isnan(vals)], 40)

                        # start with comparing 1D plot
                        axis_list.append(plt.subplot(cf_plot_row, cf_plot_column, subplot_index))
                        subplot_index += 1
                        occ_cf = cf_all_binnocc[da_key]
                        smoothed_occ_cf = cf_all_smoothed_occ[da_key]
                        xvals_cf, rawrm_cf, smrm_cf = get_1d_ratemap(cf_factor1d[da_key], cf_bins1d[da_key],
                                                                     occ_cf, cellocc_ind)

                        ICr_cf = calculate_skinfo_rate(rawrm_cf, occ_cf)
                        tit = ((da_key)[2:]).replace('_', ' ')
                        tit = 'Cf. %s (%f)' % (tit, ICr_cf)
                        plot_1d_values(xvals_cf, smrm_cf, rawrm_cf, occ_cf,
                                       (cf_shuffled_values_1d_means[da_key])[:, cellnum],
                                       (cf_shuffled_values_1d_stds[da_key])[:, cellnum], occupancy_thresh_1d, tit)
                        # plot_1d_value_shalves(factor1d_comparing[da_key], captureframerate, tracking_ts, bins1d_comparing[da_key],
                        #                       tempoffsets[i], da_cell_data, startaltbins, endaltbins,
                        #                       occupancy_thresh_1d, num_bins_1d)
                        occ_p1_cf = cf_p1_binnocc[da_key]
                        smocc_p1_cf = cf_p1_smoothed_occ[da_key]
                        xvals_cf, rawrm_p1_cf, smrm_p1_cf = get_1d_ratemap(cf_factor1d[da_key], cf_bins1d[da_key],
                                                                           occ_p1_cf, cellocc_ind_p1)
                        plt.plot(xvals_cf[occ_p1_cf > occupancy_thresh_1d],
                                 smrm_p1_cf[occ_p1_cf > occupancy_thresh_1d], 'o', color='green')
                        plt.plot(xvals_cf[occ_p1_cf > occupancy_thresh_1d],
                                 rawrm_p1_cf[occ_p1_cf > occupancy_thresh_1d], '+', color='green')
                        occ_p2_cf = cf_p2_binnocc[da_key]
                        smocc_p2_cf = cf_p2_smoothed_occ[da_key]
                        xvals_cf, rawrm_p2_cf, smrm_p2_cf = get_1d_ratemap(cf_factor1d[da_key], cf_bins1d[da_key],
                                                                           occ_p2_cf, cellocc_ind_p2)
                        plt.plot(xvals_cf[occ_p2_cf > occupancy_thresh_1d],
                                 smrm_p2_cf[occ_p2_cf > occupancy_thresh_1d], 'o', color='red')
                        plt.plot(xvals_cf[occ_p2_cf > occupancy_thresh_1d],
                                 rawrm_p2_cf[occ_p2_cf > occupancy_thresh_1d], '+', color='red')
                        # if (max_ylim < (axis_list[count_1d].get_ylim())[1]):
                        #     max_ylim = (axis_list[count_1d].get_ylim())[1]
                        count_1d += 1

                        plt.xlabel('%s' % (cf_xaxis1d[da_key]))
                        plt.subplot(cf_plot_row, cf_plot_column, subplot_index)
                        subplot_index += 1
                        vals = cf_factor1d[da_key]
                        plt.hist(vals[~np.isnan(vals)], 40)

                        onethingy = np.zeros((15, len(xvals_cf)))
                        onethingy[0, :] = xvals_cf
                        onethingy[1, :] = rawrm_cf
                        onethingy[2, :] = occ_cf
                        onethingy[3, :] = smrm_cf
                        onethingy[4, :] = (cf_shuffled_values_1d_means[da_key])[:, cellnum]
                        onethingy[5, :] = (cf_shuffled_values_1d_stds[da_key])[:, cellnum]
                        onethingy[6, :] = smoothed_occ_cf
                        onethingy[7, :] = rawrm_p1_cf
                        onethingy[8, :] = smrm_p1_cf
                        onethingy[9, :] = occ_p1_cf
                        onethingy[10, :] = smocc_p1_cf
                        onethingy[11, :] = rawrm_p2_cf
                        onethingy[12, :] = smrm_p2_cf
                        onethingy[13, :] = occ_p2_cf
                        onethingy[14, :] = smocc_p2_cf
                        cf_ratemaps_1d_data['%s-%s-data' % (cell_names[cellnum], (da_key)[2:])] = onethingy
                        cf_ratemaps_1d_data['%s-%s-ICr' % (cell_names[cellnum], (da_key)[2:])] = ICr_cf

                    for j in range(count_1d):
                        axis_list[j].set_ylim([0, good_max_ylim])

                    # print('debug')
                    tit = "%s : %s : %f\nTop 1D info rates from: %s, %s, %s, %s" % (
                        output_file_prefix, cell_names[cellnum], tempoffsets[i], name1st, name2nd, name3rd, name4th)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.gcf().suptitle(tit, fontsize=40)
                    if abs(tempoffsets[i] - 0) < 0.001:
                        plt.savefig(
                            '%s_%04d_%s_at_zero_comparing.png' % (output_file_prefix, print_cell, cell_names[cellnum]))

                pdf.savefig()
                plt.close()

    onethingy_name = ['xvals', 'raw_ratemap', 'occ', 'smoothed_ratemap',
                      'shuffled_mean', 'shuffled_std', 'smoothed_occ',
                      'rawrm_p1', 'smrm_p1', 'occ_p1', 'smocc_p1',
                      'rawrm_p2', 'smrm_p2', 'occ_p2', 'smocc_p2']

    if save_1d_ratemaps:
        if isinstance(cell_index, int):
            output_file_prefix = '%s_%04d_%s' % (output_file_prefix, int(cell_index), data['cell_names'][cell_index])
        else:
            output_file_prefix = '%s_%04d' % (output_file_prefix, int(cell_index[0]))

        ratemaps_1d_data['data-names'] = onethingy_name
        scipy.io.savemat('%s_for_recreating_1D_ratemaps.mat' % output_file_prefix, ratemaps_1d_data)
        if comparing:
            cf_ratemaps_1d_data['data-names'] = onethingy_name
            scipy.io.savemat('%s_for_recreating_1D_ratemaps_comparing.mat' % output_file_prefix, cf_ratemaps_1d_data)

    print("\n")
    print("         \\|||||/        ")
    print("         ( O O )         ")
    print("|--ooO-----(_)----------|")
    print("|                       |")
    print("|    Rate Maps Ferdig   |")
    print("|                       |")
    print("|------------------Ooo--|")
    print("         |__||__|        ")
    print("          ||  ||         ")
    print("         ooO  Ooo        ")
