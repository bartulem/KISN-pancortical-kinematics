"""
Prepares data for rate-map analyses.
@author: JingyiGF
"""

from .toolkits import *


def get_data(data, use_imu=False, max_num_spk=4):
    # max_num_spk, THRESHOLD_FOR_MAX_NUMBER_OF_SPIKES_IN_INTERVAL
    session_ts = data['session_ts']
    tracking_ts = data['tracking_ts']
    framerate = data['framerate']
    
    cell_names = data['cell_names']
    cell_activity = data['cell_activities']

    if not use_imu:
        possible_covariates = data['possiblecovariates'].copy()
        bounds = data['possiblecovariatesbounds'].copy()
    else:
        possible_covariates = data['possiblecovariates_imu'].copy()
        bounds = data['possiblecovariatesbounds_imu'].copy()
    
    possible_covariates['selfmotion_x'] = data['dxs'][:, 0].copy()
    possible_covariates['selfmotion_y'] = data['dys'][:, 0].copy()
    XYZofAnimal = data['animal_location'].copy()
    possible_covariates['position_x'] = XYZofAnimal[:, 0]
    possible_covariates['position_y'] = XYZofAnimal[:, 1]
    
    covar_keys = list(possible_covariates.keys())
    covar_keys = sorted(covar_keys)
    for i in range(len(covar_keys)):
        da_key = covar_keys[i]
        values = possible_covariates[da_key].copy()
        print(
            ('For', da_key, 'the percent of non-nan values is',
             100. * float(np.sum(~np.isnan(values))) / float(len(values))))
    
    nf = len(possible_covariates[(list(possible_covariates.keys()))[0]])
    starttime = tracking_ts[0]
    
    spk_mat = np.zeros((len(cell_names), nf))
    for i in range(len(cell_names)):
        cell_ts = (cell_activity[i] - tracking_ts[0])
        print((cell_names[i], 'has', len(cell_ts), 'identified events, with average firing rate',
               len(cell_ts) / (session_ts[1] - session_ts[0]), 'Hertz'))
        for t in cell_ts:
            indy = int(np.floor(t * framerate))  # this is the same as timebin = 1000/framerate = 8.33333333333333 ms
            if indy > nf - 1 or indy < 0:
                print(('Time point %f, of cell %s, is outside of range (0, %f)' % (
                    t, cell_names[i], tracking_ts[1] - tracking_ts[0])))
                continue
            spk_mat[i, indy] += 1
    
    spk_mat[spk_mat > max_num_spk] = max_num_spk
    for i in range(len(cell_names)):
        print(('For', cell_names[i],
               '(estimated firing rate = %f Hertz), there are' % (np.mean(spk_mat[i, :]) * framerate),))
        for j in range(1, 50, 1):
            n = np.sum(spk_mat[i, :] == j)
            if n > 0:
                print((n, 'bins with', j, 'spikes,',))
                if j > 5:
                    vv = np.argwhere(spk_mat[i, :] == j)
                    print('This happened at bin number', vv, 'which corresponds to time',
                          vv / framerate + tracking_ts[0])
                    print('Remember TS is', tracking_ts[0], 'and session info is', session_ts)
        print('')
    
    return possible_covariates, bounds, spk_mat, cell_names, framerate


def shift_spikes_in_time(possible_covariates, spk_mat, toff):
    # nf = len(possible_covariates[list(possible_covariates.keys())[0]])
    nf = len(spk_mat[0, :])
    if toff < 0:
        ii = 0
        jj = nf + toff
        aa = abs(toff)
        bb = nf
    else:
        ii = toff
        jj = nf
        aa = 0
        bb = nf - toff
    
    spk_mat_new = spk_mat[:, ii:jj]
    new_pos_cov = {}
    for k in possible_covariates:
        new_pos_cov[k] = (possible_covariates[k])[aa:bb]
    
    return new_pos_cov, spk_mat_new


def randomize_stuff(possible_covariates, spk_mat):
    good_times = list(range(len(spk_mat[0, :])))
    random.shuffle(good_times)
    new_order = np.argsort(good_times)
    
    new_spk_mat = spk_mat[:, new_order]
    new_pos_cov = {}
    for k in possible_covariates:
        new_pos_cov[k] = (possible_covariates[k])[new_order]
    
    return new_pos_cov, new_spk_mat


def get_bins_with_enough_in_each(values, minval, maxval, nbins):
    nf = len(values)
    bins = np.linspace(minval, maxval, nbins + 1)
    centers = 0.5 * (bins[:(-1)] + bins[1:])
    occupancy = np.zeros(len(centers))
    
    chopped_up_guys = np.zeros((nf, len(centers)))
    for i in range(len(centers)):
        whiches = (values >= bins[i]) * (values < bins[i + 1])
        chopped_up_guys[whiches, i] = 1
        occupancy[i] = np.sum(whiches)
    return centers, occupancy, chopped_up_guys


def bin_covariate_1d(values, ips, nbins, min_occupancy=100):
    minval = ips[0]
    maxval = ips[1]
    if ips[2]:
        if abs(minval) < 0.0001 and abs(maxval) < 0.0001:
            minval = np.nanmin(values)
            maxval = np.nanmax(values)
        for i in range(10000):
            centers, occ, chopped_up_guys = get_bins_with_enough_in_each(values, minval, maxval, nbins)
            goodtogo = True
            if occ[0] < min_occupancy:
                minval = minval + 0.05 * (centers[0] - minval)
                goodtogo = False
            if occ[-1] < min_occupancy:
                maxval = maxval + 0.05 * (centers[-1] - maxval)
                goodtogo = False
            if goodtogo:
                break
            if not goodtogo and i > 9999:
                for j in range(10):
                    raise Exception(('SHIT! Could not find good bounds for the variable!!'))
    
    centers, occ, chopped_up_guys = get_bins_with_enough_in_each(values, minval, maxval, nbins)
    counts = np.sum(chopped_up_guys, 1)
    print((': using min and max of', minval, maxval, 'and the fraction covered =', np.sum(occ) / len(values)))
    
    return centers, occ, chopped_up_guys


def super_convert(smguy, size_axis_bins):
    vals = np.zeros(len(smguy))
    vals[:] = np.nan
    vals[~np.isnan(smguy)] = (np.floor(smguy[~np.isnan(smguy)] / size_axis_bins)).astype(int)
    vals[~np.isnan(smguy)] -= min(vals[~np.isnan(smguy)])
    return vals


def bin_covariate_2d(xvals, yvals, size_axis_bins, min_occupancy=100):
    ixes = super_convert(xvals, size_axis_bins)
    jyes = super_convert(yvals, size_axis_bins)
    
    outputvector = []
    totpos = 0
    for i in np.arange(np.nanmax(ixes)):
        for j in np.arange(np.nanmax(jyes)):
            totpos += 1
            stuff = (ixes == i) * (jyes == j) * 1.
            if np.sum(stuff) > min_occupancy:
                outputvector.append(stuff)
    print("number of good bins is", len(outputvector), 'of a possible', totpos)
    outputvector = np.transpose(np.array(outputvector))
    whichgood = np.sum(outputvector, 1) > 0.5
    print('Shape of output vector is', np.shape(outputvector), 'whichgood', np.sum(whichgood) / float(len(whichgood)))
    return [outputvector, whichgood]


def preprocess_covariates(covariates, bounds, use_bins=True, nbins=15, temporal_offsets=0, min_occupancy_1d=100, min_occupancy_2d=100):
    covar_keys = list(covariates.keys())
    covar_keys.sort()
    
    covar_land = {}
    
    if use_bins:
        for da_key in covar_keys:
            print(('Key', da_key))
            values = covariates[da_key].copy()
            
            if da_key in bounds:
                ips = [bounds[da_key][0], bounds[da_key][1], False, temporal_offsets]
            else:
                ips = [0, 0, True, temporal_offsets]
                bounds[da_key] = ips
            
            centers, occ, chopped_up_guys = bin_covariate_1d(values, ips, nbins, min_occupancy_1d)
            which_good = np.sum(chopped_up_guys, 1) > 0.5
            covar_land[da_key] = [chopped_up_guys, which_good]
        
        print("SELF MOTION")
        covar_land['Z Self_motion'] = bin_covariate_2d(covariates['selfmotion_x'], covariates['selfmotion_y'], 5, min_occupancy_2d)
        
        print("POSITION!")
        covar_land['Z Position'] = bin_covariate_2d(100. * covariates['position_x'], 100. * covariates['position_y'], 10, min_occupancy_2d)
    else:
        print('use spline, not implemented yet, haha')
    
    return covar_land


def prepare_data4glms(data, use_bins=True, nbins=15, temporal_offsets=0, time_shift=0, randomize_time=False,
                      feature_list=None, use_imu=False, min_occupancy_1d=100, min_occupancy_2d=100):
    """
    Prepares the data for running GLMs.

    Parameters
    ----------
    data : the file for generating rate maps.

    use_bins : boolin type,
        default is True, prepare covariates into bins. Use spline if False.

    nbins : integer,
        number of bins.

    time_shift : integer,
        time shift in a bin.
    """
    settings = data['settings'].copy()
    settings['glm_use_bins'] = use_bins
    settings['glm_nbins'] = nbins
    settings['glm_temporal_offsets'] = temporal_offsets
    settings['glm_time_shift'] = time_shift
    settings['glm_randomize_time'] = randomize_time
    
    features, bounds, spk_mat, cell_names, framerate = get_data(data, use_imu=use_imu)
    
    if abs(time_shift) > 0.00001:
        features, spk_mat = shift_spikes_in_time(features, spk_mat, time_shift)
    
    if randomize_time:
        features, spk_mat = randomize_stuff(features, spk_mat)
    
    features_mat = preprocess_covariates(features, bounds, use_bins, nbins, temporal_offsets, min_occupancy_1d, min_occupancy_2d)
    
    data4glms = {'framerate': framerate,
                 'settings': settings,
                 'cell_names': cell_names,
                 'spk_mat': spk_mat,
                 'features_mat': features_mat}
    
    if feature_list is not None:
        # need to check feature_list though
        features_submat = {}
        for da_key in feature_list:
            features_submat[da_key] = features_mat[da_key]
        data4glms['features_mat'] = features_submat
    
    return data4glms
