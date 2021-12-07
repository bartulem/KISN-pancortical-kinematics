import scipy
import scipy.io
import sys
import os
import time
import copy
import warnings
import random
import pickle
from copy import deepcopy
from scipy.optimize import minimize
from scipy.stats import wilcoxon

from typing import Any, Optional
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.special import log1p
from scipy.special import expit, loggamma
from scipy.stats import norm




# try:
#     from tqdm.auto import tqdm
#     has_tqdm = True
# except ImportError:
#     print(
#         "tqdm library not found. Falling back to non-interactive progress "
#         "visualization.")
#     has_tqdm = False


def dataframe_like(value, name, optional=False, strict=False):
    """
    Convert to dataframe or raise if not dataframe_like
    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow dataframe. If False, allow types that support
        casting to dataframe.
    Returns
    -------
    converted : dataframe
        value converted to a dataframe
    """
    if optional and value is None:
        return None
    if not isinstance(value, dict) or (
        strict and not (isinstance(value, pd.DataFrame))
    ):
        extra_text = "If not None, " if optional else ""
        strict_text = " or dataframe_like " if strict else ""
        msg = "{0}{1} must be a dict{2}".format(extra_text, name, strict_text)
        raise TypeError(msg)
    return pd.DataFrame(value)



def bool_like(value, name, optional=False, strict=False):
    """
    Convert to bool or raise if not bool_like.
    
    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow bool. If False, allow types that support
        casting to bool.
    Returns
    -------
    converted : bool
        value converted to a bool
    """
    if optional and value is None:
        return value
    extra_text = " or None" if optional else ""
    if strict:
        if isinstance(value, bool):
            return value
        else:
            raise TypeError("{0} must be a bool{1}".format(name, extra_text))

    if hasattr(value, "squeeze") and callable(value.squeeze):
        value = value.squeeze()
    try:
        return bool(value)
    except Exception:
        raise TypeError(
            "{0} must be a bool (or bool-compatible)"
            "{1}".format(name, extra_text)
        )


def int_like(
    value: Any, name: str, optional: bool = False, strict: bool = False
) -> Optional[int]:
    """
    Convert to int or raise if not int_like
    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow int or np.integer that are not bool. If False,
        allow types that support integer division by 1 and conversion to int.
    Returns
    -------
    converted : int
        value converted to a int
    """
    if optional and value is None:
        return None
    is_bool_timedelta = isinstance(value, (bool, np.timedelta64))

    if hasattr(value, "squeeze") and callable(value.squeeze):
        value = value.squeeze()

    if isinstance(value, (int, np.integer)) and not is_bool_timedelta:
        return int(value)
    elif not strict and not is_bool_timedelta:
        try:
            if value == (value // 1):
                return int(value)
        except Exception:
            pass
    extra_text = " or None" if optional else ""
    raise TypeError(
        "{0} must be integer_like (int or np.integer, but not bool"
        " or timedelta64){1}".format(name, extra_text)
    )


def required_int_like(value: Any, name: str, strict: bool = False) -> int:
    """
    Convert to int or raise if not int_like
    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow int or np.integer that are not bool. If False,
        allow types that support integer division by 1 and conversion to int.
    Returns
    -------
    converted : int
        value converted to a int
    """
    _int = int_like(value, name, optional=False, strict=strict)
    assert _int is not None
    return _int



def float_like(value, name, optional=False, strict=False):
    """
    Convert to float or raise if not float_like
    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow int, np.integer, float or np.inexact that are
        not bool or complex. If False, allow complex types with 0 imag part or
        any other type that is float like in the sense that it support
        multiplication by 1.0 and conversion to float.
    Returns
    -------
    converted : float
        value converted to a float
    """
    if optional and value is None:
        return None
    is_bool = isinstance(value, bool)
    is_complex = isinstance(value, (complex, np.complexfloating))
    if hasattr(value, "squeeze") and callable(value.squeeze):
        value = value.squeeze()

    if isinstance(value, (int, np.integer, float, np.inexact)) and not (
        is_bool or is_complex
    ):
        return float(value)
    elif not strict and is_complex:
        imag = np.imag(value)
        if imag == 0:
            return float(np.real(value))
    elif not strict and not is_bool:
        try:
            return float(value / 1.0)
        except Exception:
            pass
    extra_text = " or None" if optional else ""
    raise TypeError(
        "{0} must be float_like (float or np.inexact)"
        "{1}".format(name, extra_text)
    )


def string_like(value, name, optional=False, options=None, lower=True):
    """
    Check if object is string-like and raise if not
    Parameters
    ----------
    value : object
        Value to verify.
    name : str
        Variable name for exceptions.
    optional : bool
        Flag indicating whether None is allowed.
    options : tuple[str]
        Allowed values for input parameter `value`.
    lower : bool
        Convert all case-based characters in `value` into lowercase.
    Returns
    -------
    str
        The validated input
    Raises
    ------
    TypeError
        If the value is not a string or None when optional is True.
    ValueError
        If the input is not in ``options`` when ``options`` is set.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        extra_text = " or None" if optional else ""
        raise TypeError("{0} must be a string{1}".format(name, extra_text))
    if lower:
        value = value.lower()
    if options is not None and value not in options:
        extra_text = "If not None, " if optional else ""
        options_text = "'" + "', '".join(options) + "'"
        msg = "{0}{1} must be one of: {2}".format(
            extra_text, name, options_text
        )
        raise ValueError(msg)
    return value



def list_like(value, name, optional=False, options=None):
    """
    Check if object is list-like and raise if not
    Parameters
    ----------
    value : object
        Value to verify.
    name : str
        Variable name for exceptions.
    optional : bool
        Flag indicating whether None is allowed.
    options : tuple[str]
        Allowed values for input parameter `value`.
    Returns
    -------
    str
        The validated input
    Raises
    ------
    TypeError
        If the value is not a string or None when optional is True.
    ValueError
        If the input is not in ``options`` when ``options`` is set.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        extra_text = " or None" if optional else ""
        raise TypeError("{0} must be a list{1}".format(name, extra_text))
    if options is not None and value not in options:
        extra_text = "If not None, " if optional else ""
        options_text = "'" + "', '".join(options) + "'"
        msg = "{0}{1} must be one of: {2}".format(
            extra_text, name, options_text
        )
        raise ValueError(msg)
    return value



def dict_like(value, name, optional=False, strict=True):
    """
    Check if dict_like (dict, Mapping) or raise if not
    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow dict. If False, allow any Mapping-like object.
    Returns
    -------
    converted : dict_like
        value
    """
    if optional and value is None:
        return None
    if not isinstance(value, Mapping) or (
        strict and not (isinstance(value, dict))
    ):
        extra_text = "If not None, " if optional else ""
        strict_text = " or dict_like (i.e., a Mapping)" if strict else ""
        msg = "{0}{1} must be a dict{2}".format(extra_text, name, strict_text)
        raise TypeError(msg)
    return value


def softplus(z):
    """Numerically stable version of log(1 + exp(z))."""
    # see stabilizing softplus: http://sachinashanbhag.blogspot.com/2014/05/numerically-approximation-of-log-1-expy.html # noqa
    mu = z.copy()
    mu[z > 35] = z[z > 35]
    mu[z < -10] = np.exp(z[z < -10])
    mu[(z >= -10) & (z <= 35)] = log1p(np.exp(z[(z >= -10) & (z <= 35)]))
    return mu


def link_func(x, link='logit'):
    link = string_like(link, 'link', False, ('logit', 'log', 'cloglog', 'probit'))
    if(link == 'logit'):
        y = np.log(x) - np.log(1-x)
    elif(link == 'log'):
        y = np.log(x)
    elif(link == 'cloglog'):
        y = np.log(-np.log(1-x))
    elif(link == 'probit'):
        y = norm.ppf(x, loc=0, scale=1)
    return y


def inv_link(x, link='logit'):
    link = string_like(link, 'link', False, ('logit', 'log', 'cloglog', 'probit'))
    if (link == 'logit'):
        y = expit(x)
    elif (link == 'log'):
        y = np.exp(x)
    elif (link == 'cloglog'):
        y = 1 - np.exp(-np.exp(x))
    elif(link == 'probit'):
        y = norm.cdf(x, loc=0, scale=1)
    return y


def grad_invlink(x, link):
    link = string_like(link, 'link', False, ('logit', 'log', 'cloglog', 'probit'))
    if (link == 'logit'):
        y = expit(x) * (1 - expit(x))
    elif (link == 'log'):
        y = np.exp(x)
    elif (link == 'cloglog'):
        y = np.exp(x - np.exp(x))
    elif (link == 'probit'):
        y = norm.pdf(x, loc=0, scale=1)
    return y


def check_params(distr, max_iter, fit_intercept):
    ALLOWED_DISTRS = ['poisson', 'logistic', 'bernoulli', 'gaussian', 'neg-binomial']

    if distr not in ALLOWED_DISTRS:
        raise ValueError('distr must be one of %s, Got '
                         '%s' % (', '.join(ALLOWED_DISTRS), distr))

    if not isinstance(max_iter, int):
        raise ValueError('max_iter must be of type int')

    if not isinstance(fit_intercept, bool):
        raise ValueError('fit_intercept must be bool, got %s'
                         % type(fit_intercept))
    
    
def check_penalty(penalty, group_index, alpha, gamma):
    ALLOWED_PENALTIES = ['lasso', 'ridge', 'g-lasso', 'elastic-net', 'sg-lasso',
                         'scad', 'mcp', 'g-enet', 'g-scad', 'g-mcp']

    if penalty not in ALLOWED_PENALTIES + ['l1','l2','group-lasso','group-enet','group-scad','group-mcp']:
        raise ValueError('penalty must be one of %s, Got '
                         '%s' % (', '.join(ALLOWED_PENALTIES), penalty))

    if 'g-' in penalty or 'group' in penalty:
        if group_index is None:
            raise ValueError('When using group penalty, group_index must be given.')

    if (gamma <= 1 & penalty in ["g-mcp", "g-mcp"]):
        raise ValueError("gamma must be greater than 1 for the MC penalty")
    if (gamma <= 2 and penalty == "g-scad"):
        raise ValueError("gamma must be greater than 2 for the SCAD penalty")
    if (alpha > 1 or alpha <= 0):
        raise ValueError("alpha must be in (0, 1]")
    
    
def check_solver(solver):
    ALLOWED_SOLVER = ['batch-gradient','cd-fast', 'l-bfgs', 'cd-naive', 'cd-covariance', 'cd-weighted']

    if solver not in ALLOWED_SOLVER + ['gradient-descent', 'gd']:
        raise ValueError('penalty must be one of %s, Got '
                         '%s' % (', '.join(ALLOWED_SOLVER), solver))

    
    
# def verbose_iterable(data):
#     """Wrap an iterable object with tqdm.
#     If tqdm is not available or if we did not set the appropriate
#     log level, then we fall back to the classical method.
#     Parameters
#     ----------
#     data: range, list \n
#         This will be data which will wrapped by tqdm (if available).
#     Returns
#     -------
#     wrapped_data: \n
#         Data object wrapped with tqdm.
#     """
#     wrapped_data = data
#     if has_tqdm:
#         wrapped_data = tqdm(data, leave=False)
#     return wrapped_data



def construct_model(xkeys, exist_keys=None, special_group=None, model_start_index=0):
    """
    Construct model with given information.
    
    Parameters
    ----------
    xkeys: array like strings,
    
    add_keys: list like
    
    exist_keys: list like
    
    Returns
    -------

    """
    model = {}
    if (exist_keys is None):
        xkeys_in = xkeys.copy()
        print('Layer 1 models (contain 1 covariate) are constructing.')
        for i in range(len(xkeys)):
            model_ind = model_start_index + i
            model['m%d' % model_ind] = [xkeys[i]]
    else:
        exist_keys = list_like(exist_keys, 'exist_keys', True)
        layer_ind = len(exist_keys)
        print('Layer %d models are constructing.' % int(layer_ind+1))
        if(special_group is not None):
            exist_special = False
            n_group = len(special_group)
            if(n_group < 2.):
                raise ValueError('special_group has length larger than 1.')
            for i in range(n_group):
                exist_special = (len([da_key for da_key in exist_keys if special_group[i] in da_key]) > 0)
                if(exist_special):
                    special_key = special_group[i]
                    break
            if(exist_special):
                not_special_key = [da_key for da_key in special_group if special_key not in da_key]
                ignore_ind = []
                for i in range(len(not_special_key)):
                    ignore_ind = ignore_ind + [ind for ind in range(len(xkeys)) if not_special_key[i] in xkeys[ind]]
                ind_temp = np.zeros(len(xkeys)) < 1
                ind_temp[ignore_ind] = False
                xkeys_in = xkeys[ind_temp]
            else:
                xkeys_in = xkeys.copy()
        else:
            xkeys_in = xkeys.copy()
            
        delete_ind = [ind for ind in range(len(xkeys_in)) if xkeys_in[ind] in exist_keys]
        x_keys_ind = np.zeros(len(xkeys_in)) < 1.
        x_keys_ind[delete_ind] = False
        xkeys_in = xkeys_in[x_keys_ind]
        for i in range(len(xkeys_in)):
            model_ind = model_start_index + i
            model['m%d' % model_ind] = exist_keys + [xkeys_in[i]]
    return model, xkeys_in



def soft_thresholding_operator(z, l):
    """
    Soft-thresholding operator.

    Parameters
    ----------
    

    Returns
    -------

    """
    if z > l:
      val = z-l
    elif z < -l:
      val = z+l
    else:
      val = 0
    return val



def firm_func(z, l1, l2, gamma):
    """
    Firm-thresholding operator.

    Parameters
    ----------


    Returns
    -------

    """
    if z > 0:
        s = 1
    elif z < 0:
        s = -1
    if (abs(z) <= l1):
        val = 0
    elif (abs(z) <= gamma * l1 * (1 + l2)):
        val = s * (abs(z) - l1) / (1 + l2 - 1/gamma)
    else:
        val = z / (1 + l2)
    return val


def block_stand(x, group):
    """
    Standardised the blocks.

    Parameters
    ----------


    Returns
    -------

    """
    
    n_samples, n_features = x.shape
    xx = x.copy()
    if np.any(group == 0):
        one = np.repeat(1, n_samples)
        not_penalised = np.zeros(n_features) > 1.
        not_penalised[group == 0] = True
        scale_notpen = np.sqrt(np.sum(x[:,not_penalised]**2, 1) / n_samples)
        xx[:,not_penalised] = x[:,not_penalised] / scale_notpen
   
    group_ind = np.unique(group[group != 0])
    scale_pen = np.zeros(len(group_ind))
    scale_pen[:] = np.nan
    for j in range(len(group_ind)):
        ind = np.where(group == group_ind[j])[0]
        if(np.linalg.matrix_rank(x[:, ind]) < len(ind)):
            raise ValueError("Block %d has not full rank! \n" % group_ind[j])
        decomp_q, decomp_r = np.linalg.qr(x[:, ind])
        scale_pen[j] = decomp_r / np.sqrt(n_samples)
        xx[:, ind] = decomp_q * np.sqrt(n_samples)
    
    return xx, scale_pen, scale_notpen


def center_scale(x_mat, standardize):
    """
    Center x_mat, may scale x_mat if standardize == True.
    
    Parameters
    ----------
    x_mat : np.ndarray (n_samples, n_features), the covariates matrix

    standardize : bool like.
    
    Returns
    -------
    x_out: the center(scaled) x_mat
    
    x_transform : the mean of x_mat of each feature

    x_scale : the standard deviation of each feature
    """
    n_samples, n_features = x_mat.shape
    x = np.zeros((n_samples, n_features))
    
    means = np.mean(x_mat, 0)
    for i in range(n_features):
        x[:, i] = x_mat[:, i] - means[i]
    x_transform = means
    x_out = x.copy()
    
    if standardize:
        x_scale = np.std(x, 0)
        cind = np.where(x_scale > 1e-8)[0]
        nleft = len(cind)
        x_out = np.zeros((n_samples, nleft))
        for i in range(nleft):
            x_out[:, i] = x[:, cind[i]] / x_scale[cind[i]]
    else:
        x_scale = 1
    
    return x_out, x_transform, x_scale, cind


def partition_data(y, nfold=10, fold_method='sublock', nrepeat=None):
    """
        partition data for K-Fold Cross-Validation.

        Parameters
        ----------
        nobs : int type.
            Number of observations.

        nfold : int type,
            default is 10. number of folds.

        method : str type,
            could be block, random, sublock.
            sublock, A sub fold of each big fold will be chosen to form a fold.

        n_repeat : int type.
            If not None, the cross-validataion will be re-runned for n_repeat times.
            Each time with a new folding.


    Return
    ----------
        fold_index : list type,
            the index of fold for each observation. e.g. if observations 1,4,5 in fold 2, and observations 2,3,6 in fold 1,
            the returned list will be [2, 1, 1, 2, 2, 1].

    """
    nobs = len(y)
    nfold = int_like(nfold, 'nfold')
    fold_method = string_like(fold_method, 'fold_method')
    nrepeat = int_like(nrepeat, 'nrepeat', True)

    not_zeros_ind = np.where(y != 0)[0]
    fold_index = np.zeros(nobs)
    fold_index[:] = np.nan
    fold_index[not_zeros_ind[0:nfold]] = np.arange(nfold)

    fill_ind = (np.zeros(nobs) < 1.)
    fill_ind[not_zeros_ind[0:nfold]] = False

    left_ind = np.arange(nobs)
    left_ind = left_ind[fill_ind]

    left_nobs = nobs - nfold

    nbasis = int(np.floor(left_nobs / nfold))
    leftover = left_nobs % nfold
    nobs_in_fold = [nbasis + 1 if i <= leftover - 1 else nbasis for i in range(nfold)]

    fold_method = fold_method.lower()
    if (fold_method == 'sublock'):
        n_subfold = nfold * nfold
        nbasis_subfold = int(np.floor(left_nobs / n_subfold))
        leftover_subfold = left_nobs % n_subfold
        nobs_in_subfold = [nbasis_subfold + 1 if i <= leftover_subfold - 1 else nbasis_subfold for i in
                           range(n_subfold)]
    
        fold_index_temp = []
        for i in range(nfold):
            fold_index_temp = fold_index_temp + list(
                np.concatenate([np.repeat(j, nobs_in_subfold[j + i * nfold]) for j in range(nfold)]))
        fold_index[left_ind] = fold_index_temp

    elif (fold_method == 'random'):
        obs_index = np.random.permutation(left_nobs)
        temp_index = list(np.concatenate([np.repeat(j, nobs_in_fold[j]) for j in range(nfold)]))
        index_order = np.argsort(obs_index)
        fold_index_temp = [temp_index[i] for i in index_order]
        fold_index[left_ind] = fold_index_temp
    else:
        fold_index_temp = list(np.concatenate([np.repeat(j, nobs_in_fold[j]) for j in range(nfold)]))
        fold_index[left_ind] = fold_index_temp
        # a = [ind for ind in range(len(fold_index)) if fold_index[ind] == 0]
        # print(len(a))
    # fold_index = np.asarray(fold_index_temp)

    msg = '[Each fold contains spikes: {:s}]'.format(
        ', '.join(['{:}'.format(np.sum(y[fold_index == i]).astype(int)) for i in range(nfold)]))
    print(msg)


    return fold_index


def process_group_info(group_index, x_mat):
    has_unpen = False
    is_lasso = False
    n_samples, n_features = x_mat.shape
    
    # group_index = np.array([0, 0, 0, 1, 1, 1, 1, 3, 2, 2, 3, 3, 3, 4, 4, 4])
    # n_features = len(group_index)
    
    uni_groups = np.unique(group_index)
    sorted_uni_groups = np.sort(uni_groups)
    num_groups = len(uni_groups)
    ix = []
    for i in range(num_groups):
        gr_number = sorted_uni_groups[i]
        ix = ix + list(np.where(group_index == gr_number)[0])
    ix = np.ravel(ix).astype(int)
    sorted_groups = group_index[ix]
    x_mat_sorted = x_mat[:, ix].copy()
    
    if np.any(group_index == 0):
        has_unpen = True
    
    group_range_vec = np.zeros(num_groups + 1)
    for i in range(1, num_groups + 1):
        step_group_ind = len(np.where(sorted_groups == uni_groups[i - 1])[0])
        group_range_vec[i] = group_range_vec[i - 1] + step_group_ind
    
    group_range_vec = group_range_vec.astype(int)
    group_length_vec = np.diff(group_range_vec).astype(int)
    
    if np.all(group_length_vec == 1):
        is_lasso = True
    
    group_info = {'has_unpen': has_unpen,
                  'sorted_groups': sorted_groups,
                  'num_groups': num_groups,
                  'group_range_vec': group_range_vec,
                  'group_length_vec': group_length_vec,
                  'is_lasso': is_lasso,
                  'x_mat_sorted': x_mat_sorted}
    
    return group_info




# def keep_high_quantity_data(x_mat, level=0.9, delete_feature='Ego2_head_azimuth'):
#     feature_keys = np.sort(list(x_mat.keys()))
#
#     select_feature = feature_keys
#     n_feature = len(feature_keys)
#     select_feature_ind = np.zeros(n_feature) < 1.
#     res = np.zeros(n_feature)
#     for i in range(n_feature):
#         good_ind = x_mat[feature_keys[i]][1]
#         good_ind = good_ind > 0.5
#         res[i] = float(np.sum(good_ind)) / float(len(good_ind))
#         print("Good ind for", feature_keys[i], feature_keys[j], "are", np.sum(good_ind), 'out of', len(good_ind),
#               ', which is', res[i])
#     select_feature_ind[res < level] = False
#     select_feature = feature_keys[select_feature_ind]
#
#     n_feature = len(select_feature)
#     select_feature_ind = np.zeros(n_feature) < 1.
#     res = np.zeros((n_feature, n_feature))
#     combo_list = []
#     for i in range(n_feature):
#         good_ind = x_mat[feature_keys[i]][1]
#         for j in range(n_feature):
#             if j <= i:
#                 continue
#             good_ind = good_ind * x_mat[feature_keys[j]][1]
#             good_ind = good_ind > 0.5
#             res[i,j] = float(np.sum(good_ind)) / float(len(good_ind))
#             print("Good ind for", feature_keys[i], feature_keys[j], "are", np.sum(good_ind), 'out of', len(good_ind),
#                   ', which is', res[i,j])
#             if(res[i,j] >= 0.9):
#                 combo_list.append([feature_keys[i],feature_keys[j]])
#
#     combo_string = []
#     for i in range(len(combo_list)):
#         print(combo_list[i])
#         combo_string = combo_string + combo_list[i]
#
#     combo_unique_list = list(set(combo_string))
#     len(combo_unique_list)
#
#
#
#     select_feature_ind[res < level] = False
#
#
#     store_res = []
    