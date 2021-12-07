from .toolkits import *
from .family import *
from .metrics import *
from .solver import *


class CrossValidation(object):
    def __init__(self, family='bernoulli', kappa=2., theta=1., tik_tau=None, group=None,
                 nfold=10, fold_method='sublock', n_repeat=None, alpha=0.5, reg_lambda=None,
                 solver='L-BFGS', learning_rate=2e-1, max_iter=1000, xtol=1e-6,
                 fit_intercept=True, random_state=0, verbose=False):
        
        self.family = family
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.n_fold = nfold
        self.fold_index = None
        self.fold_method = fold_method
        self.n_repeat = n_repeat
        self.tik_tau = tik_tau
        self.group = group
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.beta0_ = None
        self.beta_ = None
        self.reg_lambda_opt_ = None
        self.glm_ = None
        self.scores_ = None
        self.ynull_ = None
        self.tol = xtol
        self.kappa = kappa
        self.theta = theta
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
        self.is_success_ = False


    def __repr__(self):
        """Description of the object."""
        reg_lambda = self.reg_lambda
        s = '<\nDistribution | %s' % self.family
        s += '\nalpha | %0.2f' % self.alpha
        s += '\nmax_iter | %0.2f' % self.max_iter
        if len(reg_lambda) > 1:
            s += ('\nlambda: %0.2f to %0.2f\n>'
                  % (reg_lambda[0], reg_lambda[-1]))
        else:
            s += '\nlambda: %0.2f\n>' % reg_lambda[0]
        return s
        

    def partition_data(self, y):
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
        nobs = int_like(nobs, 'nobs')
        # nfold = int_like(nfold, 'nfold')
        # method = string_like(method, 'method')
        # n_repeat = int_like(n_repeat, 'n_repeat')
        
        not_zeros_ind = np.where(y != 0)[0]
        fold_index = np.zeros(nobs)
        fold_index[:] = np.nan
        fold_index[not_zeros_ind[0:self.n_fold]] = np.arange(self.n_fold)

        fill_ind = (np.zeros(nobs) < 1.)
        fill_ind[not_zeros_ind[0:self.n_fold]] = False

        left_ind = np.arange(nobs)
        left_ind = left_ind[fill_ind]
        
        left_nobs = nobs - 10

        nbasis = int(np.floor(left_nobs / 10))
        leftover = left_nobs % 10
        nobs_in_fold = [nbasis + 1 if i <= leftover - 1 else nbasis for i in range(10)]

        fold_method = self.fold_method.lower()
        if (fold_method == 'sublock'):
            n_subfold = 10 * 10
            nbasis_subfold = int(np.floor(left_nobs / n_subfold))
            leftover_subfold = left_nobs % n_subfold
            nobs_in_subfold = [nbasis_subfold + 1 if i <= leftover_subfold - 1 else nbasis_subfold for i in
                               range(n_subfold)]
    
            fold_index_temp = []
            for i in range(10):
                fold_index_temp = fold_index_temp + list(
                    np.concatenate([np.repeat(j, nobs_in_subfold[j + i * 10]) for j in range(10)]))
            fold_index[left_ind] = fold_index_temp

        elif (fold_method == 'random'):
            obs_index = np.random.permutation(left_nobs)
            temp_index = list(np.concatenate([np.repeat(j, nobs_in_fold[j]) for j in range(self.n_fold)]))
            index_order = np.argsort(obs_index)
            fold_index_temp = [temp_index[i] for i in index_order]
            fold_index[left_ind] = fold_index_temp
        else:
            fold_index_temp = list(np.concatenate([np.repeat(j, nobs_in_fold[j]) for j in range(10)]))
            fold_index[left_ind] = fold_index_temp
            # a = [ind for ind in range(len(fold_index)) if fold_index[ind] == 0]
            # print(len(a))
        # fold_index = np.asarray(fold_index_temp)
        
        msg = '[Each fold contains spikes: {:s}]'.format(', '.join(['{:}'.format(np.sum(y[fold_index == i]).astype(int)) for i in range(10)]))
        print(msg)
            
        self.fold_index = fold_index
        
        return fold_index

    

    def cv(self, y, x):
        """
            Performs cross validation for a particular model.

            Parameters
            ----------
            y : array type.
                spike train for a single cell.

            x : dict like or data frame like,
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
        # x = dataframe_like(x, 'x')
        n_samples, n_features = x.shape
    
        if (n_samples != nobs):
            raise ValueError('')
    
        # fold_index = list_like(self.fold_index, 'fold_index') # should be array like
        
        if (self.fold_index is None):
            fold_index = self.partition_data(y)
            print(self.fold_index)
            fold_index = np.ravel(fold_index)
        else:
            fold_index = np.ravel(self.fold_index)

        glm = Solver()

    
        pred_vals = np.zeros((self.n_fold, n_features))
        pr2 = np.zeros(self.n_fold)  # pseudo_R2
        norm_ll_ratio = np.zeros(self.n_fold)
        
        params = np.zeros((self.n_fold, n_features + int(self.fit_intercept)))
        scores = np.zeros((self.n_fold, 4))
        scores_per_spikes = np.zeros((self.n_fold, 2))
        fitted = np.zeros((n_samples))
        fitted[:] = np.nan

        not_success = np.zeros(self.n_fold) > 1
        for i in range(self.n_fold):
            test_inds = np.zeros(nobs) > 1
            test_inds[fold_index == i] = True

            x_temp = x[~test_inds, :] + 0.
        
            # choose not all zeros covariates and store the index
            x_pred_temp = np.zeros(n_features)
            x_pred_temp[:] = np.nan
            for j in range(n_features):
                if (np.std(x_temp[:, j]) > 0.):
                    mean_temp = np.mean(x_temp[:, j])
                    std_temp = np.std(x_temp[:, j])
                    x[:, j] = (x[:, j] - mean_temp) / std_temp
                    x_pred_temp[j] = (1 - mean_temp) / std_temp

            y_fit = y[~test_inds] + 0.
            x_fit_temp = x[~test_inds, :] + 0.
            
            y_test = y[test_inds] + 0.
            x_test_temp = x[test_inds, :] + 0.
            
            
            # if (sum(S_test) < 1):
            #     print('No spikes in this fold!!!!')
            #     return np.nan, np.nan, np.nan, P, h

            fit_index = np.where(np.std(x_fit_temp, 0) > 0.)[0]
            if self.fit_intercept:
                param_index = np.append(0, fit_index + 1.).astype(int)
            else:
                param_index = fit_index
            x_fit = x_fit_temp[:, fit_index]
            x_test = x_test_temp[:, fit_index]
            x_pred = np.diag(x_pred_temp[fit_index])
        
            if (self.family == 'bernoulli'):
                y_fit = (y_fit > 0.5) * 1.  # binarize it!
                y_test = (y_test > 0.5) * 1.  # binarize it!
        
            if self.verbose:
                msg = '\tFold {} has {} spikes out of {} obs to fit, and {} spikes of {} obs to test'.format(i, np.sum(y_fit), len(y_fit), np.sum(y_test), len(y_test))
                print(msg)
            
            # beta_null, llnull, intercept_converged = glm.fit_null(y_fit)
            fit_res = glm.fit(y_fit, x_fit)
            
            y_hat = fit_res.predict(x_test)
            if fit_res.is_success_:
                loglikr = get_llr(y_test, y_hat, fit_res.y_null_, self.family, self.theta)
            else:
                not_success[i] = True
                loglikr = np.nan
            if (np.sum(y_test) < 1):
                print('No spikes in this fold!!!!')
                llr_per_spike = np.nan
            else:
                llr_per_spike = loglikr / np.sum(y_test)
            # llr_per_spike = get_llr(y_test, y_hat, np.mean(y_fit), 'bernoulli', 1) / np.sum(y_test)
            deviance, pr2, accuracy = fit_res.score(y_test, x_test)

            params[i, param_index] = fit_res.beta

            pred_vals[i, fit_index] = fit_res.predict(x_pred)
            
            fitted[test_inds] = y_hat

            scores[i] = np.array([deviance, pr2, loglikr, accuracy])
            
            scores_per_spikes[i] = np.array([np.sum(y_test), llr_per_spike])
            
        return params, pred_vals, fitted, scores, scores_per_spikes, not_success

