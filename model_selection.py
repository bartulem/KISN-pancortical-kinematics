"""
Performs model selection.
@author: JingyiGF
"""

from .family import *
from .cross_validation import *


class ForwardSelection(object):
    def __init__(self, family='bernoulli', models=None, kappa=2., theta=1., tik_tau=None, group=None,
                 nfold=10, fold_method='sublock', n_repeat=None, alpha=1, reg_lambda=0.0001,
                 solver='L-BFGS', learning_rate=2e-1, max_iter=1000, xtol=1e-6,
                 significance=0.01, fit_intercept=True, score_metric='llr',
                 seed=142, verbose=False, save_file=True):

        self.family = family
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.nfold = nfold
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
        self.model = None
        self.best_model = None
        self.xtol = xtol
        self.kappa = kappa
        self.theta = theta
        self.significance = significance
        self.score_metric = score_metric
        self.fit_intercept = fit_intercept
        self.seed = seed
        self.verbose = verbose
        self.save_file = save_file
        self.models = models

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

    def fit(self, data, cell_index, models=None, avoid_feature=None, special_group=None):
        """
        Performs a forward model selection procedure for a single cell.

        Parameters
        ----------
        data : the mat file from function prepare_data4glms.

        """
        models = dict_like(models, 'model', True)

        y = data['spk_mat'][cell_index].copy()
        x_mat = data['features_mat'].copy()
        # check if good_ind included ???
        feature_keys = np.sort(list(x_mat.keys()))
        if avoid_feature is not None:
            keep_ind = [ind for ind in range(len(feature_keys)) if feature_keys[ind] not in avoid_feature]
            feature_keys = feature_keys[keep_ind]

        output_dict = {}
        output_dict['cellnames'] = data['cell_names'][cell_index].copy()

        cv_obj = CrossValidation(self.family, self.kappa, self.theta, self.tik_tau, self.group,
                                 self.nfold, self.fold_method, self.n_repeat,
                                 self.alpha, self.reg_lambda,
                                 self.solver, self.learning_rate, self.max_iter, self.xtol,
                                 self.fit_intercept, self.seed, self.verbose)
        # cv_obj = CrossValidation()

        if (models is None):
            save_name = 'full'
            # model = None
            good_ind = x_mat[feature_keys[0]][1]
            for i in range(1, len(feature_keys), 1):
                good_ind = good_ind * x_mat[feature_keys[i]][1]
            good_ind = good_ind > 0.5
            print("Good ind are", np.sum(good_ind), 'out of', len(good_ind), ', which is',
                  float(np.sum(good_ind)) / float(len(good_ind)))

            y_filtered = y[good_ind]

            best_model = 'null_model'
            best_model_pvalue = 0
            possible_keys = feature_keys.copy()
            exist_keys = None
            model_end_index = 0
            best_scores_goodind = (np.zeros(10) < 1.)
            best_test_scores = np.zeros(10)
            best_score = 0.

            while best_model_pvalue < 0.01:
                model_start_index = model_end_index + 0.
                model, possible_keys = construct_model(possible_keys, exist_keys, special_group, model_start_index)
                # da_model = 'm11'
                n_models = len(model)
                if n_models == 0:
                    print('reach all the covariates !!! ')
                    break
                avg_score = np.zeros(n_models)
                test_stats = np.zeros(n_models)
                test_pvalue = np.zeros(n_models)

                model_ind = 0
                mkeys = list(model.keys())
                for da_model in mkeys:
                    # print(da_model)
                    xkeys = model[da_model]
                    msg = '\tProcessing for model {}: '.format(da_model) + ','.join(xkeys)
                    print(msg)
                    nx = len(xkeys)
                    x = x_mat[xkeys[0]][0]
                    for i in range(1, nx, 1):
                        x = np.concatenate((x, x_mat[xkeys[i]][0]), axis=1)

                    x_filtered = x[good_ind, :]
                    params, pred_vals, fitted, scores, scores_per_spikes, not_success = cv_obj.cv(y_filtered, x_filtered)

                    # check convergence
                    if np.any(not_success):
                        print('cell index: {}, model: {}, not converge for fold: {}'.format(cell_index, da_model, np.where(not_success)[0]))
                        break

                    test_scores = scores_per_spikes[:, 1] + 0.
                    test_scores_goodind = (scores_per_spikes[:, 0] != 0)
                    test_goind = test_scores_goodind * best_scores_goodind

                    # check test scores
                    if np.all(np.isnan(test_scores)):
                        print('cell index: {}, model: {}, scores all nan.'.format(cell_index, da_model))
                        break

                    if np.any(np.isnan(test_scores)):
                        print('cell index: {}, model: {}, scores all nan for fold: '
                              '{}'.format(cell_index, da_model, np.where(np.isnan(test_scores))[0]))

                    # do test
                    test_stats[model_ind], test_pvalue[model_ind] = \
                        wilcoxon(test_scores[test_goind], best_test_scores[test_goind], alternative='greater')

                    # calculate mean scores
                    avg_score[model_ind] = np.nanmean(test_scores)

                    # save result
                    output_dict['%s-keys' % da_model] = xkeys
                    output_dict['%s-params' % da_model] = params + 0.
                    output_dict['%s-predvals' % da_model] = pred_vals + 0.
                    output_dict['%s-scores' % da_model] = scores + 0.
                    output_dict['%s-scores_per_spikes' % da_model] = scores_per_spikes + 0.

                    model_ind += 1
                model_end_index += model_ind + 0.

                sig_bool = test_pvalue < 0.01
                if np.any(sig_bool):
                    sig_models = np.where(sig_bool)[0]
                    best_model_score = np.nanmax(avg_score[sig_models])
                    if best_model_score > best_score:
                        best_model_index_temp = np.where(avg_score[sig_models] == best_model_score)[0]
                        if len(best_model_index_temp) > 1.:
                            warnings.warn("At least 2 models have the same score.")
                        best_model_index = sig_models[best_model_index_temp[0]]
                        best_model = mkeys[best_model_index]
                        best_model_covariates = output_dict['%s-keys' % best_model]
                        exist_keys = best_model_covariates
                        best_model_pvalue = test_pvalue[best_model_index] + 0.
                        best_test_scores = output_dict['%s-scores_per_spikes' % best_model][:, 1] + 0.
                        best_scores_goodind = (output_dict['%s-scores_per_spikes' % best_model][:, 0] != 0)
                        best_score = best_model_score
                    else:
                        break
                else:
                    break

            output_dict['best-model'] = best_model

        else:
            save_name = 'pre'
            n_models = len(models)
            model_keys = list(models.keys())

            test_stats = np.zeros(n_models)
            test_pvalue = np.zeros(n_models)
            avg_score = np.zeros(n_models)
            test_scores = np.zeros((n_models, 10))

            best_model = 'null model'
            # best_test_scores_goodind = (np.zeros(self.nfold) < 1.)
            best_test_scores = np.zeros(10)
            # best_test_success = np.zeros(self.nfold) < 1.

            good_ind = x_mat[feature_keys[0]][1]
            for i in range(1, len(feature_keys), 1):
                good_ind = good_ind * x_mat[feature_keys[i]][1]
            good_ind = good_ind > 0.5
            print("Good ind are", np.sum(good_ind), 'out of', len(good_ind), ', which is',
                  float(np.sum(good_ind)) / float(len(good_ind)))

            y_filtered = y[good_ind]

            model_ind = 0
            for da_model in model_keys:
                xkeys = models[da_model]
                msg = '\tProcessing for model {}: '.format(da_model) + ','.join(xkeys)
                print(msg)
                nx = len(xkeys)
                x = x_mat[xkeys[0]][0]
                for i in range(1, nx, 1):
                    x = np.concatenate((x, x_mat[xkeys[i]][0]), axis=1)

                x_filtered = x[good_ind, :]

                params, pred_vals, fitted, scores, scores_per_spikes, not_success = cv_obj.cv(y_filtered, x_filtered)
                # print(scores)
                # print(not_success)
                # print(scores_per_spikes)
                # print(params)
                # print(pred_vals)
                # print(fitted)
                # check convergence
                if np.any(not_success):
                    print('cell index: {}, model: {}, not converge for fold: {}'.format(cell_index, da_model,
                                                                                        np.where(not_success)[0]))
                    break

                test_scores = scores_per_spikes[:, 1] + 0.
                test_scores_goodind = (scores_per_spikes[:, 0] != 0)
                test_scores_success = ~np.isnan(test_scores)
                test_goind = test_scores_goodind * test_scores_success

                if np.all(np.isnan(test_scores)):
                    test_stats[model_ind] = np.nan
                    test_pvalue[model_ind] = np.nan
                    avg_score[model_ind] = np.nan
                else:
                    test_stats[model_ind], test_pvalue[model_ind] = \
                        wilcoxon(test_scores[test_goind], best_test_scores[test_goind], alternative='greater')

                # check test scores
                if np.any(np.isnan(test_scores)):
                    print('cell index: {}, model: {}, not converge for fold: '
                          '{}'.format(cell_index, da_model, np.where(np.isnan(test_scores))[0]))
                    break

                avg_score[model_ind] = np.nanmean(test_scores)

                output_dict['%s-keys' % da_model] = xkeys
                output_dict['%s-params' % da_model] = params + 0.
                output_dict['%s-predvals' % da_model] = pred_vals + 0.
                output_dict['%s-scores' % da_model] = scores + 0.
                output_dict['%s-scores_per_spikes' % da_model] = scores_per_spikes + 0.
                model_ind += 1

                if np.any(test_pvalue < self.significance):
                    best_model_score = np.nanmax(avg_score)
                    best_model_index = np.where(avg_score == best_model_score)[0]
                    if len(best_model_index) > 1.:
                        warnings.warn("At least 2 models have the same score.")
                    best_model = list(models.keys())[best_model_index[0]]

            output_dict['best-model'] = best_model

        if self.save_file:
            file_name = data['file_name']
            cell_name = data['cell_names'][cell_index]
            scipy.io.savemat('glmres_%s_%04d_%s_%s.mat' % (file_name, cell_index, cell_name, save_name), output_dict)
            print("\n")
            print("         \\|||||/        ")
            print("         ( O O )         ")
            print("|--ooO-----(_)----------|")
            print("|                       |")
            print("|        Ferdig         |")
            print("|                       |")
            print("|------------------Ooo--|")
            print("         |__||__|        ")
            print("          ||  ||         ")
            print("         ooO  Ooo        ")
        else:
            return output_dict, best_model
