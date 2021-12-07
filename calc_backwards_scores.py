from .toolkits import *
from .family import *
from .metrics import *
from .solver import Solver
import scipy.io


def calc_xpr(y, yhat, nt):
    tpath = np.linspace(0, 1, nt)
    tp = np.zeros(nt)
    tn = np.zeros(nt)
    fp = np.zeros(nt)
    fn = np.zeros(nt)
    for i in range(nt):
        yt = (yhat >= tpath[i]) * 1
        # pval = np.where(yt == 1)[0]
        # nval = np.where(yt == 0)[0]
        tp[i] = np.sum(y[yt == 1])
        fn[i] = np.sum(y[yt == 0])
        fp[i] = len(np.where(y[yt == 1] == 0)[0])
        tn[i] = len(np.where(y[yt == 0] == 0)[0])
    
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    return tpr, fpr
    



def calc_backwards_scores(family, data, cell_index, final_model):
    # cell_index = 14
    cell_name = data['cell_names'][cell_index]
    y = data['spk_mat'][cell_index].copy()
    x_mat = data['features_mat'].copy()
    n_samples = len(y)
    print('%s_start' % data['cell_names'][cell_index])
    if (family == 'bernoulli'):
        y = (y > 0.5) * 1.  # binarize it!
    
    full_model = final_model[cell_name]
    # full_model = final_model['imec0_cl0017_ch005']
    
    feature_keys = []
    for i in range(len(full_model)):
        feature_keys.append(full_model[i].strip())
    print(feature_keys)
    
    if (np.any(full_model == 'null')):
        return []
    else:
        n_model = len(feature_keys)
        
        # get new good_ind
        x = x_mat[feature_keys[0]][0]
        groups = list(np.zeros(x.shape[1]).astype(int))
        if n_model > 1:
            for j in range(1, len(feature_keys), 1):
                x_par = x_mat[feature_keys[j]][0]
                groups = groups + list(np.ones(x_par.shape[1]).astype(int) * j)
                x = np.concatenate((x, x_par), axis=1)
        
        n_samples, total_features = x.shape
        
        uni_groups = np.unique(groups)
        n_groups = len(uni_groups)
        good_ind = np.zeros(n_samples) < 1
        for i in range(n_groups):
            cind = np.zeros(total_features) < 1
            cind[groups == i] = False
            x_temp = x[:, cind].copy()
            good_ind_temp = np.zeros(n_samples) < 1
            bad_ind = np.ravel([ind for ind in range(n_samples) if np.all(x_temp[ind, :] == 0)])
            if (len(bad_ind) > 0):
                good_ind_temp[bad_ind] = False
            good_ind = good_ind * good_ind_temp
        
        good_ind = good_ind > 0.5
        print("Good ind are", np.sum(good_ind), 'out of', len(good_ind), ', which is',
              float(np.sum(good_ind)) / float(len(good_ind)))
        
        groups = np.ravel(groups)
        
        output_dict = {}
        output_dict['cellnames'] = data['cell_names'][cell_index].copy()
        
        y_in = y[good_ind]
        
        glm = Solver()
        
        # run full model
        x_in_use = x.copy()
        x_filtered = x_in_use[good_ind, :]
        n_samples, n_features = x_filtered.shape
        print(n_features)
        x_temp = x_filtered + 0.
        # choose not all zeros covariates and store the index
        x_pred_temp = np.zeros(n_features)
        x_pred_temp[:] = np.nan
        for j in range(n_features):
            if (np.std(x_temp[:, j]) > 0.):
                mean_temp = np.mean(x_temp[:, j])
                std_temp = np.std(x_temp[:, j])
                x_filtered[:, j] = (x_filtered[:, j] - mean_temp) / std_temp
                x_pred_temp[j] = (1 - mean_temp) / std_temp
        
        fit_index = np.where(np.std(x_filtered, 0) > 0.)[0]
        x_in = x_filtered[:, fit_index] + 0.
        
        fit_res = glm.fit(y_in, x_in)
        eta = linear_predictor(fit_res.beta, x_in, fit_intercept=True)
        mu = latent_mu(family, 'logit', eta, 2, fit_intercept=True)
        yhat = fit_res.predict(x_in)
        log_lik_full = loglik(family, y_in, eta, mu, theta=1.0)
        log_lik_null = loglik(family, y_in, None, fit_res.y_null_, theta=1.0)
        
        dscore = get_deviance(y_in, yhat, family, theta=1.0)
        rscore = get_pseudo_R2(y_in, yhat, fit_res.y_null_, family, theta=1.0)
        lscore = get_llr(y_in, yhat, fit_res.y_null_, family, theta=1.0)
        sqPearson = np.corrcoef(y_in, yhat) ** 2
        sqPearson = sqPearson[0, 1]
        Tjur = np.mean(yhat[y_in == 1]) - np.mean(yhat[y_in == 0])
        ytilde = (yhat > 0.5) * 1
        ascore = get_accuracy(y_in, ytilde)
        
        
        # CoxSnell2 = 1 - np.power(np.exp(log_lik_full - log_lik_null), (2 / n_samples))
        CoxSnell = 1 - np.exp((-2 / n_samples) * (log_lik_full - log_lik_null))
        Negelkerke = CoxSnell / (1 - np.exp((2 * n_samples)**(-1) * log_lik_null))
        adj_r2 = 1 - ((n_samples - 1) / (n_samples - n_features) * (1 - rscore))
        print(ascore)
        
        tpr, fpr = calc_xpr(y_in, yhat, 100)
        auc = get_auc(fpr, tpr)

        output_dict['data'] = y_in
        output_dict['null_loglik'] = log_lik_null
        output_dict['full_model_keys'] = feature_keys
        output_dict['full_model_predict'] = yhat
        output_dict['full_model_auc'] = auc
        output_dict['full_model_loglik'] = log_lik_full
        output_dict['full_model_scores'] = np.array([ascore, dscore, rscore, lscore])
        output_dict['full_model_new_scores'] = np.array([CoxSnell, Negelkerke, sqPearson, adj_r2, Tjur])
        
        if n_groups > 1:
            for i in range(n_groups):
                cind = np.zeros(total_features) < 1
                cind[groups == i] = False
                x_in_use = x[:, cind].copy()
                
                x_filtered = x_in_use[good_ind, :]
                
                n_samples, n_features = x_filtered.shape
                print(n_features)
                x_temp = x_filtered + 0.
                # choose not all zeros covariates and store the index
                x_pred_temp = np.zeros(n_features)
                x_pred_temp[:] = np.nan
                for j in range(n_features):
                    if (np.std(x_temp[:, j]) > 0.):
                        mean_temp = np.mean(x_temp[:, j])
                        std_temp = np.std(x_temp[:, j])
                        x_filtered[:, j] = (x_filtered[:, j] - mean_temp) / std_temp
                        x_pred_temp[j] = (1 - mean_temp) / std_temp
                
                fit_index = np.where(np.std(x_filtered, 0) > 0.)[0]
                x_in = x_filtered[:, fit_index] + 0.
                
                fit_res = glm.fit(y_in, x_in)
                eta = linear_predictor(fit_res.beta, x_in, fit_intercept=True)
                mu = latent_mu(family, 'logit', eta, 2, fit_intercept=True)
                yhat = fit_res.predict(x_in)
                log_lik = loglik(family, y_in, eta, mu, theta=1.0)
                if (np.isnan(log_lik)):
                    print('%s_problem' % data['cell_names'][cell_index])
                    raise ValueError('log_lik for group %d' % i)
                
                dscore = get_deviance(y_in, yhat, family, theta=1.0)
                rscore = get_pseudo_R2(y_in, yhat, fit_res.y_null_, family, theta=1.0)
                lscore = get_llr(y_in, yhat, fit_res.y_null_, family, theta=1.0)
                sqPearson = np.corrcoef(y_in, yhat) ** 2
                sqPearson = sqPearson[0, 1]
                Tjur = np.mean(yhat[y_in == 1]) - np.mean(yhat[y_in == 0])
                ytilde = (yhat > 0.5) * 1
                ascore = get_accuracy(y_in, ytilde)
                
                
                CoxSnell = 1 - np.exp((- 2 / n_samples) * (log_lik_full - log_lik_null))
                Negelkerke = CoxSnell / (1 - np.exp((2 * n_samples) ** (-1) * log_lik_null))
                adj_r2 = 1 - ((n_samples - 1) / (n_samples - n_features) * (1 - rscore))
                print(ascore)

                tpr, fpr = calc_xpr(y_in, yhat, 100)
                auc = get_auc(fpr, tpr)
                
                output_dict['sub_model_%d_delete_key' % i] = [feature_keys[i]]
                output_dict['sub_model_%d_yhat' % i] = yhat
                output_dict['sub_model_%d_auc' % i] = auc
                output_dict['sub_model_%d_loglik' % i] = log_lik
                output_dict['sub_model_%d_scores' % i] = np.array([ascore, dscore, rscore, lscore])
                output_dict['sub_model_new_scores'] = np.array([CoxSnell, Negelkerke, sqPearson, adj_r2, Tjur])

        file_name = data['file_name']
        cell_name = data['cell_names'][cell_index]
        print('%s_finished' % data['cell_names'][cell_index])
        scipy.io.savemat('backwards_glmres_%s_%04d_%s.mat' % (file_name, cell_index, cell_name), output_dict)
        
        return output_dict