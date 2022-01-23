"""
Performs various GLM-related computations.
@author: SolVind
"""

from .toolkits import *


def linear_predictor(beta, x_mat, fit_intercept):
    """Compute linear predictor. eta = beta0 + X*beta"""
    if x_mat is None:
        if not fit_intercept:
            raise ValueError('When x_mat is None, we will fit intercept only.')
        else:
            eta = np.ones_like(beta)
            eta[:] = beta[0]
    else:
        n_samples, n_features = x_mat.shape
        if fit_intercept:
            x_mat_new = np.column_stack((np.ones(n_samples), x_mat))
        else:
            x_mat_new = x_mat.copy()
        eta = np.dot(x_mat_new, beta)
    return eta


def latent_mu(family, link, eta, kappa, fit_intercept=True):
    """The non-linearity (inverse link)."""
    if family == 'neg-binomial':
        mu = softplus(eta)
    elif family == 'poisson':
        mu = eta.copy()
        beta0 = (1 - kappa) * np.exp(kappa) if fit_intercept else 0.
        mu[eta > kappa] = eta[eta > kappa] * np.exp(kappa) + beta0
        mu[eta <= kappa] = np.exp(eta[eta <= kappa])
    elif family == 'gaussian':
        mu = eta
    elif family == 'bernoulli':
        mu = inv_link(eta, link=link)
    return mu


def grad_latent(family, link, eta, kappa):
    """Derivative of the non-linearity."""
    if family == 'neg-binomial':
        grad_mu = expit(eta)
    elif family == 'poisson':
        grad_mu = eta.copy()
        grad_mu[eta > kappa] = np.ones_like(eta)[eta > kappa] * np.exp(kappa)
        grad_mu[eta <= kappa] = np.exp(eta[eta <= kappa])
    elif family == 'gaussian':
        grad_mu = np.ones_like(eta)
    elif family == 'bernoulli':
        grad_mu = grad_invlink(eta, link=link)
    return grad_mu


def loglik(family, y, eta, mu, theta=1.0):
    """The log likelihood."""
    family = string_like(family, 'family', options=('poisson', 'gaussian', 'bernoulli', 'neg-binomial'))
    flow_control = False
    if family == 'poisson':
        eps = np.spacing(1)
        log_lik = np.sum(y * np.log(mu + eps) - mu)
    elif family == 'gaussian':
        log_lik = -0.5 * np.sum((y - mu)**2)
    elif family == 'bernoulli':
        if np.any(mu == 1.0):
            flow_control = True
            mu[mu == 1.0] = 0.9999999999
        if np.any(mu == 0.0):
            flow_control = True
            mu[mu == 0.0] = 0.0000000001
        if (eta is not None) and (not flow_control):
            log_lik = np.sum(y * eta - np.log(1. + np.exp(eta)))
        else:
            log_lik = np.sum(y * np.log(mu) + (1 - y) * np.log(1. - mu))
            
    elif family == 'neg-binomial':
        log_lik = np.sum(loggamma(y + theta) - loggamma(theta) - loggamma(y + 1.) +
                        theta * np.log(theta) + y * np.log(mu) - (theta + y) * np.log(mu + theta))
    return log_lik


def gradhess_negloglik_1d(family:str, link, y, xk, eta, kappa, theta, fit_intercept=True):
    """
    Compute gradient (1st derivative) and Hessian (2nd derivative)
    of negative log-likelihood for a single coordinate.

    Parameters
    ----------
    xk: float
        (n_samples)
    y: float
        (n_samples)
    eta: float
        (n_samples)

    Returns
    -------
    gk: gradient, float:
        (n_features + 1)
    hk: float:
        (n_features + 1)
    """
    n_samples = xk.shape[0]
        
    mu = latent_mu(family, link, eta, kappa, fit_intercept)
    
    if family == 'poisson':
        s = expit(eta)
        gk = np.sum((mu[eta <= kappa] - y[eta <= kappa]) * xk[eta <= kappa]) + \
             np.exp(kappa) * np.sum((1 - y[eta > kappa] / mu[eta > kappa]) * xk[eta > kappa])
        hk = np.sum(mu[eta <= kappa] * xk[eta <= kappa] ** 2) + \
             np.exp(kappa) ** 2 * np.sum(y[eta > kappa] / (mu[eta > kappa] ** 2) * (xk[eta > kappa] ** 2))
    
    elif family == 'gaussian':
        gk = np.sum((eta - y) * xk)
        hk = np.sum(xk * xk)
    
    elif family == 'bernoulli':
        gk = np.sum((mu - y) * xk)
        hk = np.sum(mu * (1.0 - mu) * xk * xk)
    
    elif family == 'neg-binomial':
        grad_mu = grad_latent(family, eta, kappa)
        hess_mu = np.exp(-eta) * expit(eta) ** 2
        
        gradient_beta_j = -grad_mu * (y / mu - (y + theta) / (mu + theta))
        partial_beta_0_1 = hess_mu * (y / mu - (y + theta) / (mu + theta))
        partial_beta_0_2 = grad_mu ** 2 * ((y + theta) / (mu + theta) ** 2 - y / mu ** 2)
        partial_beta_0 = -(partial_beta_0_1 + partial_beta_0_2)
        gk = np.dot(gradient_beta_j.T, xk)
        hk = np.dot(partial_beta_0.T, xk ** 2)
    
    standard_gk = 1. / n_samples * gk
    standard_hk = 1. / n_samples * hk
    
    return standard_gk, standard_hk
    

# def quadratic_loglik(family, y, eta, mu0, eta0, ntrials=1, theta=1.0):
#     """The quadratic log-likelihood."""
#     family = string_like(family, 'family', options=('poisson', 'gaussian', 'binomial', 'bernoulli', 'neg-binomial'))
#     if family == 'poisson':
#         wii = mu0
#         zi = eta0 + (y - mu0) / mu0
#     elif family == 'gaussian':
#         wii = 1
#         zi = 1
#     elif family in ['binomial', 'bernoulli']:
#         if (family == 'bernoulli'):
#             ntrials = 1
#         wii = mu0 * (1 - mu0)
#         zi = eta0 + (y - mu0) / (mu0 * (1 - mu0))
#     elif family == 'neg-binomial':
#         wii = 1
#         zi = 1
#     qloglik = - 0.5 * np.sum(wii * (zi - eta)**2)
#     return qloglik



def l1_penalty(beta, group=None):
    """The L1 penalty."""
    # Compute the L1 penalty
    if group is None:
        # Lasso-like penalty
        l1_reg = np.linalg.norm(beta, 1)
    else:
        # Group sparsity case: apply group sparsity operator
        group_ids = np.unique(group)
        l1_reg = 0.0
        for group_id in group_ids:
            if group_id != 0:
                l1_reg += np.linalg.norm(beta[group == group_id], 2)
        l1_reg += np.linalg.norm(beta[group == 0], 1)
    return l1_reg


def grad_l1penalty(beta, group=None):
    """The gradient of L1 penalty."""
    # Compute the L1 penalty
    if group is None:
        # Lasso-like penalty
        grad_l1 = np.sign(beta)
    else:
        # Group sparsity case: apply group sparsity operator
        # group_ids = np.unique(group)
        # l1_reg = 0.0
        # for group_id in group_ids:
        #     if group_id != 0:
        #         l1_reg += np.linalg.norm(beta[group == group_id], 2)
        # l1_reg += np.linalg.norm(beta[group == 0], 1)
        grad_l1 = 1
    return grad_l1
    


def l2_penalty(beta, tik_tau):
    """The L2 penalty."""
    # Compute the L2 penalty
    if tik_tau is None:
        # Ridge=like penalty
        l2_reg = np.linalg.norm(beta, 2) ** 2
    else:
        # Tikhonov penalty
        if (tik_tau.shape[0] != beta.shape[0] or
                tik_tau.shape[1] != beta.shape[0]):
            raise ValueError('Tau should be (n_features x n_features)')
        else:
            l2_reg = np.linalg.norm(np.dot(tik_tau, beta), 2) ** 2
    return l2_reg


def mcp_penalty(beta, gamma):
    """The MCP penalty."""
    # Compute the MCP penalty
    mcp_reg = 1
    return mcp_reg


def scad_penalty(beta, gamma):
    """The MCP penalty."""
    # Compute the MCP penalty
    scad_reg = 1
    return scad_reg


def penalty(beta, style='enet', alpha=0.5, tik_tau=None, group=None):
    """The penalty."""
    penalty_options = ('enet', 'elastic-net', 'l1', 'lasso', 'l2', 'ridge', 'tikhonov', 'mnet', 'snet', 'mcp', 'scad')
    style = string_like(style, 'style', options = penalty_options)
    if style == 'enet':
        # Combine L1 and L2 penalty terms
        pen_wt_lam = 0.5 * (1 - alpha) * l2_penalty(beta, tik_tau) + alpha * l1_penalty(beta, group)
    elif style == 'mnet':
        pen_wt_lam = 0.5 * (1 - alpha) * l2_penalty(beta, tik_tau) + alpha * mcp_penalty(beta, group)
    return pen_wt_lam


def loss_func(family, link, y, x_mat, beta, style, alpha, tik_tau, lambdas, kappa, theta, group, fit_intercept=True):
    """Define the objective function for elastic net."""
    n_samples, n_features = x_mat.shape
    eta = linear_predictor(beta[0], beta[1:], x_mat, fit_intercept)
    y_hat = latent_mu(family, link, eta, kappa, fit_intercept)
    standardized_loglik = 1. / n_samples * loglik(family, y, eta, y_hat, theta)
    if fit_intercept:
        penal_t = penalty(beta[1:], style, alpha, tik_tau, group)
    else:
        penal_t = penalty(beta, style, alpha, tik_tau, group)
    loss_value = - standardized_loglik + lambdas * penal_t  # negative log-likelihood + penalty
    return loss_value


def l2_loss(family, link, y, x_mat, beta, alpha, lambdas, tik_tau, kappa, theta, fit_intercept=True):
    """Define the objective function."""
    n_samples, n_features = x_mat.shape
    eta = linear_predictor(beta, x_mat, fit_intercept)
    mu = latent_mu(family, link, eta, kappa, fit_intercept)
    standard_loglik = 1. / n_samples * loglik(family, y, eta, mu, theta)
    if fit_intercept:
        pen_wt_lam = 0.5 * (1 - alpha) * l2_penalty(beta[1:], tik_tau)
    else:
        pen_wt_lam = 0.5 * (1 - alpha) * l2_penalty(beta, tik_tau)
    obj_func = - standard_loglik + lambdas * pen_wt_lam
    return obj_func


def grad_l2loss(family, link, y, x_mat, beta, lambdas, alpha=0.5, kappa=None, theta=None, tik_tau=None, fit_intercept=True):
    """The gradient."""
    n_samples, n_features = x_mat.shape
    n_samples = np.float(n_samples)

    if tik_tau is None:
        if fit_intercept:
            tik_tau = np.eye(beta[1:].shape[0])
        else:
            tik_tau = np.eye(beta.shape[0])
    InvCov = np.dot(tik_tau.T, tik_tau)

    eta = linear_predictor(beta, x_mat, fit_intercept)
    mu = latent_mu(family, link, eta, kappa, fit_intercept)
    grad_mu = grad_latent(family, link, eta, kappa)

    grad_beta0 = 0.
    if family == 'poisson':
        if fit_intercept:
            grad_beta0 = np.sum(grad_mu) - np.sum(y * grad_mu / mu)
        grad_beta = ((np.dot(grad_mu.T, x_mat) - np.dot((y * grad_mu / mu).T, x_mat)).T)

    elif family == 'gaussian':
        if fit_intercept:
            grad_beta0 = np.sum((mu - y) * grad_mu)
        grad_beta = np.dot((mu - y).T, x_mat * grad_mu[:, None]).T

    elif family == 'bernoulli':
        if fit_intercept:
            grad_beta0 = np.sum(mu - y)
        grad_beta = np.dot((mu - y).T, x_mat).T

    elif family == 'neg-binomial':
        partial_beta_0 = grad_mu * ((theta + y) / (mu + theta) - y / mu)

        if fit_intercept:
            grad_beta0 = np.sum(partial_beta_0)

        grad_beta = np.dot(partial_beta_0.T, x_mat)

    grad_beta0 *= 1. / n_samples
    grad_beta *= 1. / n_samples
    if fit_intercept:
        grad_beta += lambdas * (1 - alpha) * np.dot(InvCov, beta[1:])
        g = np.zeros((n_features + 1, ))
        g[0] = grad_beta0
        g[1:] = grad_beta
    else:
        grad_beta += lambdas * (1 - alpha) * np.dot(InvCov, beta)
        g = grad_beta

    return g


def quadratic_loss(family, y, x_mat, beta, style, alpha, stau, lambdas, kappa, theta, group, fit_intercept=True):
    """Define the objective function for elastic net."""
    n_samples, n_features = x_mat.shape
    eta = linear_predictor(beta, x_mat, fit_intercept)
    y_hat = latent_mu(family, eta, kappa, fit_intercept)
    standardized_loglik = 1. / n_samples * loglik(family, y, eta, y_hat, eta, theta)
    if fit_intercept:
        penal_t = penalty(beta[1:], style, alpha, stau, group)
    else:
        penal_t = penalty(alpha, beta, stau, group)
    loss_value = - standardized_loglik + lambdas * penal_t  # negative log-likelihood + penalty
    return loss_value










    




    
    
    


