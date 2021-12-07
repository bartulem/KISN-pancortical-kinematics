
from .toolkits import *
from .family import *
from .metrics import *


def null_solver(family, link, y, beta0, eta, kappa=2, theta=None, learning_rate=2e-1,
                max_iter=1e5, xtol=1e-6, verbose=False):
    if (beta0 is None):
        beta0 = np.random.uniform(0, 0.5, 1)

    xk = np.ones(len(y))
    if (eta is None):
        eta = beta0 * xk
    convergence_list = []
    n_iter_ = 0
    converged = False
    for t in np.arange(0, max_iter, 1):
        n_iter_ += 1
        beta0_old = beta0.copy()
        standard_gk, standard_hk = gradhess_negloglik_1d(family, link, y, xk, eta, kappa, theta, fit_intercept=False)
        update = learning_rate * standard_gk / standard_hk
        beta0 = beta0 - update
        eta = eta + update

        if (np.mod(t, 10) == 0 and verbose):
            mu = latent_mu(family, link, eta, kappa)
            log_lik = loglik(family, y, eta, mu, theta)
            print('\titer {} : loglik {}, params {}'.format(t, log_lik, beta0))

        # Convergence by relative parameter change tolerance
        norm_update = np.linalg.norm(beta0 - beta0_old)
        norm_update /= np.linalg.norm(beta0)
        convergence_list.append(norm_update)
        if t > 1 and convergence_list[-1] < xtol:
            converged = True
            print('\tIntercept update for tolerance. ' + 'Converged in {0:d} iterations'.format(t))
            break
    if n_iter_ == max_iter:
        warnings.warn("Reached max number of iterations without convergence for intercept.")
    mu = latent_mu(family, link, eta, kappa)
    neg_loglik = -loglik(family, y, eta, mu, theta)
    return beta0, neg_loglik, converged


class Solver(object):
    def __init__(self, alpha=1, lambdas=0.0001, learning_rate=2e-1, max_iter=1000,
                 xtol=1e-6, fit_intercept=True, verbose=False, seed=142):
        
        self.family = 'bernoulli'
        self.link = 'logit'
        self.alpha = alpha
        self.ntrials = 1
        self.lambdas = lambdas
        self.tik_tau = None
        self.group = None
        self.beta = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.xtol = xtol
        self.kappa = 2.
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.theta = 1.
        self.seed = seed
        self.y_null_ = None
        self.is_fitted_ = False
        self.is_success_ = False
    
    def __repr__(self):
        """Description of the object."""
        lambdas = self.lambdas
        s = '<\nDistribution | %s' % self.family
        s += '\nalpha | %0.2f' % self.alpha
        s += '\nmax_iter | %0.2f' % self.max_iter
        s += '\nlambda: %0.2f\n>' % lambdas
        return s
    
    def fit_null(self, y):
        beta_null, llnull, intercept_converged = \
            null_solver(self.family, y, self.link, self.kappa, self.theta, self.ntrials,
                        self.learning_rate, self.max_iter, self.xtol, self.verbose)

        return beta_null, llnull, intercept_converged

    def fit(self, y, x_mat):
        n_iter_ = 0
        random_state_ = np.random.RandomState(self.seed)
        
        self.beta = None
        
        n_observations, n_features = x_mat.shape

        if self.fit_intercept:
            beta = 1 / (n_features + 1) * random_state_.normal(0.0, 1.0, (n_features + 1,))
            beta[0] = inv_link(np.mean(y), self.link)
        else:
            beta = 1 / n_features * random_state_.normal(0.0, 1.0, (n_features,))
        
        # define single iteration
        family = self.family
        kappa = self.kappa
        theta = self.theta
        tik_tau = self.tik_tau
        fit_intercept = self.fit_intercept
        alpha = self.alpha
        lambdas = self.lambdas

        def single_iter(val):
            grad_beta = grad_l2loss(family, 'logit', y, x_mat, val, lambdas, alpha, kappa,
                                    theta, tik_tau, fit_intercept)
            grad_beta += lambdas * np.sign(val)
            neg_loglik = l2_loss(family, 'logit', y, x_mat, val, alpha, lambdas, tik_tau, kappa, theta, fit_intercept)
            neg_loglik += lambdas * l1_penalty(val, None)
            return neg_loglik, grad_beta
        
        vals = beta.copy()
        res = minimize(single_iter, vals, method='L-BFGS-B', jac=True,
                       options={'ftol': 1e-10, 'gtol': 1e-6, 'disp': False, 'maxiter': 5000})
        beta = res.x + 0.
        if not res.success:
            warnings.warn("Reached max number of iterations without convergence.")
        else:
            self.is_success_ = True
        self.beta = beta
        self.y_null_ = np.mean(y)
        self.is_fitted_ = True
        return self

    def predict(self, x_mat):
        """Predict targets.
        Parameters
        ----------
        X: array
            Input data for prediction, of shape (n_samples, n_features)
        Returns
        -------
        yhat: array
            The predicted targets of shape (n_samples,)
        """
        # x_mat = check_array(x_mat, accept_sparse=False)
        # check_is_fitted(self, 'is_fitted_')
        
        if not isinstance(x_mat, np.ndarray):
            raise ValueError('Input data should be of type ndarray (got %s).' % type(x_mat))
        
        if not self.is_fitted_:
            raise ValueError('Model must be fit before ' +
                             'prediction can be scored')
        
        if self.is_success_:
            eta = linear_predictor(self.beta, x_mat, self.fit_intercept)
            yhat = latent_mu(self.family, self.link, eta, self.kappa, self.fit_intercept)
        else:
            yhat = np.nan
        
        yhat = np.asarray(yhat)
        return yhat

    def score(self, y, x_mat):
        """Score the model.
        Parameters
        ----------
        X: array
            The input data whose prediction will be scored,
            of shape (n_samples, n_features).
        y: array
            The true targets against which to score the predicted targets,
            of shape (n_samples,).
        Returns
        -------
        score: float
            The score metric
        """
        if not self.is_fitted_:
            raise ValueError('Model must be fit before ' +
                             'prediction can be scored')

        accuracy = np.nan
        if self.is_success_:
            y = np.asarray(y).ravel()

            yhat = self.predict(x_mat)
            deviance = get_deviance(y, yhat, self.family, self.theta)
            pr2 = get_pseudo_R2(y, yhat, self.y_null_, self.family, self.theta)
            
            if self.family in ['binomial', 'multinomial', 'bernoulli']:
                yhat = (yhat > 0.5) * 1
                accuracy = get_accuracy(y, yhat)
        else:
            deviance = np.nan
            pr2 = np.nan
            
        return deviance, pr2, accuracy
