import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize


def t(x, tau, mu1, sigma1, mu2, sigma2):
    x = np.asarray(x)
    t_1 = tau / np.sqrt(2 * np.pi * sigma1) * np.exp(-0.5 * (x - mu1) ** 2 / sigma1)
    t_2 = (1 - tau) / np.sqrt(2 * np.pi * sigma2) * np.exp(-0.5 * (x - mu2) ** 2 / sigma2)
    return np.vstack((t_1 / (t_1 + t_2), t_2 / (t_1 + t_2)))


def theta(x, old):
    t_1, t_2 = t(x, *old)
    tau = np.sum(t_1) / (np.sum(t_1 + t_2))
    mu1 = np.sum(t_1 * x) / (tau*np.sum(t_1 + t_2))
    mu2 = np.sum(t_2 * x) / ((1-tau)*np.sum(t_2 + t_1))
    sigma1 = np.sum(t_1 * (x - mu1) ** 2) / (tau*np.sum(t_1 + t_2))
    sigma2 = np.sum(t_2 * (x - mu2) ** 2) / ((1-tau)*np.sum(t_1 + t_2))
    return tau, mu1, sigma1, mu2, sigma2


def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    th = np.array((tau, mu1, sigma1, mu2, sigma2))
    while True:
        th = np.array(theta(x, th))
        if np.linalg.norm(np.array(theta(x, th)) - th)/np.linalg.norm(th) < rtol:
            break
    return th[0], th[1], np.sqrt(th[2]), th[3], np.sqrt(th[4])


def regressLL(x, params):
    N1 = params[0] / np.sqrt(2 * np.pi * params[2]) * np.exp(-0.5 * (x - params[1]) ** 2 / (params[2]))
    N2 = (1 - params[0]) / np.sqrt(2 * np.pi * params[4]) * np.exp(-0.5 * (x - params[3]) ** 2 / (params[4]))
    logLike = -np.sum(np.log(N1+N2))
    return logLike


def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    x = np.asarray(x)
    initParams = [tau, mu1, sigma1, mu2, sigma2]
    results = minimize(fun=lambda z: regressLL(x, z), x0=initParams, method='nelder-mead', tol=rtol)
    return results.x[0], results.x[1], np.sqrt(results.x[2]), results.x[3], np.sqrt(results.x[4])


def e(x, tau1, mu1, sigma1, tau2, mu2, sigma2):
    x = np.asarray(x)
    t_1 = tau1 * multivariate_normal.pdf(x, mean=mu1, cov=sigma1, allow_singular=True)
    t_2 = tau2 * multivariate_normal.pdf(x, mean=mu2, cov=sigma2, allow_singular=True)
    t_3 = np.ones_like(t_2)*(1-tau1-tau2) * uni
    return np.vstack((t_1 / (t_1 + t_2 + t_3), t_2 / (t_1 + t_2 + t_3), t_3/(t_1 + t_2 + t_3)))


def m(x, old):
    t_1, t_2, t_3 = e(x, *old)
    tau1 = np.sum(t_1) / (np.sum(t_1 + t_2 + t_3))
    tau2 = np.sum(t_2) / (np.sum(t_1 + t_2 + t_3))
    mu1 = t_1 @ x / (tau1*np.sum(t_1+t_2 + t_3))
    mu2 = t_2 @ x / (tau2*np.sum(t_1 + t_2 + t_3))
    sigma1 = t_1 @ (x - mu1) ** 2 / (tau1*np.sum(t_1 + t_2 + t_3))
    sigma2 = t_2 @ (x - mu2) ** 2 / (tau2*np.sum(t_1 + t_2 + t_3))
    return tau1, mu1, sigma1, tau2, mu2, sigma2


def em_double_cluster(x, uniform_dens, tau1, mu1, sigma1, tau2, mu2, sigma2, rtol=1e-5):
    global uni
    uni = uniform_dens
    th = np.array((tau1, mu1, sigma1, tau2, mu2, sigma2))
    while True:
        th = np.array(m(x, th))
        if np.linalg.norm(np.array(m(x, th)).any() - th.any())/np.linalg.norm(th.any()) < rtol:
            break
    return th[0], th[1], np.sqrt(th[2]), th[3], th[4], np.sqrt(th[5])
