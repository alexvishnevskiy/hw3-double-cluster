import unittest
import mixfit
import numpy as np

class TestNormalDistribution(unittest.TestCase):

    def test1(self):
        tau, mu1, sigma1, mu2, sigma2 = np.random.random(5)
        n = 10000
        n_n = int(tau * n)
        n_u = n - n_n
        x_n = np.random.normal(mu1, sigma1, n_n)
        x_u = np.random.normal(mu2, sigma2, n_u)
        x = np.concatenate((x_n, x_u))
        m = mixfit.em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2)
        self.assertAlmostEqual(np.linalg.norm(m), np.linalg.norm([tau, mu1, sigma1, mu2, sigma2]), places=1)

    def test2(self):
        tau, mu1, sigma1, mu2, sigma2 = np.random.random(5)
        n = 10000
        n_n = int(tau * n)
        n_u = n - n_n
        x_n = np.random.normal(mu1, sigma1, n_n)
        x_u = np.random.normal(mu2, sigma2, n_u)
        x = np.concatenate((x_n, x_u))
        m = mixfit.max_likelihood(x, tau, mu1, sigma1, mu2, sigma2)
        self.assertAlmostEqual(np.linalg.norm(m), np.linalg.norm([tau, mu1, sigma1, mu2, sigma2]), places=1)

    if __name__ == '__main__':
        unittest.main()
