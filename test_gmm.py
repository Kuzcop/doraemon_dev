import unittest
import numpy as np
import torch
from torch.distributions import Normal, Categorical
from copy import deepcopy
import math
from doraemon.doraemon import DomainRandDistribution
# Import your DomainRandDistribution class here
# from your_module import DomainRandDistribution

class TestGMMDistribution(unittest.TestCase):

    def setUp(self):
        # 1D GMM with 2 components
        self.gmm_distr = [
            [
                {'m': -2, 'M': 2, 'mean': 0.5, 'variance': 0.1, 'weight': 0.7},
                {'m': -2, 'M': 2, 'mean': -1, 'variance': 0.2, 'weight': 0.3}
            ]
        ]
        self.beta_distr = [
            {'m': -2, 'M': 2, 'a': 2, 'b': 2,}
        ]
        self.gmm_dist_obj = DomainRandDistribution('GMM', self.gmm_distr)
        self.beta_dist_obj = DomainRandDistribution('beta', self.beta_distr)

    def test_initialization(self):
        self.assertEqual(self.gmm_dist_obj.ndims, 1)
        self.assertEqual(self.gmm_dist_obj.num_mixture_models, 2)
        self.assertEqual(len(self.gmm_dist_obj.to_distr), 2)
        self.assertTrue(isinstance(self.gmm_dist_obj.to_distr[0], Normal))
        self.assertEqual(self.gmm_distr[0][0]['variance'], self.gmm_dist_obj.distr[0][0]['variance'])
    def test_multivariate_sampling(self):
        samples = self.gmm_dist_obj.sample(n_samples=10)
        self.assertEqual(samples.shape, (10, 1))
        self.assertTrue(np.all(samples[:, 0] >= -2))
        self.assertTrue(np.all(samples[:, 0] <= 2))

    def test_univariate_sampling(self):
        samples = self.gmm_dist_obj.sample_univariate(0, n_samples=5)
        self.assertEqual(samples.shape, (5, 1))
        self.assertTrue(np.all(samples[:, 0] >= -2))
        self.assertTrue(np.all(samples[:, 0] <= 2))

    def test_pdf_evaluation(self):
        x = torch.tensor([[0.5]])
        pdf_vals = self.gmm_dist_obj.pdf(x)
        self.assertEqual(pdf_vals.shape[0], 1)
        self.assertTrue(torch.all(pdf_vals > 0))

        log_pdf_vals = self.gmm_dist_obj.pdf(x, log=True)
        self.assertEqual(log_pdf_vals.shape[0], 1)
        self.assertTrue(torch.all(log_pdf_vals <= torch.log(pdf_vals + 1e-12)))

    def test_entropy(self):
        entropy = self.gmm_dist_obj.entropy(num_samples=100)
        print(entropy)
        self.assertIsInstance(entropy, torch.Tensor)
        self.assertTrue(entropy.item() > 0)

    def test_kl_divergence(self):
        # KL(self || self) should be >= 0
        kl = self.gmm_dist_obj.kl_divergence(self.gmm_dist_obj, num_samples=100)
        print(kl)
        self.assertTrue(kl >= 0)

        kl = self.gmm_dist_obj.kl_divergence(q=self.beta_dist_obj, p_params=self.gmm_dist_obj, num_samples=100)
        print(kl)
        self.assertTrue(kl >= 0)

    def test_update_parameters(self):
        # Arrange new parameters: [mean1, var1, weight1, mean2, var2, weight2]
        new_params = np.array([0.1, 0.1, 0.5, 1.9, 0.2, 0.5])
        self.gmm_dist_obj.update_parameters(new_params)
        self.assertAlmostEqual(self.gmm_dist_obj.distr[0][0]['mean'], 0.1)
        self.assertAlmostEqual(self.gmm_dist_obj.distr[0][1]['mean'], 1.9)
        self.assertAlmostEqual(self.gmm_dist_obj.distr[0][0]['weight'], 0.5)
        self.assertAlmostEqual(self.gmm_dist_obj.distr[0][1]['weight'], 0.5)

if __name__ == '__main__':
    unittest.main()
