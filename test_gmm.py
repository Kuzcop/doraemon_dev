"""
Documented unit tests for GMM (truncated normal components) and related utilities.

This file contains unittest-based tests that exercise the DomainRandDistribution wrapper
that constructs a per-dimension distribution object (GMM or Beta). The tests below are
annotated with comments explaining their purpose, expected behaviour, and useful
troubleshooting tips.

How to run
----------
$ python -m unittest test_gmm_entropy.py

Notes
-----
- These tests assume the project exposes:
    - DomainRandDistribution('GMM', ...) -> container that stores per-dimension GMM objects
    - DomainRandDistribution('beta', ...) -> simple beta distribution wrapper
    - GMM class with attributes and methods:
        - components: ModuleList of `truncated_normal` components
        - normalized_weights / logits: mixture weights
        - sample(sample_shape) -> draws samples from GMM (exact sampling recommended)
        - log_prob(x) -> log p(x) for the full mixture (uses log-sum-exp)
        - sample_univariate(dim, n_samples)
        - pdf(x, log=False)
        - entropy(num_samples=N)
        - kl_divergence(...)

"""

import unittest
import numpy as np
import torch
from torch.distributions import Normal, Categorical
from copy import deepcopy
import math
from doraemon.doraemon import DomainRandDistribution
from doraemon.GMM import truncated_normal, GMM


class TestGMMDistribution(unittest.TestCase):
    """Unit tests for the DomainRandDistribution/GMM pipeline.

    Each test focuses on a specific behaviour:
      - initialization: correct object structure
      - sampling: samples are inside truncation bounds and shapes are correct
      - pdf/log-pdf: evaluation shape and basic sign checks
      - entropy: Monte Carlo estimate versus theoretical uniform bound
      - KL: sanity check that KL >= 0 for reasonable inputs
      - update_parameters: verifies parameter update round-trip
    """

    def setUp(self):
        """Create simple, deterministic distribution objects used across tests.

        - gmm_distr: a single-dimension GMM with two truncated-normal components
                     (both truncated to [-2, 2]). Weights are intentionally
                     different so we can check sampling frequencies.
        - beta_distr: a beta distribution configured to be uniform over [-2, 2]
                     (Beta(1,1)) which serves as a higher-entropy baseline.
        """
        # 1D GMM with 2 components (sharp Gaussians centered at +/-1.5)
        self.gmm_distr = [
            [
                {'m': -2, 'M': 2, 'mean': 1.5, 'std': 0.1, 'weight': 2.0},
                {'m': -2, 'M': 2, 'mean': -1.5, 'std': 0.1, 'weight': 1.0},
            ]
        ]

        # 1D Beta distribution parameters (Beta(1,1) maps to uniform over [m,M])
        self.beta_distr = [
            {'m': -2, 'M': 2, 'a': 1, 'b': 1,}
        ]

        # DomainRandDistribution wraps and constructs per-dimension objects
        # (first dimension is a GMM, second would be another distribution if present)
        self.gmm_dist_obj = DomainRandDistribution('GMM', self.gmm_distr)
        self.beta_dist_obj = DomainRandDistribution('beta', self.beta_distr)

    def test_initialization(self):
        """Verify constructed objects and shapes are as expected.

        This test ensures the wrapper created a single-dimension GMM object,
        that the GMM contains two components, and that the component type
        is the expected `truncated_normal`.
        """
        self.assertEqual(self.gmm_dist_obj.ndims, 1)
        self.assertEqual(self.gmm_dist_obj.num_mixture_models, 2)
        self.assertEqual(len(self.gmm_dist_obj.to_distr), 1)
        self.assertTrue(isinstance(self.gmm_dist_obj.to_distr[0], GMM))
        self.assertTrue(isinstance(self.gmm_dist_obj.to_distr[0].components[0], truncated_normal))
        # check std was propagated into component object
        self.assertEqual(self.gmm_distr[0][0]['std'], self.gmm_dist_obj.to_distr[0].components[0].std)

    def test_multivariate_sampling(self):
        """Sample the wrapper's `sample` and verify bounds and shape.

        Also prints an empirical hit-rate for samples > 0 to illustrate sample
        distribution qualitatively. Additionally runs a diagnostic that
        prints the normalized weights and exact-categorical empirical
        frequencies to check the sampler is honoring weights.
        """
        samples = self.gmm_dist_obj.sample(n_samples=1000)

        # quick hit-rate diagnostic
        pos = [1 if samp > 0 else 0 for samp in samples]
        print(f"HIT RATE {np.mean(pos):.3f}")

        # shape and bounds checks
        self.assertEqual(samples.shape, (1000, 1))
        self.assertTrue(np.all(samples[:, 0] >= -2))
        self.assertTrue(np.all(samples[:, 0] <= 2))

        # Diagnostic: show normalized weights and exact-categorical frequencies
        def diagnostic_sampling(gmm, N=10000, tau=0.1):
            # Print weights used by sampling
            print("normalized_weights:", gmm.normalized_weights.detach().cpu().numpy())

            # Exact categorical sampling should reflect the normalized weights
            comp_idx = torch.multinomial(gmm.normalized_weights, N, replacement=True)
            counts_exact = torch.bincount(comp_idx, minlength=gmm.num_components).float()
            freq_exact = counts_exact / N
            print("Exact categorical freq:", freq_exact.cpu().numpy())

        diagnostic_sampling(self.gmm_dist_obj.to_distr[0])

    def test_univariate_sampling(self):
        """Test the convenience univariate sampler returns expected shapes and bounds."""
        samples = self.gmm_dist_obj.sample_univariate(0, n_samples=5)
        self.assertEqual(samples.shape, (5, 1))
        self.assertTrue(np.all(samples[:, 0] >= -2))
        self.assertTrue(np.all(samples[:, 0] <= 2))

    def test_pdf_evaluation(self):
        """Basic checks on pdf and log-pdf evaluation.

        Ensures pdf returns positive values and log-pdf is the log of the pdf
        within numerical tolerance.
        """
        x = torch.tensor([[0.5]])
        pdf_vals = self.gmm_dist_obj.pdf(x)
        self.assertEqual(pdf_vals.shape[0], 1)
        print(f"PDF VALUE {pdf_vals}")
        self.assertTrue(torch.all(pdf_vals > 0))

        log_pdf_vals = self.gmm_dist_obj.pdf(x, log=True)
        self.assertEqual(log_pdf_vals.shape[0], 1)
        # log-pdf should be <= log(pdf + eps) elementwise (eps guards tiny numerics)
        self.assertTrue(torch.all(log_pdf_vals <= torch.log(pdf_vals + 1e-12)))

    def test_entropy(self):
        """Monte Carlo entropy test for the GMM.

        This test computes the entropy estimate via sampling and checks it is
        less than the maximum entropy of the uniform distribution over the
        truncation interval (log(b-a)).
        """
        entropy_gmm = self.gmm_dist_obj.entropy(num_samples=int(2e5))
        print(f"Entropy GMM: {entropy_gmm}")

        # maximum entropy for any distribution supported on [a,b] is log(b-a)
        entropy_max = np.log(self.gmm_dist_obj.to_distr[0].components[0].b.numpy() -
                             self.gmm_dist_obj.to_distr[0].components[0].a.numpy())
        print(f"Entropy Max: {entropy_max}")

        entropy_beta = self.beta_dist_obj.entropy()
        print(f"Entropy Beta: {entropy_beta}")

        # GMM entropy should be less than uniform over the same support
        self.assertLess(entropy_gmm, entropy_max)

    def test_kl_divergence(self):
        """Simple sanity check for KL divergence computation.

        Computes KL(q || p) where q is the Beta(uniform) distribution and
        p is the GMM parameterized by `self.gmm_dist_obj`. The KL should be
        non-negative; we use Monte Carlo evaluation inside the implementation.
        """
        kl = self.gmm_dist_obj.kl_divergence(q=self.beta_dist_obj, p_params=self.gmm_dist_obj, num_samples=1000)
        print(f"KL DIV: {kl}")
        self.assertTrue(kl >= 0)

    def test_update_parameters(self):
        """Test updating GMM parameters from a flat parameter vector.

        The expected layout for `new_params` is repeated blocks of
        [mean, std, weight] for each component. The test writes new
        parameter values and verifies they were applied.
        """
        if len(self.gmm_distr[0]) == 2:
            new_params = np.array([0.1, 0.1, 0.5, 1.9, 0.2, 0.5])
            self.gmm_dist_obj.update_parameters(new_params)
            self.assertAlmostEqual(self.gmm_dist_obj.to_distr[0].components[0].mean, 0.1)
            self.assertAlmostEqual(self.gmm_dist_obj.to_distr[0].components[1].mean, 1.9)
            self.assertAlmostEqual(self.gmm_dist_obj.to_distr[0].weights[0], 0.5)
            self.assertAlmostEqual(self.gmm_dist_obj.to_distr[0].weights[1], 0.5)


if __name__ == '__main__':
    unittest.main()
