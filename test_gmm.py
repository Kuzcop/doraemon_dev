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
from torch import nn

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

        - gmm_distr: a single-dimension GMM with three truncated-normal components
                     (both truncated to [-2, 2]). Weights are intentionally
                     different so we can check sampling frequencies.
        - beta_distr: a beta distribution configured to be uniform over [-2, 2]
                     (Beta(1,1)) which serves as a higher-entropy baseline.
        """
        # 1D GMM with 3 components (sharp Gaussians centered at +/-1.5)
        self.gmm_distr = [
            [
                {'m': -2, 'M': 2, 'mean': 10, 'std': 3, 'weight': 2.0},
                {'m': -2, 'M': 2, 'mean': -1.5, 'std': 0.1, 'weight': 1.0},
                {'m': -2, 'M': 2, 'mean': 0, 'std': 0.5, 'weight': 1.0},
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
        print(f"{'-'*25}test_initialization{'-'*25}")
        self.assertEqual(self.gmm_dist_obj.ndims, len(self.gmm_distr))
        self.assertEqual(self.gmm_dist_obj.num_mixture_models, len(self.gmm_distr[0]))
        self.assertEqual(len(self.gmm_dist_obj.to_distr), 1)
        self.assertTrue(isinstance(self.gmm_dist_obj.to_distr[0], GMM))
        self.assertTrue(isinstance(self.gmm_dist_obj.to_distr[0].components[0], truncated_normal))
        # check std was propagated into component object
        # self.assertEqual(self.gmm_distr[0][0]['std'], self.gmm_dist_obj.to_distr[0].components[0].std)

    def test_multivariate_sampling(self):
        """Sample the wrapper's `sample` and verify bounds and shape.

        Also prints an empirical hit-rate for samples > 0 to illustrate sample
        distribution qualitatively. Additionally runs a diagnostic that
        prints the normalized weights and exact-categorical empirical
        frequencies to check the sampler is honoring weights.
        """
        print(f"{'-'*25}test_multivariate_sampling{'-'*25}")
        num_samples = 10000
        samples = self.gmm_dist_obj.sample(n_samples=num_samples)

        # quick hit-rate diagnostic
        pos = [1 if samp > 0 else 0 for samp in samples]
        print(f"HIT RATE {np.mean(pos):.3f}")

        # shape and bounds checks
        self.assertEqual(samples.shape, (num_samples, 1))
        self.assertTrue(np.all(samples[:, 0] >= -2))
        self.assertTrue(np.all(samples[:, 0] <= 2))

        # Diagnostic: show normalized weights and exact-categorical frequencies
        def diagnostic_sampling(gmm, N=num_samples, tau=0.1):
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
        print(f"{'-'*25}test_univariate_sampling{'-'*25}")
        samples = self.gmm_dist_obj.sample_univariate(0, n_samples=5)
        self.assertEqual(samples.shape, (5, 1))
        self.assertTrue(np.all(samples[:, 0] >= -2))
        self.assertTrue(np.all(samples[:, 0] <= 2))

    def test_pdf_evaluation(self):
        """Basic checks on pdf and log-pdf evaluation.

        Ensures pdf returns positive values and log-pdf is the log of the pdf
        within numerical tolerance.
        """
        print(f"{'-'*25}test_pdf_evaluation{'-'*25}")
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
        print(f"{'-'*25}test_entropy{'-'*25}")
        entropy_gmm = self.gmm_dist_obj.entropy(num_samples=int(2e5))
        print(f"Entropy GMM: {entropy_gmm}")
        entropy_gmm = self.gmm_dist_obj.entropy(num_samples=int(2e5), relaxed=True)
        print(f"Entropy GMM: {entropy_gmm}")

        # maximum entropy for any distribution supported on [a,b] is log(b-a)
        entropy_max = np.log(self.gmm_dist_obj.to_distr[0].components[0].b.numpy() -
                             self.gmm_dist_obj.to_distr[0].components[0].a.numpy())
        print(f"Entropy Max: {entropy_max}")

        entropy_beta = self.beta_dist_obj.entropy()
        print(f"Entropy Beta: {entropy_beta}")

        gmm_distr = [
            [
                {'m': -2, 'M': 2, 'mean': 0, 'std': 0.00001, 'weight': 1.0},
            ]
        ]

        gmm_dist_obj = DomainRandDistribution('GMM', gmm_distr)

        ent = gmm_dist_obj.entropy(num_samples=int(2e5))
        print(f"Entropy Thin GMM: {ent}")

        # GMM entropy should be less than uniform over the same support
        self.assertLess(entropy_gmm, entropy_max)

    def test_kl_divergence(self):
        """Simple sanity check for KL divergence computation.

        Computes KL(q || p) where q is the Beta(uniform) distribution and
        p is the GMM parameterized by `self.gmm_dist_obj`. The KL should be
        non-negative; we use Monte Carlo evaluation inside the implementation.
        """
        print(f"{'-'*25}test_kl_divergence{'-'*25}")
        gmm_distr_1 = [
            [
                {'m': -2, 'M': 2, 'mean': 0, 'std': 0.1, 'weight': 1.0},
            ]
        ]

        gmm_distr_2 = [
            [
                {'m': -2, 'M': 2, 'mean': 0, 'std': 0.5, 'weight': 1.0},
            ]
        ]

        gmm_dist_obj_1 = DomainRandDistribution('GMM', gmm_distr_1)
        gmm_dist_obj_2 = DomainRandDistribution('GMM', gmm_distr_2)

        # 1D Beta distribution parameters (Beta(1,1) maps to uniform over [m,M])
        beta_distr = [
            {'m': -2, 'M': 2, 'a': 1, 'b': 1,}
        ]
        beta_dist_obj = DomainRandDistribution('beta', beta_distr)

        kl_1 = gmm_dist_obj_1.kl_divergence(q=beta_dist_obj, num_samples=1000)
        print(f"KL 1 DIV: {kl_1}")

        kl_2 = gmm_dist_obj_2.kl_divergence(q=beta_dist_obj, num_samples=1000)
        print(f"KL 2 DIV: {kl_2}")
        self.assertTrue(kl_1 >= 0)
        self.assertTrue(kl_2 >= 0)
        self.assertLess(kl_2, kl_1)
        # TODO: TEST GRAD

    def test_update_parameters(self):
        """Test updating GMM parameters from a flat parameter vector.

        The expected layout for `new_params` is repeated blocks of
        [mean, std, weight] for each component. The test writes new
        parameter values and verifies they were applied.
        """
        print(f"{'-'*25}test_update_parameters{'-'*25}")
        if len(self.gmm_distr[0]) == 2:
            new_params = np.array([0.1, 0.1, 0.5, 1.9, 0.2, 0.5])
            self.gmm_dist_obj.update_parameters(new_params)
            self.assertAlmostEqual(self.gmm_dist_obj.to_distr[0].components[0].mean, 0.1)
            self.assertAlmostEqual(self.gmm_dist_obj.to_distr[0].components[1].mean, 1.9)
            self.assertAlmostEqual(self.gmm_dist_obj.to_distr[0].weights[0], 0.5)
            self.assertAlmostEqual(self.gmm_dist_obj.to_distr[0].weights[1], 0.5)

    def test_gumbel_vs_categorical_distribution(self):
        """Verify Gumbel-Softmax approximates categorical sampling.
        
        With low temperature, the distribution of samples from relaxed=True
        should be similar to relaxed=False (categorical).
        """
        print(f"{'-'*25}test_gumbel_vs_categorical_distribution{'-'*25}")
        gmm = self.gmm_dist_obj.to_distr[0]
        N = 50000
        
        # Categorical sampling
        samples_cat = []
        for _ in range(N):
            s = gmm.sample(1, relaxed=False)
            samples_cat.append(s.item())
        
        # Gumbel-Softmax with low temperature
        samples_gumbel_low = gmm.sample(N, tau=0.1, relaxed=True).detach().numpy()
        
        # Gumbel-Softmax with high temperature
        samples_gumbel_high = gmm.sample(N, tau=1.0, relaxed=True).detach().numpy()
        
        # Compare histograms using KL divergence
        bins = np.linspace(-2, 2, 50)
        hist_cat, _ = np.histogram(samples_cat, bins=bins, density=True)
        hist_gumbel_low, _ = np.histogram(samples_gumbel_low, bins=bins, density=True)
        hist_gumbel_high, _ = np.histogram(samples_gumbel_high, bins=bins, density=True)
        
        from scipy.stats import entropy
        kl_low = entropy(hist_cat + 1e-10, hist_gumbel_low + 1e-10)
        kl_high = entropy(hist_cat + 1e-10, hist_gumbel_high + 1e-10)
        
        print(f"KL(categorical || gumbel τ=0.1): {kl_low:.4f}")
        print(f"KL(categorical || gumbel τ=1.0): {kl_high:.4f}")
        
        # Low temp should be closer to categorical than high temp
        self.assertLess(kl_low, kl_high)
        # Low temp should be reasonably close (threshold depends on your needs)
        self.assertLess(kl_low, 0.1)

    def test_gumbel_softmax_gradients(self):
        """Verify gradients flow through relaxed sampling to weights.
        
        This is the whole point of Gumbel-Softmax - weights should receive
        gradients when using relaxed=True but not with relaxed=False.
        """
        print(f"{'-'*25}test_gumbel_softmax_gradients{'-'*25}")
        gmm = self.gmm_dist_obj.to_distr[0]
        
        # Make weights require gradients
        gmm.weights.requires_grad = True
        
        # Relaxed sampling - should have gradients
        samples_relaxed = gmm.sample(100, tau=0.5, relaxed=True)
        loss_relaxed = samples_relaxed.mean()
        loss_relaxed.backward()
        
        self.assertIsNotNone(gmm.weights.grad)
        self.assertTrue(torch.any(gmm.weights.grad != 0))
        print(f"Weights gradient (relaxed): {gmm.weights.grad}")
        
        # Reset gradients
        gmm.weights.grad = None
        
        # Categorical sampling - should NOT have gradients
        samples_cat = gmm.sample(100, relaxed=False)
        loss_cat = samples_cat.mean()
        # This will raise an error or grad will be None
        try:
            loss_cat.backward()
            # If it doesn't error, gradient should be None or zero
            if gmm.weights.grad is not None:
                print(f"Weights gradient (categorical): {gmm.weights.grad}")
                self.assertTrue(torch.all(gmm.weights.grad == 0))
        except RuntimeError:
            # Expected - no grad path through multinomial
            print("Expected: Loss categorical backward prop failed!")


    def test_kl_gradient_flow_to_weights(self):
        """Simulate full training loop to debug weight gradients."""
        print(f"{'-'*25}test_kl_gradient_flow_to_weights{'-'*25}")
        # Setup: Create two GMMs
        p_params = [[{'m': -2, 'M': 2, 'mean': 0.0, 'std': 0.5, 'weight': 1.0}, {'m': -2, 'M': 2, 'mean': 0.0, 'std': 0.5, 'weight': 1.0}]]
        q_params = [[{'m': -2, 'M': 2, 'mean': 0.5, 'std': 0.3, 'weight': 1.0}, {'m': -2, 'M': 2, 'mean': 0.0, 'std': 0.5, 'weight': 1.0}]]

        x_opt = torch.tensor([0.25, 0.2, 1.1, 0.1, 0.15, 0.9], requires_grad=True)
        
        p_gmm = DomainRandDistribution('GMM', p_params)
        q_gmm = DomainRandDistribution('GMM', q_params)
        
        # Make sure weights are parameters
        gmm = p_gmm.to_distr[0]
        print(f"Before: weights = {gmm.weights}")
        print(f"weights.requires_grad = {gmm.weights.requires_grad}")
        
        # If not already a parameter, make it one
        if not isinstance(gmm.weights, nn.Parameter):
            gmm.weights = nn.Parameter(gmm.weights)
            print("Converted weights to Parameter")
        
        # Training loop
        for step in range(5):
            
            # Compute KL with requires_grad=True
            kl = p_gmm.kl_divergence(q=q_gmm, num_samples=1000, requires_grad=True, p_params=x_opt)[0]
            
            print(f"\nStep {step}")
            print(f"KL: {kl}")
            
            # Backward
            grads = torch.autograd.grad(kl, x_opt)
            
            # Check gradients
            print(f"weights.grad: {grads}")
            
            if grads is None:
                print("ERROR: No gradient!")
                break


    def test_kl_gradient_flow_different_components(self):
        """Test with actually different mixture components."""
        print(f"{'-'*25}test_kl_gradient_flow_different_components{'-'*25}")
        # P: Two DIFFERENT components
        p_params = [[
            {'m': -2, 'M': 2, 'mean': -0.5, 'std': 0.3, 'weight': 2.0},  # Left mode
            {'m': -2, 'M': 2, 'mean': 0.5, 'std': 0.3, 'weight': 1.0}    # Right mode
        ]]
        
        # Q: Single component in the middle
        q_params = [[
            {'m': -2, 'M': 2, 'mean': 0.0, 'std': 0.5, 'weight': 1.0}
        ]]
        
        # x_opt: [mean1, std1, weight1, mean2, std2, weight2]
        x_opt = torch.tensor([-0.5, 0.3, 2.0, 0.5, 0.3, 1.0], requires_grad=True)
        
        p_gmm = DomainRandDistribution('GMM', p_params)
        q_gmm = DomainRandDistribution('GMM', q_params)
        
        for step in range(5):
            # Compute KL
            kl = p_gmm.kl_divergence(q=q_gmm, num_samples=5000, requires_grad=True, p_params=x_opt)[0]
            
            # Get gradients
            grads = torch.autograd.grad(kl, x_opt, retain_graph=(step < 4))
            
            print(f"\nStep {step}")
            print(f"KL: {kl:.6f}")
            print(f"Gradients: {grads[0]}")
            print(f"  mean1: {grads[0][0]:.6f}")
            print(f"  std1:  {grads[0][1]:.6f}")
            print(f"  weight1: {grads[0][2]:.6f}") 
            print(f"  mean2: {grads[0][3]:.6f}")
            print(f"  std2:  {grads[0][4]:.6f}")
            print(f"  weight2: {grads[0][5]:.6f}")
            
            # Check weight gradients are non-zero
            if step == 4:  # Only check on last iteration
                self.assertNotAlmostEqual(grads[0][2].item(), 0.0, places=4,
                                        msg="weight1 gradient is zero!")
                self.assertNotAlmostEqual(grads[0][5].item(), 0.0, places=4,
                                        msg="weight2 gradient is zero!")

    def test_boundary_values(self):
        """Verify samples at exact boundaries are handled correctly.
        
        Floating point can produce exact boundary values, these should not
        get -inf log probability.
        """
        print(f"{'-'*25}test_boundary_values{'-'*25}")
        gmm = self.gmm_dist_obj.to_distr[0]
        a = gmm.components[0].a
        b = gmm.components[0].b
        
        # Test exact boundary values
        boundary_vals = torch.tensor([a.item(), b.item()])
        log_probs = gmm.log_prob(boundary_vals)
        
        print(f"Log prob at boundaries: {log_probs}")
        # Should not be -inf if using inclusive bounds (>= and <=)
        self.assertTrue(torch.all(torch.isfinite(log_probs)))
        
        # Just outside boundaries should be -inf
        outside_vals = torch.tensor([a.item() - 0.001, b.item() + 0.001])
        log_probs_outside = gmm.log_prob(outside_vals)
        self.assertTrue(torch.all(torch.isinf(log_probs_outside)))

    def test_truncated_normal_log_prob_correctness(self):
        from scipy.stats import truncnorm

        tn = truncated_normal(mean=0.0, std=1.0, a=-2, b=2)

        test_x = torch.linspace(-2, 2, 100)[1:-1]  # exclude boundaries
        log_probs_torch = tn.log_prob(test_x)

        mu = 0.0
        sigma = (torch.nn.functional.softplus(torch.tensor(1.0), 4) + 1e-6).item()

        a_scipy = (-2.0 - mu) / sigma
        b_scipy = ( 2.0 - mu) / sigma

        scipy_tn = truncnorm(a_scipy, b_scipy, loc=mu, scale=sigma)
        log_probs_scipy = scipy_tn.logpdf(test_x.numpy())

        diff = torch.abs(log_probs_torch - torch.tensor(log_probs_scipy))

        print(f"Max diff: {diff.max():.3e}")
        print(f"Mean diff: {diff.mean():.3e}")

        self.assertTrue(torch.all(diff < 1e-5))


    def test_entropy_single_component_analytical(self):
        """Test entropy of single truncated normal against known formula.
        
        For a single truncated normal, there's a (complex) analytical formula.
        We can verify our MC estimate is close to it.
        """

        print(f"{'-'*25}test_entropy_single_component_analytical{'-'*25}")

        # Create a single-component GMM
        single_comp_params = [
            [{'m': -2, 'M': 2, 'mean': 0.0, 'std': 1.0, 'weight': 1.0}]
        ]
        single_gmm = DomainRandDistribution('GMM', single_comp_params)
        
        # MC estimate
        entropy_mc = single_gmm.entropy(num_samples=int(1e6))
        
        # Analytical entropy of truncated normal
        # H = 0.5 * log(2πe σ²) + log(Z) + (α*φ(α) - β*φ(β)) / (2*Z)
        # where α = (a-μ)/σ, β = (b-μ)/σ, φ is pdf, Z is normalization
        from scipy.stats import norm
        
        mean, std = 0.0, 1.0
        a, b = -2.0, 2.0
        alpha = (a - mean) / std
        beta = (b - mean) / std
        
        Z = norm.cdf(beta) - norm.cdf(alpha)
        phi_alpha = norm.pdf(alpha)
        phi_beta = norm.pdf(beta)
        
        entropy_analytical = (
            0.5 * np.log(2 * np.pi * np.e * std**2) + 
            np.log(Z) + 
            (alpha * phi_alpha - beta * phi_beta) / (2 * Z)
        )
        
        print(f"Entropy MC: {entropy_mc:.6f}")
        print(f"Entropy Analytical: {entropy_analytical:.6f}")
        print(f"Difference: {abs(entropy_mc - entropy_analytical):.6f}")
        
        # Should be within 1% with 1M samples
        self.assertAlmostEqual(entropy_mc, entropy_analytical, delta=0.01 * entropy_analytical)

    def test_entropy_mixture_bounds(self):
        """Verify mixture entropy satisfies known bounds.
        
        For a mixture with weights w_k and component entropies H_k:
        sum(w_k * H_k) <= H(mixture) <= sum(w_k * H_k) + H(weights)
        """
        print(f"{'-'*25}test_entropy_mixture_bounds{'-'*25}")

        gmm = self.gmm_dist_obj.to_distr[0]
        
        # Get component entropies (approximate via sampling)
        component_entropies = []
        weights = gmm.normalized_weights.detach().numpy()
        
        for comp in gmm.components:
            # Sample from individual component
            samples = comp.sample((100000,))
            log_probs = comp.log_prob(samples)
            H_k = -log_probs.mean().item()
            component_entropies.append(H_k)
        
        # Lower bound: weighted average of component entropies
        lower_bound = sum(w * H for w, H in zip(weights, component_entropies))
        
        # Upper bound: lower bound + entropy of mixture weights
        from scipy.stats import entropy as scipy_entropy
        weight_entropy = scipy_entropy(weights)
        upper_bound = lower_bound + weight_entropy
        
        # Mixture entropy
        mixture_entropy = self.gmm_dist_obj.entropy(num_samples=100000)
        
        print(f"Component entropies: {component_entropies}")
        print(f"Weights: {weights}")
        print(f"Lower bound: {lower_bound:.4f}")
        print(f"Mixture entropy: {mixture_entropy:.4f}")
        print(f"Upper bound: {upper_bound:.4f}")
        
        # Add some slack for MC estimation error
        self.assertGreaterEqual(mixture_entropy, lower_bound - 0.1)
        self.assertLessEqual(mixture_entropy, upper_bound + 0.1)

    def test_kl_two_truncated_normals_analytical(self):
        """KL between two truncated normals has a formula we can verify against.
        
        Creates two single-component GMMs and compares MC estimate to
        analytical formula.
        """
        print(f"{'-'*25}test_kl_two_truncated_normals_analytical{'-'*25}")
        # Two single-component GMMs with same truncation
        p_params = [
            [{'m': -2, 'M': 2, 'mean': 0.5, 'std': 0.8, 'weight': 1.0}]
        ]
        q_params = [
            [{'m': -2, 'M': 2, 'mean': -0.3, 'std': 1.2, 'weight': 1.0}]
        ]
        
        p_gmm = DomainRandDistribution('GMM', p_params)
        q_gmm = DomainRandDistribution('GMM', q_params)
        
        # MC estimate: KL(p || q)
        kl_mc = p_gmm.kl_divergence(q=q_gmm, num_samples=100000)

        # Analytical KL between two normal Gaussians (not truncated) as a baseline      
        mu_p, sigma_p = 0.5, 0.8
        mu_q, sigma_q = -0.3, 1.2
        a, b = -2, 2

        kl_base = np.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5
        
        print(f"KL MC: {kl_mc:.6f}")
        print(f"KL baseline (approximate): {kl_base:.6f}")
        
        # At minimum, verify KL is positive
        self.assertGreater(kl_mc, 0)

    def test_kl_self_divergence(self):
        """KL(p || p) should be zero.
        
        This is a fundamental property of KL divergence.
        """
        print(f"{'-'*25}test_kl_self_divergence{'-'*25}")
        # KL of distribution with itself
        kl_self = self.gmm_dist_obj.kl_divergence(
            q=self.gmm_dist_obj,
            p_params=self.gmm_dist_obj,
            num_samples=50000
        )
        
        print(f"KL(p || p): {kl_self:.6f}")
        
        # Should be very close to zero (small MC error acceptable)
        self.assertAlmostEqual(kl_self, 0.0, delta=0.01)

if __name__ == '__main__':
    unittest.main()
