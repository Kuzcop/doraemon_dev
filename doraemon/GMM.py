import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Categorical

class truncated_normal(nn.Module):
    """A 1-D (univariate) truncated normal distribution implemented with PyTorch.

    Parameters
    ----------
    mean : float or torch.Tensor
        Location parameter (mu) of the base Normal distribution.
    std : float or torch.Tensor
        Scale parameter (sigma) (must be positive). Internally clamped to eps.
    a, b : float
        Left and right truncation limits (support is [a, b]).
    eps : float
        Small numerical constant used to prevent division by zero.

    Methods
    -------
    sample(sample_shape)
        Return `sample_shape` i.i.d. draws from the truncated normal.
    log_prob(x)
        Compute log p(x) for inputs x (returns -inf outside [a, b]).
    """
    def __init__(self, mean, std, a, b, eps=1e-12):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std  = torch.as_tensor(std,  dtype=torch.float32)
        self.mean = mean
        self.std  = torch.clamp(std, min=eps)
        self.a    = torch.as_tensor(a, dtype=torch.float32, device=self.mean.device)
        self.b    = torch.as_tensor(b, dtype=torch.float32, device=self.mean.device)
        self.eps  = eps
        
        # Base normal distribution (used for cdf/icdf/log_prob)
        self.base = torch.distributions.Normal(self.mean, torch.clamp(self.std, min=self.eps))

    def sample(self, sample_shape):
        """Draw reparameterized samples using inverse-CDF sampling.
        Based on the math from the wikipedia article: 
        https://en.wikipedia.org/wiki/Truncated_normal_distribution#:~:text=Generating%20values%20from%20the%20truncated%20normal%20distribution%5Bedit%5D

        Notes
        -----
        - This uses inverse transform sampling via the Normal.icdf an   d is
          differentiable w.r.t. mean and std for pathwise gradients (if u is
          treated as the randomness).
        """
        u     = torch.rand(sample_shape, dtype=self.mean.dtype, device=self.mean.device)
        CDF_b = self.base.cdf(self.b)
        CDF_a = self.base.cdf(self.a)
        Z     = torch.clamp(CDF_b - CDF_a, min=self.eps)
        u     = CDF_a + u * Z
        return self.base.icdf(u)

    def log_prob(self, x):
        """Log probability of x under the truncated normal.

        Returns -inf for values outside [a,b]. The returned tensor has shape
        (N,) after flattening the input.
        """
        x          = torch.as_tensor(x, dtype=self.mean.dtype, device=self.mean.device).view(-1)
        log_prob_x = self.base.log_prob(x)
        # Normalizing constant for truncation
        CDF_b      = self.base.cdf(self.b)
        CDF_a      = self.base.cdf(self.a)
        Z          = torch.clamp(CDF_b - CDF_a, min=self.eps)
        # Only valid inside the closed interval [a,b]. Use inclusive bounds to
        # avoid dropping exact-boundary values due to floating point error.
        in_support = (x > self.a) & (x < self.b)
        res        = log_prob_x - torch.log(Z)
        # Outside support return -inf (caller may clamp to avoid NaNs)
        return torch.where(in_support, res, torch.full_like(res, -float("inf")))
    
class GMM():
    """A simple 1-D Gaussian Mixture Model composed of truncated normals.

    Weights are stored as non-normalized
    positive values and normalized via softmax when needed. Components are
    instances of `truncated_normal`.
    """
    def __init__(self, num_components, means, stds, a, b, eps=1e-12, weights = None):
        assert num_components == len(means) == len(stds)
        assert a < b
        self.num_components = num_components
        self.weights = torch.as_tensor([1.0 for i in range(num_components)], dtype=torch.float32) if not weights else torch.as_tensor(weights, dtype=torch.float32)
        self.softmax = nn.Softmax(dim=0)

        self.components = nn.ModuleList(
            [truncated_normal(means[i], stds[i], a, b) for i in range(num_components)]
        )

    @property
    def normalized_weights(self):
        """Return mixture probabilities normalized to sum to 1."""
        return self.softmax(self.weights)
    
    def log_prob(self, x):
        """Log probability of x under the full mixture.

        This computes log-sum-exp across components in a numerically stable way:
            log p(x) = log sum_k w_k p_k(x) = logsumexp( log w_k + log p_k(x) ).
        """
        x    = torch.as_tensor(x, dtype=self.weights.dtype, device=self.weights.device).view(-1)
        logw = torch.log(self.normalized_weights + 1e-12)
        terms = []
        for i in range(self.num_components):
            terms.append(logw[i] + self.components[i].log_prob(x))    # (N,)
        stacked = torch.stack(terms, dim=0)                           # (K,N)
        return torch.logsumexp(stacked, dim=0)   
    
    def get_parameters(self):
        """Return a flattened parameter vector [mean, std, weight] for each comp.

        This is a convenience that helps interop with optimizers or black-box
        solvers that expect a flat numpy vector. Returned tensor is float32.
        """
        parameters = torch.zeros(self.num_components * 3, dtype=torch.float32, device=self.weights.device)
        for i in range(self.num_components):
            s = 3 * i
            parameters[s:s+3] = torch.tensor(
                [self.components[i].mean.item(),
                 self.components[i].std.item(),
                 self.normalized_weights[i].item()],
                dtype=torch.float32, device=self.weights.device
            )
        return parameters

    def sample(self, sample_shape, tau=0.1, relaxed=False):
        
        """Draw samples from the mixture.

        Parameters
        ----------
        sample_shape : int or shape-like
            Number of samples to draw (kept simple: int accepted).
        tau : float
            Temperature used for Gumbel-softmax when `relaxed=True`.
        relaxed : bool
            If True, perform a relaxed (differentiable) Gumbel-softmax mixture
            sampling. If False, perform exact categorical sampling per sample.

        Returns
        -------
        samples : Tensor
            Tensor of shape (N,) with sampled values from the mixture.
        """

        # Resolve N and device/dtype
        N = int(sample_shape) if isinstance(sample_shape, int) else int(sample_shape[0])
        device = self.weights.device
        dtype = self.weights.dtype

        if relaxed:
            # Gumbel-softmax sampling
            g = -torch.log(-torch.log(torch.rand(N, self.num_components, device=device, dtype=dtype)))
            A = F.softmax((self.weights.unsqueeze(0) + g) / tau, dim=-1)    # (N,K)
            Xk = torch.stack([self.components[k].sample((N,)) for k in range(self.num_components)], dim=1)  # (N,K)
            x  = (A * Xk).sum(dim=1)                                       # (N,)
            samples = x
        else:
            # Exact categorical sampling: choose component per sample, then draw
            # that many samples from each component and place them into the
            # preallocated tensor.
            weights = self.normalized_weights.detach()  # shape (K,)
            # Draw component indices for each sample
            comp_idx = torch.multinomial(weights, N, replacement=True).to(device)
            samples = torch.empty(N, device=device, dtype=dtype)
            # For each component, sample the number needed and assign
            for k, comp in enumerate(self.components):
                mask = (comp_idx == k)
                n_k = int(mask.sum().item())
                if n_k > 0:
                    # comp.sample expects sample_shape tuple
                    samples_k = comp.sample((n_k,))
                    # make sure samples_k is a 1D tensor of length n_k
                    samples[mask] = samples_k

        return samples