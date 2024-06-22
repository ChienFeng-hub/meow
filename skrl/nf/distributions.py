import torch
import torch.nn as nn
import numpy as np

class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model
    Parameters do not depend of target variable (as is the case for a VAE encoder)
    """

    def __init__(self):
        super().__init__()

    def forward(self, num_samples=1):
        """Samples from base distribution and calculates log probability

        Args:
          num_samples: Number of samples to draw from the distriubtion

        Returns:
          Samples drawn from the distribution, log probability
        """
        raise NotImplementedError

    def log_prob(self, z):
        """Calculate log probability of batch of samples

        Args:
          z: Batch of random variables to determine log probability for

        Returns:
          log probability for each batch element
        """
        raise NotImplementedError

    def sample(self, num_samples=1, **kwargs):
        """Samples from base distribution

        Args:
          num_samples: Number of samples to draw from the distriubtion

        Returns:
          Samples drawn from the distribution
        """
        z, _ = self.forward(num_samples, **kwargs)
        return z
    
class ConditionalDiagLinearGaussian(BaseDistribution):
    """
    Conditional multivariate Gaussian distribution with diagonal
    covariance matrix, parameters are obtained by a context encoder,
    context meaning the variable to condition on
    """
    def __init__(self, shape, loc=None, log_scale=None, SIGMA_MIN=-5, SIGMA_MAX=-0.3):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          context_encoder: Computes mean and log of the standard deviation
          of the Gaussian, mean is the first half of the last dimension
          of the encoder output, log of the standard deviation the second
          half
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.const = torch.tensor(-0.5 * np.prod(shape) * np.log(2 * np.pi))
        self.loc = loc
        self.log_scale = log_scale
        self.SIGMA_MIN = SIGMA_MIN
        self.SIGMA_MAX = SIGMA_MAX

    def get_mean_std(self, z, context):
        mean = self.loc(context) if self.loc is not None else torch.zeros_like(z)
        sigma = self.log_scale(context) if self.log_scale is not None else torch.zeros_like(z)
        sigma = torch.tanh(sigma)
        sigma = self.SIGMA_MIN + 0.5 * (self.SIGMA_MAX - self.SIGMA_MIN) * (sigma+1)
        sigma = sigma.exp()
        return mean, sigma

    def forward(self, z, context):
        return self.get_mean_std(z, context)
    
    @torch.jit.export
    def log_prob(self, z, context):
        mean, std = self.get_mean_std(z, context)
        log_p = self.const - torch.sum(
            torch.log(std) + 0.5 * torch.pow((z - mean) / std, 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p
    
    @torch.jit.export
    def get_qv(self, z, context):
        mean, std = self.get_mean_std(z, context)
        q = -torch.sum(0.5 * torch.pow((z - mean) / std, 2),
                       list(range(1, self.n_dim + 1)))
        v = self.const - torch.sum(torch.log(std), list(range(1, self.n_dim + 1)))
        return q, -v
    
    @torch.jit.ignore
    def sample(self, num_samples=1, context=None):
        eps = torch.randn((num_samples,) + self.shape, dtype=context.dtype, device=context.device)
        mean, std = self.get_mean_std(eps, context)
        z = mean + std * eps
        log_p = self.const - torch.sum(
            torch.log(std) + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p
