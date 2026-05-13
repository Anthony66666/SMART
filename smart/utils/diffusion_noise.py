import torch
import torch.nn as nn


class LogLinearNoise(nn.Module):
    """Log-linear noise schedule for discrete mask diffusion.

    Built such that mask_prob = 1 - exp(-sigma(t)) interpolates
    between ~0 and ~1 as t varies from eps to 1.0.

    From MDLM (Sahoo et al., NeurIPS 2024):
      sigma(t) = -log(1 - (1 - eps) * t)
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, t: torch.Tensor):
        """Compute noise level, mask probability, and rate.

        Args:
            t: (B,) in [eps, 1.0]

        Returns:
            sigma_t:  (B,) noise level
            mask_prob: (B,) P[mask] = 1 - exp(-sigma_t)
            dsigma_t: (B,) rate of change of noise wrt t
        """
        sigma_t = -torch.log1p(-(1.0 - self.eps) * t)
        mask_prob = 1.0 - torch.exp(-sigma_t)
        dsigma_t = (1.0 - self.eps) / (1.0 - (1.0 - self.eps) * t)
        return sigma_t, mask_prob, dsigma_t
