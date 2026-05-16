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


class BD3Noise(nn.Module):
    """BD3-LM-style noise interface returning loss scaling and move chance.

    Adapted from the Apache-2.0 kuleshov-group/bd3lms noise schedule API.  The
    official implementation uses ``t`` directly as the mask/move probability
    for log-linear BD3-LM training; SMART-Diffusion keeps that interface for
    block training while preserving ``LogLinearNoise`` above for legacy full
    diffusion and sampling code.
    """

    def forward(self, t: torch.Tensor):
        return self.compute_loss_scaling_and_move_chance(t)

    def compute_loss_scaling_and_move_chance(self, t: torch.Tensor):
        raise NotImplementedError


class BD3LogLinearNoise(BD3Noise):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def total_noise(self, t: torch.Tensor):
        return -torch.log1p(-(1.0 - self.eps) * t)

    def rate_noise(self, t: torch.Tensor):
        return (1.0 - self.eps) / (1.0 - (1.0 - self.eps) * t)

    def compute_loss_scaling_and_move_chance(self, t: torch.Tensor):
        t = t.clamp_min(self.eps)
        loss_scaling = -1.0 / t
        move_chance = t
        return loss_scaling, move_chance


class BD3ExpNoise(BD3Noise):
    def __init__(self, exp: float, eps: float = 1e-3):
        super().__init__()
        self.exp = exp
        self.eps = eps

    def compute_loss_scaling_and_move_chance(self, t: torch.Tensor):
        t = t.clamp_min(self.eps)
        move_chance = torch.pow(t, self.exp).clamp_min(self.eps)
        loss_scaling = -(self.exp * torch.pow(t, self.exp - 1.0)) / move_chance
        return loss_scaling, move_chance


class BD3LogarithmicNoise(BD3Noise):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def compute_loss_scaling_and_move_chance(self, t: torch.Tensor):
        t = t.clamp_min(self.eps)
        log2 = torch.log(torch.tensor(2.0, device=t.device, dtype=t.dtype))
        move_chance = torch.log1p(t) / log2
        loss_scaling = -1.0 / (move_chance.clamp_min(self.eps) * log2 * (1.0 + t))
        return loss_scaling, move_chance


class BD3CosineNoise(BD3Noise):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def compute_loss_scaling_and_move_chance(self, t: torch.Tensor):
        cos = -(1.0 - self.eps) * torch.cos(t * torch.pi / 2.0)
        sin = -(1.0 - self.eps) * torch.sin(t * torch.pi / 2.0)
        move_chance = cos + 1.0
        loss_scaling = sin / (move_chance + self.eps) * torch.pi / 2.0
        return loss_scaling, move_chance


def get_bd3_noise(noise_type: str = 'loglinear', eps: float = 1e-3) -> BD3Noise:
    noise_type = str(noise_type).lower()
    if noise_type == 'loglinear':
        return BD3LogLinearNoise(eps=eps)
    if noise_type == 'square':
        return BD3ExpNoise(exp=2.0, eps=eps)
    if noise_type == 'square_root':
        return BD3ExpNoise(exp=0.5, eps=eps)
    if noise_type == 'log':
        return BD3LogarithmicNoise(eps=eps)
    if noise_type == 'cosine':
        return BD3CosineNoise(eps=eps)
    raise ValueError(f'{noise_type} is not a valid BD3 noise schedule.')
