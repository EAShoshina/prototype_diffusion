"""
Forward and reverse diffusion processes (DDPM).
Implements the noise schedule and sampling steps.
"""
import torch
import numpy as np


def make_schedule(beta_0: float, beta_T: float, T: int, device):
    """
    Create a linear noise schedule from beta_0 to beta_T.

    Args:
        beta_0: Starting noise level
        beta_T: Ending noise level
        T: Number of diffusion steps
        device: torch device

    Returns:
        Tuple (betas, alphas, bar_alphas) — all torch.Tensor on device
    """
    betas_np = np.linspace(beta_0, beta_T, T)
    alphas_np = 1 - betas_np
    betas = torch.tensor(betas_np).to(torch.float32).to(device)
    alphas = torch.tensor(alphas_np).to(torch.float32).to(device)
    bar_alphas = torch.cumprod(alphas, axis=0).to(torch.float32).to(device)
    return betas, alphas, bar_alphas


def p_forward(x0: torch.Tensor, t: int, bar_alphas, device) -> torch.Tensor:
    """
    Sample x_t from q(x_t | x_0) via closed-form formula:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

    Args:
        x0: Original data, shape (N,)
        t: Timestep index
        bar_alphas: Cumulative product of alphas
        device: torch device

    Returns:
        Noisy sample x_t
    """
    alpha_bar_t = bar_alphas[t]
    mean = torch.sqrt(alpha_bar_t) * x0
    sigma2 = torch.eye(x0.shape[0]).to(device) * (1 - alpha_bar_t)
    return torch.distributions.MultivariateNormal(mean, sigma2).sample().to(device)


def compute_tilde_mu(x0, x_t, t, alphas, bar_alphas):
    """
    Compute the analytical posterior mean mu_tilde(x_t, x_0).
    Used as training target for the denoiser network.

    Args:
        x0: Clean data
        x_t: Noisy data at step t
        t: Timestep
        alphas: Alpha schedule
        bar_alphas: Cumulative alpha schedule

    Returns:
        mu_tilde: Posterior mean tensor
    """
    beta_t = 1 - alphas[t]
    alpha_bar_t = bar_alphas[t]
    alpha_bar_tm1 = bar_alphas[t - 1]
    mu = (
        x0 * torch.sqrt(alpha_bar_tm1) * beta_t / (1 - alpha_bar_t)
        + x_t * torch.sqrt(alphas[t]) * (1 - alpha_bar_tm1) / (1 - alpha_bar_t)
    )
    return mu


def reverse_step(model, x_t, t, alphas, bar_alphas, len_x, device):
    """
    One step of the reverse diffusion (denoising):
        x_{t-1} ~ N(mu_theta(x_t, t), beta_t * I)

    Args:
        model: Trained denoiser
        x_t: Current noisy state
        t: Current timestep
        alphas: Alpha schedule
        bar_alphas: Cumulative alpha schedule
        len_x: Data dimensionality
        device: torch device

    Returns:
        x_{t-1}: Less noisy sample
    """
    beta_t = 1 - alphas[t]
    mu = model(x_t, t)
    return torch.distributions.MultivariateNormal(
        mu, torch.diag(beta_t.repeat(len_x))
    ).sample().to(device)
