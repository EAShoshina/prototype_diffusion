"""
Denoiser neural network for the diffusion model.
Uses sinusoidal positional encoding for timestep embedding.
"""
import torch
import torch.nn as nn
import numpy as np


def position_encoding_init(seq_len: int, d: int, n: int = 10000) -> torch.Tensor:
    """
    Compute sinusoidal positional encoding matrix.

    Args:
        seq_len: Number of diffusion timesteps T
        d: Embedding dimension (equals data length)
        n: Base for frequency computation

    Returns:
        Tensor of shape (seq_len, d)
    """
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return torch.from_numpy(P).to(torch.float32)


class Denoise(nn.Module):
    """
    MLP denoiser with sinusoidal timestep embedding.
    Predicts mu_theta(x_t, t) — the mean of the reverse diffusion step.

    Args:
        len_x: Data dimensionality (number of samples)
        T: Total number of diffusion steps
    """

    def __init__(self, len_x: int, T: int):
        super(Denoise, self).__init__()
        self.linear1 = nn.Linear(len_x, len_x)
        self.emb = position_encoding_init(T, len_x)
        self.linear2 = nn.Linear(len_x, len_x)
        self.linear3 = nn.Linear(len_x, len_x)
        self.relu = nn.ReLU()

    def forward(self, input_x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_x: Noisy data x_t, shape (len_x,)
            t: Current diffusion timestep (int)

        Returns:
            Predicted mean mu_theta, shape (len_x,)
        """
        emb_t = self.emb[t]
        x = self.linear1(input_x + emb_t)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
