"""
Observation encoder/decoder for continuous control.
Replaces TEM's two-hot compression (f_c) and categorical decompression (f_c_star).
"""

import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    """Compress raw continuous observations to a latent vector.
    Replaces the two-hot lookup table in original TEM."""

    def __init__(self, cfg):
        super().__init__()
        obs_dim = cfg['obs_dim']
        n_x_c = cfg['n_x_c']
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, n_x_c),
        )

    def forward(self, x_raw):
        """x_raw: (batch, obs_dim) -> x_c: (batch, n_x_c)"""
        return self.net(x_raw)


class ObservationDecoder(nn.Module):
    """Decode grounded location p back to observation prediction.
    Replaces f_x (sum over entorhinal preferences) + MLP_c_star (decompression).
    Outputs Gaussian parameters instead of categorical logits."""

    def __init__(self, cfg):
        super().__init__()
        obs_dim = cfg['obs_dim']
        n_x_total = cfg['n_x_c'] * cfg['n_f']  # all frequency modules concatenated

        # Learned weights for summing over entorhinal preferences (same as original w_x, b_x)
        # These are stored in the main TEM model; this decoder receives the
        # already-transformed input of shape (batch, n_x_c * n_f)

        # Decompression: from multi-frequency sensory space to observation space
        self.mu_net = nn.Sequential(
            nn.Linear(n_x_total, 64),
            nn.ELU(),
            nn.Linear(64, obs_dim),
        )
        self.log_var_net = nn.Sequential(
            nn.Linear(n_x_total, 64),
            nn.ELU(),
            nn.Linear(64, obs_dim),
        )

    def forward(self, x_decompressed):
        """x_decompressed: (batch, n_x_f[0]) -> mu, sigma: (batch, obs_dim)"""
        mu = self.mu_net(x_decompressed)
        log_var = self.log_var_net(x_decompressed)
        # Clamp log_var so NLL stays non-negative: log(σ²) >= -log(2π) ≈ -1.84
        # This sets σ_min ≈ 0.40, making the per-dim NLL floor ≈ 0
        log_var = torch.clamp(log_var, min=-1.84, max=4.0)
        sigma = torch.exp(0.5 * log_var)
        return mu, sigma
