# LAVAR: Latent Autoencoder VAR
# Combination of Encoder and Decoder with VAR as latent factor dynamics (nonlinear + VAR)
# - The model is non linear in the observation space, but linear in the latent state dynamics
# Intuition
# Mathmatics
# - $LatentDynamics(linear): z_t = f(z_{t-1}, \epsilon_t)$
# - $ObservationModel(nonlinear): x_t = g(z_t, \eta_t)$
# - z_t is a low dimentional and structured
# - g is a neural network that maps non linear observation space to latent space
# - f is a learnable VAR model with matrix parameters
#   - Latent dynamics are often approximately linear
#   - Non linearity is pushed into the observation model
#   - Linear latent dynamics will give us the interpretability

import torch
import torch.nn as nn
from pipeline.lavar.network import MLP, SupplyModel
from pipeline.lavar.dynamics import VARDynamics
from typing import List, Optional, Dict


class LAVAR(nn.Module):
    r"""
    LAVAR - Latent Autoencoder VAR

    Observation:
      x_t = g_theta(z_t) + \eta_t

    Latent dynamics (VAR(p)):
      z_t = sum_{i=1}^{p} A_i z(t-i)
    """
    def __init__(self, 
                 input_dim: int, 
                 latent_dim: int, 
                 transition_order: int = 1, 
                 eps_zero: float = 1e-6,
                 encoder_hidden_dims: Optional[List[int]] = [128, 64],
                 decoder_hidden_dims: Optional[List[int]] = [64, 128],
                 activation: nn.Module = nn.ReLU()):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.transition_order = transition_order
        self.eps_zero = eps_zero

        if encoder_hidden_dims is None:
            h1 = max(32, 2 * input_dim)
            h2 = max(16, h1 // 2)
            encoder_hidden_dims = [h1, h2]
        if decoder_hidden_dims is None:
            h1 = max(32, 2 * latent_dim)
            h2 = max(16, h1 // 2)
            decoder_hidden_dims = [h2, h1]

        self.encoder = MLP(input_dim, encoder_hidden_dims, latent_dim, activation)
        self.decoder = MLP(latent_dim, decoder_hidden_dims, input_dim, activation)
        self.dynamics = VARDynamics(latent_dim, transition_order)

    @staticmethod
    def replace_zeros(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Replace exact zeros with eps to avoid dead gradients when subsequent ops
        include division, log, or gating that can become inactive at 0.
        """
        if eps <= 0.0:
            return x
        return torch.where(x == 0, torch.full_like(x, eps), x)  # replace zeros with eps (full like) where x == 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.replace_zeros(x, self.eps_zero)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x_seq: (B, p+1, input_dim)
               Where last index is the current time and previous p are the history.
          - B: batch size
          - T: number of time steps in the window
          - D: number of features
        """
        B, T, D = x_seq.shape
        p = self.transition_order  # VAR order
        if T != p + 1:
            raise ValueError(f"x_seq must have shape (B, p+1, D). Got {x_seq.shape}")
        
        # Encode each time step independently 
        # Temporal structure is enforced via latent dynamics loss
        z_seq = self.encode(x_seq.reshape(-1, D)).reshape(B, T, self.latent_dim)
        
        # VAR(p) predicts z_t from previous p latents
        z_pred = self.dynamics(z_seq[:, :-1, :])  # (B, k)

        # Reconstruct x_t from z_t (current latent)
        z_true = z_seq[:, -1, :]
        x_hat = self.decode(z_true)

        return {
            "x_hat": x_hat,
            "z_pred": z_pred,
            "z_true": z_true,
        }

    def rollout_latent(self, z_history: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        z_history: (B, past, k), where k is the latent dimension
        returns: z_future: (B, horizon, k)
        """
        B, p, k = z_history.shape
        z_future = []

        z_buffer = z_history.clone()
        for _ in range(horizon):
            z_next = self.dynamics(z_buffer)  # (B, k)
            z_future.append(z_next)
            z_buffer = torch.cat(
                [z_buffer[:, 1:, :], z_next.unsqueeze(1)],
                dim=1
            )

        return torch.stack(z_future, dim=1)


class LAVARWithSupply(nn.Module):
    def __init__(self, 
                 lavar: LAVAR, 
                 supply_dim: int,
                 horizon: int):
        super().__init__()
        self.lavar = lavar
        self.horizon = horizon
        self.supply_head = SupplyModel(
            latent_dim=lavar.latent_dim, 
            output_dim=supply_dim
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (B, p, Dx) or (B, p+1, Dx)
          - If (p+1), the last step is treated as "current" and we use the last p steps
            as VAR history to roll out the future.
        returns: (B, horizon, n_supply dimension)
        """
        B, T, D = x_seq.shape
        p = self.lavar.transition_order

        if T == p + 1:
            # Drop the oldest step; keep the most recent p steps ending at "current"
            x_hist = x_seq[:, 1:, :]
        elif T == p:
            x_hist = x_seq
        else:
            raise ValueError(
                f"x_seq must have shape (B, p, D) or (B, p+1, D) with p={p}. Got {x_seq.shape}"
            )

        z_history = self.lavar.encode(x_hist.reshape(-1, D)).reshape(B, p, -1)
        z_future = self.lavar.rollout_latent(z_history, self.horizon)
        
        # Decode supplies
        B, H, k = z_future.shape
        y_hat = self.supply_head(
            z_future.reshape(B * H, k)
        ).reshape(B, H, -1)

        return y_hat

