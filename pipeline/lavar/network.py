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
from typing import List, Optional

# Encoder and Decoder object
class MLP(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int], 
                 output_dim: int, 
                 activation: nn.Module = nn.ReLU(),
                 dropout_rate: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims

        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(activation)
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SupplyModel(nn.Module):
    """
    Map latent state z_t to supply usage y_t

    Supply Model is independent of hospital: only dimension will change

    Easily extendable to probabilistic heads later
    """
    def __init__(self, 
                 latent_dim: int, 
                 output_dim: int, 
                 hidden_dims: Optional[List[int]] = [64, 64], 
                 activation: nn.Module = nn.ReLU(), 
                 use_softplus: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.use_softplus = use_softplus

        layers: List[nn.Module] = []
        dims = [latent_dim] + hidden_dims
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(activation)
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)
        
        # Softplus gurantees non-negative output
        self.use_softplus = use_softplus
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim k)
        returns: (B, n_supply), non-negative when softplus is used
        """
        out = self.model(z)

        if self.use_softplus:
            return self.softplus(out)
        
        return out