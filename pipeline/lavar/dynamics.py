import torch
import torch.nn as nn

# VAR(p) latent dynamics
class VARDynamics(nn.Module):
    """
    z(t) = sum_{i=1}^{p} A_i z(t-i) + B \epsilon(t)

    - A_i is a matrix of size k x k
    - B is a matrix of size k x 1
    - k is the dimension of the latent state
    - p is the order of the VAR model
    - \epsilon(t) is a noise term
    - z(t) is the latent state at time t
    - z(t-i) is the latent state at time t-i
    """
    def __init__(self, latent_dim: int, order: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.order = order
        
        A = torch.zeros(order, latent_dim, latent_dim)
        A[0] = torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim)  # Initialize close to identity
        for i in range(1, order):
            A[i] = 0.05 * torch.randn(latent_dim, latent_dim)
        self.A = nn.Parameter(A)  # (order, latent_dim, latent_dim). Register them as parameters making them learnable.
    
    def forward(self, z_history: torch.Tensor) -> torch.Tensor:
        """
        z_history: (B, p, k)
        A:         (p, k, k)
        return:    (B, k)
        """
        # z_t = sum_{lag=1..p} z_{t-lag} @ A_lag^T
        # (B,p,k) x (p,k,k) -> (B,k)
        return torch.einsum("bpk,pkh->bh", z_history, self.A)