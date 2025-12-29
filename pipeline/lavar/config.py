from pydantic import BaseModel, Field
from typing import List, Literal

class LAVARConfig(BaseModel):
    device: Literal["cpu", "mps", "cuda"] = Field(default="cpu", description="Device to use for training")

    # Window
    p: int = 7  # History length used as VAR order input
    horizon: int = 14  # 14 days

    # model_size
    latent_dim: int = 8
    encoder_hidden: List[int] = [32, 16]
    decoder_hidden: List[int] = [16, 32]
    supply_hidden: List[int] = []

    # training
    batch_size: int = 64
    num_workers: int = 0

    # stage 1 (LAVAR)
    lr_lavar: float = 1e-3
    epochs_lavar: int = 50
    lambda_dyn: float = 1.0  # weight for latent dynamics loss
    lambda_recon: float = 1.0  # weight for reconstruction loss
    multi_step_latent_supervision: bool = True  # z rollout against future encoded z

    # stage 2 (Supply Model)
    lr_supply: float = 1e-3
    epochs_supply: int = 50
    supply_nonneg: bool = True  # Softplus gurantee?
    supply_loss: str = "mse"
    
    # Split
    train_days: int = 365 * 3  # ~3 years for training.