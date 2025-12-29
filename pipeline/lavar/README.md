# LAVAR (Latent Autoencoder VAR) + Supply Head

This module trains a **two-stage** model:

- **Stage 1**: Learn a latent state \(z_t\) from non-supply features \(x_t\) with an autoencoder, while enforcing **linear VAR(p)** dynamics in latent space.
- **Stage 2**: Freeze the learned latent model and train a **SupplyHead** to predict future supplies \(y_{t+1:t+H}\) from rolled-out future latents.

## System diagram (Mermaid)

```mermaid
flowchart TD
  %% =========================
  %% Data / windowing
  %% =========================
  X["Non-supply features\nx: T x Dx"] -->|Rolling window| W["x_past: B x p_plus_1 x Dx"]
  X -->|Future supervision opt| XF["x_future: B x H x Dx"]
  Y["Supply targets\ny: T x Dy"] -->|Future targets| YF["y_future: B x H x Dy"]

  %% =========================
  %% Stage 1: LAVAR training
  %% =========================
  subgraph S1[Stage 1 - Train LAVAR]
    direction TB

    W -->|Encode each step| ENC["Encoder MLP\nDx -> k"]
    ENC --> ZSEQ["z_seq: B x p_plus_1 x k"]

    ZSEQ -->|Use history z t-p to t-1| VAR["VAR p dynamics\nA1..Ap"]
    VAR --> ZP["z_pred: B x k"]

    ZSEQ -->|Current latent z_t| ZT["z_true: B x k"]
    ZT --> DEC["Decoder MLP\nk -> Dx"]
    DEC --> XH["x_hat: B x Dx"]

    XH --> LREC["Reconstruction loss\nMSE(x_hat, x_t)"]
    ZP --> LDYN["Latent dynamics loss\nMSE(z_pred, z_true)"]

    %% optional multi-step latent supervision
    ZSEQ -->|Optional: encode past p steps| ZH["z_hist: B x p x k"]
    ZH -->|rollout_latent horizon H| ZR["z_roll: B x H x k"]
    XF -->|Encode future steps| ZF["z_fut_true: B x H x k"]
    ZR --> LMS["Multi-step latent loss\nMSE(z_roll, z_fut_true)"]

    LREC --> SUM1((Weighted sum))
    LDYN --> SUM1
    LMS -.-> SUM1
    SUM1 --> OPT1["Adam optimizer\nupdates Encoder/Decoder/VAR"]
  end

  %% =========================
  %% Stage 2: Supply head training
  %% =========================
  subgraph S2[Stage 2 - Train SupplyHead]
    direction TB

    W -->|Take last p steps as history| HIST["x_hist: B x p x Dx"]
    HIST -->|Encode| ENC2["Encoder MLP\nfrozen"]
    ENC2 --> ZH2["z_hist: B x p x k"]
    ZH2 -->|rollout_latent H| ZFUT["z_future: B x H x k"]
    ZFUT -->|Flatten| FLAT["B*H x k"]
    FLAT --> HEAD["SupplyHead\nk -> Dy"]
    HEAD --> YH["y_hat: B x H x Dy"]

    YH --> LSUP["Supply loss\nMSE or Poisson NLL"]
    YF --> LSUP
    LSUP --> OPT2["Adam optimizer\nupdates SupplyHead only"]
  end

  %% =========================
  %% Saved artifacts
  %% =========================
  S1 --> CK1[[lavar_best.pth]]
  S2 --> CK2[[lavar_supply_best.pth]]
```

## Code map

- `dataset.py`: `RollingXYDataset` produces `(x_past, x_future, y_future)` rolling windows.
- `models.py`:
  - `LAVAR`: encoder/decoder + `VARDynamics`
  - `LAVARWithSupply`: latent rollout + `SupplyModel` head
- `dynamics.py`: `VARDynamics` implements linear VAR(p) in latent space.
- `train_stage1.py`: optimizes reconstruction + latent dynamics (+ optional multi-step latent supervision).
- `train_stage2.py`: freezes `lavar` and trains only `supply_head`.


