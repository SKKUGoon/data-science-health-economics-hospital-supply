import torch
from torch.utils.data import DataLoader
from pipeline.lavar.config import LAVARConfig
from pipeline.lavar.models import LAVARWithSupply
from pipeline.lavar.losses import poisson_nll


def stage2_train_supply(model: LAVARWithSupply, train_loader: DataLoader, val_loader: DataLoader, cfg: LAVARConfig) -> None:
    device = torch.device(cfg.device)
    model.to(device)

    # Freeze LAVAR weights
    for p in model.lavar.parameters():
        p.requires_grad = False
    model.lavar.eval()

    # Supply model optimizer
    opt = torch.optim.Adam(model.supply_head.parameters(), lr=cfg.lr_supply)

    # Training start
    best_val = float("inf")
    for epoch in range(1, cfg.epochs_supply + 1):
        model.train()
        tr_loss = 0.0
        n = 0

        for x_past, _x_future, y_future in train_loader:
            x_past = x_past.to(device)  # (B, p, Dx)
            y_future = y_future.to(device)  # (B, horizon, n_supply)

            y_hat = model(x_past)  # (B, horizon, n_supply)

            if cfg.supply_loss == "poisson_nll":
                loss = poisson_nll(y_hat, y_future)
            else:
                loss = torch.mean((y_hat - y_future) ** 2)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.supply_head.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item() * x_past.size(0)
            n += x_past.size(0)
        
        tr_loss /= max(n, 1)
        
        model.eval()
        with torch.no_grad():
            va_loss = 0.0
            vn = 0

            for x_past, _x_future, y_future in val_loader:
                x_past = x_past.to(device)
                y_future = y_future.to(device)

                y_hat = model(x_past)
                if cfg.supply_loss == "poisson_nll":
                    loss = poisson_nll(y_hat, y_future)
                else:
                    loss = torch.mean((y_hat - y_future) ** 2)
                
                va_loss += loss.item() * x_past.size(0)
                vn += x_past.size(0)

            va_loss /= max(vn, 1)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, "lavar_supply_best.pth")
            print(f"Epoch {epoch}: New best validation loss {best_val:.6f}")
        else:
            print(f"Epoch {epoch}: Validation loss {va_loss:.6f} (no improvement)")
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Training loss {tr_loss:.6f}, Validation loss {va_loss:.6f}")
        
        model.load_state_dict(best_state)
        print(f"Loaded best model from epoch {epoch}, best validation loss {best_val:.6f}")
            