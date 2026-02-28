import torch


def dhrp_loss(layer, xb, Sb, rb, hrp_w, is_em=False, lam_hrp=0.3):
    """Multi-objective loss: CRRA utility + Sharpe + HRP regularization."""
    port_r, wts = [], []
    for t in range(rb.shape[0]):
        w = layer(xb[t], Sb[t])
        wts.append(w)
        port_r.append((w * rb[t]).sum())
    port_r = torch.stack(port_r)
    wts = torch.stack(wts)
    port_r = torch.clamp(port_r, -0.5, 0.5)

    gamma = 1.2 if is_em else 2.5
    crra = (
        (torch.clamp(1 + port_r, min=0.1) ** (1 - gamma) - 1) / (1 - gamma)
    ).mean()

    sharpe = port_r.mean() / (port_r.std() + 1e-6) * (0.09 if is_em else 0.18)

    hrp_target = torch.from_numpy(hrp_w).to(xb.device).float()
    hrp_reg = ((wts - hrp_target) ** 2).mean() * lam_hrp * (1.5 if is_em else 0.8)

    risk = (
        torch.stack([wts[t] @ Sb[t] @ wts[t] for t in range(rb.shape[0])]).mean()
        * (0.004 if is_em else 0.001)
    )

    entropy = (
        (-(wts * torch.log(wts + 1e-8)).sum(1).mean() * 0.15) if is_em else 0
    )

    loss = -(crra + sharpe + entropy) + hrp_reg + risk
    if torch.isnan(loss):
        return torch.tensor(0.0, device=xb.device, requires_grad=True)
    return loss
