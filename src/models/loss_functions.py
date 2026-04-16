import torch


def dhrp_loss(layer, xb, Sb, rb, hrp_w, is_em=False, lam_hrp=0.3,
              prev_weights=None, lam_turnover=0.0):
    """Multi-objective loss: CRRA utility + Sharpe + HRP regularization + turnover penalty."""
    port_r, wts = [], []
    for t in range(rb.shape[0]):
        w = layer(xb[t], Sb[t])
        wts.append(w)
        port_r.append((w * rb[t]).sum())
    port_r = torch.stack(port_r)
    wts = torch.stack(wts)

    gamma = 1.2 if is_em else 2.5
    crra = (
        (torch.clamp(1 + port_r, min=0.1) ** (1 - gamma) - 1) / (1 - gamma)
    ).mean()

    sharpe = port_r.mean() / (port_r.std() + 1e-6) * (0.5 if is_em else 1.0)

    hrp_target = torch.from_numpy(hrp_w).to(xb.device).float()
    hrp_reg = ((wts - hrp_target) ** 2).mean() * lam_hrp * (0.5 if is_em else 0.2)

    risk = (
        torch.stack([wts[t] @ Sb[t] @ wts[t] for t in range(rb.shape[0])]).mean()
        * (0.004 if is_em else 0.001)
    )

    entropy = (
        (-(wts * torch.log(wts + 1e-8)).sum(1).mean() * 0.15) if is_em else 0
    )

    hhi = (wts ** 2).sum(1).mean()
    concentration_pen = hhi * (0.3 if is_em else 0.1)

    # Turnover penalty: penalize large weight changes between consecutive samples
    turnover_pen = 0.0
    if lam_turnover > 0:
        if prev_weights is not None:
            turnover_pen = torch.mean(torch.abs(wts - prev_weights)) * lam_turnover
        # Also penalize within-batch consecutive weight changes
        if wts.shape[0] > 1:
            intra_turnover = torch.mean(torch.abs(wts[1:] - wts[:-1])) * lam_turnover * 0.5
            turnover_pen = turnover_pen + intra_turnover

    loss = -(crra + sharpe + entropy) + hrp_reg + risk + concentration_pen + turnover_pen
    if torch.isnan(loss):
        return torch.tensor(0.0, device=xb.device, requires_grad=True)
    return loss
