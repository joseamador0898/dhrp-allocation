"""Training loops for DHRP, LLM-DHRP, and deep baseline models."""

import numpy as np
import torch
import torch.optim as optim

from ..models.dhrp_layer import DHRPLayer
from ..models.llm_dhrp_layer import LLMDHRPLayer
from ..models.loss_functions import dhrp_loss
from ..data.feature_engineering import build_dataset, make_features, DEFAULT_FDIM
from ..data.universe_config import get_universe_config


def _set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_dhrp(prices, device="cpu", is_em=False, volume=None, fdim=DEFAULT_FDIM, seed=42,
               train_end=None):
    """Train DHRP model on historical price data."""
    _set_seed(seed)
    X, S, R, H = build_dataset(prices, is_em=is_em, volume=volume, fdim=fdim,
                                train_end=train_end)
    if X.ndim == 1 or X.shape[0] < 50:
        raise ValueError(f"Insufficient data: {X.shape[0] if X.ndim > 1 else 0}")
    n_samp, fdim_actual, n_assets = X.shape[0], X.shape[1], prices.shape[1]
    mkt = "EM" if is_em else "DM"
    print(f"  [{mkt}] {n_samp} samples, {n_assets} assets, fdim={fdim_actual}")

    hid, wd, lr = (32, 1e-3, 1.5e-4) if is_em else (64, 3e-4, 6e-4)
    epochs, clip = (50, 0.5) if is_em else (60, 1.0)
    hrp_s, hrp_e = (0.3, 0.05) if is_em else (0.15, 0.02)

    model = DHRPLayer(n_assets, fdim_actual, hidden_dim=hid, is_em=is_em).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)

    Xt = torch.from_numpy(X).to(device)
    St = torch.from_numpy(S).to(device)
    Rt = torch.from_numpy(R).to(device)

    # Turnover penalty ramps up over training to encourage stable weights
    lam_turnover = 0.05 if not is_em else 0.02

    best_loss, best_st = float("inf"), None
    for ep in range(epochs):
        perm = torch.randperm(n_samp)
        Xs, Ss, Rs = Xt[perm], St[perm], Rt[perm]
        Hs = H[perm.cpu().numpy()]
        ep_loss, nb = 0.0, 0
        lam = hrp_s - (hrp_s - hrp_e) * (ep / epochs)
        # Ramp turnover penalty: 0 for first 20% of training, then linear to full
        turnover_scale = max(0.0, (ep / epochs - 0.2) / 0.8)

        prev_wts = None
        for s in range(0, n_samp, 32):
            e = min(s + 32, n_samp)
            opt.zero_grad()
            loss = dhrp_loss(
                model, Xs[s:e], Ss[s:e], Rs[s:e], Hs[s:e],
                is_em=is_em, lam_hrp=lam,
                lam_turnover=lam_turnover * turnover_scale,
            )
            if not torch.isnan(loss) and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                ep_loss += loss.item()
                nb += 1
        sched.step()

        if nb > 0:
            avg = ep_loss / nb
            if avg < best_loss:
                best_loss = avg
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  [{mkt}] Epoch {ep + 1}/{epochs}, loss={avg:.6f}")

    if best_st:
        model.load_state_dict({k: v.to(device) for k, v in best_st.items()})
    return model


def train_dhrp_multiseed(prices, seeds=None, device="cpu", is_em=False,
                         volume=None, fdim=DEFAULT_FDIM, train_end=None):
    """Train DHRP models across multiple seeds for robustness analysis.

    Returns list of trained models (one per seed).
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    models = []
    for s in seeds:
        print(f"\n--- Seed {s} ---")
        m = train_dhrp(prices, device=device, is_em=is_em, volume=volume,
                       fdim=fdim, seed=s, train_end=train_end)
        models.append(m)
    return models


def train_llm_dhrp(
    prices,
    text_features=None,
    macro_features=None,
    device="cpu",
    is_em=False,
    text_dim=768,
    use_text=True,
    use_macro=False,
    macro_dim=4,
    fusion_type="cross_attention",
    depth=3,
    hidden_dim=64,
    epochs=60,
    lr=3e-4,
    weight_decay=3e-4,
    batch_size=32,
    grad_clip=1.0,
    hrp_lam_start=0.3,
    hrp_lam_end=0.05,
    use_hrp_reg=True,
    volume=None,
    fdim=DEFAULT_FDIM,
    seed=42,
    train_end=None,
    universe=None,
):
    """Train LLM-enhanced DHRP model.

    Args:
        prices: DataFrame of adjusted close prices
        text_features: dict with 'finbert' (n_samples, n_assets, 768) and/or
                       'sentiment' (n_samples, n_assets, 16), or None
        macro_features: np.ndarray (n_samples, macro_dim), or None
        device: compute device
        is_em: emerging markets flag
        text_dim: LLM embedding dimension (768 for FinBERT)
        use_text / use_macro: feature toggles
        fusion_type: "cross_attention", "concat", or "additive"
        depth: tree depth
        epochs, lr, weight_decay, batch_size, grad_clip: training hyperparams
        hrp_lam_start / hrp_lam_end: HRP regularization schedule
        use_hrp_reg: whether to use HRP regularization
        volume: DataFrame of daily volume, or None
        fdim: feature dimension
        seed: random seed for reproducibility
        train_end: if provided, only use data up to this date for training
    Returns:
        trained LLMDHRPLayer model
    """
    _set_seed(seed)
    # Resolve per-universe overrides (lookback, text LR, gate bias, dropout,
    # HRP schedule, tree depth). When `universe` is None the defaults match
    # current DM/EM hard-coded behavior, so existing callers are unaffected.
    cfg = get_universe_config(universe)
    lookback = cfg.get("lookback_window", 252)
    text_lr_scale = cfg.get("text_lr_scale", 0.3)
    modality_dropout = cfg.get("modality_dropout", 0.2)
    gate_bias_init = cfg.get("gate_bias_init", -2.0)
    if universe is not None:
        hrp_lam_start = cfg.get("hrp_lam_start", hrp_lam_start)
        hrp_lam_end = cfg.get("hrp_lam_end", hrp_lam_end)
        depth = cfg.get("tree_depth", depth)

    X, S, R, H = build_dataset(prices, window=lookback, is_em=is_em,
                                volume=volume, fdim=fdim, train_end=train_end)
    if X.ndim == 1 or X.shape[0] < 50:
        raise ValueError(f"Insufficient data: {X.shape[0] if X.ndim > 1 else 0}")
    n_samp, fdim_actual, n_assets = X.shape[0], X.shape[1], prices.shape[1]
    mkt = universe if universe else ("EM" if is_em else "DM")
    print(f"  [{mkt}] {n_samp} samples, {n_assets} assets, fdim={fdim_actual} (LLM-DHRP)")

    # Prepare text embeddings — average across assets for a global signal
    text_embs = None
    if use_text and text_features is not None:
        if "finbert" in text_features and text_features["finbert"] is not None:
            fb = text_features["finbert"]  # (n_dates, n_assets, 768)
            n_text = min(fb.shape[0], n_samp)
            text_embs = fb[:n_text].mean(axis=1)  # (n_text, 768)
            if n_text < n_samp:
                pad = np.zeros((n_samp - n_text, text_embs.shape[1]), dtype=np.float32)
                text_embs = np.vstack([text_embs, pad])
            print(f"  [{mkt}] Text features: {text_embs.shape}")

    # Prepare macro features
    macro_feats = None
    if use_macro and macro_features is not None:
        n_macro = min(macro_features.shape[0], n_samp)
        macro_feats = macro_features[:n_macro]
        if n_macro < n_samp:
            pad = np.zeros((n_samp - n_macro, macro_feats.shape[1]), dtype=np.float32)
            macro_feats = np.vstack([macro_feats, pad])
        macro_dim = macro_feats.shape[1]
        print(f"  [{mkt}] Macro features: {macro_feats.shape}")

    model = LLMDHRPLayer(
        n_assets=n_assets, feature_dim=fdim_actual, text_dim=text_dim,
        hidden_dim=hidden_dim, depth=depth, is_em=is_em,
        use_text=use_text and text_embs is not None,
        use_macro=use_macro and macro_feats is not None,
        macro_dim=macro_dim, fusion_type=fusion_type,
        modality_dropout=modality_dropout,
        gate_bias_init=gate_bias_init,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{mkt}] Model params: {n_params:,}")

    # Differential LR: text/fusion params learn slower to avoid corrupting price signals
    text_param_names = {"text_proj", "fusion", "fusion_proj"}
    text_params = []
    other_params = []
    for name, param in model.named_parameters():
        if any(tp in name for tp in text_param_names):
            text_params.append(param)
        else:
            other_params.append(param)
    opt = optim.AdamW([
        {"params": other_params, "lr": lr},
        {"params": text_params, "lr": lr * text_lr_scale},
    ], weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)

    Xt = torch.from_numpy(X).to(device)
    St = torch.from_numpy(S).to(device)
    Rt = torch.from_numpy(R).to(device)
    Tt = torch.from_numpy(text_embs).to(device) if text_embs is not None else None
    Mt = torch.from_numpy(macro_feats.astype(np.float32)).to(device) if macro_feats is not None else None

    # Turnover penalty ramps up over training
    lam_turnover = 0.05 if not is_em else 0.02

    best_loss, best_st = float("inf"), None
    for ep in range(epochs):
        perm = torch.randperm(n_samp)
        Xs, Ss, Rs = Xt[perm], St[perm], Rt[perm]
        Hs = H[perm.cpu().numpy()]
        Ts = Tt[perm] if Tt is not None else None
        Ms = Mt[perm] if Mt is not None else None
        ep_loss, nb = 0.0, 0
        lam = hrp_lam_start - (hrp_lam_start - hrp_lam_end) * (ep / epochs) if use_hrp_reg else 0.0
        turnover_scale = max(0.0, (ep / epochs - 0.2) / 0.8)

        for s in range(0, n_samp, batch_size):
            e = min(s + batch_size, n_samp)
            opt.zero_grad()
            loss = _llm_dhrp_loss(
                model, Xs[s:e], Ss[s:e], Rs[s:e], Hs[s:e],
                text_embs=Ts[s:e] if Ts is not None else None,
                macro_feats=Ms[s:e] if Ms is not None else None,
                is_em=is_em, lam_hrp=lam,
                lam_turnover=lam_turnover * turnover_scale,
            )
            if not torch.isnan(loss) and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                ep_loss += loss.item()
                nb += 1
        sched.step()

        if nb > 0:
            avg = ep_loss / nb
            if avg < best_loss:
                best_loss = avg
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  [{mkt}] Epoch {ep + 1}/{epochs}, loss={avg:.6f}")

    if best_st:
        model.load_state_dict({k: v.to(device) for k, v in best_st.items()})
    return model


def train_llm_dhrp_warmstart(
    prices,
    pretrained_dhrp,
    text_features=None,
    macro_features=None,
    device="cpu",
    is_em=False,
    text_dim=768,
    use_text=True,
    use_macro=False,
    macro_dim=4,
    fusion_type="cross_attention",
    epochs=40,
    lr=1e-4,
    batch_size=32,
    grad_clip=0.5,
    volume=None,
    fdim=DEFAULT_FDIM,
    seed=42,
    train_end=None,
    universe=None,
):
    """Train LLM-DHRP with warm-start: load pretrained DHRP weights, freeze
    price pathway for first phase, then fine-tune everything with lower LR.

    Phase 1 (60% of epochs): Only text/fusion params train
    Phase 2 (40% of epochs): All params train with reduced LR

    This prevents the text pathway from corrupting learned price signals.
    """
    _set_seed(seed)
    cfg = get_universe_config(universe)
    lookback = cfg.get("lookback_window", 252)
    modality_dropout = cfg.get("modality_dropout", 0.1)  # Lower dropout for warmstart
    gate_bias_init = cfg.get("gate_bias_init", -1.0)  # Less aggressive suppression

    X, S, R, H = build_dataset(prices, window=lookback, is_em=is_em,
                                volume=volume, fdim=fdim, train_end=train_end)
    if X.ndim == 1 or X.shape[0] < 50:
        raise ValueError(f"Insufficient data: {X.shape[0] if X.ndim > 1 else 0}")
    n_samp, fdim_actual, n_assets = X.shape[0], X.shape[1], prices.shape[1]
    mkt = universe if universe else ("EM" if is_em else "DM")
    print(f"  [{mkt}] {n_samp} samples (warm-start LLM-DHRP)")

    # Prepare text embeddings
    text_embs = None
    if use_text and text_features is not None:
        if "finbert" in text_features and text_features["finbert"] is not None:
            fb = text_features["finbert"]
            n_text = min(fb.shape[0], n_samp)
            text_embs = fb[:n_text].mean(axis=1)
            if n_text < n_samp:
                pad = np.zeros((n_samp - n_text, text_embs.shape[1]), dtype=np.float32)
                text_embs = np.vstack([text_embs, pad])
            print(f"  [{mkt}] Text features: {text_embs.shape}")

    # Prepare macro features
    macro_feats = None
    if use_macro and macro_features is not None:
        n_macro = min(macro_features.shape[0], n_samp)
        macro_feats = macro_features[:n_macro]
        if n_macro < n_samp:
            pad = np.zeros((n_samp - n_macro, macro_feats.shape[1]), dtype=np.float32)
            macro_feats = np.vstack([macro_feats, pad])
        macro_dim = macro_feats.shape[1]

    depth = cfg.get("tree_depth", 3)
    hidden_dim_val = 64

    model = LLMDHRPLayer(
        n_assets=n_assets, feature_dim=fdim_actual, text_dim=text_dim,
        hidden_dim=hidden_dim_val, depth=depth, is_em=is_em,
        use_text=use_text and text_embs is not None,
        use_macro=use_macro and macro_feats is not None,
        macro_dim=macro_dim, fusion_type=fusion_type,
        modality_dropout=modality_dropout,
        gate_bias_init=gate_bias_init,
    ).to(device)

    # Transfer pretrained DHRP weights (shared params: leaf_assign, gates, cov_proj, etc.)
    dhrp_state = pretrained_dhrp.state_dict()
    model_state = model.state_dict()
    transferred = 0
    for key in dhrp_state:
        if key in model_state and dhrp_state[key].shape == model_state[key].shape:
            model_state[key] = dhrp_state[key].to(device)
            transferred += 1
    model.load_state_dict(model_state)
    print(f"  [{mkt}] Transferred {transferred} params from pretrained DHRP")

    # Identify text vs price params
    text_param_names = {"text_proj", "fusion", "fusion_proj"}
    text_params = []
    price_params = []
    for name, param in model.named_parameters():
        if any(tp in name for tp in text_param_names):
            text_params.append(param)
        else:
            price_params.append(param)

    Xt = torch.from_numpy(X).to(device)
    St = torch.from_numpy(S).to(device)
    Rt = torch.from_numpy(R).to(device)
    Tt = torch.from_numpy(text_embs).to(device) if text_embs is not None else None
    Mt = torch.from_numpy(macro_feats.astype(np.float32)).to(device) if macro_feats is not None else None

    phase1_epochs = int(epochs * 0.6)
    phase2_epochs = epochs - phase1_epochs
    hrp_lam = cfg.get("hrp_lam_start", 0.15)

    # Phase 1: Freeze price params, only train text/fusion
    print(f"  [{mkt}] Phase 1: training text pathway only ({phase1_epochs} epochs)")
    for p in price_params:
        p.requires_grad = False
    opt1 = optim.AdamW(text_params, lr=lr, weight_decay=1e-4)
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=phase1_epochs, eta_min=lr / 10)

    best_loss, best_st = float("inf"), None
    for ep in range(phase1_epochs):
        perm = torch.randperm(n_samp)
        Xs, Ss, Rs = Xt[perm], St[perm], Rt[perm]
        Hs = H[perm.cpu().numpy()]
        Ts = Tt[perm] if Tt is not None else None
        Ms = Mt[perm] if Mt is not None else None
        ep_loss, nb = 0.0, 0

        for s in range(0, n_samp, batch_size):
            e = min(s + batch_size, n_samp)
            opt1.zero_grad()
            loss = _llm_dhrp_loss(
                model, Xs[s:e], Ss[s:e], Rs[s:e], Hs[s:e],
                text_embs=Ts[s:e] if Ts is not None else None,
                macro_feats=Ms[s:e] if Ms is not None else None,
                is_em=is_em, lam_hrp=hrp_lam,
            )
            if not torch.isnan(loss) and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(text_params, grad_clip)
                opt1.step()
                ep_loss += loss.item()
                nb += 1
        sched1.step()

        if nb > 0:
            avg = ep_loss / nb
            if avg < best_loss:
                best_loss = avg
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  [{mkt}] Phase 1 Epoch {ep + 1}/{phase1_epochs}, loss={avg:.6f}")

    # Phase 2: Unfreeze all, fine-tune with lower LR
    print(f"  [{mkt}] Phase 2: fine-tuning all params ({phase2_epochs} epochs)")
    for p in price_params:
        p.requires_grad = True
    opt2 = optim.AdamW([
        {"params": price_params, "lr": lr * 0.1},
        {"params": text_params, "lr": lr * 0.5},
    ], weight_decay=1e-4)
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=phase2_epochs, eta_min=lr / 50)

    for ep in range(phase2_epochs):
        perm = torch.randperm(n_samp)
        Xs, Ss, Rs = Xt[perm], St[perm], Rt[perm]
        Hs = H[perm.cpu().numpy()]
        Ts = Tt[perm] if Tt is not None else None
        Ms = Mt[perm] if Mt is not None else None
        ep_loss, nb = 0.0, 0

        for s in range(0, n_samp, batch_size):
            e = min(s + batch_size, n_samp)
            opt2.zero_grad()
            loss = _llm_dhrp_loss(
                model, Xs[s:e], Ss[s:e], Rs[s:e], Hs[s:e],
                text_embs=Ts[s:e] if Ts is not None else None,
                macro_feats=Ms[s:e] if Ms is not None else None,
                is_em=is_em, lam_hrp=hrp_lam * 0.5,
                lam_turnover=0.03,
            )
            if not torch.isnan(loss) and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt2.step()
                ep_loss += loss.item()
                nb += 1
        sched2.step()

        if nb > 0:
            avg = ep_loss / nb
            if avg < best_loss:
                best_loss = avg
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  [{mkt}] Phase 2 Epoch {ep + 1}/{phase2_epochs}, loss={avg:.6f}")

    if best_st:
        model.load_state_dict({k: v.to(device) for k, v in best_st.items()})
    return model


def train_llm_dhrp_multiseed(prices, seeds=None, **kwargs):
    """Train LLM-DHRP models across multiple seeds for robustness analysis."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    models = []
    for s in seeds:
        print(f"\n--- Seed {s} ---")
        m = train_llm_dhrp(prices, seed=s, **kwargs)
        models.append(m)
    return models


def _llm_dhrp_loss(model, xb, Sb, rb, hrp_w, text_embs=None, macro_feats=None,
                   is_em=False, lam_hrp=0.3, lam_turnover=0.0):
    """Multi-objective loss for LLM-DHRP."""
    port_r, wts = [], []
    for t in range(rb.shape[0]):
        te = text_embs[t] if text_embs is not None else None
        mf = macro_feats[t] if macro_feats is not None else None
        w = model(xb[t], Sb[t], text_emb=te, macro_feat=mf)
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
    entropy = (-(wts * torch.log(wts + 1e-8)).sum(1).mean() * 0.15) if is_em else 0
    hhi = (wts ** 2).sum(1).mean()
    concentration_pen = hhi * (0.3 if is_em else 0.1)

    # Turnover penalty on consecutive weight changes within batch
    turnover_pen = 0.0
    if lam_turnover > 0 and wts.shape[0] > 1:
        turnover_pen = torch.mean(torch.abs(wts[1:] - wts[:-1])) * lam_turnover

    loss = -(crra + sharpe + entropy) + hrp_reg + risk + concentration_pen + turnover_pen
    if torch.isnan(loss):
        return torch.tensor(0.0, device=xb.device, requires_grad=True)
    return loss


def dhrp_weights(model, rets, is_em=False, volume=None):
    """Compute portfolio weights from a trained DHRP model."""
    try:
        device = next(model.parameters()).device
        cov = (rets.cov().values * 252).astype(np.float32)
        if is_em:
            cov += np.eye(cov.shape[0]) * 0.01
        feat = make_features(rets, model.feature_dim, is_em, volume=volume).astype(np.float32)
        with torch.no_grad():
            w = model(
                torch.from_numpy(np.nan_to_num(feat)).to(device),
                torch.from_numpy(np.nan_to_num(cov)).to(device),
            ).cpu().numpy()
        w = np.nan_to_num(np.clip(w, 0, 1))
        return w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
    except Exception as e:
        import warnings
        warnings.warn(f"dhrp_weights failed for {type(model).__name__}: {e}")
        return np.ones(rets.shape[1]) / rets.shape[1]
