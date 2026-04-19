"""LLM feature extraction: FinBERT embeddings + Qwen3-8B structured sentiment.

Designed to run on Google Colab free T4 (15GB VRAM):
- FinBERT: ~2GB VRAM, batch inference
- Qwen3-8B (4-bit): ~5GB VRAM, structured JSON output
"""

import json
import os

import numpy as np


def get_gemini_embeddings(headlines, model="text-embedding-004", batch_size=100):
    """Extract embeddings via Google Gemini API.

    Uses GOOGLE_API_KEY from .env.  Returns shape (len(headlines), 768).
    Falls back to zeros if the API is unavailable.

    Args:
        headlines: list of headline strings
        model: Gemini embedding model name
        batch_size: headlines per API call
    Returns:
        np.ndarray of shape (len(headlines), 768)
    """
    import os
    import time

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not set. Returning zero embeddings.")
        return np.zeros((len(headlines), 768), dtype=np.float32)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
    except ImportError:
        print("Install google-generativeai: pip install google-generativeai")
        return np.zeros((len(headlines), 768), dtype=np.float32)

    embeddings = []
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        for attempt in range(3):
            try:
                result = genai.embed_content(
                    model=f"models/{model}",
                    content=batch,
                    task_type="SEMANTIC_SIMILARITY",
                )
                emb = np.array(result["embedding"], dtype=np.float32)
                # Gemini may return different dims; truncate/pad to 768
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                if emb.shape[1] > 768:
                    emb = emb[:, :768]
                elif emb.shape[1] < 768:
                    pad = np.zeros((emb.shape[0], 768 - emb.shape[1]), dtype=np.float32)
                    emb = np.hstack([emb, pad])
                embeddings.append(emb)
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait = 2 ** (attempt + 1)
                    print(f"  Gemini rate limit, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Gemini embedding error: {e}")
                    embeddings.append(np.zeros((len(batch), 768), dtype=np.float32))
                    break

    if embeddings:
        return np.vstack(embeddings)
    return np.zeros((len(headlines), 768), dtype=np.float32)


def apply_soft_zca_whitening(embeddings, alpha=0.05):
    """Apply Soft-ZCA whitening to fix BERT anisotropy (cosine sim collapse).

    Reference: "Isotropy Matters: Soft-ZCA Whitening of Embeddings"
    (arXiv 2411.17538). BERT mean-pooled embeddings cluster into a narrow
    cone (cosine sim ~1.0). Whitening restores isotropy and recovers
    discriminative signal.

    Args:
        embeddings: (N, D) array of raw BERT embeddings
        alpha: regularization strength (0=full ZCA, 1=identity); 0.05 is
               the documented sweet spot from the paper
    Returns:
        (N, D) whitened embeddings with restored variance
    """
    if embeddings.shape[0] < 2:
        return embeddings  # need ≥2 samples to compute covariance

    # Center
    mu = embeddings.mean(axis=0, keepdims=True)
    X = embeddings - mu

    # Covariance + regularized eigendecomposition
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)

    # Soft ZCA: blend whitening with identity by alpha
    # W = U * diag(eigvals^{-(1-alpha)/2}) * U^T
    inv_sqrt = eigvals ** (-(1 - alpha) / 2)
    W = eigvecs @ np.diag(inv_sqrt) @ eigvecs.T

    return (X @ W).astype(np.float32)


def get_finbert_embeddings(headlines, batch_size=None, device="cuda",
                            apply_whitening=True, pooling="mean"):
    """Extract 768-dim FinBERT embeddings from financial headlines.

    By default applies Soft-ZCA whitening to fix the documented BERT
    anisotropy problem (without whitening, all embeddings have cosine
    similarity ~1.0 — see arXiv 2411.17538).

    Batch size auto-scales: 256 on A100 (80GB), 64 on T4 (16GB).

    Args:
        headlines: list of headline strings
        batch_size: inference batch size (None = auto by VRAM)
        device: "cuda" or "cpu"
        apply_whitening: if True, apply Soft-ZCA whitening (recommended)
        pooling: "mean" (default), "cls", or "max"
    Returns:
        np.ndarray of shape (len(headlines), 768)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    if batch_size is None:
        if device == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
            batch_size = 256 if vram_gb >= 40 else 64
        else:
            batch_size = 32

    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        if pooling == "cls":
            emb = outputs.last_hidden_state[:, 0, :]
        elif pooling == "max":
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            hidden = outputs.last_hidden_state * mask + (1 - mask) * (-1e9)
            emb = hidden.max(dim=1).values
        else:  # mean
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

        embeddings.append(emb.cpu().numpy())

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    raw = np.vstack(embeddings) if embeddings else np.empty((0, 768))

    if apply_whitening and raw.shape[0] >= 2:
        return apply_soft_zca_whitening(raw, alpha=0.05)
    return raw


def aggregate_text_per_timestep(fb, method="norm_mean_max_concat"):
    """Aggregate per-asset text embeddings into a per-timestep feature vector.

    Critical fix for the anisotropy collapse: simple mean across n_assets
    of high-dim BERT embeddings collapses to ~zero, making every timestep
    look identical. Instead, L2-normalize each asset embedding first, then
    use [mean, max] concatenation to preserve per-asset variance signal.

    Args:
        fb: (n_dates, n_assets, D) per-asset text embeddings
        method:
            "mean" — legacy (broken): simple mean across assets, returns (N, D)
            "norm_mean" — L2-normalize first, then mean. Returns (N, D)
            "norm_mean_max_concat" — L2-norm, then [mean, max] concat. Returns (N, 2D)
    Returns:
        (n_dates, D_out) timestep-level text features
    """
    if method == "mean":
        return fb.mean(axis=1).astype(np.float32)

    norms = np.linalg.norm(fb, axis=-1, keepdims=True)
    fb_norm = fb / (norms + 1e-8)

    if method == "norm_mean":
        return fb_norm.mean(axis=1).astype(np.float32)

    # Default: norm_mean_max_concat preserves variance + extreme signals
    mean_part = fb_norm.mean(axis=1)
    max_part = fb_norm.max(axis=1)
    return np.concatenate([mean_part, max_part], axis=-1).astype(np.float32)


def get_finance_sentence_embeddings(headlines, device="cuda",
                                     model_name="FinLang/finance-embeddings-investopedia"):
    """Alternative: use a finance-domain sentence-transformer.

    FinLang/finance-embeddings-investopedia is fine-tuned from BAAI/bge-base-en-v1.5
    on Investopedia finance text. Built with sentence-transformers, so it
    natively avoids the BERT anisotropy problem.

    Use this as a drop-in replacement for get_finbert_embeddings() when
    text quality matters more than backwards compatibility with the
    ProsusAI/finbert sentiment classifier.

    Args:
        headlines: list of headline strings
        device: "cuda" or "cpu"
        model_name: HuggingFace model ID (default: FinLang investopedia)
    Returns:
        np.ndarray of shape (len(headlines), 768)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers required: pip install sentence-transformers"
        )

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        headlines,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
    )

    del model
    import torch
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings.astype(np.float32)


def _select_qwen_model():
    """Select best Qwen3 model for available VRAM.

    A100 80GB  -> Qwen3-32B  (4-bit ~20GB, best structured reasoning)
    T4/V100    -> Qwen3-8B   (4-bit ~5GB, good baseline)
    CPU        -> None        (skip)
    """
    import torch
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    vram_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
    if vram_gb >= 70:
        return "Qwen/Qwen3-32B"   # A100 80GB / H100: best reasoning + JSON
    elif vram_gb >= 12:
        return "Qwen/Qwen3-8B"    # A100 40GB / T4 / V100
    return None


def get_qwen3_sentiment(headlines_batch, device="cuda", max_new_tokens=256,
                        model_name=None):
    """Extract structured sentiment using Qwen3-32B (A100) or Qwen3-8B (T4).

    Auto-selects model based on GPU VRAM. Returns JSON with:
    sentiment (-1 to 1), risk_level, regime, key_factors, confidence.

    Args:
        headlines_batch: list of lists -- each inner list is headlines for one prompt
        device: "cuda" or "cpu"
        max_new_tokens: max generation length
        model_name: override model (None = auto-select by VRAM)
    Returns:
        list of dicts with structured sentiment data
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    if model_name is None:
        model_name = _select_qwen_model()
    if model_name is None:
        print("  No GPU available for Qwen3. Returning neutral sentiment.")
        return [{"sentiment": 0.0, "risk_level": "medium",
                 "regime": "transitioning", "key_factors": [],
                 "confidence": 0.0}] * len(headlines_batch)

    print(f"  Loading {model_name} (4-bit)...")

    # Free VRAM from any prior models (e.g. FinBERT) before loading
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"  {model_name} loaded. VRAM: {vram_used:.1f} GB")

    PROMPT_TEMPLATE = (
        "You are an expert financial analyst. Analyze these financial headlines "
        "and return ONLY valid JSON (no explanation, no markdown):\n\n"
        "Headlines:\n{headlines}\n\n"
        "Return exactly this JSON structure:\n"
        '{{"sentiment": <float -1.0 to 1.0>, '
        '"risk_level": "low" | "medium" | "high", '
        '"regime": "risk_on" | "risk_off" | "transitioning", '
        '"key_factors": [<up to 5 short strings>], '
        '"confidence": <float 0.0 to 1.0>}}'
    )

    results = []
    for headlines in headlines_batch:
        prompt = PROMPT_TEMPLATE.format(headlines="\n".join(headlines[:15]))
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.1, do_sample=True,
            )
        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Extract JSON from response (handle markdown code blocks)
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

        try:
            parsed = json.loads(text)
            # Validate expected fields
            parsed.setdefault("sentiment", 0.0)
            parsed.setdefault("risk_level", "medium")
            parsed.setdefault("regime", "transitioning")
            parsed.setdefault("key_factors", [])
            parsed.setdefault("confidence", 0.5)
            results.append(parsed)
        except json.JSONDecodeError:
            results.append({
                "sentiment": 0.0, "risk_level": "medium",
                "regime": "transitioning", "key_factors": [],
                "confidence": 0.0,
            })

    del model
    torch.cuda.empty_cache()

    return results


def sentiment_to_features(sentiment_data, feature_dim=16):
    """Convert structured sentiment output to a fixed-size feature vector.

    Args:
        sentiment_data: dict from get_qwen3_sentiment
        feature_dim: output dimension
    Returns:
        np.ndarray of shape (feature_dim,)
    """
    feat = np.zeros(feature_dim, dtype=np.float32)

    # Sentiment score
    feat[0] = sentiment_data.get("sentiment", 0.0)

    # Risk level encoding
    risk_map = {"low": -1.0, "medium": 0.0, "high": 1.0}
    feat[1] = risk_map.get(sentiment_data.get("risk_level", "medium"), 0.0)

    # Regime encoding (one-hot)
    regime = sentiment_data.get("regime", "transitioning")
    feat[2] = 1.0 if regime == "risk_on" else 0.0
    feat[3] = 1.0 if regime == "risk_off" else 0.0
    feat[4] = 1.0 if regime == "transitioning" else 0.0

    # Number of key factors as complexity signal
    n_factors = len(sentiment_data.get("key_factors", []))
    feat[5] = min(n_factors / 5.0, 1.0)

    # Confidence (from Qwen3-32B)
    feat[6] = sentiment_data.get("confidence", 0.5)

    return feat


def build_text_features(
    headlines_by_date_ticker,
    rebalance_dates,
    tickers,
    use_finbert=True,
    use_qwen=True,
    finbert_dim=768,
    sentiment_dim=16,
    device="cuda",
):
    """Build per-asset, per-date text feature matrix.

    Args:
        headlines_by_date_ticker: dict mapping (date, ticker) -> list of headlines
        rebalance_dates: list of rebalance dates
        tickers: list of tickers
        use_finbert: whether to compute FinBERT embeddings
        use_qwen: whether to compute Qwen3 sentiment
        finbert_dim: FinBERT embedding dimension
        sentiment_dim: structured sentiment feature dimension
        device: compute device
    Returns:
        dict with keys:
            "finbert": np.ndarray (n_dates, n_assets, finbert_dim) or None
            "sentiment": np.ndarray (n_dates, n_assets, sentiment_dim) or None
    """
    n_dates = len(rebalance_dates)
    n_assets = len(tickers)

    finbert_features = np.zeros((n_dates, n_assets, finbert_dim), dtype=np.float32) if use_finbert else None
    sentiment_features = np.zeros((n_dates, n_assets, sentiment_dim), dtype=np.float32) if use_qwen else None

    for i, rd in enumerate(rebalance_dates):
        for j, ticker in enumerate(tickers):
            key = (pd.Timestamp(rd) if not isinstance(rd, pd.Timestamp) else rd, ticker)
            headlines = headlines_by_date_ticker.get(key, [])
            if not headlines:
                continue

            if use_finbert:
                embs = get_finbert_embeddings(headlines, device=device)
                finbert_features[i, j] = embs.mean(axis=0)  # Average over headlines

            if use_qwen:
                sentiments = get_qwen3_sentiment([headlines], device=device)
                if sentiments:
                    sentiment_features[i, j] = sentiment_to_features(sentiments[0], sentiment_dim)

    return {"finbert": finbert_features, "sentiment": sentiment_features}


# Avoid circular import — only import pandas when needed
import pandas as pd
