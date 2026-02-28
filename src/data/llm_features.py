"""LLM feature extraction: FinBERT embeddings + Qwen3-8B structured sentiment.

Designed to run on Google Colab free T4 (15GB VRAM):
- FinBERT: ~2GB VRAM, batch inference
- Qwen3-8B (4-bit): ~5GB VRAM, structured JSON output
"""

import json
import os

import numpy as np


def get_finbert_embeddings(headlines, batch_size=64, device="cuda"):
    """Extract 768-dim FinBERT embeddings from financial headlines.

    Uses mean pooling over token representations for dense embeddings.

    Args:
        headlines: list of headline strings
        batch_size: inference batch size (64 fits comfortably on T4)
        device: "cuda" or "cpu"
    Returns:
        np.ndarray of shape (len(headlines), 768)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

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
        # Mean pooling over tokens (excluding padding)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings.append(emb.cpu().numpy())

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return np.vstack(embeddings) if embeddings else np.empty((0, 768))


def get_qwen3_sentiment(headlines_batch, device="cuda", max_new_tokens=200):
    """Extract structured sentiment from headlines using Qwen3-8B (4-bit).

    Returns JSON with: sentiment (-1 to 1), risk_level, regime, key_factors.

    Args:
        headlines_batch: list of lists — each inner list is headlines for one prompt
        device: "cuda" or "cpu"
        max_new_tokens: max generation length
    Returns:
        list of dicts with structured sentiment data
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    PROMPT_TEMPLATE = (
        "Analyze these financial headlines and return ONLY valid JSON "
        "(no explanation):\n\nHeadlines:\n{headlines}\n\n"
        'Return: {{"sentiment": float(-1 to 1), '
        '"risk_level": "low"|"medium"|"high", '
        '"regime": "risk_on"|"risk_off"|"transitioning", '
        '"key_factors": [str]}}'
    )

    results = []
    for headlines in headlines_batch:
        prompt = PROMPT_TEMPLATE.format(headlines="\n".join(headlines[:10]))
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.1, do_sample=True,
            )
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            parsed = json.loads(response.strip())
            results.append(parsed)
        except json.JSONDecodeError:
            results.append({
                "sentiment": 0.0, "risk_level": "medium",
                "regime": "transitioning", "key_factors": [],
            })

    del model
    if device == "cuda":
        import torch
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
