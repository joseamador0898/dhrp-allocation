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


def get_finbert_embeddings(headlines, batch_size=None, device="cuda"):
    """Extract 768-dim FinBERT embeddings from financial headlines.

    Uses mean pooling over token representations for dense embeddings.
    Batch size auto-scales: 256 on A100 (80GB), 64 on T4 (16GB).

    Args:
        headlines: list of headline strings
        batch_size: inference batch size (None = auto by VRAM)
        device: "cuda" or "cpu"
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
        # Mean pooling over tokens (excluding padding)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings.append(emb.cpu().numpy())

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return np.vstack(embeddings) if embeddings else np.empty((0, 768))


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
    if vram_gb >= 40:
        return "Qwen/Qwen3-32B"   # A100/H100: best reasoning + JSON output
    elif vram_gb >= 12:
        return "Qwen/Qwen3-8B"    # T4/V100: good baseline
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
