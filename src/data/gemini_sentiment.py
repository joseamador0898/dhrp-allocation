"""Gemini-powered structured financial sentiment extraction.

Replaces FinBERT + Qwen3-8B with Gemini API calls:
- 6-dimensional structured output per headline (vs FinBERT's 3-class scalar)
- Schema-enforced JSON (vs Qwen3's free-text parsing)
- Zero GPU compute (API calls)
- Works with any Gemini model (3.1 Pro, 3.1 Flash-Lite, 2.5 Flash, etc.)

Usage:
    from src.data.gemini_sentiment import extract_sentiment_batch
    features = extract_sentiment_batch(headlines_df, api_key=..., model='gemini-3.1-pro-preview')
    # features: dict[ticker, np.ndarray of shape (6,)]
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel


class HeadlineSentiment(BaseModel):
    """Structured sentiment label for a single financial headline."""
    sentiment_score: float       # -1.0 (very bearish) to 1.0 (very bullish)
    risk_level: str              # "low", "medium", "high", "extreme"
    regime: str                  # "risk_on", "risk_off", "neutral", "crisis"
    confidence: float            # 0.0 to 1.0
    sector_impact: str           # "broad_market", "sector_specific", "macro", "idiosyncratic"
    rate_sensitivity: float      # -1.0 (benefits from rate cuts) to 1.0 (benefits from rate hikes)


# Rebuild for Pydantic + Python 3.14 compatibility
HeadlineSentiment.model_rebuild()


# Encode categorical fields as floats for the feature vector
RISK_LEVEL_MAP = {"low": 0.0, "medium": 0.33, "high": 0.67, "extreme": 1.0}
REGIME_MAP = {"risk_on": 1.0, "neutral": 0.0, "risk_off": -0.5, "crisis": -1.0}
SECTOR_MAP = {"idiosyncratic": 0.0, "sector_specific": 0.33, "macro": 0.67, "broad_market": 1.0}


def sentiment_to_vector(s: HeadlineSentiment) -> np.ndarray:
    """Convert a HeadlineSentiment to a 6-dimensional float vector."""
    return np.array([
        s.sentiment_score,
        RISK_LEVEL_MAP.get(s.risk_level, 0.5),
        REGIME_MAP.get(s.regime, 0.0),
        s.confidence,
        SECTOR_MAP.get(s.sector_impact, 0.5),
        s.rate_sensitivity,
    ], dtype=np.float32)


GEMINI_FEATURE_DIM = 6  # number of features per headline


def _build_prompt(headlines: list[str]) -> str:
    """Build the prompt for batch headline analysis."""
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    return (
        "You are a quantitative financial analyst. Analyze the financial sentiment "
        "of each headline below for portfolio allocation purposes. For each headline, "
        "assess: (1) overall sentiment score from -1.0 (very bearish) to 1.0 (very "
        "bullish), (2) risk level, (3) market regime implied, (4) your confidence in "
        "the assessment, (5) whether the impact is broad-market, sector-specific, "
        "macro, or idiosyncratic, and (6) rate sensitivity from -1.0 (benefits from "
        "rate cuts) to 1.0 (benefits from rate hikes).\n\n"
        f"Headlines:\n{numbered}"
    )


def _call_gemini(
    client,
    headlines: list[str],
    model: str = "gemini-3.1-pro-preview",
    max_retries: int = 3,
) -> list[HeadlineSentiment]:
    """Call Gemini API with structured output for a batch of headlines."""
    from google import genai

    prompt = _build_prompt(headlines)

    # Raw JSON schema (avoids Pydantic + Python 3.14 Literal incompatibility
    # with the Gemini SDK's schema transformer)
    raw_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "sentiment_score": {"type": "number"},
                "risk_level": {"type": "string", "enum": ["low", "medium", "high", "extreme"]},
                "regime": {"type": "string", "enum": ["risk_on", "risk_off", "neutral", "crisis"]},
                "confidence": {"type": "number"},
                "sector_impact": {"type": "string", "enum": ["broad_market", "sector_specific", "macro", "idiosyncratic"]},
                "rate_sensitivity": {"type": "number"},
            },
            "required": ["sentiment_score", "risk_level", "regime", "confidence", "sector_impact", "rate_sensitivity"],
        },
    }

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=raw_schema,
                    temperature=0.1,
                ),
            )
            results = json.loads(response.text)
            return [HeadlineSentiment(**r) for r in results]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Gemini error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Gemini FAILED after {max_retries} attempts: {e}")
                # Return neutral defaults
                return [HeadlineSentiment(
                    sentiment_score=0.0, risk_level="medium", regime="neutral",
                    confidence=0.0, sector_impact="broad_market", rate_sensitivity=0.0,
                ) for _ in headlines]


def extract_sentiment_batch(
    headlines_df,
    api_key: str | None = None,
    model: str = "gemini-3.1-pro-preview",
    max_headlines_per_ticker: int = 15,
    batch_size: int = 10,
    cache_path: str | None = None,
    delay_between_calls: float = 1.0,
) -> dict[str, np.ndarray]:
    """Extract structured sentiment features for all tickers using Gemini API.

    Args:
        headlines_df: DataFrame with columns ['ticker', 'headline']
        api_key: Gemini API key (reads from GOOGLE_API_KEY env var if None)
        model: Gemini model name
        max_headlines_per_ticker: max headlines to analyze per ticker
        batch_size: headlines per API call (Gemini handles multi-headline prompts)
        cache_path: path to cache JSON (avoids re-calling API on re-runs)
        delay_between_calls: seconds between API calls (rate limiting)

    Returns:
        dict mapping ticker -> np.ndarray of shape (GEMINI_FEATURE_DIM,)
        Each vector is the MEAN of per-headline sentiment vectors for that ticker.
    """
    from google import genai

    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("No Gemini API key. Set GOOGLE_API_KEY or pass api_key=")

    # Load cache if exists
    cache_file = Path(cache_path) if cache_path else Path("results/features/gemini_sentiment.json")
    cached = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"  Loaded {len(cached)} cached ticker sentiments from {cache_file}")
        except Exception:
            cached = {}

    client = genai.Client(api_key=api_key)

    # Group headlines by ticker
    ticker_headlines = (
        headlines_df.groupby("ticker")["headline"]
        .apply(lambda x: list(x)[:max_headlines_per_ticker])
        .to_dict()
    )

    results = {}
    n_api_calls = 0
    n_cached = 0

    for ticker, hdls in sorted(ticker_headlines.items()):
        if len(hdls) < 3:
            continue

        # Check cache
        if ticker in cached:
            results[ticker] = np.array(cached[ticker], dtype=np.float32)
            n_cached += 1
            continue

        # Call Gemini in batches
        all_vectors = []
        for i in range(0, len(hdls), batch_size):
            batch = hdls[i:i + batch_size]
            sentiments = _call_gemini(client, batch, model=model)
            for s in sentiments:
                all_vectors.append(sentiment_to_vector(s))
            n_api_calls += 1
            if delay_between_calls > 0:
                time.sleep(delay_between_calls)

        if all_vectors:
            mean_vec = np.mean(all_vectors, axis=0).astype(np.float32)
            results[ticker] = mean_vec
            cached[ticker] = mean_vec.tolist()
            print(f"  {ticker}: {len(all_vectors)} headlines -> "
                  f"sentiment={mean_vec[0]:.2f}, risk={mean_vec[1]:.2f}, "
                  f"regime={mean_vec[2]:.2f}, rate_sens={mean_vec[5]:.2f}")

    # Save cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cached, f, indent=2)
    print(f"  Cached {len(cached)} tickers to {cache_file}")
    print(f"  API calls: {n_api_calls}, cached: {n_cached}")

    return results


def build_gemini_text_tensor(
    sentiment_by_ticker: dict[str, np.ndarray],
    n_samples: int,
    tickers: list[str],
) -> np.ndarray:
    """Build a (n_samples, GEMINI_FEATURE_DIM) tensor from per-ticker sentiment.

    The sentiment is STATIC per ticker (same value across all time steps),
    but the DHRP gating network can learn to route based on the cross-sectional
    dispersion of sentiment across assets at each timestep.

    For time-varying sentiment, call extract_sentiment_batch with date-windowed
    headlines.

    Args:
        sentiment_by_ticker: dict from extract_sentiment_batch()
        n_samples: number of time steps in the training dataset
        tickers: ordered list of ticker symbols matching the price DataFrame columns

    Returns:
        np.ndarray of shape (n_samples, GEMINI_FEATURE_DIM)
        The per-ticker vectors are averaged across tickers to produce a
        global sentiment signal per timestep (consistent with the existing
        LLM-DHRP pipeline that averages FinBERT embeddings across assets).
    """
    # Average across all tickers to get a single global sentiment vector
    vecs = []
    for t in tickers:
        if t in sentiment_by_ticker:
            vecs.append(sentiment_by_ticker[t])
        else:
            vecs.append(np.zeros(GEMINI_FEATURE_DIM, dtype=np.float32))

    global_vec = np.mean(vecs, axis=0).astype(np.float32)
    # Tile across all samples (static sentiment — same for all timesteps)
    return np.tile(global_vec, (n_samples, 1))
