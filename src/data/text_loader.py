"""Financial news headline loading for LLM feature extraction.

Sources:
- Yahoo Finance news headlines (via yfinance)
- HuggingFace datasets: financial_phrasebank, FiQA
- SEC EDGAR filings (10-K/10-Q summaries)
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def load_yfinance_news(tickers, start, end, max_headlines=100):
    """Load recent news headlines for each ticker via yfinance.

    Args:
        tickers: list of ticker symbols
        start: start date string
        end: end date string
        max_headlines: max headlines per ticker
    Returns:
        DataFrame with columns [ticker, date, headline]
    """
    import yfinance as yf

    records = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            news = t.news
            if news is None:
                continue
            for item in news[:max_headlines]:
                # Handle both old and new yfinance API formats
                content = item.get("content", item)  # new format nests under 'content'
                title = content.get("title", "") if isinstance(content, dict) else item.get("title", "")
                # Parse publication date
                pub_str = content.get("pubDate", "") if isinstance(content, dict) else ""
                if pub_str:
                    try:
                        pub_date = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).strftime("%Y-%m-%d")
                    except Exception:
                        pub_date = datetime.now().strftime("%Y-%m-%d")
                elif "providerPublishTime" in item:
                    pub_date = datetime.fromtimestamp(item["providerPublishTime"]).strftime("%Y-%m-%d")
                else:
                    pub_date = datetime.now().strftime("%Y-%m-%d")
                provider = content.get("provider", {}) if isinstance(content, dict) else {}
                source = provider.get("displayName", "") if isinstance(provider, dict) else item.get("publisher", "")
                if title:
                    records.append({
                        "ticker": ticker,
                        "date": pub_date,
                        "headline": title,
                        "source": source,
                    })
        except Exception:
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df


def load_rss_headlines(tickers=None, max_per_feed=50):
    """Load financial headlines from public RSS feeds.

    Sources: Reuters, CNBC, MarketWatch, Yahoo Finance RSS.
    Returns DataFrame with [ticker, date, headline, source].
    """
    try:
        import feedparser
    except ImportError:
        print("Install feedparser: pip install feedparser")
        return pd.DataFrame(columns=["ticker", "date", "headline", "source"])

    feeds = {
        "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
        "CNBC Top News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    }

    records = []
    for source_name, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get("title", "")
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub:
                    pub_date = datetime(*pub[:6]).strftime("%Y-%m-%d")
                else:
                    pub_date = datetime.now().strftime("%Y-%m-%d")

                if not title:
                    continue

                # Map headline to tickers via keyword matching
                matched_tickers = _match_headline_to_tickers(title, tickers) if tickers else ["MARKET"]
                for ticker in matched_tickers:
                    records.append({
                        "ticker": ticker,
                        "date": pub_date,
                        "headline": title,
                        "source": source_name,
                    })
        except Exception:
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _match_headline_to_tickers(headline, tickers):
    """Match a headline to relevant tickers via keyword matching."""
    if tickers is None:
        return ["MARKET"]

    headline_upper = headline.upper()
    # Keyword mappings for common ETFs
    keywords = {
        "SPY": ["S&P", "SPY", "SP500", "S&P 500"],
        "QQQ": ["NASDAQ", "QQQ", "TECH STOCKS"],
        "EFA": ["INTERNATIONAL", "EAFE", "EUROPE", "DEVELOPED"],
        "EEM": ["EMERGING", "EM MARKET"],
        "GLD": ["GOLD", "PRECIOUS METAL"],
        "USO": ["OIL", "CRUDE", "PETROLEUM", "OPEC"],
        "TLT": ["TREASURY", "BOND", "YIELD", "RATE"],
        "UNG": ["NATURAL GAS", "NATGAS"],
        "EWZ": ["BRAZIL"],
        "FXI": ["CHINA", "CHINESE"],
        "EWY": ["KOREA", "KOREAN"],
        "INDA": ["INDIA", "INDIAN"],
        "XLF": ["BANK", "FINANCIAL"],
        "XLK": ["TECH", "TECHNOLOGY", "SOFTWARE"],
        "XLE": ["ENERGY", "OIL"],
        "DBA": ["AGRICULTURE", "CROP", "FARM"],
        "SLV": ["SILVER"],
        "CPER": ["COPPER"],
        "WEAT": ["WHEAT"],
        "CORN": ["CORN"],
    }

    matched = []
    tickers_list = list(tickers.values()) if isinstance(tickers, dict) else list(tickers)
    for ticker in tickers_list:
        if ticker in headline_upper:
            matched.append(ticker)
        elif ticker in keywords:
            for kw in keywords[ticker]:
                if kw in headline_upper:
                    matched.append(ticker)
                    break

    return matched if matched else ["MARKET"]


def load_all_headlines(tickers, start, end, max_headlines=100, use_rss=True,
                       use_phrasebank=True, use_fiqa=True, use_gdelt=True,
                       gdelt_max_per_ticker=500, gdelt_chunk_months=3):
    """Load headlines from all available sources and combine.

    Target: 5000+ unique dated headlines for robust FinBERT embeddings.

    Sources (in order of historical coverage):
      1. GDELT DOC 2.0 — 10-year historical coverage, free, no API key
      2. yfinance — recent per-ticker news
      3. RSS feeds — current headlines from Reuters, CNBC, MarketWatch
      4. Financial PhraseBank — static academic dataset (undated)
      5. FiQA — static financial QA sentiment dataset (undated)
    """
    all_dfs = []

    # GDELT historical headlines (primary source for dated coverage)
    if use_gdelt:
        print("  Loading GDELT historical headlines...")
        try:
            from src.data.gdelt_loader import load_gdelt_headlines
            gdelt_df = load_gdelt_headlines(
                tickers, start, end,
                max_per_ticker=gdelt_max_per_ticker,
                chunk_months=gdelt_chunk_months,
            )
            if not gdelt_df.empty:
                # Standardize columns
                if "tone" in gdelt_df.columns:
                    gdelt_df = gdelt_df.drop(columns=["tone"])
                print(f"    GDELT: {len(gdelt_df)} headlines")
                all_dfs.append(gdelt_df)
        except Exception as e:
            print(f"    GDELT failed: {e}")

    # Yahoo Finance per-ticker news
    print("  Loading yfinance headlines...")
    yf_df = load_yfinance_news(tickers, start, end, max_headlines=max_headlines)
    if not yf_df.empty:
        print(f"    yfinance: {len(yf_df)} headlines")
        all_dfs.append(yf_df)

    # RSS feeds
    if use_rss:
        print("  Loading RSS headlines...")
        rss_df = load_rss_headlines(tickers)
        if not rss_df.empty:
            print(f"    RSS: {len(rss_df)} headlines")
            all_dfs.append(rss_df)

    # Financial PhraseBank (static dataset — generic financial sentences)
    if use_phrasebank:
        print("  Loading Financial PhraseBank...")
        fpb = load_financial_phrasebank()
        if not fpb.empty:
            fpb_df = pd.DataFrame({
                "ticker": "MARKET",
                "date": pd.Timestamp.now(),
                "headline": fpb["sentence"],
                "source": "FinancialPhraseBank",
            })
            print(f"    PhraseBank: {len(fpb_df)} sentences")
            all_dfs.append(fpb_df)

    # FiQA sentiment dataset
    if use_fiqa:
        print("  Loading FiQA dataset...")
        fiqa = load_fiqa_sentiment()
        if not fiqa.empty:
            fiqa_df = pd.DataFrame({
                "ticker": "MARKET",
                "date": pd.Timestamp.now(),
                "headline": fiqa["sentence"],
                "source": "FiQA",
            })
            print(f"    FiQA: {len(fiqa_df)} sentences")
            all_dfs.append(fiqa_df)

    if not all_dfs:
        return pd.DataFrame(columns=["ticker", "date", "headline", "source"])

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["headline"])
    print(f"  Total unique headlines: {len(combined)}")
    return combined


def load_financial_phrasebank(split="sentences_allagree"):
    """Load the Financial PhraseBank dataset from HuggingFace.

    Args:
        split: agreement level — "sentences_allagree", "sentences_75agree", etc.
    Returns:
        DataFrame with columns [sentence, label]
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("financial_phrasebank", split)
        df = pd.DataFrame(ds["train"])
        df.columns = ["sentence", "label"]
        return df
    except Exception as e:
        print(f"Financial PhraseBank unavailable: {e}")
        return pd.DataFrame(columns=["sentence", "label"])


def load_fiqa_sentiment():
    """Load the FiQA sentiment dataset from HuggingFace.

    Returns:
        DataFrame with columns [sentence, sentiment_score]
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("pauri32/fiqa-2018")
        records = []
        for split_name in ["train", "validation", "test"]:
            if split_name in ds:
                for row in ds[split_name]:
                    records.append({
                        "sentence": row.get("sentence", ""),
                        "sentiment_score": row.get("sentiment_score", 0.0),
                    })
        return pd.DataFrame(records)
    except ImportError:
        print("Install datasets: pip install datasets")
        return pd.DataFrame(columns=["sentence", "sentiment_score"])


def aggregate_headlines_for_rebalance(
    headlines_df, rebalance_dates, tickers, lookback_days=30,
):
    """Aggregate headlines into per-asset, per-rebalance-date groups.

    For each (rebalance_date, ticker), collects all headlines from the
    lookback window. This creates the input for LLM feature extraction.

    Args:
        headlines_df: DataFrame with [ticker, date, headline]
        rebalance_dates: list of rebalance dates
        tickers: list of tickers
        lookback_days: number of days to look back for headlines
    Returns:
        dict mapping (date, ticker) -> list of headline strings
    """
    if headlines_df.empty:
        return {}

    headlines_df = headlines_df.copy()
    headlines_df["date"] = pd.to_datetime(headlines_df["date"])
    result = {}

    for rd in rebalance_dates:
        rd = pd.Timestamp(rd)
        window_start = rd - timedelta(days=lookback_days)
        window = headlines_df[
            (headlines_df["date"] >= window_start)
            & (headlines_df["date"] <= rd)
        ]
        for ticker in tickers:
            ticker_headlines = window[window["ticker"] == ticker]["headline"].tolist()
            if ticker_headlines:
                result[(rd, ticker)] = ticker_headlines

    return result


def create_text_features_placeholder(n_dates, n_assets, text_dim=768):
    """Create zero-valued text feature placeholder when no headlines available.

    Used for ablation A1 (price-only, no LLM features).
    """
    return np.zeros((n_dates, n_assets, text_dim), dtype=np.float32)
