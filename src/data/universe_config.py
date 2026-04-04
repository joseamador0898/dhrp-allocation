"""Per-universe data modality configuration.

Defines which data sources each asset-class universe receives during
LLM-DHRP training and backtesting.  DM gets full US FRED macro;
EM and Commodities get text-only conditioning (no US macro).
"""

UNIVERSE_DATA_CONFIG = {
    "DM": {
        "use_text": True,
        "use_macro": True,
        "macro_source": "fred_us",
        "spgci": False,
    },
    "EM": {
        "use_text": True,
        "use_macro": False,
        "macro_source": None,
        "spgci": False,
    },
    "Commodities": {
        "use_text": True,
        "use_macro": False,
        "macro_source": None,
        "spgci": True,
    },
}
