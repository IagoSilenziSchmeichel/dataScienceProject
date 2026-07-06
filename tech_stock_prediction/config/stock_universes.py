"""
Central stock universe definitions for paper trading.
"""

ORIGINAL_TECH = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "GOOGL",
    "META",
    "AMZN",
    "TSLA",
    "INTC",
    "ADBE",
]

TECH_NO_NVDA = [
    "AAPL",
    "MSFT",
    "AMD",
    "GOOGL",
    "META",
    "AMZN",
    "TSLA",
    "INTC",
    "ADBE",
]

NEW_TECH = [
    "PLTR",
    "SNOW",
    "CRWD",
    "NET",
    "NOW",
    "ORCL",
    "CSCO",
    "MU",
    "SMCI",
    "DELL",
]

DEFENSIVE_NON_TECH = [
    "WMT",
    "COST",
    "PG",
    "KO",
    "PEP",
    "JNJ",
    "MRK",
    "ABBV",
    "XOM",
    "CVX",
]

UNIVERSES = {
    "original_tech": ORIGINAL_TECH,
    "tech_no_nvda": TECH_NO_NVDA,
    "new_tech": NEW_TECH,
    "defensive_non_tech": DEFENSIVE_NON_TECH,
}

BENCHMARKS = {
    "original_tech": "QQQ",
    "tech_no_nvda": "QQQ",
    "new_tech": "QQQ",
    "defensive_non_tech": "SPY",
}


def list_available_universes() -> list[str]:
    """Return all supported universe names."""
    return list(UNIVERSES.keys())


def get_universe(name: str) -> list[str]:
    """Return tickers for one universe."""
    normalized_name = name.lower().strip()

    if normalized_name not in UNIVERSES:
        valid_names = ", ".join(list_available_universes())
        raise ValueError(
            f"Unknown universe '{name}'. Valid universe names are: {valid_names}"
        )

    return UNIVERSES[normalized_name]


def get_benchmark_for_universe(name: str) -> str:
    """Return benchmark ticker for one universe."""
    normalized_name = name.lower().strip()

    if normalized_name not in BENCHMARKS:
        valid_names = ", ".join(list_available_universes())
        raise ValueError(
            f"Unknown universe '{name}'. Valid universe names are: {valid_names}"
        )

    return BENCHMARKS[normalized_name]
