"""
Alpaca Paper Trading configuration.

API keys are loaded from environment variables or a local .env file. Keys are
never hardcoded in the project.
"""

from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False


DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"


@dataclass
class AlpacaSettings:
    api_key_id: str
    api_secret_key: str
    api_base_url: str
    dry_run: bool = True
    top_k: int = 1
    order_type: str = "market"
    time_in_force: str = "day"


def parse_bool(value: str | None, default: bool) -> bool:
    """Parse common string values into bool."""
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_alpaca_settings(
    *,
    require_keys: bool = True,
    dry_run_override: bool | None = None,
    top_k_override: int | None = None,
) -> AlpacaSettings:
    """
    Load Alpaca settings from .env and environment variables.

    In signals-only mode, use require_keys=False because no Alpaca connection is
    needed.
    """
    load_dotenv()

    api_key_id = os.getenv("APCA_API_KEY_ID", "").strip()
    api_secret_key = os.getenv("APCA_API_SECRET_KEY", "").strip()
    api_base_url = os.getenv("APCA_API_BASE_URL", DEFAULT_BASE_URL).strip()

    if require_keys:
        missing = []
        if not api_key_id:
            missing.append("APCA_API_KEY_ID")
        if not api_secret_key:
            missing.append("APCA_API_SECRET_KEY")

        if missing:
            raise RuntimeError(
                "Missing Alpaca API configuration: "
                + ", ".join(missing)
                + ". Add them to a local .env file or environment variables."
            )

    if "paper-api.alpaca.markets" not in api_base_url:
        raise RuntimeError(
            "Only Alpaca Paper Trading is allowed. "
            f"APCA_API_BASE_URL must point to paper-api.alpaca.markets, got: {api_base_url}"
        )

    dry_run = parse_bool(os.getenv("DRY_RUN"), default=True)
    if dry_run_override is not None:
        dry_run = dry_run_override

    top_k = int(os.getenv("TOP_K", "1"))
    if top_k_override is not None:
        top_k = top_k_override

    return AlpacaSettings(
        api_key_id=api_key_id,
        api_secret_key=api_secret_key,
        api_base_url=api_base_url,
        dry_run=dry_run,
        top_k=top_k,
        order_type=os.getenv("ORDER_TYPE", "market").strip().lower(),
        time_in_force=os.getenv("TIME_IN_FORCE", "day").strip().lower(),
    )
