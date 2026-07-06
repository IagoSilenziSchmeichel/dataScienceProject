"""
Small wrapper around alpaca-py for Paper Trading.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from alpaca_trading.alpaca_config import AlpacaSettings


class AlpacaPaperClient:
    """Alpaca Paper Trading client with a safe dry-run mode."""

    def __init__(self, settings: AlpacaSettings):
        self.settings = settings
        self.trading_client = None

        if settings.api_key_id and settings.api_secret_key:
            self.trading_client = self._create_trading_client()

    def _create_trading_client(self):
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as error:
            raise RuntimeError(
                "alpaca-py is not installed. Run: pip install alpaca-py"
            ) from error

        return TradingClient(
            api_key=self.settings.api_key_id,
            secret_key=self.settings.api_secret_key,
            paper=True,
            url_override=self.settings.api_base_url,
        )

    def get_account_summary(self) -> dict[str, Any]:
        """Return cash and portfolio value."""
        if self.settings.dry_run and self.trading_client is None:
            return {
                "portfolio_value": 100000.0,
                "cash": 100000.0,
                "buying_power": 100000.0,
                "dry_run": True,
                "note": "Simulated account values because DRY_RUN=True.",
            }

        account = self.trading_client.get_account()
        return {
            "portfolio_value": float(account.portfolio_value),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "dry_run": self.settings.dry_run,
        }

    def get_positions(self) -> list[dict[str, Any]]:
        """Return current Alpaca positions."""
        if self.settings.dry_run and self.trading_client is None:
            return []

        positions = self.trading_client.get_all_positions()
        rows = []

        for position in positions:
            rows.append(
                {
                    "ticker": position.symbol,
                    "quantity": float(position.qty),
                    "entry_price": float(position.avg_entry_price),
                    "current_price": float(position.current_price),
                    "market_value": float(position.market_value),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                }
            )

        return rows

    def submit_order(
        self,
        *,
        ticker: str,
        side: str,
        quantity: float | None = None,
        notional: float | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """
        Submit or simulate a market order.

        In dry-run mode, no Alpaca order is sent.
        """
        order_preview = {
            "ticker": ticker,
            "side": side,
            "quantity": quantity,
            "notional": notional,
            "order_type": self.settings.order_type,
            "time_in_force": self.settings.time_in_force,
            "dry_run": self.settings.dry_run,
            "alpaca_order_id": "",
            "reason": reason,
        }

        if self.settings.dry_run:
            print(f"DRY RUN order: {order_preview}")
            return order_preview

        try:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest
        except ImportError as error:
            raise RuntimeError(
                "alpaca-py is not installed. Run: pip install alpaca-py"
            ) from error

        if self.settings.order_type != "market":
            raise ValueError("Only market orders are supported in this project.")

        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif_enum = TimeInForce.DAY

        if self.settings.time_in_force != "day":
            raise ValueError("Only TIME_IN_FORCE=day is supported in this project.")

        request = MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            notional=notional,
            side=side_enum,
            time_in_force=tif_enum,
        )
        order = self.trading_client.submit_order(order_data=request)
        order_preview["alpaca_order_id"] = str(order.id)
        order_preview["alpaca_order_status"] = str(order.status)

        return order_preview

    def as_dict(self) -> dict[str, Any]:
        """Return settings without secret values."""
        data = asdict(self.settings)
        data["api_secret_key"] = "***"
        return data
