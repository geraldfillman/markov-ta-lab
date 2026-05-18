"""Financial Modeling Prep REST client for market data ingestion."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


FMP_BASE_URL = "https://financialmodelingprep.com/stable"
FMP_API_KEY_ENV_NAMES = ("FMP_API_KEY", "FINANCIAL_MODELING_PREP_API_KEY")
FMP_OHLCV_FIELDS = {
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}

Transport = Callable[[str], Any]


class FMPError(RuntimeError):
    """Raised when FMP configuration or API responses cannot be used safely."""


class FMPClient:
    """Small FMP REST client focused on repeatable market-data ingestion."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = FMP_BASE_URL,
        transport: Transport | None = None,
    ) -> None:
        self.api_key = api_key or load_fmp_api_key()
        self.base_url = base_url.rstrip("/")
        self.transport = transport

    def __repr__(self) -> str:
        return f"FMPClient(base_url={self.base_url!r}, api_key=<redacted>)"

    def fetch_historical_eod(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily EOD OHLCV from FMP and normalize to the project schema."""
        records = self._get_json(
            "historical-price-eod/full",
            {
                "symbol": symbol.upper().strip(),
                "from": start,
                "to": end,
            },
        )
        return normalize_fmp_ohlcv(_extract_records(records), symbol)

    def _get_json(self, endpoint: str, params: Mapping[str, str]) -> Any:
        query = urlencode({**params, "apikey": self.api_key})
        url = f"{self.base_url}/{endpoint.lstrip('/')}?{query}"

        if self.transport is not None:
            return self.transport(url)

        try:
            with urlopen(url, timeout=30) as response:  # noqa: S310 - user-configured market data endpoint
                payload = response.read()
        except HTTPError as exc:
            raise FMPError(f"FMP request failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise FMPError("FMP request failed before a response was received") from exc

        try:
            return json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise FMPError("FMP returned invalid JSON") from exc


def load_fmp_api_key(
    dotenv_path: str | Path = ".env",
    environ: Mapping[str, str] | None = None,
) -> str:
    """Load the FMP key from environment variables or a local `.env` file."""
    env = os.environ if environ is None else environ
    for name in FMP_API_KEY_ENV_NAMES:
        value = env.get(name)
        if value:
            return value.strip()

    path = Path(dotenv_path)
    if path.exists():
        variables = _parse_dotenv(path)
        for name in FMP_API_KEY_ENV_NAMES:
            value = variables.get(name)
            if value:
                return value.strip()

    names = " or ".join(FMP_API_KEY_ENV_NAMES)
    raise FMPError(f"Missing FMP API key. Set {names} in the environment or .env file.")


def normalize_fmp_ohlcv(records: Sequence[Mapping[str, Any]], symbol: str | None = None) -> pd.DataFrame:
    """Normalize FMP OHLCV records to Date-indexed Open/High/Low/Close/Volume columns."""
    if not records:
        name = f" for {symbol}" if symbol else ""
        raise ValueError(f"No FMP OHLCV rows returned{name}")

    frame = pd.DataFrame(records)
    missing_fields = [field for field in FMP_OHLCV_FIELDS if field not in frame.columns]
    if missing_fields:
        raise ValueError(f"Missing required FMP OHLCV fields: {missing_fields}")

    result = frame.loc[:, list(FMP_OHLCV_FIELDS)].rename(columns=FMP_OHLCV_FIELDS)
    result["Date"] = pd.to_datetime(result["Date"])
    result = result.set_index("Date")
    result.index.name = "Date"
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    return result.dropna(subset=numeric_columns)


def download_fmp_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV for one symbol from FMP."""
    return FMPClient().fetch_historical_eod(symbol, start, end)


def _extract_records(payload: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("historical", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise FMPError("FMP response did not contain a usable OHLCV record list")


def _parse_dotenv(path: Path) -> dict[str, str]:
    variables: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        clean_name = name.strip()
        clean_value = value.strip().strip('"').strip("'")
        if clean_name:
            variables[clean_name] = clean_value
    return variables

