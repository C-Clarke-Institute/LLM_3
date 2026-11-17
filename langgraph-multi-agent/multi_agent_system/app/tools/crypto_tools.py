from typing import Any

import httpx
from langchain_core.tools import tool

from app.core.config import settings


def _coingecko_headers() -> dict[str, str]:
    headers = {"accept": "application/json"}
    if settings.coingecko_api_key:
        # CoinGecko Demo API key header. Public endpoints may work without it.
        headers["x-cg-demo-api-key"] = settings.coingecko_api_key
    return headers


@tool
async def get_crypto_price(coin_ids: str, vs_currency: str = "usd") -> dict[str, Any]:
    """Get current cryptocurrency prices. Use CoinGecko IDs, e.g. bitcoin, ethereum.
    Pass multiple comma-separated IDs in coin_ids."""
    ids = ",".join(part.strip().lower() for part in coin_ids.split(",") if part.strip())
    if not ids:
        return {"error": "At least one CoinGecko coin ID is required."}
    url = f"{settings.coingecko_base_url.rstrip('/')}/simple/price"
    params = {"ids": ids, "vs_currencies": vs_currency.lower(), "include_24hr_change": "true"}
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.get(url, params=params, headers=_coingecko_headers())
            response.raise_for_status()
            return {"source": "CoinGecko", "data": response.json()}
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": f"CoinGecko price lookup failed: {type(exc).__name__}"}


@tool
async def get_crypto_market_information(
    coin_ids: str, vs_currency: str = "usd"
) -> dict[str, Any]:
    """Get market cap, rank, trading volume, current price and 24-hour change
    for comma-separated CoinGecko IDs such as bitcoin,ethereum."""
    ids = ",".join(part.strip().lower() for part in coin_ids.split(",") if part.strip())
    if not ids:
        return {"error": "At least one CoinGecko coin ID is required."}
    url = f"{settings.coingecko_base_url.rstrip('/')}/coins/markets"
    params = {
        "vs_currency": vs_currency.lower(),
        "ids": ids,
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "sparkline": "false",
    }
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.get(url, params=params, headers=_coingecko_headers())
            response.raise_for_status()
            return {"source": "CoinGecko", "data": response.json()}
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": f"CoinGecko market lookup failed: {type(exc).__name__}"}


@tool
def get_cryptocurrency_explanation(topic: str) -> str:
    """Return a short reference note about a cryptocurrency or blockchain concept.
    Use this for fundamentals, not live prices."""
    return (
        f"Explain {topic} using established cryptocurrency and blockchain concepts. "
        "This tool is an explanatory reference, not a source of live market data."
    )


CRYPTO_TOOLS = [
    get_crypto_price,
    get_crypto_market_information,
    get_cryptocurrency_explanation,
]
