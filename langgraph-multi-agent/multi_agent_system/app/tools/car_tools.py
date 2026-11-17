import asyncio
import re
from typing import Any
from urllib.parse import quote

import httpx
from langchain_core.tools import tool

from app.core.config import settings


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")


@tool
async def lookup_vehicle_listings(car_model: str) -> dict[str, Any]:
    """Search Sri Lankan vehicle listing pages for a model. Returns extracted page
    text and source URLs; listings are advertisements, not verified sale prices."""
    if not settings.vehicle_sources_enabled:
        return {"error": "Vehicle listing lookup is disabled in configuration."}
    model = _slug(car_model)
    urls = [
        f"https://patpat.lk/en/sri-lanka/vehicle/car/toyota/{quote(model)}",
        f"https://ikman.lk/en/ads?query={quote(car_model)}",
        f"https://riyasewana.com/search/{quote(model)}",
    ]
    try:
        from langchain_community.document_loaders import WebBaseLoader

        def load_pages():
            loader = WebBaseLoader(urls)
            return loader.load()

        docs = await asyncio.to_thread(load_pages)
        results = []
        for doc in docs:
            content = (doc.page_content or "").strip()
            results.append({
                "source": doc.metadata.get("source", "unknown"),
                "extracted_text": content[:9000],
            })
        return {
            "notice": "Advertised listing information may be outdated or inaccurate. "
                      "Verify with the seller before making a purchase.",
            "results": results,
        }
    except Exception as exc:
        return {
            "error": f"Vehicle listing sources could not be loaded ({type(exc).__name__}).",
            "sources": urls,
        }


@tool
async def get_vehicle_specifications(make: str, model: str, year: int | None = None) -> dict[str, Any]:
    """Look up vehicle specifications using the public NHTSA vPIC API.
    Data coverage is primarily vehicles identified in the US market."""
    url = "https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMakeYear/make/"
    params = {"format": "json"}
    if year is not None:
        params["year"] = year
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.get(
                f"{url}{quote(make)}", params=params
            )
            response.raise_for_status()
            payload = response.json()
        models = payload.get("Results", [])
        matches = [
            item for item in models
            if model.lower() in str(item.get("Model_Name", "")).lower()
        ]
        return {
            "source": "NHTSA vPIC",
            "make": make,
            "requested_model": model,
            "year": year,
            "matches": matches[:20],
            "notice": "NHTSA coverage is not a complete Sri Lankan vehicle-specification database.",
        }
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": f"Vehicle specification lookup failed ({type(exc).__name__})."}


@tool
async def get_fuel_economy_information(make: str, model: str, year: int | None = None) -> dict[str, Any]:
    """Search available Sri Lankan listings for fuel-economy claims for a vehicle.
    Report only values present in retrieved source text; do not estimate."""
    search = await lookup_vehicle_listings.ainvoke({"car_model": f"{make} {model} {year or ''}"})
    return {
        "source_lookup": search,
        "instruction": (
            "Extract fuel economy only if an explicit figure and unit appear in the "
            "source text. Otherwise report that no verified figure was found."
        ),
    }


@tool
async def compare_vehicles(vehicle_a: str, vehicle_b: str) -> dict[str, Any]:
    """Retrieve Sri Lankan listing evidence for two vehicles for a source-based comparison."""
    first, second = await asyncio.gather(
        lookup_vehicle_listings.ainvoke({"car_model": vehicle_a}),
        lookup_vehicle_listings.ainvoke({"car_model": vehicle_b}),
    )
    return {
        "vehicle_a": {"query": vehicle_a, "evidence": first},
        "vehicle_b": {"query": vehicle_b, "evidence": second},
        "notice": "Compare only facts present in the retrieved evidence; disclose missing data.",
    }


CAR_TOOLS = [
    lookup_vehicle_listings,
    get_vehicle_specifications,
    get_fuel_economy_information,
    compare_vehicles,
]
