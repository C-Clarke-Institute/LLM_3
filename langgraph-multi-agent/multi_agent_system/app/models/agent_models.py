from enum import Enum
from typing import List, Literal

from pydantic import BaseModel, Field


class AgentName(str, Enum):
    CRYPTO = "crypto_agent"
    CAR_ANALYZER = "car_analyzer_agent"
    GENERAL = "general_agent"


class RouterDecision(BaseModel):
    """Structured output produced by the Router Agent."""

    user_question: str = Field(description="The original user question.")
    selected_agent: AgentName = Field(
        description="Exactly one agent that should answer the question."
    )
    routing_reason: str = Field(
        min_length=1,
        description="A concise explanation for the routing decision.",
    )


# Adapted from the uploaded crypto_models.py.
class MarketFactor(BaseModel):
    factor: str = Field(description="Market factor influencing the coin.")
    impact: str = Field(description="How the factor affects price or sentiment.")


class CryptoInsights(BaseModel):
    prediction: str = Field(description="A cautious possible price movement insight.")
    confidence: int = Field(ge=0, le=100, description="Confidence from 0 to 100.")


class CoinMarketAnalysis(BaseModel):
    coin: str
    summary: str
    sentiment: Literal["bullish", "bearish", "neutral"]
    key_factors: List[MarketFactor] = Field(default_factory=list, max_length=3)
    insights: List[str] = Field(default_factory=list, max_length=3)


class CryptoAnalysisResponse(BaseModel):
    analysis: List[CoinMarketAnalysis]


class CryptoRequest(BaseModel):
    coin_list: List[str] = Field(min_length=1)
