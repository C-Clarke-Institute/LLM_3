from typing import Literal, List

from pydantic import BaseModel, Field

class MarketFactor(BaseModel):
    factor: str = Field(description="Name or a short description of the market factor influencing the coin"
                                    "eg : (ETF Approval, Trump truth social post)")
    impact: str = Field(description="Explanation of how this factor affects the coins price or sentiment")

class CryptoInsights(BaseModel):
    prediction: str = Field(description="specific prediction or insight about the coin's possible price"
                                        "movement")
    confidence: int = Field(..., ge=0, le=100, description="Confidence level (0 -100)"
                                                           "indicating how strongly "
                                                           "the prediction is supported")

class CoinMarketAnalysis(BaseModel):
    coin: str = Field(description="Cryptocurrency symbol or "
                                  "name being analyzed ex: (bitcoin, ethereum)")
    summary: str = Field(description="A short summary describing the current state of the market"
                                     "for this coin")
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(description="overall market sentiment for the coin")
    key_factors: List[MarketFactor] = Field(description="List of major factors currently influencing the"
                                                        "coin limit 3")
    insights: List[str] = Field(description="3 key insights and predictions with confidence score")

class CryptoAnalysisResponse(BaseModel):
    analysis : List[CoinMarketAnalysis] = Field(description="List of crypto analysis for each "
                                                            "cryptocurrency")

class CryptoRequest(BaseModel):
    coin_list: List[str] = Field(min_length=1)
