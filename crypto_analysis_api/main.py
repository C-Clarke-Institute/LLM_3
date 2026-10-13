import json
import os
from typing import List

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from crypto_models import CryptoRequest, CryptoAnalysisResponse

load_dotenv()

app = FastAPI(title="Crypto analysis API")

COINGECKO_URL = os.getenv("COINGECKO_URL")

@tool
async def get_crypto_insights(coin_list: List[str]):
    params = {
        "ids" : ",".join(coin_list),
        "vs_currency" : "usd"
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(COINGECKO_URL, params=params)
        return resp.json()

response_prompt = ChatPromptTemplate.from_template(
    """
    You are a CryptoAnalys AI - a professional crypto market analyst
    You will be given recent market data for several crypto currencies (price, market cap, volume
    24h change)
    
    {market_data}
    
    provide a comprehensive analysis based on the provided metrics
    """
)

@app.post("/crypto/insights", response_model=CryptoAnalysisResponse)
async def crypto_insights(request: CryptoRequest):
    try:
        market_data = await get_crypto_insights(request.coin_list)
        response_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(CryptoAnalysisResponse)
        response_chain = response_prompt  | response_llm

        return await response_chain.ainvoke({"market_data": json.dumps(market_data, indent=2)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error - {str(e)}")