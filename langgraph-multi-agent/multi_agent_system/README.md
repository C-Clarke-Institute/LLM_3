# Multi-Agent AI System

A modular multi-agent application built with Python 3.12+, LangChain, LangGraph,
FastAPI, Pydantic v2, and OpenAI chat models.

## 1. Architecture

A request is sent to `POST /api/v1/ask`. The Router Agent uses an LLM with
Pydantic structured output to choose exactly one specialist:

- `crypto_agent` — cryptocurrency and blockchain questions.
- `car_analyzer_agent` — automotive research and vehicle-market questions.
- `general_agent` — all other questions.

LangGraph executes the router first, then uses `add_conditional_edges()` to execute
only the selected specialist. Each specialist returns its result into the shared
graph state. The service layer converts the graph result into an `AgentResponse`.

## 2. Project structure

```text
multi_agent_system/
├── app/
│   ├── main.py
│   ├── api/routes.py
│   ├── agents/
│   │   ├── router_agent.py
│   │   ├── crypto_agent.py
│   │   ├── car_analyzer_agent.py
│   │   ├── general_agent.py
│   │   └── tool_runner.py
│   ├── graph/{state.py,routing.py,workflow.py}
│   ├── models/{request_models.py,response_models.py,agent_models.py}
│   ├── tools/{crypto_tools.py,car_tools.py,general_tools.py}
│   ├── core/{config.py,llm.py}
│   └── services/agent_service.py
├── tests/test_api.py
├── .env.example
├── requirements.txt
└── README.md
```

## 3. Requirements

- Python 3.12 or newer
- An OpenAI API key
- Internet access for CoinGecko and vehicle listing lookups

## 4. Installation

Create and activate a virtual environment.

### macOS / Linux

```bash
cd multi_agent_system
python3.12 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
cd multi_agent_system
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create your local environment file:

```bash
cp .env.example .env
```

On Windows, copy `.env.example` to `.env` manually or run:

```powershell
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Do not commit `.env` to source control.

## 5. Run the API

From the project root:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

- API base: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- Health: http://127.0.0.1:8000/health

## 6. Test the API

### Request

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Bitcoin and how does cryptocurrency trading work?"}'
```

### Example response

```json
{
  "success": true,
  "question": "What is Bitcoin and how does cryptocurrency trading work?",
  "selected_agent": "crypto_agent",
  "answer": "Bitcoin is a decentralized digital currency ...",
  "routing_reason": "The question asks about Bitcoin and cryptocurrency trading.",
  "error": null
}
```

The answer above is illustrative; actual output is generated at request time.

Try other routes:

```json
{"question":"Compare a Toyota Prius and Toyota Axio for fuel economy in Sri Lanka."}
```

```json
{"question":"Explain the difference between Python and Java."}
```

Use Swagger UI at `/docs`, open `POST /api/v1/ask`, click **Try it out**, enter a JSON request, and click **Execute**.

## 7. How the modules work together

1. `api/routes.py` validates the request with `UserQuestionRequest`.
2. `services/agent_service.py` invokes the compiled graph asynchronously.
3. `agents/router_agent.py` returns a `RouterDecision` using structured output.
4. `graph/routing.py` validates the selected route.
5. `graph/workflow.py` sends execution to exactly one specialist node.
6. The selected agent uses its dedicated tools and the shared LLM.
7. The graph result is converted to `AgentResponse`.
8. FastAPI serializes the response.

## 8. Tools and data-source notes

### Cryptocurrency

`tools/crypto_tools.py` calls CoinGecko's public API for current prices and market
metrics. Coin IDs are expected, for example `bitcoin` and `ethereum`. CoinGecko may
apply rate limits; configure `COINGECKO_API_KEY` if you have a Demo API key.

### Automotive

`tools/car_tools.py` adapts the uploaded vehicle-market research approach and checks
Patpat, ikman, and Riyasewana listing pages. It also exposes an NHTSA vPIC lookup.
These sources do not guarantee complete or current Sri Lankan vehicle data.
Advertised prices are not verified transaction prices. The system must report
missing values rather than invent them. Website markup, access rules, and availability
can change; use only sources you are permitted to access.

### General

The general agent uses the shared LLM and has a small UTC-time utility tool. Add
additional tools to `tools/general_tools.py` when a real use case requires them.

## 9. Error handling and security

- Blank questions are rejected with HTTP 422.
- Router decisions are validated against the `AgentName` enum.
- Tool errors are returned to the agent as tool results so it can explain unavailable data.
- Workflow/provider failures are logged server-side and returned as a generic HTTP 502.
- Secrets are read from environment variables and are not returned by API endpoints.
- For production, add authentication, rate limiting, request tracing, structured logs,
  and a deployment-specific secret manager.

## 10. Run tests

```bash
pytest -q
```

The included tests cover health and request validation. Add mocked graph/service
tests for routing and specialist behavior before production deployment.

## 11. Notes for extension

To add another specialist:
1. Add an `AgentName` enum value.
2. Create its tools in `app/tools/`.
3. Implement the agent in `app/agents/`.
4. Add a graph node and conditional mapping in `app/graph/workflow.py`.
5. Extend the router prompt and response tests.

The LLM model is configured once in `app/core/llm.py` and shared by all agents.
