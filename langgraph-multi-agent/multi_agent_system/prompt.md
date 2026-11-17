# Project: Multi-Agent AI System Using LangGraph and FastAPI

## 1. Project Overview

Build a modular, production-ready Multi-Agent AI application using **Python, LangGraph, LangChain, FastAPI, and Pydantic**.



The application should receive a user's question through a FastAPI REST API and intelligently route the question to the appropriate specialized AI agent.



The system must follow this workflow:



**Workflow:**

1. User submits a question.
2. The Router Agent analyzes the question.
3. The Router Agent determines which specialized agent should handle the question.
4. LangGraph uses conditional edges to route the request.
5. The selected agent processes the question using its dedicated tools.
6. The agent generates a response.
7. The response is returned to the user through FastAPI.

## 2. System Architecture

Implement the following LangGraph workflow:

### Step 1: User Question

Accept a natural-language question from the user through a FastAPI POST endpoint.



Example questions:

- "What is Bitcoin and how does cryptocurrency trading work?"
- "Analyze the current price and performance of a Toyota Prius."
- "Explain the difference between Python and Java."
- "What are the latest developments in artificial intelligence?"

### Step 2: Router Agent

Create a Router Agent responsible for understanding the user's question and determining the appropriate agent.



The Router Agent must classify incoming questions into one of three categories:

- `crypto_agent`
- `car_analyzer_agent`
- `general_agent`

Use an LLM with structured output to produce a valid routing decision.



The router must return a structured result containing:

- User question
- Selected agent
- Routing reason

### Step 3: Conditional Edge

Implement LangGraph conditional routing using `add_conditional_edges()`.



Routing logic:

- If the selected agent is `crypto_agent`, execute the Crypto Agent.
- If the selected agent is `car_analyzer_agent`, execute the Car Analyzer Agent.
- If the selected agent is `general_agent`, execute the General Agent.

### Step 4: Specialized Agents

#### A. Crypto Agent

Responsible for cryptocurrency-related questions.



Examples:

- Cryptocurrency fundamentals
- Bitcoin and Ethereum
- Cryptocurrency market information
- Cryptocurrency comparisons
- Blockchain technology

The Crypto Agent must use its own dedicated tools.

#### B. Car Analyzer Agent

Responsible for automotive-related questions.



Examples:

- Vehicle specifications
- Car comparisons
- Fuel economy
- Electric and hybrid vehicles
- Vehicle performance analysis
- Car recommendations

The Car Analyzer Agent must use its own dedicated tools.

#### C. General Agent

Responsible for questions that do not belong to cryptocurrency or automotive domains.



Examples:

- Programming
- Software engineering
- General knowledge
- Education
- Technology
- Science

The General Agent should use an LLM to generate helpful responses and can have its own dedicated tools when necessary.

## 3. Required Technology Stack

Use the following technologies:

- Python 3.12+
- LangChain
- LangGraph
- OpenAI Chat Models
- FastAPI
- Pydantic v2
- Uvicorn
- Python-dotenv

Use environment variables for API keys and configuration.

## 4. Required Project Structure

Follow a clean, modular architecture.



Do not put all application logic into a single Python file.



Use the following folder structure:
```markdown
multi_agent_system/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── router_agent.py
│   │   ├── crypto_agent.py
│   │   ├── car_analyzer_agent.py
│   │   └── general_agent.py
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── workflow.py
│   │   └── routing.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request_models.py
│   │   ├── response_models.py
│   │   └── agent_models.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── crypto_tools.py
│   │   ├── car_tools.py
│   │   └── general_tools.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── llm.py
│   │
│   └── services/
│       ├── __init__.py
│       └── agent_service.py
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 5. Module Responsibilities

### models/

All Pydantic models must be defined in this module.



Do not define request or response Pydantic models inside FastAPI routes, agent files, or graph files.



Create the following models:



**request_models.py**

- UserQuestionRequest
- Validate the question field.
- Reject empty questions.

**response_models.py**

- AgentResponse
- Include the original question.
- Include the selected agent.
- Include the generated answer.
- Include success status.

**agent_models.py**

- RouterDecision
- Define the available agent names using an Enum or Literal.
- Include the routing reason.

### agents/

Each agent must be implemented in its own Python module.



Responsibilities:

- `router_agent.py`: Classify the user's question and determine the appropriate agent.
- `crypto_agent.py`: Process cryptocurrency questions using crypto-specific tools.
- `car_analyzer_agent.py`: Process automotive questions using car-specific tools.
- `general_agent.py`: Process general questions.

Each agent should expose a clear function that can be called from the LangGraph workflow.



Keep prompts, model invocation, and agent-specific processing inside the corresponding agent modules.

### tools/

All tools must be implemented separately from agents.



Create:



**crypto_tools.py**

- Cryptocurrency information tools.
- Cryptocurrency price lookup tool.
- Cryptocurrency market information tool.

**car_tools.py**

- Vehicle specification lookup tool.
- Fuel economy information tool.
- Vehicle comparison tool.

**general_tools.py**

- General information search tool, if required.
- Other general-purpose tools when necessary.

Use LangChain's `@tool` decorator where appropriate.



Tools must be reusable and must not contain agent orchestration logic.



For external APIs, use configurable API credentials and handle API failures gracefully. Do not fabricate live market prices or vehicle specifications.

### graph/

This module must contain all LangGraph workflow implementation.



**state.py**



Define the shared TypedDict state used by LangGraph.



The state should include:

- user_question
- selected_agent
- routing_reason
- agent_response
- error

**routing.py**



Implement the routing function used by LangGraph conditional edges.



**workflow\.py**



Build the complete LangGraph workflow.



Use:

- StateGraph
- START
- END
- add_node()
- add_edge()
- add_conditional_edges()
- compile()

The graph should follow this sequence:



START → Router Agent → Conditional Routing



Conditional branches:

- Crypto Agent → END
- Car Analyzer Agent → END
- General Agent → END

Ensure the graph is compiled in this module and expose a reusable compiled graph instance or factory function.

### core/

**config.py**



Manage application configuration using Pydantic Settings.



Load environment variables from `.env`.



**llm.py**



Initialize and expose the shared LLM configuration.



Do not create unnecessary duplicate LLM instances across agents.

### services/

**agent_service.py**



Create a service layer responsible for:

- Accepting the user question.
- Invoking the compiled LangGraph workflow.
- Extracting the selected agent and response.
- Returning structured application data.
- Handling workflow errors.

Keep business orchestration out of the FastAPI route handlers.

### api/

**routes.py**



Create the REST API endpoint:



`POST /api/v1/ask`



Request:
```json
{
  "question": "What is Bitcoin?"
}
```

Response:
```json
{
  "success": true,
  "question": "What is Bitcoin?",
  "selected_agent": "crypto_agent",
  "answer": "Bitcoin is a decentralized digital currency...",
  "routing_reason": "The question is related to cryptocurrency."
}
```

Use the Pydantic request and response models from the models module.

### main.py

Create the FastAPI application.



Requirements:

- Initialize FastAPI.
- Register API routers.
- Add application metadata.
- Add a health-check endpoint.
- Configure appropriate exception handling.

## 6. LangGraph Implementation Requirements

Implement the workflow using a shared state.



The Router Agent must execute first.



The conditional edge must examine the `selected_agent` state field and select the correct agent node.



Each specialized agent must update the shared state with its response.



The graph should terminate after the selected agent finishes processing.



Do not execute all three specialized agents for a single request.



Only the selected agent should execute.

## 7. Agent Prompt Engineering

Give each agent a dedicated system prompt.



Router Agent:

- Act as an intelligent query classifier.
- Understand the user's intent.
- Select exactly one specialized agent.
- Never generate the final answer.
- Return a structured routing decision.

Crypto Agent:

- Act as a cryptocurrency specialist.
- Answer cryptocurrency-related questions.
- Use available tools when real-time information is requested.
- Distinguish verified information from general explanations.

Car Analyzer Agent:

- Act as an automotive specialist.
- Analyze vehicles and compare specifications.
- Use available tools for factual vehicle information.
- Do not invent specifications.

General Agent:

- Act as a helpful general-purpose assistant.
- Answer questions outside the specialized domains.
- Provide clear and accurate explanations.

## 8. Error Handling and Validation

Implement proper error handling for:

- Empty user questions.
- Invalid router decisions.
- LLM API failures.
- Tool execution failures.
- LangGraph execution failures.
- Missing environment variables.

If the router returns an invalid agent name, handle it safely rather than silently executing an arbitrary node.



Return appropriate HTTP status codes from FastAPI.



Never expose API keys, internal exceptions, or sensitive configuration in API responses.

## 9. Additional Requirements

- Follow Python PEP 8 conventions.
- Use type hints throughout the project.
- Use meaningful function and variable names.
- Add docstrings to important functions.
- Keep agents, tools, models, graph, and API responsibilities separate.
- Use dependency injection or shared configuration where appropriate.
- Avoid circular imports.
- Avoid unnecessary abstractions.
- Make the code easy to extend with additional agents in the future.
- Use asynchronous execution where appropriate and supported.
- Include logging for routing decisions and execution failures.
- Keep the implementation beginner-friendly but architecturally sound.

## 10. Expected Deliverables

Generate the complete working project.



Provide:

1. Complete folder structure.
2. All Python source files.
3. All Pydantic models.
4. All agent implementations.
5. All specialized tools.
6. Complete LangGraph workflow.
7. FastAPI application and API routes.
8. Configuration management.
9. requirements.txt.
10. .env.example.
11. README.md with setup instructions.
12. Example API requests and responses.
13. Instructions for running the application using Uvicorn.
14. Instructions for testing the API using Swagger UI.

## 11. Implementation Approach

Build the project step by step.



First, explain the architecture and folder structure.



Then implement the models module.



Next, implement the core configuration and shared LLM.



Then implement the tools module.



After that, implement each agent independently.



Next, implement the LangGraph state, routing logic, and workflow.



Finally, integrate the graph with FastAPI through the service layer and API routes.



For each file:

- Show the complete code.
- Explain its responsibility.
- Explain how it connects to the other modules.

Ensure all imports are correct and the complete application can run without missing modules or undefined functions.



**Important:** Do not provide only pseudocode or isolated examples. The final deliverable must be a complete, executable, modular Python application following the architecture described above.