from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel, Field



load_dotenv()

USERS = {
    "1": {
        "name": "Amal",
        "monthly_income": 100000,
        "savings_goal": 20000
    },
    "2": {
        "name": "Kamal",
        "monthly_income": 150000,
        "savings_goal": 30000
    },
    "3": {
        "name": "Nimal",
        "monthly_income": 200000,
        "savings_goal": 50000
    },
    "4": {
        "name": "Saman",
        "monthly_income": 120000,
        "savings_goal": 25000
    },
    "5": {
        "name": "Ruwan",
        "monthly_income": 250000,
        "savings_goal": 60000
    },
    "6": {
        "name": "Kasun",
        "monthly_income": 180000,
        "savings_goal": 40000
    },
    "7": {
        "name": "Tharindu",
        "monthly_income": 90000,
        "savings_goal": 15000
    },
    "8": {
        "name": "Dinesh",
        "monthly_income": 300000,
        "savings_goal": 75000
    }
}

TRANSACTIONS = {
    "1": [
        {
            "description": "Food",
            "category": "Food",
            "amount": 1500
        },
        {
            "description": "Train and Bus",
            "category": "Transport",
            "amount": 1500
        },
        {
            "description": "Groceries",
            "category": "Food",
            "amount": 8500
        }
    ],

    "2": [
        {
            "description": "Lunch",
            "category": "Food",
            "amount": 2500
        },
        {
            "description": "Uber",
            "category": "Transport",
            "amount": 1800
        },
        {
            "description": "Movie",
            "category": "Entertainment",
            "amount": 3000
        }
    ],

    "3": [
        {
            "description": "Restaurant",
            "category": "Food",
            "amount": 4500
        },
        {
            "description": "Fuel",
            "category": "Transport",
            "amount": 5000
        },
        {
            "description": "Shopping",
            "category": "Shopping",
            "amount": 7500
        }
    ],

    "4": [
        {
            "description": "Breakfast",
            "category": "Food",
            "amount": 1200
        },
        {
            "description": "Bus",
            "category": "Transport",
            "amount": 800
        },
        {
            "description": "Clothes",
            "category": "Shopping",
            "amount": 6500
        }
    ],

    "5": [
        {
            "description": "Groceries",
            "category": "Food",
            "amount": 7500
        },
        {
            "description": "Fuel",
            "category": "Transport",
            "amount": 6000
        },
        {
            "description": "Dinner",
            "category": "Food",
            "amount": 3500
        }
    ],

    "6": [
        {
            "description": "Lunch",
            "category": "Food",
            "amount": 2200
        },
        {
            "description": "Train",
            "category": "Transport",
            "amount": 1000
        },
        {
            "description": "Online Shopping",
            "category": "Shopping",
            "amount": 4500
        }
    ],

    "7": [
        {
            "description": "Groceries",
            "category": "Food",
            "amount": 5000
        },
        {
            "description": "Bus",
            "category": "Transport",
            "amount": 1000
        },
        {
            "description": "Coffee",
            "category": "Food",
            "amount": 1200
        }
    ],

    "8": [
        {
            "description": "Restaurant",
            "category": "Food",
            "amount": 5500
        },
        {
            "description": "Fuel",
            "category": "Transport",
            "amount": 7000
        },
        {
            "description": "Shopping",
            "category": "Shopping",
            "amount": 12000
        }
    ]
}

RECURRING_EXPENSES = {
    "1": [
        {
            "name": "Internet",
            "amount": 5000
        },
        {
            "name": "Rent",
            "amount": 50000
        }
    ],

    "2": [
        {
            "name": "Internet",
            "amount": 4500
        },
        {
            "name": "Rent",
            "amount": 40000
        },
        {
            "name": "Netflix",
            "amount": 1500
        }
    ],

    "3": [
        {
            "name": "Rent",
            "amount": 60000
        },
        {
            "name": "Internet",
            "amount": 6000
        },
        {
            "name": "Mobile Phone",
            "amount": 2500
        }
    ],

    "4": [
        {
            "name": "Rent",
            "amount": 35000
        },
        {
            "name": "Internet",
            "amount": 4500
        }
    ],

    "5": [
        {
            "name": "Rent",
            "amount": 55000
        },
        {
            "name": "Internet",
            "amount": 5000
        },
        {
            "name": "Gym",
            "amount": 4000
        }
    ],

    "6": [
        {
            "name": "Rent",
            "amount": 45000
        },
        {
            "name": "Internet",
            "amount": 5000
        },
        {
            "name": "Netflix",
            "amount": 1500
        }
    ],

    "7": [
        {
            "name": "Rent",
            "amount": 30000
        },
        {
            "name": "Internet",
            "amount": 4000
        }
    ],

    "8": [
        {
            "name": "Rent",
            "amount": 70000
        },
        {
            "name": "Internet",
            "amount": 6000
        },
        {
            "name": "Gym",
            "amount": 5000
        },
        {
            "name": "Streaming Services",
            "amount": 2500
        }
    ]
}


#User Context
@dataclass
class Context :

    user_id : str

@tool
def get_user_profile(runtime : ToolRuntime[Context]) :
    """
    Get the current user's financial profile.

    Returns the user's name, monthly income and savings goal.

    Use this tool when information about the current user is required.
    """

    user = USERS.get(runtime.context.user_id)

    if user is None:
        return "User not found"
    return {
        "name" : user["name"],
        "monthly_income" : user["monthly_income"],
        "savings_goal" : user["savings_goal"]
    }
@tool
def get_transactions(runtime : ToolRuntime[Context]):

    """
    Get all transactions for the current user.

    Returns transaction descriptions, categories and amounts
    """

    transactions = TRANSACTIONS.get(runtime.context.user_id)

    if transactions is None :
        return "No Transaction found"

    return transactions

@tool
def calculate_total_expenses(runtime : ToolRuntime[Context]):

    """
    Calculate the total amount spent by the current user.
    """
    transactions = TRANSACTIONS.get(runtime.context.user_id)

    if not transactions:
        return {
            "total_expenses" : 0.0
        }
    total = sum(transaction["amount"] for transaction in transactions)

    return {
        "total_expenses": total
    }
@tool
def calculate_savings(runtime: ToolRuntime[Context]) :
    """
    Calculate the user's remaining money after the expenses.
    """

    user = USERS.get(runtime.context.user_id)

    if user is None :
        return "User not Found"

    transactions = TRANSACTIONS.get(runtime.context.user_id)

    if not transactions :
        total_expenses = 0

    else :
        total_expenses = sum( transaction["amount"] for transaction in transactions)

    income = user["monthly_income"]

    savings = income - total_expenses

    savings_percentage = (savings / income) * 100

    return {
        "monthly_income" : income,
        "total_expenses" : total_expenses,
        "available_savings" : savings,
        "savings_percentage" : round(savings_percentage, 2)
    }

class FinancialAnalysis(BaseModel):
    request_type : str = Field(description="Type of financial assistance requested")
    user_name : str = Field("Name off the user")
    summary : str = Field("short summary of the financial analysis")
    monthly_income : float
    total_expenses : float
    available_savings : float
    savings_percentage : float
    highest_expense_category : str
    recommendations : List[str] = Field(description=
                                        "useful financial recommendations")

SYSTEM_PROMPT = """
You are a Personal Finance Assistant.

Your job is to help users understand and mange their personal finance.

You have access to tool that provide:

- User profile
- Transactions
- Total expenses
- Category expenses
- Savings 
- Spending analysis
- Recurring expenses
- Affordability calculations
- Budget creation

IMPORTANT RULES :

1. Never invent financial information.
2. Use Tools when actual user information is required.
3. Do not perform important financial calculations yourself 
when a calculation tool is exists. 
4. Use the appropriate  tool based on the user's request.
5. You may call multiple tools when necessary.
6. if the user asks about their finances, 
retrieve their actual financial information.
7.  If the user asks about savings, use the savings calculation tool.
8. if the user asks about spending, analyze their transactions.
9.  If the  users asks about a specific category, 
use the category expense tool.
10. If the user asks whether they can  afford something, 
obtain their available money first and then use the affordability tool.
11. If the user asks for a budget, obtain their income and savings goal 
and use the budget tool
12. Return the final response  using the required structured output format.

Currency: Sri Lankan Rupees (LKR)

"""
model = ChatOpenAI(
    model = "gpt-6-luna",

    max_tokens= 16000,
    use_responses_api=True
)

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_profile,
           get_transactions,
           calculate_savings,calculate_total_expenses],
    context_schema=Context,
    response_format=ToolStrategy(FinancialAnalysis),
    checkpointer=InMemorySaver()
)

def ask(question : str, user_id : str , thread_id : str) :

    result = agent.invoke(
        {
            "messages" : [
                {"role" : "user", "content" : question}
            ]
        },
        config= {
            "configurable" : {
                "thread_id" : thread_id
            }
        },

        context= Context(user_id=user_id)

    )

    return result["structured_response"]

print(ask("what is my monthly total expense", "1", "amal-finance"))

# if __name__ == "__main__" :
#     result = ask(
#         "How mouch money did I spend on this months?",
#         "1",
#         "amal-finance"
#     )
#     print(f"\n{result}")