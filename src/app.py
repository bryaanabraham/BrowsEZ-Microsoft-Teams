import asyncio
import re
from typing import Dict, Any

from azure.identity import ManagedIdentityCredential
from microsoft.teams.api import MessageActivity, TypingActivityInput
from microsoft.teams.apps import ActivityContext, App
from config import Config

import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta, timezone
IST = timezone(timedelta(hours=5, minutes=30))
now_ist = datetime.now(IST)


from openai import OpenAI
import json

from tools import check_bank_balance, \
                    get_bank_statement, \
                    imps_status, \
                    neft_status, \
                    rtgs_status, \
                    transaction_status_tl, \
                    upi_status_ml

TOOLS = [
    {
        "type": "function",
        "name": "check_bank_balance",
        "description": "This tool checks the users bank balance. No Input is needed for this tool",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "get_bank_statement",
        "description": "This function gets the bank statement of the user for a particular time range.",
        "parameters": {
            "type": "object",
            "properties": {
                "fromDate": {
                    "type": "string",
                    "description": r"Start date for fetching the bank statement. Pattern: ^\d{4}-\d{2}-\d{2}$"
                },
                "toDate": {
                    "type": "string",
                    "description": r"End date for fetching the bank statement. Pattern: ^\d{4}-\d{2}-\d{2}$"
                }
            },
            "required": ["fromDate", "toDate"],
            "additionalProperties": False
        },
        
    },
    {
        "type": "function",
        "name": "imps_status_check",
        "description": "This tool facilitates the merchant to check the status of IMPS transactions using RRN or TxnId",
        "parameters": {
            "type": "object",
            "properties": {
                "txnid": {
                    "type": "string",
                    "description": "TransXT Transaction ID obtained in the txnId parameter, outside the IMPS Payout API response object."
                },
                "rrn": {
                    "type": "string",
                    "description": "UTR/RRN (Transaction reference number). LENGTH=12"
                }
            },
            "additionalProperties": False
        },
        
    },
    {
        "type": "function",
        "name": "neft_status_check",
        "description": "This tool facilitates the merchant to check the status of NEFT transactions using TransXT txnId.",
        "parameters": {
            "type": "object",
            "properties": {
                "txnid": {
                    "type": "string",
                    "description": "TransXT Transaction ID obtained in the txnId parameter, outside the IMPS Payout API response object."
                },
                "rrn": {
                    "type": "string",
                    "description": "UTR/RRN (Transaction reference number). LENGTH=12"
                }
            },
            "additionalProperties": False
        },
        
    },
    {
        "type": "function",
        "name": "rtgs_status_check",
        "description": "This tool facilitates the merchant to check the status of RTGS transactions using TransXT txnId.",
        "parameters": {
            "type": "object",
            "properties": {
                "txnid": {
                    "type": "string",
                    "description": "TransXT Transaction ID obtained in the txnId parameter, outside the IMPS Payout API response object."
                },
                "rrn": {
                    "type": "string",
                    "description": "UTR/RRN (Transaction reference number). LENGTH=12"
                }
            },
            "additionalProperties": False
        },
        
    },
    {
        "type": "function",
        "name": "transaction_status_tl",
        "description": "Through this tool, a Merchant can check the status of transactions at TransXT Layer.",
        "parameters": {
            "type": "object",
            "properties": {
                "txnid": {
                    "type": "string",
                    "description": "TransXT Transaction ID obtained in the txnId parameter, outside the IMPS Payout API response object."
                }
            },
            "required": ["txnid"],
            "additionalProperties": False
        },
        
    },
    {
        "type": "function",
        "name": "upi_status_ml",
        "description": "Through this tool, a Merchant can check the status of UPI transactions at the Merchant Layer.",
        "parameters": {
            "type": "object",
            "properties": {
                "txnid": {
                    "type": "string",
                    "description": "TransXT Transaction ID obtained in the txnId parameter, outside the IMPS Payout API response object. MAX LENGTH=35"
                }
            },
            "required": ["txnid"],
            "additionalProperties": False
        },
        
    }
]

TOOL_REGISTRY = {
    "check_bank_balance": check_bank_balance,
    "get_bank_statement": get_bank_statement,
    "imps_status_check": imps_status,
    "neft_status_check": neft_status,
    "rtgs_status_check": rtgs_status,
    "transaction_status_tl": transaction_status_tl,
    "upi_status_ml": upi_status_ml,
}
def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    tool = TOOL_REGISTRY.get(tool_name)
    if not tool:
        raise ValueError(f"Tool '{tool_name}' not implemented.")
    return tool.run(arguments)

SYSTEM_PROMPT=f"""
You are **BrowsEZ**, a digital teammate inside Microsoft Teams that helps users get work done through simple, natural conversation.

You can execute tools, retrieve enterprise data, and trigger workflows across cloud or on-prem systems. Based on the user's intent, context, and permissions, you must:

* Identify the correct tool from the available list
* Extract required parameters from the conversation
* Execute the most appropriate action
* Respond clearly with the result or next step

Always prioritize accuracy, security, and minimal back-and-forth. Act like a proactive, reliable enterprise assistant. not just a chatbot.
Current date and time (IST) is: {now_ist    }
"""

import json
import re

def beautify_response(
    data,
    max_total_chars=25000,
    max_rows=40,
    max_cell_width=100
):
    """
    Enterprise-grade Teams formatter.

    Features:
    - Groups nested sections
    - Renders list[dict] as tables
    - Limits rows
    - Truncates long cells
    - Enforces total output size (prevents 413)
    """

    # -------------------------------------------------
    # SAFE JSON PARSING
    # -------------------------------------------------
    if isinstance(data, str):
        cleaned = re.sub(r",\s*([}\]])", r"\1", data)
        data = json.loads(cleaned)

    # -------------------------------------------------
    # HELPERS
    # -------------------------------------------------
    def truncate_text(text):
        text = "" if text is None else str(text).replace("\n", " ")
        if len(text) > max_cell_width:
            return text[: max_cell_width - 3] + "..."
        return text

    def enforce_total_limit(output):
        if len(output) > max_total_chars:
            truncated = output[: max_total_chars - 120]
            truncated += (
                "\n\n⚠️ Output truncated as it exceeded Microsoft Teams size limits."
            )
            return truncated
        return output

    def render_table(rows):
        if not rows:
            return "_No data available_\n"

        headers = sorted({k for r in rows for k in r.keys()})
        header = "| " + " | ".join(headers) + " |\n"
        header += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        body = ""
        for i, row in enumerate(rows):
            if i >= max_rows:
                body += (
                    f"\n⚠️ Showing top {max_rows} rows as output was too large.\n"
                )
                break

            body += "| " + " | ".join(
                truncate_text(row.get(h, "")) for h in headers
            ) + " |\n"

        return header + body + "\n"

    def render_section(title, content, level=2):
        prefix = "#" * level
        output = f"\n{prefix} {title}\n\n"

        if isinstance(content, dict):
            for k, v in content.items():
                if isinstance(v, dict):
                    output += render_section(k, v, level + 1)
                elif isinstance(v, list):
                    output += render_section(k, v, level + 1)
                else:
                    output += f"- **{k}**: {truncate_text(v)}\n"

        elif isinstance(content, list):
            if content and all(isinstance(x, dict) for x in content):
                output += render_table(content)
            else:
                for i, item in enumerate(content):
                    if i >= max_rows:
                        output += (
                            f"\n⚠️ Showing top {max_rows} items as output was too large.\n"
                        )
                        break
                    output += f"- {truncate_text(item)}\n"
                output += "\n"

        else:
            output += truncate_text(content) + "\n"

        return output

    # -------------------------------------------------
    # BUILD OUTPUT
    # -------------------------------------------------
    if isinstance(data, dict):
        output = ""
        for key, value in data.items():
            output += render_section(key, value, level=2)
    else:
        output = truncate_text(data)

    return enforce_total_limit(output.strip())


client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
def call_llm(query: str) -> str:
    try:
        response = client.responses.create(
            model="gpt-5",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            tools=TOOLS # type: ignore
        )

        # Handle possible multiple tool calls safely
        for item in response.output:

            if item.type == "function_call":
                
                tool_name = item.name
                arguments = json.loads(item.arguments)
                print(f"Calling Tool: {tool_name}\nArguments:{arguments}")
                try:
                    tool_result = execute_tool(tool_name, arguments)
                    print(f"Tool Result: \n{tool_result}")
                    
                    data = tool_result.get("text")
                    btfd_result = beautify_response(data)
                    print(f"Beautified Tool Result: \n{btfd_result}")
                    return btfd_result
                except Exception:
                    return "There was an error executing your request."

        # If no tool call happened
        return f"I'm here to help. I can only run your APIs, Let me know how I can make you life E-Z.\n\nLLM: {response.output}"

    except Exception as e:
        print(f"Error occurred: {e}")
        return f"Something went wrong while processing your request.\n\nError: {e}"
    
config = Config()

def create_token_factory():
    def get_token(scopes, tenant_id=None):
        credential = ManagedIdentityCredential(client_id=config.APP_ID)
        if isinstance(scopes, str):
            scopes_list = [scopes]
        else:
            scopes_list = scopes
        token = credential.get_token(*scopes_list)
        return token.token
    return get_token

app = App(
    token=create_token_factory() if config.APP_TYPE == "UserAssignedMsi" else None
)

@app.on_message_pattern(re.compile(r"Hello|Hi"))
async def handle_greeting(ctx: ActivityContext[MessageActivity]) -> None:
    """Handle greeting messages."""
    await ctx.send("Hello! How can I assist you today?")

@app.on_message_pattern(re.compile(r"Who are you?|Who is this?"))
async def handle_inquiry(ctx: ActivityContext[MessageActivity]) -> None:
    """Handle greeting messages."""
    await ctx.send("Hey! I'm BrowsEZ. Need something? Just ask—I'll make it easy.")

@app.on_message
async def handle_message(ctx: ActivityContext[MessageActivity]):
    """Handle message activities using the new generated handler system."""
    await ctx.reply(TypingActivityInput())
    query = ctx.activity.text
    print(f"User entered: {query}")
    repsonse = call_llm(query)
    await ctx.send(repsonse)

if __name__ == "__main__":
    asyncio.run(app.start())
