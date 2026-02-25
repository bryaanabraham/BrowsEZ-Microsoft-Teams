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
from typing import Any, Union

def render_response(
    data: Union[dict, list, str],
    max_total_chars: int = 25000,
    max_rows: int = 200,          # bumped default to be practical for markdown
    max_cell_width: int = 120
) -> str:
    """
    Convert raw API responses (dict/list/JSON-string) into grouped Markdown.
    - Accepts dict/list directly, or {"text": "...json..."} or {"text": <dict/list>}
    - Truncates large values and very long outputs
    - Renders lists of dicts as Markdown tables
    - Recursively groups content at all nesting depths
    """

    # -------------------------------
    # 1) Normalize input into a Python object
    # -------------------------------
    def _to_obj(x: Any) -> Any:
        # Case A: {"text": ...}
        if isinstance(x, dict) and "text" in x:
            inner = x["text"]
            # If it's already a dict/list, use it
            if isinstance(inner, (dict, list)):
                return inner
            # If it's a string, try to parse JSON
            if isinstance(inner, str):
                try:
                    return json.loads(inner)
                except Exception:
                    # If not valid JSON, return the raw string as a leaf
                    return {"text": inner}
            # Fallback
            return {"text": str(inner)}

        # Case B: Raw dict/list passed directly
        if isinstance(x, (dict, list)):
            return x

        # Case C: Raw JSON string
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                # Non-JSON string: present it as-is
                return {"text": x}

        # Anything else → stringify
        return {"text": str(x)}

    obj = _to_obj(data)

    # -------------------------------
    # 2) Truncation helpers
    # -------------------------------
    def _truncate_cell(v: Any) -> str:
        s = "" if v is None else str(v)
        if len(s) > max_cell_width:
            return s[:max_cell_width] + "...(truncated)"
        return s

    # -------------------------------
    # 3) Table builder for list-of-dicts
    # -------------------------------
    def _make_table(records: list) -> Union[str, None]:
        if not isinstance(records, list) or not records:
            return None
        if not all(isinstance(r, dict) for r in records):
            return None

        # collect union of keys across rows (stable order: as discovered)
        seen_cols = []
        for r in records:
            for k in r.keys():
                if k not in seen_cols:
                    seen_cols.append(k)

        header = "| " + " | ".join(seen_cols) + " |\n"
        sep = "| " + " | ".join(["---"] * len(seen_cols)) + " |\n"
        rows = []
        for r in records:
            row = [_truncate_cell(r.get(c, "")) for c in seen_cols]
            rows.append("| " + " | ".join(row) + " |")
        return header + sep + "\n".join(rows) + "\n"

    # -------------------------------
    # 4) Recursive renderer to Markdown
    # -------------------------------
    def _walk(node: Any, level: int = 0, name: str = "") -> str:
        md = []
        h = max(1, min(6, level + 1))

        if isinstance(node, dict):
            if name:
                md.append(f"\n{'#' * h} {name}")
            for k, v in node.items():
                md.append(_walk(v, level + 1, k))

        elif isinstance(node, list):
            # If list of dicts → table
            tbl = _make_table(node)
            if tbl is not None:
                if name:
                    md.append(f"\n{'#' * h} {name}")
                md.append(tbl)
            else:
                # scalar/mixed list → render each item
                if name:
                    md.append(f"\n{'#' * h} {name}")
                for i, item in enumerate(node):
                    md.append(_walk(item, level + 1, f"{name}[{i}]"))

        else:
            # leaf value
            val = _truncate_cell(node)
            if name is None:
                # anonymous root scalar
                md.append(f"\n{'#' * h} value\n{val}")
            else:
                md.append(f"\n{'#' * h} {name}: {val}")

        return "\n".join(part for part in md if part)

    markdown = _walk(obj)

    # -------------------------------
    # 5) Global output truncation
    # -------------------------------
    if len(markdown) > max_total_chars:
        markdown = markdown[:max_total_chars] + "\n\n...(OUTPUT TRUNCATED: size limit)..."

    # Limit lines (rows)
    lines = markdown.splitlines()
    if len(lines) > max_rows:
        markdown = "\n".join(lines[:max_rows]) + "\n...(OUTPUT TRUNCATED: row limit)..."

    return markdown
    


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

                    btfd_result = render_response(tool_result)
                    if btfd_result:
                        print(f"Beautified Tool Result: \n{btfd_result}")
                        return btfd_result
                    else:
                        return "Looks like the backend didn't send anything. Can you check the parameters when you get a sec?"
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
