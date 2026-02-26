from datetime import datetime, timedelta, timezone
import json
from tools import check_bank_balance, \
                    get_bank_statement, \
                    imps_status, \
                    neft_status, \
                    rtgs_status, \
                    transaction_status_tl, \
                    upi_status_ml

IST = timezone(timedelta(hours=5, minutes=30))
def _now_ist() -> str:
    """Return current IST timestamp (called fresh each time)."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_bank_balance",
            "description": "This tool checks the users bank balance. No Input is needed for this tool",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
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
            }
        }
    },
    {
        "type": "function",
        "function": {
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
            }
        }
    },
    {
        "type": "function",
        "function": {
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
            }
        }
    },
    {
        "type": "function",
        "function": {
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
            }
        }
    },
    {
        "type": "function",
        "function": {
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
            }
        }
    },
    {
        "type": "function",
        "function": {
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
            }
        }
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

SYSTEM_PROMPT = """
You are **BrowsEZ**, a digital teammate inside Microsoft Teams that helps users get work done through simple, natural conversation.

You can execute tools, retrieve enterprise data, and trigger workflows across cloud or on-prem systems. Based on the user's intent, context, and permissions, you must:

* Identify the correct tool from the available list
* Extract required parameters from the conversation
* Execute the most appropriate action
* Respond clearly with the result or next step

Always prioritize accuracy, security, and minimal back-and-forth. Act like a proactive, reliable enterprise assistant — not just a chatbot.
"""

def maintain_context_limit(
    data,
    max_total_chars: int = 2500,
    max_field_value: int = 1500,
    max_rows: int = 200,
) -> str:
    ''' 
    1. Takes in data (string or API response) and converts to JSON.
    2. Caps total rows/fields at max_rows.
    3. Truncates any field value above max_field_value (replaces with "output too large to display").
    4. Truncates overall output if total chars > max_total_chars.
    5. Returns reduced JSON as a string.
    '''

    # Step 1 — convert input to JSON-safe data
    try:
        if isinstance(data, str):
            json_data = json.loads(data)
        else:
            json_data = data
    except Exception:
        return "An error occured in processing the API response"

    # Ensure we are always working with a list for max_rows logic
    if isinstance(json_data, dict):
        items = [json_data]
    elif isinstance(json_data, list):
        items = json_data
    else:
        items = [{"response": str(json_data)}]

    # Step 2 — Cap number of rows
    if len(items) > max_rows:
        items = items[:max_rows]

    # Step 3 — Truncate long field values
    def truncate_values(obj):
        if isinstance(obj, dict):
            return {
                k: truncate_values(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [truncate_values(v) for v in obj]
        elif isinstance(obj, str):
            if len(obj) > max_field_value:
                return "output too large to display"
            return obj
        else:
            return obj

    items = truncate_values(items)

    # Step 4 — Convert to JSON string and check total char limit
    output = json.dumps(items, indent=2)

    if len(output) > max_total_chars:
        truncated = output[:max_total_chars] + "\n... output truncated ..."
        return truncated

    return output

