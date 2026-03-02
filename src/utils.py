from datetime import datetime, timedelta, timezone
import json
from tools import check_bank_balance, \
                    get_bank_statement, \
                    imps_status, \
                    neft_status, \
                    query_sury_va, \
                    rtgs_status, \
                    transaction_status_tl, \
                    upi_status_ml, \
                    query_sury_merchant
import pandas as pd
from collections import deque

USD_TO_INR = 91.0
PRICE_UNIT = 1_000_000

import tiktoken


def get_tokens(string: str, model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # fallback tokenizer
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(string)
    return len(tokens)

def extract_text(msg) -> str:
    """Safely extracts string content from various message formats."""
    content = ""
    if isinstance(msg, dict):
        content = msg.get("content")
    elif hasattr(msg, "content"):
        content = msg.content
    
    # 2. Handle cases where content is None (common in tool-calling assistant messages)
    if content is None:
        if isinstance(msg, dict) and msg.get("tool_calls"):
            return json.dumps(msg.get("tool_calls"))
        return ""
        
    return str(content)

def get_cost(messages: deque, dataframe: pd.DataFrame, model: str, usage_data: dict) -> str:
    """
    Calculate cost from either API usage data (preferred) or manual token counting.
    
    Args:
        messages: Message history
        dataframe: Pricing dataframe
        model: Model name
        usage_data: Optional usage dict from API response with 'prompt_tokens' and 'completion_tokens'
    """
    # 1. Validation
    required_cols = {"Model", "Input", "Output"}
    if not required_cols.issubset(dataframe.columns):
        missing = required_cols - set(dataframe.columns)
        raise ValueError(f"Missing columns: {missing}")

    # 2. Extract pricing
    row = dataframe.loc[dataframe["Model"] == model]
    if row.empty:
        return ""

    input_price_per_unit = row["Input"].iloc[0]
    output_price_per_unit = row["Output"].iloc[0]

    # 3. Get token counts - prefer API usage data
    if usage_data:
        input_tokens = usage_data.get("prompt_tokens", 0)
        output_tokens = usage_data.get("completion_tokens", 0)
    else:
        # Fallback to manual counting (less accurate, doesn't include tool definitions)
        msg_list = list(messages)
        history_text = " ".join([extract_text(m) for m in msg_list[:-1]])
        last_msg_text = extract_text(msg_list[-1]) if msg_list else ""
        
        input_tokens = get_tokens(history_text, model) 
        output_tokens = get_tokens(last_msg_text, model)
        print("Failed to get usage, defaulting to manual check.")

    # 4. Calculate costs
    input_cost = (input_tokens / PRICE_UNIT) * input_price_per_unit * USD_TO_INR
    output_cost = (output_tokens / PRICE_UNIT) * output_price_per_unit * USD_TO_INR
    total_cost = input_cost + output_cost

    return (f"""
| Input Tokens | Output Tokens | Total Cost (INR) |
| :----------- | :------------ | :--------------- |
| {input_tokens:,} | {output_tokens:,} | ₹{total_cost:.4f} |
""")

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
    },
    {
        "type": "function",
        "function": {
            "name": "query_sury_va",
            "description": "This tool allows users to write natural language queries and retrieve data from Metabase Sury-Virtual Accounts. The tool converts the user input to an SQL command which is queried over SURYODAY VIRTUAL ACCOUNTS(VA) ONLY.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": """ 
Convert the natural language questions into SQL queries. A database schema will be provided to you. Follow these rules strictly:
- Use only the tables and columns explicitly provided in the schema.
- Do not invent new tables, columns, or relationships. Use only what is defined.
- ALWAYS USE ALL THE DATA PROVIDED BY THE USER. DO NOT MISS ANY DETAILS INCLUDING TYPE/RANGE ETC. THERE MAY BE MULTIPLE.
- When joins are required, infer the relationship based on foreign-key naming conventions or explicit schema instructions. If ambiguity exists, choose the relationship that is most consistent with typical relational database design.
- Fully qualify columns when necessary to avoid ambiguity.
- Never return placeholders. For example, do not return "table_name" or "column_name". Always return actual schema elements.
- DONT USE UNNECESSARILY COMPLEX COMMANDS. RESPOND WITH THE MOST UNDERSTANABLE AND USER READABLE COMMANDS.
- Use time format as follows: BETWEEN '2026-02-01 00:00:00' AND '2026-02-19 23:59:59'
- DONT MAKE ASSUMPTIONS ABOUT. SEMANTIC MEANINGS OF THE ATTRIBUTES IN THE QUERY. EG: DONT ASSUME THAT CORPORATE ID IS THE SAME AS MERCHANT ID (MID). TELL THE USER YOU DONT HAVE THE INFOMRATION IF NECESSARY.
- Use the schema provided below for your query:
    DateTime clarifications: insertedon looks like this -> "2026-03-01T00:00:00+05:30" for 01 March 2026, 12:00 AM IST.
    CREATE TABLE apiaccess (
    apiaccessid INT UNSIGNED PRIMARY KEY, entityname VARCHAR, entitykey VARCHAR, entityenckey VARCHAR, apiaccess VARCHAR, updatedon DATETIME, insertedon DATETIME, isactive INT, nodeip VARCHAR
    );
    CREATE TABLE apimaster (
    apimasterid INT UNSIGNED PRIMARY KEY, apikey VARCHAR, apiname VARCHAR, apiurl VARCHAR, apidesc VARCHAR, apitype INT, isactive INT, insertedon DATETIME, nodeip VARCHAR
    );
    CREATE TABLE balmaster (
    balmasterid INT UNSIGNED PRIMARY KEY, balkey VARCHAR, baldesc VARCHAR, isactive INT, nodeip VARCHAR
    );
    CREATE TABLE corporate (
    corporateid INT UNSIGNED PRIMARY KEY, corporateunqid VARCHAR, corporatename VARCHAR, corporatecode VARCHAR, corporateacdtls VARCHAR, insertedon DATETIME, isactive INT, nodeip VARCHAR
    );
    CREATE TABLE corporatedtls (
    corporatedtlsid INT UNSIGNED PRIMARY KEY, corporateid INT, corpdtlskey VARCHAR, corpdtlskey1 VARCHAR, corpdtlskey2 VARCHAR, corpdtlstype INT, isactive INT, insertedon DATETIME, nodeip VARCHAR
    );
    CREATE TABLE corporateva (
    corporatevaid INT UNSIGNED PRIMARY KEY, corporateid INT, corporatevaname VARCHAR, vano VARCHAR, vanomasked VARCHAR, vanohash VARCHAR, extacid VARCHAR, sourcechk INT, isactive INT, insertedon DATETIME, nodeip VARCHAR
    );
    CREATE TABLE corporatevabal (
    corporatevabalid INT UNSIGNED PRIMARY KEY, corporateid INT, corporatevaid INT, balmasterkey VARCHAR, balance DECIMAL, createdon DATETIME, updatedon DATETIME, nodeip VARCHAR
    );
    CREATE TABLE corporatevadtls (
    corporatevadtlsid INT UNSIGNED PRIMARY KEY, corporatevaid INT, dtls1 VARCHAR, dtls2 VARCHAR, dtlskey VARCHAR, dtlstype INT, isactive INT, insertedon DATETIME, updatedon DATETIME, nodeip VARCHAR
    );
    CREATE TABLE corporatevaledger (
    corporatevaledgerid INT UNSIGNED PRIMARY KEY, corporatevaid INT, refno VARCHAR, tprefno VARCHAR, bankrefno VARCHAR, narration VARCHAR, opebal DECIMAL, amount DECIMAL, clobal DECIMAL, dctype INT, optype INT, channel INT, insertedon DATETIME, nodeip VARCHAR, payeedtls VARCHAR, status VARCHAR, txninfo VARCHAR
    );
        In the Corporatevaledger table:
            For the column "dctype":
            - Numeric value 1 represents "Payout"
            - Numeric value 2 represents "Payin"
            For the column "optype":
            - Numeric value 1 represents "Top-up (credit into a VA)"
            - Numeric value 2 represents "Debit (Payout from a VA)"
            - Numeric value 3 represents "Reversal (Reversal back to a VA)"
            For the column "channel":
            - Numeric value 1 represents "IMPS"
            - Numeric value 2 represents "NEFT"
            - Numeric value 3 represents "RTGS"
            - Numeric value 4 represents "UPI"

    CREATE TABLE mwtsysconfig (
    mwtsysconfigid INT UNSIGNED PRIMARY KEY, mwtsyskeyname VARCHAR, mwtsyskeyvalue VARCHAR, isactive INT, nodeip VARCHAR, insertedon DATETIME
    );
    CREATE TABLE sysaudit (
    sysauditid INT UNSIGNED PRIMARY KEY, sysid VARCHAR, cid INT, cvaid INT, bankprocess VARCHAR, sysprocess VARCHAR, areq VARCHAR, arsp VARCHAR, astatus INT, createdon DATETIME PRIMARY KEY, nodeip VARCHAR
    );
    CREATE TABLE txnmapping (
    id INT PRIMARY KEY, refno VARCHAR, bankrefno VARCHAR
    );
    CREATE TABLE vacgen (
    vacgenid INT PRIMARY KEY, corporatecode VARCHAR, seqno INT
    );
    CREATE TABLE waudit (
    wauditid INT UNSIGNED PRIMARY KEY, wentityid INT, servicename VARCHAR, wrefno VARCHAR, narration VARCHAR, tprefno VARCHAR, opebal DECIMAL, amount DECIMAL, charge DECIMAL, tds DECIMAL, gst DECIMAL, clobal DECIMAL, optype INT, dctype INT, insertedat DATETIME, issettled INT, settleddat DATETIME, nodeip VARCHAR
    );
    CREATE TABLE wentity (
    wentitiyid INT UNSIGNED PRIMARY KEY, bankifsc VARCHAR, entityname VARCHAR, balance DECIMAL, createdat DATETIME, updatedat DATETIME, isactive INT, nodeip VARCHAR
    );
"""
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_sury_merchant",
            "description": "This tool allows users to write natural language queries and retrieve data from Metabase FROM SURYODAY MERCHANTS DB. The tool converts the user input to an SQL command which is queried over SURYODAY MERCHANT DTABASE ONLY.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": """ 
Convert the natural language questions into SQL queries. A database schema will be provided to you. Follow these rules strictly:
- Use only the tables and columns explicitly provided in the schema. If the user asks for information that does not exist in the schema, return a SQL query that uses only the available fields in the most reasonable way.
- Do not invent new tables, columns, or relationships. Use only what is defined.
- When joins are required, infer the relationship based on foreign-key naming conventions or explicit schema instructions. If ambiguity exists, choose the relationship that is most consistent with typical relational database design.
- Fully qualify columns when necessary to avoid ambiguity.
- Never return placeholders. For example, do not return "table_name" or "column_name". Always return actual schema elements.
- DONT USE UNNECESSARILY COMPLEX COMMANDS. RESPOND WITH THE MOST UNDERSTANABLE AND USER READABLE COMMANDS.
- DONT MAKE ASSUMPTIONS ABOUT. SEMANTIC MEANINGS OF THE ATTRIBUTES IN THE QUERY. EG: DONT ASSUME THAT CORPORATE ID IS THE SAME AS MERCHANT ID (MID). TELL THE USER YOU DONT HAVE THE INFOMRATION IF NECESSARY.
- Use the schema provided below for your query:
"""
                    }
                },
                "required": ["query"],
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
    "query_sury_va": query_sury_va,
    "query_sury_merchant": query_sury_merchant,
}

SYSTEM_PROMPT = """
You are **BrowsEZ**, a digital teammate inside Microsoft Teams that helps users get work done through simple, natural conversation.

You can execute tools, retrieve enterprise data, and trigger workflows across cloud or on-prem systems. Based on the user's intent and context you must:

* Run tools only when necessary to fulfill user requests, and always with the most specific parameters you can extract from the conversation. Avoid unnecessary tool calls.
* Identify the correct tool from the available list
* Extract required parameters from the conversation
* Dont truncate any information. Show as much information you can under 20,000 characters.
* You are working in INDIA for all numeric notations follow indian standards. For example, 1000000 should be shown as 10,00,000.
* TABULATE DATA WHENEVER POSSIBLE AS PER THE MICROSOFT TEAMS MARKDOWN TABLE FORMAT.

Always prioritize accuracy, security, and minimal back-and-forth. Act like a proactive, reliable enterprise assistant — not just a chatbot.
"""

def maintain_context_limit(
    data,
    max_total_chars: int = 20000,
    max_field_value: int = 10000,
    max_rows: int = 50,
) -> str:
    ''' 
    1. Takes in data (string or API response) and converts to JSON.
    2. Caps total rows at max_rows.
    3. Truncates any field value above max_field_value (keeps original but cuts it).
    4. Truncates overall output if total chars > max_total_chars.
    5. Returns reduced JSON as a string.
    '''

    try:
        if isinstance(data, str):
            json_data = json.loads(data)
        else:
            json_data = data
    except Exception:
        print("An error occurred in processing the API response")
        return "An error occurred in processing the API response"

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

    # Step 3 — Truncate long field values (keep original, just slice)
    def truncate_values(obj):
        if isinstance(obj, dict):
            return {k: truncate_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [truncate_values(v) for v in obj]
        elif isinstance(obj, str):
            if len(obj) > max_field_value:
                return obj[:max_field_value] + "...(truncated)"
            return obj
        else:
            return obj

    items = truncate_values(items)
    output = json.dumps(items, indent=2)

    if len(output) > max_total_chars:
        output = output[:max_total_chars] + "\n... output truncated ..."

    print(f"Reduced data of {len(str(data))} to {len(str(output))}")
    return output

