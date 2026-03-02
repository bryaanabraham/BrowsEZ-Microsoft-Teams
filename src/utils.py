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

def get_cost(messages: deque, dataframe: pd.DataFrame, model: str) -> str:
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

    # Convert deque to a list once for indexing
    msg_list = list(messages)
    
    # Extract text from all messages except the last one
    history_text = " ".join([extract_text(m) for m in msg_list[:-1]])
    
    # Extract text from the very last message
    last_msg_text = extract_text(msg_list[-1]) if msg_list else ""

    # Token calculation
    input_tokens = get_tokens(history_text, model) 
    output_tokens = get_tokens(last_msg_text, model)

    # Math
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
            "description": "This tool allows users to write natural language queries and retrieve data from Metabase Sury-Virtual Accounts. The tool converts the user input to an SQL command which is queried over the database mentioned by the user.",
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
- Use time format as follows: BETWEEN '2026-02-01 00:00:00' AND '2026-02-19 23:59:59'
- Use the schema provided below for your query:
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
            "description": "This tool allows users to write natural language queries and retrieve data from Metabase FROM SURYODAY MERCHANTS DB. The tool converts the user input to an SQL command which is queried over the database mentioned by the user.",
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
- Use the schema provided below for your query:
CREATE TABLE accno (
  accno VARCHAR
);
CREATE TABLE aggregator_details (
  aggregator_id INT PRIMARY KEY, aggunique_id VARCHAR, aggregator_name VARCHAR, encryption_key VARCHAR, callbackurl_1 VARCHAR, callbackurl_2 VARCHAR, created_at TIMESTAMP, updated_at TIMESTAMP, is_active INT
);
CREATE TABLE apiaccess (
  apiaccessid INT UNSIGNED PRIMARY KEY, mid INT, apiaccess VARCHAR, accesstype INT, updatedon DATETIME, insertedon DATETIME, isactive INT, nodeip VARCHAR
);
CREATE TABLE apiauditlog (
  apiauditlogid INT PRIMARY KEY, reqid VARCHAR, userid INT, serviceid INT, reqintime DATETIME, reqouttime DATETIME, channel INT, createddate TIMESTAMP, sessionid VARCHAR, node VARCHAR, udf1 VARCHAR, udf2 VARCHAR, udf3 VARCHAR, udf4 VARCHAR, udf5 VARCHAR, udf6 VARCHAR, udf7 VARCHAR, apiurl VARCHAR, payload MEDIUMTEXT, errorcode VARCHAR
);
CREATE TABLE apimaster (
  apimasterid INT UNSIGNED PRIMARY KEY, apikey VARCHAR, apiname VARCHAR, apidesc VARCHAR, isactive INT, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE apipartnerdetails (
  apipartnerdetailsid INT PRIMARY KEY, name VARCHAR, email VARCHAR, contactno VARCHAR, partnerkey VARCHAR, authorization VARCHAR, salt VARCHAR, status INT, userid INT, createddate DATETIME, udf1 VARCHAR, udf2 VARCHAR, udf3 VARCHAR, encryptpayload INT
);
CREATE TABLE arv_merchants_txn (
  id INT PRIMARY KEY, mid INT UNSIGNED, narration VARCHAR, payer_vpa VARCHAR, payer_code VARCHAR, payer_name VARCHAR, payer_mobile_no VARCHAR, payeracctype VARCHAR, payer_account_no VARCHAR, payer_ifsc_code VARCHAR, payer_adhar VARCHAR, payee_add VARCHAR, payee_code VARCHAR, payee_name VARCHAR, payee_mobileno VARCHAR, payee_actype VARCHAR, payee_account_details VARCHAR, payee_adhar VARCHAR, payee_account_name VARCHAR, collect_expiry VARCHAR, amount VARCHAR, ref_id VARCHAR, devices VARCHAR, txn_id VARCHAR, hdn_order_id VARCHAR, rrn VARCHAR, txn_type VARCHAR, settlement_status VARCHAR, settlement_date_MSF VARCHAR, MSF_amount VARCHAR, MSF_tax_amount VARCHAR, payout_status VARCHAR, status VARCHAR, error_code VARCHAR, npci_error_code VARCHAR, error_description VARCHAR, created_at DATETIME, updated_at DATETIME, approval_no VARCHAR, customer_ref_id VARCHAR, ybl_txn_id VARCHAR, uniquemid VARCHAR, upi_txn_id VARCHAR, api_id VARCHAR, channel VARCHAR, org_txn_id VARCHAR, org_hdn_order_id VARCHAR, org_rrn VARCHAR, orgtxndate VARCHAR, msgflow INT, internal_settlement_status INT, seqno INT, mandateid INT, mdrcharge DECIMAL, ismdrprec INT, mdr DECIMAL, sgst DECIMAL, cgst DECIMAL, igst DECIMAL, settamount DECIMAL, ttumtype INT, nodeip VARCHAR
);
CREATE TABLE arv_npciraw (
  npcirawid INT UNSIGNED PRIMARY KEY, pre VARCHAR, txntype VARCHAR, txnid VARCHAR, rrn VARCHAR, respcode VARCHAR, txndate DATE, txntime VARCHAR, settamt DECIMAL, umn VARCHAR, mapperid VARCHAR, initiationmode VARCHAR, purposecode VARCHAR, pyrcode VARCHAR, pyrmcc VARCHAR, pyrvpa VARCHAR, remcode VARCHAR, remifsc VARCHAR, remactype VARCHAR, remacno VARCHAR, pyecode VARCHAR, pyemcc VARCHAR, pyevpa VARCHAR, bencode VARCHAR, benifsc VARCHAR, benactype VARCHAR, benacno VARCHAR, lrn VARCHAR, resfield1 VARCHAR, resfield2 VARCHAR, resfield3 VARCHAR, npcihdr VARCHAR, npcifilename VARCHAR, npcicycle VARCHAR, optype INT, nodeip VARCHAR, insertedon DATETIME
);
CREATE TABLE arv_vpaaudit (
  vpaauditid INT UNSIGNED PRIMARY KEY, mid INT, refid VARCHAR, vpa VARCHAR, benvpa VARCHAR, status INT, isonline TINYINT, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE auditlog (
  auditlogid INT PRIMARY KEY, tablename VARCHAR, operation VARCHAR, rowid INT, methodname VARCHAR, oldvalue TEXT, newvalue TEXT, userid INT, auditdate DATETIME, approvalstatus INT, approvedby INT, approveddate DATETIME, rejectreason VARCHAR, node VARCHAR
);
CREATE TABLE audittrail (
  audittrailid INT PRIMARY KEY, operation VARCHAR, servicename VARCHAR, recordkey VARCHAR, email VARCHAR, oldvalue TEXT, newvalue TEXT, userid INT, auditdate DATETIME, approvalstatus INT, approvalby INT, approvaldate DATETIME, rejectreason VARCHAR, nodeip VARCHAR, branchcode VARCHAR, reqid VARCHAR, createddate DATETIME, requestfor INT, mid INT, name VARCHAR, virtualvpadetailsid VARCHAR, vpa VARCHAR, hashname VARCHAR, invtid INT, invtidentifier1 VARCHAR, invtidentifier2 VARCHAR, merchantmandateid INT
);
CREATE TABLE autouploaddetails (
  autouploaddetailsid INT PRIMARY KEY, lastonboardctn INT, createddate DATETIME, updateddate DATETIME
);
CREATE TABLE bank_master (
  id INT PRIMARY KEY, bank_name VARCHAR, vpa_handle VARCHAR, live_url TEXT, uat_url TEXT, is_active BIT, created_at TIMESTAMP, updated_at TIMESTAMP
);
CREATE TABLE branchmaster (
  branchmasterid INT PRIMARY KEY, branchcode VARCHAR, branchname VARCHAR, createddate DATETIME, soleid VARCHAR, ifsccode VARCHAR, address VARCHAR, statecode VARCHAR, districtcode VARCHAR, district VARCHAR, city VARCHAR, classification VARCHAR, pincode VARCHAR
);
CREATE TABLE bu_failed_records (
  ID INT PRIMARY KEY, FILEID VARCHAR, FILEDATA VARCHAR, ERRORRESPONSE VARCHAR, ROWNO VARCHAR, CREATEDDATE TIMESTAMP
);
CREATE TABLE bulkupload_details (
  fileid INT PRIMARY KEY, filename VARCHAR, filepath VARCHAR, filetype VARCHAR, batchid VARCHAR, createddate DATETIME, uploadedfilename VARCHAR, userid INT, successcnt INT, totrecordcnt INT, responsecode VARCHAR, responsemsg VARCHAR, errormsg VARCHAR, updateddate DATETIME, channel INT, location VARCHAR, chekeruserid INT, approvedate DATETIME, branchcode VARCHAR, parentmerchantsid INT, status INT, reason VARCHAR, merchantsid INT, vpaid INT
);
CREATE TABLE bussrule (
  bussruleid INT UNSIGNED PRIMARY KEY, rulename VARCHAR, ruledesc VARCHAR, rulepath VARCHAR, ruleaction INT, npcierrorcode VARCHAR, ruletype INT, forall INT, isactive INT, createdon DATETIME, nodeip VARCHAR, shortrulename VARCHAR, featuresupportedvalue VARCHAR, fscommercialvalue VARCHAR, isfeaturesupported INT
);
CREATE TABLE bussrulemapping (
  bussrulemappingid INT UNSIGNED PRIMARY KEY, merchantvpaid INT, bussruleid INT, isactive INT, createdon DATETIME, nodeip VARCHAR, createdby INT, updatedon DATETIME, updatedby INT
);
CREATE TABLE cb_merchant_txn (
  mid INT, date DATETIME, amount DECIMAL, status VARCHAR, msgflow INT
);
CREATE TABLE chargebackdetails (
  chargebackdetailsid INT PRIMARY KEY, merchantsid INT, txnuid VARCHAR, adjdate DATE, adjtype VARCHAR, txndate DATE, txntime TIME, rrn VARCHAR, benmobileno VARCHAR, remmobileno VARCHAR, txnamount DECIMAL, adjamount DECIMAL, bankadjref VARCHAR, benaccountno VARCHAR, remaccountno VARCHAR, disputeflag VARCHAR, reasoncode VARCHAR, status INT, createddate DATETIME, paymentstatus INT, clientrefid VARCHAR, apprejdate DATETIME, comments VARCHAR, userid INT, filename VARCHAR, paymentamount DECIMAL, paymenttxnid VARCHAR, approvetype INT, documentname VARCHAR, referenceno VARCHAR, chargebacktypeid INT, merchanttatdate DATE, npcitatdate DATE, upitransactionid VARCHAR, errorcode VARCHAR, errordescription VARCHAR, bankcomment VARCHAR, paymentdate DATETIME, payervpa VARCHAR, blockresponse VARCHAR, updatedate DATETIME
);
CREATE TABLE chargebackfiledata (
  chargebackfiledataid INT PRIMARY KEY, chargebackdetailsid INT, uid VARCHAR, remitter VARCHAR, beneficiery VARCHAR, response VARCHAR, terminalid VARCHAR, chbdate VARCHAR, chbref VARCHAR, rempayeepspfee DECIMAL, benfee DECIMAL, benfeesw DECIMAL, adjfee DECIMAL, npcifee DECIMAL, remfeetax DECIMAL, benfeetax DECIMAL, npcitax DECIMAL, adjref VARCHAR, adjproof VARCHAR, compensationamount DECIMAL, adjustmentraisedtime TIME, noofdaysforpenalty INT, shdt73 VARCHAR, shdt74 VARCHAR, shdt75 VARCHAR, shdt76 VARCHAR, shdt77 VARCHAR, transactiontype VARCHAR, transactionindicator VARCHAR, aadharno VARCHAR, mobileno VARCHAR, payerpsp VARCHAR, payeepsp VARCHAR, upitransactionid VARCHAR, virtualaddress VARCHAR, mcc VARCHAR, originatingchannel VARCHAR, createddate DATETIME
);
CREATE TABLE chargebacktype (
  chargebacktypeid INT PRIMARY KEY, name VARCHAR, merchanttat INT, npcitat INT, isactive INT, createddate DATETIME, adjtype VARCHAR
);
CREATE TABLE chargeprofile (
  chargeprofileid INT UNSIGNED PRIMARY KEY, chargedesc VARCHAR, chargename VARCHAR, charge DECIMAL, chargetype INT, maxcharge DECIMAL, ismaxcharge INT, lowerlimit DECIMAL, upperlimit DECIMAL, insertedon DATETIME, isactive INT, nodeip VARCHAR
);
CREATE TABLE chargescheme (
  chargeschemeid INT UNSIGNED PRIMARY KEY, chargeschemename VARCHAR, chargeprofileid VARCHAR, insertedon DATETIME, isactive INT, nodeip VARCHAR
);
CREATE TABLE config_params (
  id INT PRIMARY KEY, key_name VARCHAR, description TEXT, file_name VARCHAR, is_file ENUM, created_at DATETIME, updated_at DATETIME
);
CREATE TABLE crondetails (
  crondetailsid INT PRIMARY KEY, cronname VARCHAR, description VARCHAR, isactive INT, createddate DATETIME, nodeip VARCHAR
);
CREATE TABLE cronexecdetails (
  cronexecdetailsid INT PRIMARY KEY, cronkey VARCHAR, cronstatus INT, insertedon DATETIME, completedon DATETIME, nodeip VARCHAR
);
CREATE TABLE csc (
  csci INT PRIMARY KEY, sdt VARCHAR, edt VARCHAR
);
CREATE TABLE dispute_replies (
  id INT PRIMARY KEY, dispute_reply TEXT, disputes_id INT, replied_by INT, replied_at DATETIME, reply_updated DATETIME
);
CREATE TABLE disputeaudit (
  disputeauditid INT UNSIGNED PRIMARY KEY, disputeid INT UNSIGNED, isactive ENUM, changedby INT UNSIGNED, createddate DATETIME
);
CREATE TABLE disputes (
  id INT UNSIGNED PRIMARY KEY, merchants_id INT UNSIGNED, query_raised TEXT, supporting_file VARCHAR, ticket_no VARCHAR, transaction_ref_no VARCHAR, txn_id VARCHAR, query_type ENUM, is_active ENUM, resolved_user_id INT UNSIGNED, created_at DATETIME, updated_at DATETIME, created_user_id INT, branchcode VARCHAR
);
CREATE TABLE documents (
  id INT UNSIGNED PRIMARY KEY, name VARCHAR, slug VARCHAR, is_active BIT, created_at DATETIME
);
CREATE TABLE downloadsummary (
  downloadsummaryid INT PRIMARY KEY, filetype VARCHAR, filename VARCHAR, status BIT, createddate DATETIME, createdby INT, errorcode VARCHAR, errormsg VARCHAR, generateddate DATETIME
);
CREATE TABLE fieldpermission (
  fieldpermissionid INT PRIMARY KEY, name VARCHAR, description VARCHAR, isactive INT, createdby INT, createddate DATETIME, modifiedby INT, modifieddate DATETIME, defaultvalue VARCHAR
);
CREATE TABLE genremaster (
  genreid INT PRIMARY KEY, description VARCHAR, createddate TIMESTAMP
);
CREATE TABLE intermediarymaster (
  id INT UNSIGNED PRIMARY KEY, intermediarytype VARCHAR, description VARCHAR, createdate DATETIME, isactive INT
);
CREATE TABLE intradclog (
  intradclogid INT PRIMARY KEY, remaccountno VARCHAR, benaccountno VARCHAR, clientrefid VARCHAR, amount DECIMAL, transactiondate DATETIME, referenceno VARCHAR, status VARCHAR, errorcode VARCHAR, cbserrorcode VARCHAR, userid INT
);
CREATE TABLE intraftaudit (
  intraftauditid INT PRIMARY KEY, mid INT, txntype INT, uniquemid VARCHAR, remaccountno VARCHAR, benaccountno VARCHAR, clientrefid VARCHAR, amount DECIMAL, note VARCHAR, txnid VARCHAR, channel VARCHAR, txndate DATE, txntime TIME, txnstatus INT, errorcode VARCHAR, errordesc VARCHAR, referenceno VARCHAR, insertedon DATETIME, nodeip VARCHAR, requestid VARCHAR
);
CREATE TABLE invt (
  invtid INT PRIMARY KEY, merchantsid INT UNSIGNED, invtidentifier1 VARCHAR, invtidentifier2 VARCHAR, invtidentifier3 VARCHAR, invtmodel VARCHAR, invtdetails VARCHAR, invttype INT, isactive INT, createdby INT, updatedby INT, createddate DATETIME, updateddate DATETIME, udf1 VARCHAR, udf2 VARCHAR, udf3 VARCHAR, batchid VARCHAR, linkstatus INT, channel INT, linkedby INT, linkeddate DATETIME
);
CREATE TABLE invt_bkp_28082023 (
  invtid INT PRIMARY KEY, merchantsid INT UNSIGNED, invtidentifier1 VARCHAR, invtidentifier2 VARCHAR, invtidentifier3 VARCHAR, invtmodel VARCHAR, invtdetails VARCHAR, invttype INT, isactive INT, createdby INT, updatedby INT, createddate DATETIME, updateddate DATETIME, udf1 VARCHAR, udf2 VARCHAR, udf3 VARCHAR, batchid VARCHAR, linkstatus INT
);
CREATE TABLE invt_testing (
  invtid INT PRIMARY KEY, merchantsid INT UNSIGNED, invtidentifier1 VARCHAR, invtidentifier2 VARCHAR, invtidentifier3 VARCHAR, invtmodel VARCHAR, invtdetails VARCHAR, invttype INT, isactive INT, createdby INT, updatedby INT, createddate DATETIME, updateddate DATETIME, udf1 VARCHAR, udf2 VARCHAR, udf3 VARCHAR, batchid VARCHAR, linkstatus INT, channel INT, linkedby INT, linkeddate DATETIME
);
CREATE TABLE invthistory (
  invthistoryid INT PRIMARY KEY, invtidentifier1 VARCHAR, invtidentifier2 VARCHAR, invtidentifier3 VARCHAR, invtmodel VARCHAR, invtdetails VARCHAR, invttype INT, isactive INT, createdby INT, updatedby INT, createddate DATETIME, updateddate DATETIME, invtid INT, vpadetails VARCHAR, revokedate DATETIME, revokeby INT, udf1 VARCHAR, udf2 VARCHAR, udf3 VARCHAR, channel INT, linkedby INT, linkeddate DATETIME
);
CREATE TABLE languages (
  languageid INT PRIMARY KEY, name VARCHAR
);
CREATE TABLE mandatepresentment (
  mandatepresentmentid INT PRIMARY KEY, batchid VARCHAR, srno INT, umn VARCHAR, loanaccountno VARCHAR, amount VARCHAR, deductiondate VARCHAR, buerrorcode VARCHAR, buerrormsg VARCHAR, createddate DATETIME, createdby INT, predebitnotif INT, status INT, npcierrorcode VARCHAR
);
CREATE TABLE mandatequeue (
  mandatequeueid INT UNSIGNED PRIMARY KEY, merchantmandateid INT, umn VARCHAR, qtype INT, qcount INT, exedate DATETIME
);
CREATE TABLE mcc_codes (
  id INT UNSIGNED PRIMARY KEY, code VARCHAR, description VARCHAR, mcc_category VARCHAR, is_active BIT, created_at DATETIME
);
CREATE TABLE mdrmaster (
  mdrmasterid INT UNSIGNED PRIMARY KEY, mdrkey VARCHAR, merchantcategory VARCHAR, merchanttype VARCHAR, mcc VARCHAR, payermode VARCHAR, ruletype VARCHAR, rulevalue DECIMAL, mdr DECIMAL, ismdrprec INT, isgstreq INT, inserteddate DATETIME, updateddate DATETIME, isactive INT, nodeip VARCHAR
);
CREATE TABLE mdrmerchantconfig (
  mdrmerchantconfigid INT UNSIGNED PRIMARY KEY, merchantvpaid INT, mdrkey VARCHAR, mdr DECIMAL, ismdrprec INT, isgstreq INT, inserteddate DATETIME, isactive INT, nodeip VARCHAR, payermode VARCHAR, ruletype VARCHAR, rulevalue DECIMAL, updateddate DATETIME, updatedby INT
);
CREATE TABLE merchant_bank (
  id INT PRIMARY KEY, mid INT, bank_master_id INT, status ENUM, is_active BIT, created_at TIMESTAMP, updated_at TIMESTAMP
);
CREATE TABLE merchant_cron (
  id INT PRIMARY KEY, mid INT UNSIGNED, time VARCHAR, settlement_time VARCHAR, created_at DATETIME, is_cron INT, updated_at DATETIME
);
CREATE TABLE merchant_group (
  group_id INT PRIMARY KEY, group_name VARCHAR, isactive INT
);
CREATE TABLE merchant_notification_contact (
  notification_id INT PRIMARY KEY, merchants_id INT, notification_type TINYINT, entity_type TINYINT, entity_type_id INT, contact_id INT, is_active BIT, created_date DATETIME, updated_date DATETIME
);
CREATE TABLE merchant_notification_track (
  id INT PRIMARY KEY, subject VARCHAR, message VARCHAR, merchant_id VARCHAR, notification_type VARCHAR, notification_mode VARCHAR, created_at TIMESTAMP
);
CREATE TABLE merchant_users (
  id INT PRIMARY KEY, merchant_id INT, user_id INT, added_by INT, created_at TIMESTAMP
);
CREATE TABLE merchantcharge (
  merchantchargeid INT UNSIGNED PRIMARY KEY, product VARCHAR, mid INT, chargeschemeid INT, isactive INT, mctype INT, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE merchantcheck (
  merchantcheckid INT UNSIGNED PRIMARY KEY, eid INT, mchkkey VARCHAR, ccount INT, camt DECIMAL, bcount INT, bamt DECIMAL, bamt_bkp DECIMAL, mchecktype INT, rstd INT, rstm INT, rsty INT, insertedon DATETIME, updatedon DATETIME, nodeip VARCHAR
);
CREATE TABLE merchantext (
  merchantextid INT UNSIGNED PRIMARY KEY, mid INT, callbackurl VARCHAR, enckey VARCHAR, isactive INT, nodeip VARCHAR, createdon DATETIME
);
CREATE TABLE merchantlimits (
  merchantlimitsid INT UNSIGNED PRIMARY KEY, mid INT, isactive INT, uin INT, insertedon DATETIME, uout INT, iout INT, nodeip VARCHAR, rout INT, nout INT
);
CREATE TABLE merchantmandate (
  merchantmandateid INT UNSIGNED PRIMARY KEY, mid INT, payeevpa VARCHAR, payeename VARCHAR, payeemcc VARCHAR, payeemobno VARCHAR, payeeacdtls VARCHAR, payervpa VARCHAR, payername VARCHAR, payermcc VARCHAR, payeracno VARCHAR, payerifsc VARCHAR, note VARCHAR, clientrefid VARCHAR, amount VARCHAR, deductionamount VARCHAR, mdtexpiry VARCHAR, mdtrevokeable VARCHAR, mdtsharetopayee VARCHAR, mdtstartdate VARCHAR, mdtenddate VARCHAR, mdtamtrule VARCHAR, mdtrecpattern VARCHAR, mdtrecrulevalue VARCHAR, mdtrecruletype VARCHAR, mdtname VARCHAR, mdttype VARCHAR, mdtblockfund VARCHAR, mdtinitiatedby VARCHAR, umn VARCHAR, txnid VARCHAR, mdtstatus INT, mdtec VARCHAR, purposecode VARCHAR, rrn VARCHAR, initiationmode VARCHAR, insertedon DATETIME, nodeip VARCHAR, nextexedate DATETIME, sequenceno INT, prvacno VARCHAR, prvacifsc VARCHAR, msource INT, isexc INT, isrefupd INT, scannedfilename VARCHAR, tprefno VARCHAR, sysmdttype INT
);
CREATE TABLE merchantmdthistory (
  merchantmdthistoryid INT UNSIGNED PRIMARY KEY, mid INT, merchantmandateid INT, umn VARCHAR, reqpayload VARCHAR, rsppayload VARCHAR, status INT, amount VARCHAR, mdttype INT, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE merchants (
  id INT UNSIGNED PRIMARY KEY, uniquemid VARCHAR, mcc_codes_id INT UNSIGNED, name TEXT, spoc_name VARCHAR, contact_email VARCHAR, slug TEXT, address TEXT, group_id INT, pan_card VARCHAR, website VARCHAR, contact_no VARCHAR, merchant_rate VARCHAR, callback_service_address VARCHAR, is_callback_service BIT, is_document_verified BIT, is_live BIT, is_active BIT, parent_merchants_id INT UNSIGNED, user_id INT UNSIGNED, created_at DATETIME, updated_at DATETIME, merchant_code VARCHAR, merchant_key VARCHAR, aggunique_id VARCHAR, callback_type VARCHAR, request_origin VARCHAR, batchid VARCHAR, setttype INT, channel INT, location VARCHAR, bypasscbs INT, aggcburl VARCHAR, aggmdtcburl VARCHAR, ipr INT, nodeip VARCHAR, tpsys INT, mertype INT, pad INT, apipartnerdetailsid INT, pac INT, bankuserid INT, referralname VARCHAR, incorporationdate VARCHAR, gstn VARCHAR, cin VARCHAR, sysmdttype INT, branchcode VARCHAR, hashname VARCHAR, sendtxnreport INT, partnermerchantsid VARCHAR, statecode VARCHAR, updatedby INT, is_soundbox_onboarded INT, is_active_soundbox INT, enablevoicepod INT
);
CREATE TABLE merchants_contact (
  id INT UNSIGNED PRIMARY KEY, name VARCHAR, contact_no VARCHAR, email VARCHAR, is_active ENUM, is_sms ENUM, is_email ENUM, merchants_id INT UNSIGNED, created_at DATETIME, updated_at DATETIME, updatedby INT
);
CREATE TABLE merchants_details (
  id INT PRIMARY KEY, merchants_id INT UNSIGNED, merchant_brand VARCHAR, merchant_legal VARCHAR, merchant_franchise VARCHAR, merchant_owner_type INT, merchant_store_id VARCHAR, merchant_terminal_id VARCHAR, merchant_type INT, merchant_genre INT, onboarding_type INT, created_at DATETIME, updated_at DATETIME, updatedby INT
);
CREATE TABLE merchants_documents (
  id INT PRIMARY KEY, file_name TEXT, merchants_id INT UNSIGNED, documents_id INT UNSIGNED, verified_user_by INT UNSIGNED, is_verified BIT, created_at DATETIME, verified_at DATETIME
);
CREATE TABLE merchants_fd (
  id INT PRIMARY KEY, merchant_detail VARCHAR, is_active INT, is_default INT, merchants_id INT UNSIGNED, vpa_id INT, txn_limit VARCHAR, created_at DATETIME, updated_at DATETIME, merchant_ifsc VARCHAR, merchant_acc_no VARCHAR, settlement_account_no VARCHAR, settlement_account_ifsc VARCHAR, updatedby INT, accounttype INT, comments VARCHAR
);
CREATE TABLE merchants_mode (
  id INT UNSIGNED PRIMARY KEY, env_type ENUM, access_key VARCHAR, is_active ENUM, merchants_id INT UNSIGNED, user_id INT, created_at DATETIME, updated_at DATETIME
);
CREATE TABLE merchants_notifications (
  id INT PRIMARY KEY, merchants_id INT UNSIGNED, merchants_txn_id INT, is_type VARCHAR, status VARCHAR, message_body TEXT, response_message_body VARCHAR, recipient VARCHAR, created_at DATETIME, is_visited INT, notification_trackid INT
);
CREATE TABLE merchants_notify (
  id INT UNSIGNED PRIMARY KEY, contact_no TEXT, email TEXT, callback_service_address TEXT, merchants_id INT UNSIGNED, is_default BIT, is_sms BIT, is_email BIT, is_callback_service BIT, created_at DATETIME, updated_at DATETIME
);
CREATE TABLE merchants_preapproved (
  id INT UNSIGNED PRIMARY KEY, uniquemid VARCHAR, mcc_codes_id INT UNSIGNED, name TEXT, spoc_name VARCHAR, contact_email VARCHAR, slug TEXT, address TEXT, group_id INT, pan_card VARCHAR, website VARCHAR, contact_no VARCHAR, merchant_rate VARCHAR, callback_service_address VARCHAR, is_callback_service BIT, is_document_verified BIT, is_live BIT, is_active BIT, parent_merchants_id INT UNSIGNED, user_id INT UNSIGNED, created_at DATETIME, updated_at DATETIME, merchant_code VARCHAR, merchant_key VARCHAR, aggunique_id VARCHAR, callback_type VARCHAR, request_origin VARCHAR, batchid VARCHAR, setttype INT, channel INT, location VARCHAR, bypasscbs INT, aggcburl VARCHAR, aggmdtcburl VARCHAR, ipr INT, tpsys INT, mertype INT, pad INT, pac INT, nodeip VARCHAR, apipartnerdetailsid INT, bankuserid INT, referralname VARCHAR, incorporationdate VARCHAR, gstn VARCHAR, cin VARCHAR, sysmdttype INT
);
CREATE TABLE merchants_sdk (
  id INT UNSIGNED PRIMARY KEY, merchants_mode_id INT UNSIGNED, sdk_id INT UNSIGNED, is_active BIT, created_at DATETIME, updated_at DATETIME
);
CREATE TABLE merchants_slab (
  id INT PRIMARY KEY, slab_profile VARCHAR, slab_active INT, created_by INT, created_at DATETIME, updated_at DATETIME
);
CREATE TABLE merchants_terminal (
  terminal_id INT PRIMARY KEY, merchant_terminal_id VARCHAR, internal_terminal_id VARCHAR, vpa_id INT, verify_by INT, added_by INT, is_active BIT, created_date DATETIME, verify_date DATETIME
);
CREATE TABLE merchants_txn (
  id INT PRIMARY KEY, mid INT UNSIGNED, narration VARCHAR, payer_vpa VARCHAR, payer_code VARCHAR, payer_name VARCHAR, payer_mobile_no VARCHAR, payer_account_no VARCHAR, payeracctype VARCHAR, payer_ifsc_code VARCHAR, payer_adhar VARCHAR, payee_add VARCHAR, payee_code VARCHAR, payee_name VARCHAR, payee_mobileno VARCHAR, payee_actype VARCHAR, payee_account_details VARCHAR, payee_adhar VARCHAR, payee_account_name VARCHAR, collect_expiry VARCHAR, amount VARCHAR, ref_id VARCHAR, devices VARCHAR, txn_id VARCHAR, hdn_order_id VARCHAR, rrn VARCHAR, txn_type VARCHAR, settlement_status VARCHAR, settlement_date_MSF VARCHAR, MSF_amount VARCHAR, MSF_tax_amount VARCHAR, payout_status VARCHAR, status VARCHAR, error_code VARCHAR, npci_error_code VARCHAR, error_description VARCHAR, created_at DATETIME, updated_at DATETIME, approval_no VARCHAR, customer_ref_id VARCHAR, ybl_txn_id VARCHAR, uniquemid VARCHAR, upi_txn_id VARCHAR, api_id VARCHAR, channel VARCHAR, org_txn_id VARCHAR, org_hdn_order_id VARCHAR, org_rrn VARCHAR, internal_settlement_status INT, orgtxndate VARCHAR, msgflow INT, nodeip VARCHAR, seqno INT, mandateid INT, chg DECIMAL, mdrcharge DECIMAL, chggst DECIMAL, ismdrprec INT, mdr DECIMAL, sgst DECIMAL, cgst DECIMAL, igst DECIMAL, settamount DECIMAL, ttumtype INT, mtt INT, im VARCHAR, pc VARCHAR, npcirc VARCHAR, npcicycle VARCHAR, npcidt VARCHAR, npciadjtype VARCHAR, npciadjdt VARCHAR, adjdt DATETIME, rstatus VARCHAR, rdt DATETIME, roperation VARCHAR, rscenario INT, fstatus VARCHAR, setttype INT
);
CREATE TABLE merchants_txn_archive_22july2025 (
  id INT PRIMARY KEY, mid INT UNSIGNED, narration VARCHAR, payer_vpa VARCHAR, payer_code VARCHAR, payer_name VARCHAR, payer_mobile_no VARCHAR, payeracctype VARCHAR, payer_account_no VARCHAR, payer_ifsc_code VARCHAR, payer_adhar VARCHAR, payee_add VARCHAR, payee_code VARCHAR, payee_name VARCHAR, payee_mobileno VARCHAR, payee_actype VARCHAR, payee_account_details VARCHAR, payee_adhar VARCHAR, payee_account_name VARCHAR, collect_expiry VARCHAR, amount VARCHAR, ref_id VARCHAR, devices VARCHAR, txn_id VARCHAR, hdn_order_id VARCHAR, rrn VARCHAR, txn_type VARCHAR, settlement_status VARCHAR, settlement_date_MSF VARCHAR, MSF_amount VARCHAR, MSF_tax_amount VARCHAR, payout_status VARCHAR, status VARCHAR, error_code VARCHAR, npci_error_code VARCHAR, error_description VARCHAR, created_at DATETIME PRIMARY KEY, updated_at DATETIME, approval_no VARCHAR, customer_ref_id VARCHAR, ybl_txn_id VARCHAR, uniquemid VARCHAR, upi_txn_id VARCHAR, api_id VARCHAR, channel VARCHAR, org_txn_id VARCHAR, org_hdn_order_id VARCHAR, org_rrn VARCHAR, orgtxndate VARCHAR, msgflow INT, internal_settlement_status INT, seqno INT, mandateid INT, mdrcharge DECIMAL, ismdrprec INT, mdr DECIMAL, sgst DECIMAL, cgst DECIMAL, igst DECIMAL, settamount DECIMAL, ttumtype INT, mtt INT, im VARCHAR, pc VARCHAR, npcirc VARCHAR, npcicycle VARCHAR, npcidt VARCHAR, npciadjtype VARCHAR, npciadjdt VARCHAR, adjdt DATETIME, rstatus VARCHAR, rdt DATETIME, roperation VARCHAR, rscenario INT, fstatus VARCHAR, nodeip VARCHAR
);
CREATE TABLE merchants_txn_archive_apr_2025 (
  id INT PRIMARY KEY, mid INT UNSIGNED, narration VARCHAR, payer_vpa VARCHAR, payer_code VARCHAR, payer_name VARCHAR, payer_mobile_no VARCHAR, payeracctype VARCHAR, payer_account_no VARCHAR, payer_ifsc_code VARCHAR, payer_adhar VARCHAR, payee_add VARCHAR, payee_code VARCHAR, payee_name VARCHAR, payee_mobileno VARCHAR, payee_actype VARCHAR, payee_account_details VARCHAR, payee_adhar VARCHAR, payee_account_name VARCHAR, collect_expiry VARCHAR, amount VARCHAR, ref_id VARCHAR, devices VARCHAR, txn_id VARCHAR, hdn_order_id VARCHAR, rrn VARCHAR, txn_type VARCHAR, settlement_status VARCHAR, settlement_date_MSF VARCHAR, MSF_amount VARCHAR, MSF_tax_amount VARCHAR, payout_status VARCHAR, status VARCHAR, error_code VARCHAR, npci_error_code VARCHAR, error_description VARCHAR, created_at DATETIME, updated_at DATETIME, approval_no VARCHAR, customer_ref_id VARCHAR, ybl_txn_id VARCHAR, uniquemid VARCHAR, upi_txn_id VARCHAR, api_id VARCHAR, channel VARCHAR, org_txn_id VARCHAR, org_hdn_order_id VARCHAR, org_rrn VARCHAR, orgtxndate VARCHAR, msgflow INT, internal_settlement_status INT, seqno INT, mandateid INT, mdrcharge DECIMAL, ismdrprec INT, mdr DECIMAL, sgst DECIMAL, cgst DECIMAL, igst DECIMAL, settamount DECIMAL, ttumtype INT, mtt INT, im VARCHAR, pc VARCHAR, npcirc VARCHAR, npcicycle VARCHAR, npcidt VARCHAR, npciadjtype VARCHAR, npciadjdt VARCHAR, adjdt DATETIME, rstatus VARCHAR, rdt DATETIME, roperation VARCHAR, rscenario INT, fstatus VARCHAR, nodeip VARCHAR
);
CREATE TABLE merchants_txn_archive_aug25 (
  id INT PRIMARY KEY, mid INT UNSIGNED, narration VARCHAR, payer_vpa VARCHAR, payer_code VARCHAR, payer_name VARCHAR, payer_mobile_no VARCHAR, payeracctype VARCHAR, payer_account_no VARCHAR, payer_ifsc_code VARCHAR, payer_adhar VARCHAR, payee_add VARCHAR, payee_code VARCHAR, payee_name VARCHAR, payee_mobileno VARCHAR, payee_actype VARCHAR, payee_account_details VARCHAR, payee_adhar VARCHAR, payee_account_name VARCHAR, collect_expiry VARCHAR, amount VARCHAR, ref_id VARCHAR, devices VARCHAR, txn_id VARCHAR, hdn_order_id VARCHAR, rrn VARCHAR, txn_type VARCHAR, settlement_status VARCHAR, settlement_date_MSF VARCHAR, MSF_amount VARCHAR, MSF_tax_amount VARCHAR, payout_status VARCHAR, status VARCHAR, error_code VARCHAR, npci_error_code VARCHAR, error_description VARCHAR, created_at DATETIME PRIMARY KEY, updated_at DATETIME, approval_no VARCHAR, customer_ref_id VARCHAR, ybl_txn_id VARCHAR, uniquemid VARCHAR, upi_txn_id VARCHAR, api_id VARCHAR, channel VARCHAR, org_txn_id VARCHAR, org_hdn_order_id VARCHAR, org_rrn VARCHAR, orgtxndate VARCHAR, msgflow INT, internal_settlement_status INT, seqno INT, mandateid INT, mdrcharge DECIMAL, ismdrprec INT, mdr DECIMAL, sgst DECIMAL, cgst DECIMAL, igst DECIMAL, settamount DECIMAL, ttumtype INT, mtt INT, im VARCHAR, pc VARCHAR, npcirc VARCHAR, npcicycle VARCHAR, npcidt VARCHAR, npciadjtype VARCHAR, npciadjdt VARCHAR, adjdt DATETIME, rstatus VARCHAR, rdt DATETIME, roperation VARCHAR, rscenario INT, fstatus VARCHAR, nodeip VARCHAR, chg DECIMAL, chggst DECIMAL
);
CREATE TABLE merchants_txn_archive_jul25 (
  id INT PRIMARY KEY, mid INT UNSIGNED, narration VARCHAR, payer_vpa VARCHAR, payer_code VARCHAR, payer_name VARCHAR, payer_mobile_no VARCHAR, payeracctype VARCHAR, payer_account_no VARCHAR, payer_ifsc_code VARCHAR, payer_adhar VARCHAR, payee_add VARCHAR, payee_code VARCHAR, payee_name VARCHAR, payee_mobileno VARCHAR, payee_actype VARCHAR, payee_account_details VARCHAR, payee_adhar VARCHAR, payee_account_name VARCHAR, collect_expiry VARCHAR, amount VARCHAR, ref_id VARCHAR, devices VARCHAR, txn_id VARCHAR, hdn_order_id VARCHAR, rrn VARCHAR, txn_type VARCHAR, settlement_status VARCHAR, settlement_date_MSF VARCHAR, MSF_amount VARCHAR, MSF_tax_amount VARCHAR, payout_status VARCHAR, status VARCHAR, error_code VARCHAR, npci_error_code VARCHAR, error_description VARCHAR, created_at DATETIME PRIMARY KEY, updated_at DATETIME, approval_no VARCHAR, customer_ref_id VARCHAR, ybl_txn_id VARCHAR, uniquemid VARCHAR, upi_txn_id VARCHAR, api_id VARCHAR, channel VARCHAR, org_txn_id VARCHAR, org_hdn_order_id VARCHAR, org_rrn VARCHAR, orgtxndate VARCHAR, msgflow INT, internal_settlement_status INT, seqno INT, mandateid INT, mdrcharge DECIMAL, ismdrprec INT, mdr DECIMAL, sgst DECIMAL, cgst DECIMAL, igst DECIMAL, settamount DECIMAL, ttumtype INT, mtt INT, im VARCHAR, pc VARCHAR, npcirc VARCHAR, npcicycle VARCHAR, npcidt VARCHAR, npciadjtype VARCHAR, npciadjdt VARCHAR, adjdt DATETIME, rstatus VARCHAR, rdt DATETIME, roperation VARCHAR, rscenario INT, fstatus VARCHAR, nodeip VARCHAR
);
CREATE TABLE merchants_txn_archive_mar_2025 (
  id INT PRIMARY KEY, mid INT UNSIGNED, narration VARCHAR, payer_vpa VARCHAR, payer_code VARCHAR, payer_name VARCHAR, payer_mobile_no VARCHAR, payeracctype VARCHAR, payer_account_no VARCHAR, payer_ifsc_code VARCHAR, payer_adhar VARCHAR, payee_add VARCHAR, payee_code VARCHAR, payee_name VARCHAR, payee_mobileno VARCHAR, payee_actype VARCHAR, payee_account_details VARCHAR, payee_adhar VARCHAR, payee_account_name VARCHAR, collect_expiry VARCHAR, amount VARCHAR, ref_id VARCHAR, devices VARCHAR, txn_id VARCHAR, hdn_order_id VARCHAR, rrn VARCHAR, txn_type VARCHAR, settlement_status VARCHAR, settlement_date_MSF VARCHAR, MSF_amount VARCHAR, MSF_tax_amount VARCHAR, payout_status VARCHAR, status VARCHAR, error_code VARCHAR, npci_error_code VARCHAR, error_description VARCHAR, created_at DATETIME, updated_at DATETIME, approval_no VARCHAR, customer_ref_id VARCHAR, ybl_txn_id VARCHAR, uniquemid VARCHAR, upi_txn_id VARCHAR, api_id VARCHAR, channel VARCHAR, org_txn_id VARCHAR, org_hdn_order_id VARCHAR, org_rrn VARCHAR, orgtxndate VARCHAR, msgflow INT, internal_settlement_status INT, seqno INT, mandateid INT, mdrcharge DECIMAL, ismdrprec INT, mdr DECIMAL, sgst DECIMAL, cgst DECIMAL, igst DECIMAL, settamount DECIMAL, ttumtype INT, mtt INT, im VARCHAR, pc VARCHAR, nodeip VARCHAR
);
CREATE TABLE merchants_txn_archive_mar_2025_remaining (
  id INT PRIMARY KEY, mid INT UNSIGNED, narration VARCHAR, payer_vpa VARCHAR, payer_code VARCHAR, payer_name VARCHAR, payer_mobile_no VARCHAR, payeracctype VARCHAR, payer_account_no VARCHAR, payer_ifsc_code VARCHAR, payer_adhar VARCHAR, payee_add VARCHAR, payee_code VARCHAR, payee_name VARCHAR, payee_mobileno VARCHAR, payee_actype VARCHAR, payee_account_details VARCHAR, payee_adhar VARCHAR, payee_account_name VARCHAR, collect_expiry VARCHAR, amount VARCHAR, ref_id VARCHAR, devices VARCHAR, txn_id VARCHAR, hdn_order_id VARCHAR, rrn VARCHAR, txn_type VARCHAR, settlement_status VARCHAR, settlement_date_MSF VARCHAR, MSF_amount VARCHAR, MSF_tax_amount VARCHAR, payout_status VARCHAR, status VARCHAR, error_code VARCHAR, npci_error_code VARCHAR, error_description VARCHAR, created_at DATETIME, updated_at DATETIME, approval_no VARCHAR, customer_ref_id VARCHAR, ybl_txn_id VARCHAR, uniquemid VARCHAR, upi_txn_id VARCHAR, api_id VARCHAR, channel VARCHAR, org_txn_id VARCHAR, org_hdn_order_id VARCHAR, org_rrn VARCHAR, orgtxndate VARCHAR, msgflow INT, internal_settlement_status INT, seqno INT, mandateid INT, mdrcharge DECIMAL, ismdrprec INT, mdr DECIMAL, sgst DECIMAL, cgst DECIMAL, igst DECIMAL, settamount DECIMAL, ttumtype INT, mtt INT, im VARCHAR, pc VARCHAR, npcirc VARCHAR, npcicycle VARCHAR, npcidt VARCHAR, npciadjtype VARCHAR, npciadjdt VARCHAR, adjdt DATETIME, rstatus VARCHAR, rdt DATETIME, roperation VARCHAR, rscenario INT, fstatus VARCHAR, nodeip VARCHAR
);
CREATE TABLE merchants_txn_archive_may_2025 (
  id INT PRIMARY KEY, mid INT UNSIGNED, narration VARCHAR, payer_vpa VARCHAR, payer_code VARCHAR, payer_name VARCHAR, payer_mobile_no VARCHAR, payeracctype VARCHAR, payer_account_no VARCHAR, payer_ifsc_code VARCHAR, payer_adhar VARCHAR, payee_add VARCHAR, payee_code VARCHAR, payee_name VARCHAR, payee_mobileno VARCHAR, payee_actype VARCHAR, payee_account_details VARCHAR, payee_adhar VARCHAR, payee_account_name VARCHAR, collect_expiry VARCHAR, amount VARCHAR, ref_id VARCHAR, devices VARCHAR, txn_id VARCHAR, hdn_order_id VARCHAR, rrn VARCHAR, txn_type VARCHAR, settlement_status VARCHAR, settlement_date_MSF VARCHAR, MSF_amount VARCHAR, MSF_tax_amount VARCHAR, payout_status VARCHAR, status VARCHAR, error_code VARCHAR, npci_error_code VARCHAR, error_description VARCHAR, created_at DATETIME, updated_at DATETIME, approval_no VARCHAR, customer_ref_id VARCHAR, ybl_txn_id VARCHAR, uniquemid VARCHAR, upi_txn_id VARCHAR, api_id VARCHAR, channel VARCHAR, org_txn_id VARCHAR, org_hdn_order_id VARCHAR, org_rrn VARCHAR, orgtxndate VARCHAR, msgflow INT, internal_settlement_status INT, seqno INT, mandateid INT, mdrcharge DECIMAL, ismdrprec INT, mdr DECIMAL, sgst DECIMAL, cgst DECIMAL, igst DECIMAL, settamount DECIMAL, ttumtype INT, mtt INT, im VARCHAR, pc VARCHAR, npcirc VARCHAR, npcicycle VARCHAR, npcidt VARCHAR, npciadjtype VARCHAR, npciadjdt VARCHAR, adjdt DATETIME, rstatus VARCHAR, rdt DATETIME, roperation VARCHAR, rscenario INT, fstatus VARCHAR, nodeip VARCHAR
);
CREATE TABLE merchants_txn_archive_oct_2025 (
  id INT PRIMARY KEY, mid INT UNSIGNED, narration VARCHAR, payer_vpa VARCHAR, payer_code VARCHAR, payer_name VARCHAR, payer_mobile_no VARCHAR, payeracctype VARCHAR, payer_account_no VARCHAR, payer_ifsc_code VARCHAR, payer_adhar VARCHAR, payee_add VARCHAR, payee_code VARCHAR, payee_name VARCHAR, payee_mobileno VARCHAR, payee_actype VARCHAR, payee_account_details VARCHAR, payee_adhar VARCHAR, payee_account_name VARCHAR, collect_expiry VARCHAR, amount VARCHAR, ref_id VARCHAR, devices VARCHAR, txn_id VARCHAR, hdn_order_id VARCHAR, rrn VARCHAR, txn_type VARCHAR, settlement_status VARCHAR, settlement_date_MSF VARCHAR, MSF_amount VARCHAR, MSF_tax_amount VARCHAR, payout_status VARCHAR, status VARCHAR, error_code VARCHAR, npci_error_code VARCHAR, error_description VARCHAR, created_at DATETIME PRIMARY KEY, updated_at DATETIME, approval_no VARCHAR, customer_ref_id VARCHAR, ybl_txn_id VARCHAR, uniquemid VARCHAR, upi_txn_id VARCHAR, api_id VARCHAR, channel VARCHAR, org_txn_id VARCHAR, org_hdn_order_id VARCHAR, org_rrn VARCHAR, orgtxndate VARCHAR, msgflow INT, internal_settlement_status INT, seqno INT, mandateid INT, chg DECIMAL, chggst DECIMAL, mdrcharge DECIMAL, ismdrprec INT, mdr DECIMAL, sgst DECIMAL, cgst DECIMAL, igst DECIMAL, settamount DECIMAL, ttumtype INT, mtt INT, im VARCHAR, pc VARCHAR, npcirc VARCHAR, npcicycle VARCHAR, npcidt VARCHAR, npciadjtype VARCHAR, npciadjdt VARCHAR, adjdt DATETIME, rstatus VARCHAR, rdt DATETIME, roperation VARCHAR, rscenario INT, fstatus VARCHAR, nodeip VARCHAR, setttype INT
);
CREATE TABLE merchants_txn_archive_sep_2025 (
  id INT PRIMARY KEY, mid INT UNSIGNED, narration VARCHAR, payer_vpa VARCHAR, payer_code VARCHAR, payer_name VARCHAR, payer_mobile_no VARCHAR, payeracctype VARCHAR, payer_account_no VARCHAR, payer_ifsc_code VARCHAR, payer_adhar VARCHAR, payee_add VARCHAR, payee_code VARCHAR, payee_name VARCHAR, payee_mobileno VARCHAR, payee_actype VARCHAR, payee_account_details VARCHAR, payee_adhar VARCHAR, payee_account_name VARCHAR, collect_expiry VARCHAR, amount VARCHAR, ref_id VARCHAR, devices VARCHAR, txn_id VARCHAR, hdn_order_id VARCHAR, rrn VARCHAR, txn_type VARCHAR, settlement_status VARCHAR, settlement_date_MSF VARCHAR, MSF_amount VARCHAR, MSF_tax_amount VARCHAR, payout_status VARCHAR, status VARCHAR, error_code VARCHAR, npci_error_code VARCHAR, error_description VARCHAR, created_at DATETIME PRIMARY KEY, updated_at DATETIME, approval_no VARCHAR, customer_ref_id VARCHAR, ybl_txn_id VARCHAR, uniquemid VARCHAR, upi_txn_id VARCHAR, api_id VARCHAR, channel VARCHAR, org_txn_id VARCHAR, org_hdn_order_id VARCHAR, org_rrn VARCHAR, orgtxndate VARCHAR, msgflow INT, internal_settlement_status INT, seqno INT, mandateid INT, mdrcharge DECIMAL, ismdrprec INT, mdr DECIMAL, sgst DECIMAL, cgst DECIMAL, igst DECIMAL, settamount DECIMAL, ttumtype INT, mtt INT, im VARCHAR, pc VARCHAR, npcirc VARCHAR, npcicycle VARCHAR, npcidt VARCHAR, npciadjtype VARCHAR, npciadjdt VARCHAR, adjdt DATETIME, rstatus VARCHAR, rdt DATETIME, roperation VARCHAR, rscenario INT, fstatus VARCHAR, nodeip VARCHAR
);
CREATE TABLE merchants_txn_ttmp (
  id INT PRIMARY KEY, mid INT UNSIGNED, narration VARCHAR, payer_vpa VARCHAR, payer_code VARCHAR, payer_name VARCHAR, payer_mobile_no VARCHAR, payeracctype VARCHAR, payer_account_no VARCHAR, payer_ifsc_code VARCHAR, payer_adhar VARCHAR, payee_add VARCHAR, payee_code VARCHAR, payee_name VARCHAR, payee_mobileno VARCHAR, payee_actype VARCHAR, payee_account_details VARCHAR, payee_adhar VARCHAR, payee_account_name VARCHAR, collect_expiry VARCHAR, amount VARCHAR, ref_id VARCHAR, devices VARCHAR, txn_id VARCHAR, hdn_order_id VARCHAR, rrn VARCHAR, txn_type VARCHAR, settlement_status VARCHAR, settlement_date_MSF VARCHAR, MSF_amount VARCHAR, MSF_tax_amount VARCHAR, payout_status VARCHAR, status VARCHAR, error_code VARCHAR, npci_error_code VARCHAR, error_description VARCHAR, created_at DATETIME PRIMARY KEY, updated_at DATETIME, approval_no VARCHAR, customer_ref_id VARCHAR, ybl_txn_id VARCHAR, uniquemid VARCHAR, upi_txn_id VARCHAR, api_id VARCHAR, channel VARCHAR, org_txn_id VARCHAR, org_hdn_order_id VARCHAR, org_rrn VARCHAR, orgtxndate VARCHAR, msgflow INT, internal_settlement_status INT, seqno INT, mandateid INT, mdrcharge DECIMAL, ismdrprec INT, mdr DECIMAL, sgst DECIMAL, cgst DECIMAL, igst DECIMAL, settamount DECIMAL, ttumtype INT, mtt INT, im VARCHAR, pc VARCHAR, npcirc VARCHAR, npcicycle VARCHAR, npcidt VARCHAR, npciadjtype VARCHAR, npciadjdt VARCHAR, adjdt DATETIME, rstatus VARCHAR, rdt DATETIME, roperation VARCHAR, rscenario INT, fstatus VARCHAR, nodeip VARCHAR
);
CREATE TABLE merchants_vpa (
  id INT PRIMARY KEY, merchants_id INT UNSIGNED, vpa VARCHAR, callback_service_address VARCHAR, is_callback_service ENUM, is_active ENUM, is_primary ENUM, is_verified ENUM, vpa_type INT, valid_from DATETIME, valid_to DATETIME, created_at DATETIME, updated_at DATETIME, verified_at DATETIME, verified_by INT, invtid INT, invtlangid INT, updatedby INT, creditlimit DECIMAL, availablelimit DECIMAL, creditlimitdate DATETIME, availablelimitdate DATETIME
);
CREATE TABLE merchantskey (
  merchantskeyid INT UNSIGNED PRIMARY KEY, mid INT, merchantskey VARCHAR, merchantskeyv1 VARCHAR, merchantskeyv2 VARCHAR, isactive INT, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE merchantsrefund (
  merchantsrefundid INT UNSIGNED PRIMARY KEY, mid INT UNSIGNED, orgrrn VARCHAR, clientrefid VARCHAR, refundamount DOUBLE, errorcode VARCHAR, errordescription VARCHAR, txnstatus INT, npcierrorcode VARCHAR, txnid VARCHAR, rrn VARCHAR, orgtxnid VARCHAR, txnnote VARCHAR, msgtype VARCHAR, txntype VARCHAR, orgtxndate VARCHAR, initiationmode VARCHAR, reqadjcode VARCHAR, reqadjflag VARCHAR, txnsubtype VARCHAR, purpose VARCHAR, orgtxnamt VARCHAR, merchantstxnid INT UNSIGNED, createdat DATETIME, updatedat DATETIME, npcirc VARCHAR, fstatus INT, refundadjfiledataid INT UNSIGNED, adjupdatedat DATETIME
);
CREATE TABLE merchanttypemaster (
  merchanttypeid INT PRIMARY KEY, description VARCHAR, createddate TIMESTAMP
);
CREATE TABLE modulemaster (
  moduleid INT PRIMARY KEY, name VARCHAR, createdby INT, createddate DATETIME, isactive INT
);
CREATE TABLE mttmaster (
  mttid INT PRIMARY KEY, description VARCHAR, createddate TIMESTAMP
);
CREATE TABLE mwtmerchants (
  id INT UNSIGNED PRIMARY KEY, name TEXT
);
CREATE TABLE mwtsysconfig (
  mwtsysconfigid INT UNSIGNED PRIMARY KEY, mwtsyskeyname VARCHAR, mwtsyskeyvalue VARCHAR, isactive INT, nodeip VARCHAR, insertedon DATETIME
);
CREATE TABLE nbin (
  nbinid INT UNSIGNED PRIMARY KEY, code VARCHAR, bankname VARCHAR, memtype VARCHAR, banktype VARCHAR, nbin VARCHAR, ifsc VARCHAR, fintype INT, updatedon DATETIME, isactive INT, nodeip VARCHAR
);
CREATE TABLE notifications (
  id INT UNSIGNED PRIMARY KEY, mercahnts_id INT, merchants_txn_id INT, is_type ENUM, sms_status VARCHAR, sms_message VARCHAR, sms_data VARCHAR, sms_gateway_response TEXT, email_subject VARCHAR, email_message TEXT, created_at DATETIME, updated_at DATETIME
);
CREATE TABLE npcicodemaster (
  npcicodemasterid INT PRIMARY KEY, npcierrorcode VARCHAR, errordescription VARCHAR
);
CREATE TABLE npciraw (
  npcirawid INT UNSIGNED PRIMARY KEY, pre VARCHAR, txntype VARCHAR, txnid VARCHAR, rrn VARCHAR, respcode VARCHAR, txndate DATE, txntime VARCHAR, settamt DECIMAL, umn VARCHAR, mapperid VARCHAR, initiationmode VARCHAR, purposecode VARCHAR, pyrcode VARCHAR, pyrmcc VARCHAR, pyrvpa VARCHAR, remcode VARCHAR, remifsc VARCHAR, remactype VARCHAR, remacno VARCHAR, pyecode VARCHAR, pyemcc VARCHAR, pyevpa VARCHAR, bencode VARCHAR, benifsc VARCHAR, benactype VARCHAR, benacno VARCHAR, lrn VARCHAR, resfield1 VARCHAR, resfield2 VARCHAR, resfield3 VARCHAR, npcihdr VARCHAR, npcifilename VARCHAR, npcicycle VARCHAR, optype INT, nodeip VARCHAR, insertedon DATETIME
);
CREATE TABLE npciraw_archive_jun25 (
  npcirawid INT UNSIGNED PRIMARY KEY, pre VARCHAR, txntype VARCHAR, txnid VARCHAR, rrn VARCHAR, respcode VARCHAR, txndate DATE, txntime VARCHAR, settamt DECIMAL, umn VARCHAR, mapperid VARCHAR, initiationmode VARCHAR, purposecode VARCHAR, pyrcode VARCHAR, pyrmcc VARCHAR, pyrvpa VARCHAR, remcode VARCHAR, remifsc VARCHAR, remactype VARCHAR, remacno VARCHAR, pyecode VARCHAR, pyemcc VARCHAR, pyevpa VARCHAR, bencode VARCHAR, benifsc VARCHAR, benactype VARCHAR, benacno VARCHAR, lrn VARCHAR, resfield1 VARCHAR, resfield2 VARCHAR, resfield3 VARCHAR, npcihdr VARCHAR, npcifilename VARCHAR, npcicycle VARCHAR, optype INT, nodeip VARCHAR, insertedon DATETIME
);
CREATE TABLE npcirawfile (
  npcirawfileid INT UNSIGNED PRIMARY KEY, rawfilename VARCHAR, npcifiledate VARCHAR, npcicycle VARCHAR, nor INT, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE onboardsoundboxmapping (
  id INT PRIMARY KEY, merchantsid VARCHAR, vpa VARCHAR, createdby INT, updatedby INT, created_at DATETIME, updated_at DATETIME, invtid INT, invtlangid INT, is_active INT, parentmerchantsid INT UNSIGNED, firstname TEXT, lastname TEXT, address VARCHAR, area VARCHAR, bcagentid VARCHAR, companyname TEXT, email VARCHAR, contact_no VARCHAR, state VARCHAR, district VARCHAR, city VARCHAR, pincode VARCHAR, shopaddress VARCHAR, shoparea VARCHAR, shopcity VARCHAR, shopdistrict VARCHAR, shopname TEXT, shoppincode VARCHAR, shopstate VARCHAR, branchcode VARCHAR, uniquemid VARCHAR, is_bank_address INT, producttype VARCHAR, bcagentname VARCHAR, qrstring TEXT, iservemid VARCHAR, soundboxid INT
);
CREATE TABLE onboardtypeemaster (
  onboardtypeid INT PRIMARY KEY, description VARCHAR, createddate TIMESTAMP
);
CREATE TABLE otpauditlog (
  otpauditlogid INT PRIMARY KEY, usersid INT, otp VARCHAR, createdat DATETIME, purposecode VARCHAR, isused INT
);
CREATE TABLE ownertypeemaster (
  ownertypeid INT PRIMARY KEY, description VARCHAR, createddate TIMESTAMP
);
CREATE TABLE pac_merchants (
  mid INT, merchantname VARCHAR
);
CREATE TABLE pasett (
  pasettid INT UNSIGNED PRIMARY KEY, mid INT, amt DECIMAL, settdate DATE, settcycle VARCHAR, settype VARCHAR, settstatus INT, insertedon DATETIME PRIMARY KEY, nodeip VARCHAR, settledon DATETIME, referenceno VARCHAR, ftstatus INT
);
CREATE TABLE paymentaudit (
  paymentauditid INT UNSIGNED PRIMARY KEY, mid INT, paymentmode INT, amount DECIMAL, benname VARCHAR, benaccno VARCHAR, benifsc VARCHAR, status VARCHAR, errorcode VARCHAR, tpserrorcode VARCHAR, isprepaid INT, mdr DECIMAL, tds DECIMAL, gst DECIMAL, settamount DECIMAL, insertedon DATETIME PRIMARY KEY, nodeip VARCHAR, rrn VARCHAR, txnid VARCHAR, clientrefid VARCHAR, remaccno VARCHAR, errordescription VARCHAR, cid VARCHAR, orgstatus VARCHAR
);
CREATE TABLE paymentaudit_archive (
  paymentauditid INT UNSIGNED PRIMARY KEY, mid INT, paymentmode INT, amount DECIMAL, benname VARCHAR, benaccno VARCHAR, benifsc VARCHAR, status VARCHAR, errorcode VARCHAR, tpserrorcode VARCHAR, isprepaid INT, mdr DECIMAL, tds DECIMAL, gst DECIMAL, settamount DECIMAL, insertedon DATETIME PRIMARY KEY, nodeip VARCHAR, rrn VARCHAR, txnid VARCHAR, clientrefid VARCHAR, remaccno VARCHAR, errordescription VARCHAR, cid VARCHAR
);
CREATE TABLE paymentaudit_ttmp (
  paymentauditid INT UNSIGNED PRIMARY KEY, mid INT, paymentmode INT, amount DECIMAL, benname VARCHAR, benaccno VARCHAR, benifsc VARCHAR, status VARCHAR, errorcode VARCHAR, tpserrorcode VARCHAR, isprepaid INT, mdr DECIMAL, tds DECIMAL, gst DECIMAL, settamount DECIMAL, insertedon DATETIME PRIMARY KEY, nodeip VARCHAR, rrn VARCHAR, txnid VARCHAR, clientrefid VARCHAR, remaccno VARCHAR, errordescription VARCHAR, cid VARCHAR
);
CREATE TABLE permission (
  permissionid INT PRIMARY KEY, name VARCHAR, description VARCHAR, isactive INT, createdby INT, createddate DATETIME, modifiedby INT, modifieddate DATETIME, moduleid INT, submoduleid INT, permissionlable VARCHAR, permissiontype INT
);
CREATE TABLE permissionbkp (
  permissionid INT PRIMARY KEY, name VARCHAR, description VARCHAR, isactive INT, createdby INT, createddate DATETIME, modifiedby INT, modifieddate DATETIME
);
CREATE TABLE pwdhistory (
  id INT PRIMARY KEY, usersid INT, passwordhash VARCHAR, secretkey VARCHAR, isactive INT, isdeleted INT, createdby INT, createddate DATETIME, modifiedby INT, modifieddate DATETIME, isforgotpass INT
);
CREATE TABLE qr_data_printing (
  id INT PRIMARY KEY, vpa VARCHAR, qr_string VARCHAR, batch_id VARCHAR, created_date DATETIME
);
CREATE TABLE refundadjfiledata (
  refundadjfiledataid INT UNSIGNED PRIMARY KEY, txnuid VARCHAR, uid VARCHAR, adjdate DATE, adjtype VARCHAR, remitter VARCHAR, beneficiery VARCHAR, response VARCHAR, txndate DATE, txntime TIME, rrn VARCHAR, terminalid VARCHAR, benmobileno VARCHAR, remmobileno VARCHAR, chbdate VARCHAR, chbref VARCHAR, txnamount DOUBLE, adjamount DOUBLE, rempayeepspfee DOUBLE, benfee DOUBLE, benfeesw DOUBLE, adjfee DOUBLE, npcifee DOUBLE, remfeetax DOUBLE, benfeetax DOUBLE, npcitax DOUBLE, adjref VARCHAR, bankadjref VARCHAR, adjproof VARCHAR, compensationamount DOUBLE, adjustmentraisedtime TIME, noofdaysforpenalty INT, shdt73 VARCHAR, shdt74 VARCHAR, shdt75 VARCHAR, shdt76 VARCHAR, shdt77 VARCHAR, transactiontype VARCHAR, transactionindicator VARCHAR, benaccountno VARCHAR, remaccountno VARCHAR, aadharno VARCHAR, mobileno VARCHAR, payerpsp VARCHAR, payeepsp VARCHAR, upitransactionid VARCHAR, virtualaddress VARCHAR, disputeflag VARCHAR, reasoncode VARCHAR, mcc VARCHAR, originatingchannel VARCHAR, merchantsRefundId INT UNSIGNED, mid INT UNSIGNED, orgrrn VARCHAR, comments VARCHAR, filename VARCHAR, refundadjtypeid INT, createdat DATETIME
);
CREATE TABLE refundadjtype (
  refundadjtypeid INT UNSIGNED PRIMARY KEY, name VARCHAR, isactive INT, createddate DATETIME, adjtype VARCHAR
);
CREATE TABLE revenuesummary (
  revenuesummaryid INT UNSIGNED PRIMARY KEY, mid INT, mname VARCHAR, servicename VARCHAR, rsdate DATE, slab VARCHAR, successtxncount INT, successtxnamount DECIMAL, revenue DECIMAL, bankshare DECIMAL, sysshare DECIMAL, ratio VARCHAR, type INT, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE revenuesummary_archive (
  revenuesummaryid INT UNSIGNED PRIMARY KEY, mid INT, mname VARCHAR, servicename VARCHAR, slab VARCHAR, rsdate DATE, successtxncount INT, successtxnamount DECIMAL, revenue DECIMAL, bankshare DECIMAL, sysshare DECIMAL, ratio VARCHAR, type INT, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE role (
  roleid INT PRIMARY KEY, name VARCHAR, description VARCHAR, isactive INT, isdeleted INT, createdby INT, createddate DATETIME, modifiedby INT, modifieddate DATETIME
);
CREATE TABLE rolebkp (
  roleid INT PRIMARY KEY, name VARCHAR, description VARCHAR, isactive INT, isdeleted INT, createdby INT, createddate DATETIME, modifiedby INT, modifieddate DATETIME
);
CREATE TABLE rolepermission (
  rolepermissionid INT PRIMARY KEY, roleid INT, permissionid INT, isactive INT, isdeleted INT, createdby INT, createddate DATETIME, modifiedby INT, modifieddate DATETIME
);
CREATE TABLE rolepermissionbkp (
  rolepermissionid INT PRIMARY KEY, roleid INT, permissionid INT, isactive INT, isdeleted INT, createdby INT, createddate DATETIME, modifiedby INT, modifieddate DATETIME
);
CREATE TABLE servicemaster (
  servicemasterid INT PRIMARY KEY, url VARCHAR, servicename VARCHAR, moduleid INT, submoduleid INT, action VARCHAR, auditstatus INT, createddate TIMESTAMP
);
CREATE TABLE settlementmaster (
  settlementmasterid INT UNSIGNED PRIMARY KEY, mid INT, nickname VARCHAR, merchantdtls VARCHAR, type INT, isactive INT, createdon DATETIME, nodeip VARCHAR
);
CREATE TABLE shedlock (
  name VARCHAR PRIMARY KEY, lock_until TIMESTAMP, locked_at TIMESTAMP, locked_by VARCHAR
);
CREATE TABLE soundboxmapping (
  soundboxmappingid INT PRIMARY KEY, merchantsid INT, merchantsvpaid INT, soundboxid INT, invtid INT, langid INT, isactive INT, createdby INT, createddate DATETIME, updatedby INT, updateddate DATETIME
);
CREATE TABLE soundboxmaster (
  soundboxid INT PRIMARY KEY, name VARCHAR, isactive INT, onboardtype INT, soundboxkey VARCHAR
);
CREATE TABLE statemaster (
  statemasterid INT PRIMARY KEY, statename VARCHAR, statecode VARCHAR, createdat DATETIME, gstin VARCHAR, isactive INT
);
CREATE TABLE submodulemaster (
  submoduleid INT PRIMARY KEY, name VARCHAR, moduleid INT, createdby INT, createddate DATETIME, isactive INT
);
CREATE TABLE sysaccounts (
  sysaccountsid INT UNSIGNED PRIMARY KEY, sysid VARCHAR, mid INT, merchantid VARCHAR, narration VARCHAR, amount DECIMAL, gst DECIMAL, tds DECIMAL, status INT, sysaccdate DATE, insertedon DATETIME, respcode VARCHAR, nodeip VARCHAR, respdesc VARCHAR
);
CREATE TABLE sysaudit (
  sysauditid INT UNSIGNED PRIMARY KEY, sysid VARCHAR, mid INT, sysprocess VARCHAR, bankprocess VARCHAR, areq VARCHAR, arsp VARCHAR, astatus INT, createdon DATETIME PRIMARY KEY, nodeip VARCHAR
);
CREATE TABLE sysaudit_archive_aug25 (
  sysauditid INT UNSIGNED PRIMARY KEY, sysid VARCHAR, mid INT, sysprocess VARCHAR, bankprocess VARCHAR, areq VARCHAR, arsp VARCHAR, astatus INT, createdon DATETIME PRIMARY KEY, nodeip VARCHAR
);
CREATE TABLE sysaudit_archive_sep25 (
  sysauditid INT UNSIGNED PRIMARY KEY, sysid VARCHAR, mid INT, sysprocess VARCHAR, bankprocess VARCHAR, areq VARCHAR, arsp VARCHAR, astatus INT, createdon DATETIME PRIMARY KEY, nodeip VARCHAR
);
CREATE TABLE sysaudit_archive_till_aug_2025 (
  sysauditid INT UNSIGNED PRIMARY KEY, sysid VARCHAR, mid INT, sysprocess VARCHAR, bankprocess VARCHAR, areq VARCHAR, arsp VARCHAR, astatus INT, createdon DATETIME PRIMARY KEY, nodeip VARCHAR
);
CREATE TABLE temp (
  vpa VARCHAR, txnid VARCHAR, date DATETIME, amount VARCHAR
);
CREATE TABLE testConnection (
  a CHAR
);
CREATE TABLE tpstoken (
  tpstokenid INT UNSIGNED PRIMARY KEY, tpsidentifier VARCHAR, token1 VARCHAR, token2 VARCHAR, token3 VARCHAR, expiresin VARCHAR, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE txnaudit (
  txnauditid INT UNSIGNED PRIMARY KEY, mid INT, txnid VARCHAR, clientrefid VARCHAR, im VARCHAR, insertedon DATETIME, amount DECIMAL, updatedon DATETIME, isactive INT, nodeip VARCHAR, expireon DATETIME, prvacno VARCHAR, prvacifsc VARCHAR, errorcode VARCHAR
);
CREATE TABLE user (
  id INT UNSIGNED PRIMARY KEY, user_type INT, username VARCHAR, u_name VARCHAR, mobile_no VARCHAR, auth_key VARCHAR, password_hash VARCHAR, password_reset_token VARCHAR, email VARCHAR, status SMALLINT, login_attempts SMALLINT, is_lock ENUM, is_deleted ENUM, isBulkUploaded BIT, file_id INT, created_at TIMESTAMP, updated_at TIMESTAMP, profile_image VARCHAR
);
CREATE TABLE users (
  usersid INT PRIMARY KEY, roleid INT, email VARCHAR, passwordhash VARCHAR, mobileno VARCHAR, secretkey VARCHAR, loginattempt INT, lastlogindate DATETIME, invalidloginattempt INT, isactive INT, isdeleted INT, authtoken VARCHAR, createdby INT, createddate DATETIME, modifiedby INT, modifieddate DATETIME, succloginattempt INT, userName VARCHAR, approvalStatus INT, approvedBy INT, approvedDate TIMESTAMP, sysidentifiers VARCHAR, useridentifier VARCHAR, loginchannel INT, fcmtoken VARCHAR, branchcode VARCHAR, merchantsid INT, sessionid VARCHAR, forgotpwdcount INT, lastforgotdate DATETIME, hashusername VARCHAR, otp VARCHAR, invalidotpcount INT, createdotpcount INT, isblocked INT, lastinvalidotpdate DATETIME, blockedat DATETIME
);
CREATE TABLE users_maker (
  usersid INT PRIMARY KEY, primaryuserid INT, roleid INT, email VARCHAR, passwordhash VARCHAR, mobileno VARCHAR, secretkey VARCHAR, loginattempt INT, lastlogindate DATETIME, isactive INT, isdeleted INT, authtoken VARCHAR, createdby INT, createddate DATETIME, modifiedby INT, modifieddate DATETIME, succloginattempt INT, userName VARCHAR, approvalStatus INT, requestFor INT, approvedBy INT, approvedDate TIMESTAMP, rejectReason VARCHAR, sysidentifiers VARCHAR, udf1 INT, branchcode VARCHAR, merchantsid INT, hashusername VARCHAR
);
CREATE TABLE users_mapper (
  id INT PRIMARY KEY, mapper_id INT, user_id INT, added_by INT, created_at TIMESTAMP
);
CREATE TABLE usertoken (
  id INT PRIMARY KEY, email VARCHAR, token VARCHAR, createdby INT, createddate DATETIME, modifiedby INT, modifieddate DATETIME, isExpired INT
);
CREATE TABLE virtualvpadetails (
  virtualvpadetailsid INT PRIMARY KEY, vpa VARCHAR, fileid INT, rownum INT, status INT, activatedby INT, createddate DATETIME, activateddate DATETIME, merchantsid INT, udf1 VARCHAR, udf2 VARCHAR, udf3 VARCHAR, udf4 VARCHAR, vpatype INT
);
CREATE TABLE vpaaudit (
  vpaauditid INT UNSIGNED PRIMARY KEY, mid INT, refid VARCHAR, vpa VARCHAR, benvpa VARCHAR, status INT, charge DECIMAL, isonline TINYINT, insertedon DATETIME, ttype INT, nodeip VARCHAR
);
CREATE TABLE vpaaudit_archive_oct_2024 (
  vpaauditid INT UNSIGNED PRIMARY KEY, mid INT, refid VARCHAR, vpa VARCHAR, benvpa VARCHAR, charge DECIMAL, status INT, ttype INT, isonline TINYINT, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE vpaaudit_archive_remaining_sep_2024 (
  vpaauditid INT UNSIGNED PRIMARY KEY, mid INT, refid VARCHAR, vpa VARCHAR, benvpa VARCHAR, charge DECIMAL, status INT, ttype INT, isonline TINYINT, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE vpaaudit_archive_till_27th_Sep_2024 (
  vpaauditid INT UNSIGNED PRIMARY KEY, mid INT, refid VARCHAR, vpa VARCHAR, benvpa VARCHAR, status INT, isonline TINYINT, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE vpaaudit_archive_till_april_2025 (
  vpaauditid INT UNSIGNED PRIMARY KEY, mid INT, refid VARCHAR, vpa VARCHAR, benvpa VARCHAR, charge DECIMAL, status INT, ttype INT, isonline TINYINT, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE vparepo (
  vparepoid INT UNSIGNED PRIMARY KEY, mid INT, vpa VARCHAR, mcc VARCHAR, response VARCHAR, status INT, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE vparepo_old (
  vparepoid INT UNSIGNED PRIMARY KEY, mid INT, vpa VARCHAR, mcc VARCHAR, response VARCHAR, status INT, insertedon DATETIME, nodeip VARCHAR
);
CREATE TABLE whitelist (
  id INT UNSIGNED PRIMARY KEY, ip_alias_name VARCHAR, ip_address VARCHAR, ip_certificate VARCHAR, success_url VARCHAR, callback_url VARCHAR, slug VARCHAR, merchants_id INT UNSIGNED, resolved_users_id INT UNSIGNED, is_active ENUM, is_verified ENUM, verified_by INT, verified_at TIMESTAMP, created_at DATETIME, updated_at DATETIME
);
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

You can execute tools, retrieve enterprise data, and trigger workflows across cloud or on-prem systems. Based on the user's intent, context, and permissions, you must:

* Identify the correct tool from the available list
* Extract required parameters from the conversation
* Execute the most appropriate action
* Respond clearly with the result or next step
* Dont truncate any information. Show as much information you can under 20,000 characters.

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

    print(f"Reduced data of {len(data)} to {len(output)}")
    return output

