import asyncio
import re
import json
import os
from collections import deque
from typing import Any, Dict
from microsoft.teams.api import MessageActivity, TypingActivityInput
from microsoft.teams.apps import ActivityContext, App
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import utils

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    tool = utils.TOOL_REGISTRY.get(tool_name)
    if not tool:
        raise ValueError(f"Tool '{tool_name}' not implemented.")
    return tool.run(arguments)

DATAFRAME = pd.read_csv("api_pricing.csv")
MODEL = "gpt-4o-mini"
MESSAGES: deque = deque()
MESSAGES.append({"role": "system", "content": utils.SYSTEM_PROMPT})
MAX_HISTORY = 20

def append_messages(turn: dict) -> None:
    """Append a message and prune history to MAX_HISTORY non-system turns."""
    MESSAGES.append(turn)
    if len(MESSAGES) > MAX_HISTORY + 1:
        sys_msg = MESSAGES[0]
        recent = list(MESSAGES)[-(MAX_HISTORY):]
        MESSAGES.clear()
        MESSAGES.append(sys_msg)
        MESSAGES.extend(recent)

client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

def call_llm(query: str) -> str:
    try:
        # Inject fresh timestamp into system prompt on every call
        sys_msg = {
            "role": "system",
            "content": utils.SYSTEM_PROMPT.strip() + f"\nCurrent date and time (IST) is: {utils._now_ist()}",
        }
        MESSAGES[0] = sys_msg

        append_messages({"role": "user", "content": query})

        response = client.chat.completions.create(
            model=MODEL,
            messages=list(MESSAGES),
            tools=utils.TOOLS,  # type: ignore
            tool_choice="auto",
        )
        message = response.choices[0].message

        append_messages(message.model_dump(exclude_unset=False))

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            tool_responses = []
            for tc in tool_calls:
                tool_name = tc.function.name  # type: ignore
                arguments = json.loads(tc.function.arguments)  # type: ignore
                print(f"Calling Tool: {tool_name}\nArguments: {arguments}")
                try:
                    tool_result = execute_tool(tool_name, arguments)
                    print(f"Tool Result: \n{tool_result}")
                except Exception as tool_err:
                    print(f"Tool error: {tool_err}")
                    tool_result = {"error": str(tool_err)}

                tool_responses.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": utils.maintain_context_limit(json.dumps(tool_result)),
                })

            # Add all tool results to history
            for tr in tool_responses:
                append_messages(tr)

            final_response = client.chat.completions.create(
                model=MODEL,
                messages=list(MESSAGES)
            )
            final_message = final_response.choices[0].message
            append_messages(final_message.model_dump(exclude_unset=False))
            reply: str = final_message.content or "Looks like the backend didn't send anything. Can you check the parameters again?"
            return reply

        return message.content or "Looks like the backend didn't send anything. Can you check the parameters again?"

    except Exception as e:
        print(f"Error occurred: {e}")
        return f"Something went wrong while processing your request.\n\nError: {e}"
    
app = App()

@app.on_message_pattern(re.compile(r"Hello|Hi"))
async def handle_greeting(ctx: ActivityContext[MessageActivity]) -> None:
    """Handle greeting messages."""
    print(f"User entered: {ctx.activity.text}")
    print("Responding with: Hello! How can I assist you today?")
    await ctx.send("Hello! How can I assist you today?")

@app.on_message_pattern(re.compile(r"Who are you|Who is this"))
async def handle_inquiry(ctx: ActivityContext[MessageActivity]) -> None:
    """Handle greeting messages."""
    print(f"User entered: {ctx.activity.text}")
    print("Responding with: Hey! I'm BrowsEZ. Need something? Just ask—I'll make it easy.")
    await ctx.send("Hey! I'm BrowsEZ. Need something? Just ask—I'll make it easy.")

@app.on_message
async def handle_message(ctx: ActivityContext[MessageActivity]):
    """Handle message activities using the new generated handler system."""
    await ctx.reply(TypingActivityInput())
    query = ctx.activity.text
    print(f"User entered: {query}")
    print("Calling LLM...")
    response = call_llm(query)
    cost_table = utils.get_cost(MESSAGES, DATAFRAME, MODEL)
    await ctx.send(response+f"\n\n\n\n"+cost_table)

if __name__ == "__main__":
    asyncio.run(app.start())
