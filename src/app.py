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
MODEL = "gpt-5.1"
MESSAGES: deque = deque()
MESSAGES.append({"role": "system", "content": utils.SYSTEM_PROMPT})
MAX_HISTORY = 6

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

def call_llm(query: str) -> tuple[str, dict]:
    """
    Call LLM and return response text + usage data.
    Loops until no more tool calls are needed.
    
    Returns:
        tuple: (response_text, usage_dict)
    """
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    max_iterations = 10  # Safety limit to prevent infinite loops
    
    try:
        # Inject fresh timestamp into system prompt on every call
        sys_msg = {
            "role": "system",
            "content": utils.SYSTEM_PROMPT.strip() + f"\nCurrent date and time (IST) is: {utils._now_ist()}",
        }
        MESSAGES[0] = sys_msg

        append_messages({"role": "user", "content": query})

        # Loop until no more tool calls
        for iteration in range(max_iterations):
            response = client.chat.completions.create(
                model=MODEL,
                messages=list(MESSAGES),
                tools=utils.TOOLS,  # type: ignore
                tool_choice="auto",
            )
            message = response.choices[0].message
            
            # Accumulate usage
            if hasattr(response, 'usage') and response.usage:
                total_usage["prompt_tokens"] += response.usage.prompt_tokens
                total_usage["completion_tokens"] += response.usage.completion_tokens
                total_usage["total_tokens"] += response.usage.total_tokens

            append_messages(message.model_dump(exclude_unset=False))

            # Check if there are tool calls
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                # No more tool calls, return the final response
                reply: str = message.content or "Looks like the backend didn't send anything. Can you check the parameters again?"
                return reply, total_usage

            # Execute all tool calls
            tool_responses = []
            for tc in tool_calls:
                tool_name = tc.function.name  # type: ignore
                arguments = json.loads(tc.function.arguments)  # type: ignore
                print(f"[Iteration {iteration + 1}] Calling Tool: {tool_name}\nArguments: {arguments}")
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

            # Continue loop to check if more tool calls are needed

        # If we hit max iterations, return a warning
        print(f"Warning: Reached maximum iterations ({max_iterations})")
        return f"Processing completed but reached maximum tool call limit ({max_iterations} iterations).", total_usage

    except Exception as e:
        print(f"Error occurred: {e}")
        return f"Something went wrong while processing your request.\n\nError: {e}", total_usage
    
app = App()
print("_"*50)
print("Bot is starting...")
print("Using Model: ", MODEL)
print("_"*50)

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
    response, usage_data = call_llm(query)
    cost_table = utils.get_cost(MESSAGES, DATAFRAME, MODEL, usage_data)
    await ctx.send(response+f"\n\n\n\n"+cost_table)

if __name__ == "__main__":
    asyncio.run(app.start())
