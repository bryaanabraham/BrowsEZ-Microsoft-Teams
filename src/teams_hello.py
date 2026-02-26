from fastapi import FastAPI, Request, Response
from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext,
)
from botbuilder.schema import Activity
import os


APP_ID = os.getenv("MICROSOFT_APP_ID", "")
APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD", "")

settings = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
adapter = BotFrameworkAdapter(settings)

app = FastAPI()


async def on_message(turn_context: TurnContext):
    user_text = turn_context.activity.text

    reply = f"Hello from backend 👋\nYou said: {user_text}"

    await turn_context.send_activity(reply)
    
@app.post("/api/messages")
async def messages(req: Request):

    body = await req.json()
    activity = Activity().deserialize(body)

    auth_header = req.headers.get("Authorization", "")

    async def aux_func(turn_context: TurnContext):
        if turn_context.activity.type == "message":
            await on_message(turn_context)

    await adapter.process_activity(
        activity,
        auth_header,
        aux_func
    )

    return Response(status_code=200)