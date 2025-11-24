from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from collections import deque

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

app = FastAPI()


message_histories = {}

MAX_HISTORY = 15  

@app.post("/webhook")
async def dialogflow_webhook(request: Request):
    req = await request.json()
    
    user_text = req.get("queryResult", {}).get("queryText")
    session_id = req.get("session", "default_session")  

    if session_id not in message_histories:
        message_histories[session_id] = deque(maxlen=MAX_HISTORY)

   
    message_histories[session_id].append({"role": "user", "content": user_text})

   
    messages_for_hf = [{"role": "system", "content": "You are an helpful dialogflow telegram chatbot assistant named Mr. Bot"}]
    messages_for_hf.extend(message_histories[session_id])

    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:novita",
            messages=messages_for_hf,
            max_tokens=200,
            temperature=0.5,
        )
        reply_content = completion.choices[0].message.content

        message_histories[session_id].append({"role": "assistant", "content": reply_content})

    except Exception as e:
        print("HF API error:", e)
        reply_content = "Sorry, I could not process your request right now."

    return JSONResponse({"fulfillmentText": reply_content})
