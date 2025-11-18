from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import ask_bot

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    answer = ask_bot(query.question)
    return {"response": answer}
