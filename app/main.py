from fastapi import FastAPI
from pydantic import BaseModel
from app.agent import FactAgent

app = FastAPI(title="Fake News Fact Agent")
agent = FactAgent()

class InputPayload(BaseModel):
    text: str
    url: str | None = None

@app.post("/verify")
def verify(payload: InputPayload):
    return agent.verify_claim(payload.text, url=payload.url)

@app.get("/")
def health():
    return {"status": "ok"}
