"""Independent Reasoning microservice.

Run standalone:
    uvicorn app.services.reasoning_service:app --host 0.0.0.0 --port 8012
"""

from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agents.reasoning_agent import ReasoningAgent
from app.utils.logger import get_logger

logger = get_logger("reasoning_service", "logs/reasoning.log")

app = FastAPI(title="Reasoning Service", version="1.0.0")

_reasoner = None


def get_reasoner() -> ReasoningAgent:
    global _reasoner
    if _reasoner is None:
        _reasoner = ReasoningAgent()
    return _reasoner


class ReasonRequest(BaseModel):
    query: str
    passages: List[Dict] = []


@app.get("/health")
async def health():
    try:
        await get_reasoner().reason("ping", [])
        return {"status": "healthy", "agent": "reasoning"}
    except Exception:  # noqa: BLE001
        return {"status": "down", "agent": "reasoning"}


@app.post("/reason")
async def reason(req: ReasonRequest):
    try:
        return await get_reasoner().reason(req.query, req.passages)
    except Exception as e:  # noqa: BLE001
        logger.error("Reason failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {e}")
