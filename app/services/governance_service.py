"""Independent Governance microservice.

Run standalone:
    uvicorn app.services.governance_service:app --host 0.0.0.0 --port 8013
"""

from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agents.governance_agent import GovernanceAgent
from app.utils.logger import get_logger

logger = get_logger("governance_service", "logs/governance.log")

app = FastAPI(title="Governance Service", version="1.0.0")

_governor = None


def get_governor() -> GovernanceAgent:
    global _governor
    if _governor is None:
        _governor = GovernanceAgent()
    return _governor


class DecideRequest(BaseModel):
    reasoning_result: Dict
    retriever_confidence: Optional[float] = None


@app.get("/health")
async def health():
    get_governor()
    return {"status": "healthy", "agent": "governance"}


@app.post("/decide")
async def decide(req: DecideRequest):
    try:
        return get_governor().decide(
            req.reasoning_result,
            retriever_confidence=req.retriever_confidence,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Decide failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Governance failed: {e}")
