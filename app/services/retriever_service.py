"""Independent Retriever microservice.

Run standalone:
    uvicorn app.services.retriever_service:app --host 0.0.0.0 --port 8011
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agents.retriever_agent import RetrieverAgent
from app.utils.logger import get_logger

logger = get_logger("retriever_service", "logs/retriever.log")

app = FastAPI(title="Retriever Service", version="1.0.0")

_retriever = None


def get_retriever() -> RetrieverAgent:
    global _retriever
    if _retriever is None:
        _retriever = RetrieverAgent()
    return _retriever


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
async def health():
    r = get_retriever()
    return {"status": "healthy" if r.index is not None else "down", "agent": "retriever"}


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    try:
        passages = get_retriever().retrieve(req.query, top_k=req.top_k)
    except Exception as e:  # noqa: BLE001
        logger.error("Retrieve failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
    retriever_confidence = max((p.get("score", 0.0) for p in passages), default=0.0)
    return {"passages": passages, "retriever_confidence": float(retriever_confidence)}
