"""Standalone FastAPI microservices for each agent.

Each agent is exposed as an independent HTTP service so the system can be
deployed as separate, independently scalable Docker Compose services:

    retriever_service   -> POST /retrieve   (FAISS vector search)
    reasoning_service   -> POST /reason     (LLM / mock reasoning)
    governance_service  -> POST /decide     (policy-aware decision)

The gateway (app/main.py) calls these services over HTTP when the corresponding
*_URL environment variables are set, and otherwise falls back to in-process
agents (useful for local runs and the evaluation harness).
"""
