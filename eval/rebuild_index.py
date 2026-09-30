"""Force-rebuild FAISS (+ pickle meta) from CORPUS_DIR after evidence linking.

Usage:
    python -m eval.rebuild_index
"""

from __future__ import annotations

from pathlib import Path

from app.agents.retriever_agent import INDEX_PATH, META_PATH, RetrieverAgent
from app.config import Config


def main() -> None:
    # Remove stale index so RetrieverAgent rebuilds from the new edu2 corpus.
    for p in (INDEX_PATH, META_PATH):
        path = Path(p)
        if path.exists():
            path.unlink()
            print(f"[rebuild_index] Removed {path}")

    agent = RetrieverAgent()
    if agent.index is None or not agent.meta:
        raise SystemExit(
            f"[rebuild_index] Failed: no index built. Check CORPUS_DIR={Config.CORPUS_DIR}"
        )
    print(f"[rebuild_index] OK: {len(agent.meta)} passages indexed")


if __name__ == "__main__":
    main()
