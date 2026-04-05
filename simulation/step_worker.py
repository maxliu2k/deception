from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict

from .server import SESSION_ID_CTX, _bind_session, _persist_runtime, _run_step_job, _runtime


def _read_payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


async def _run_worker(*, session_id: str, payload: Dict[str, Any]) -> None:
    token = _bind_session(session_id)
    try:
        runtime = _runtime()
        runtime.step_task = None
        runtime.step_status["pid"] = os.getpid()
        runtime.step_status["running"] = True
        runtime.step_status["done"] = False
        runtime.step_status["error"] = None
        _persist_runtime()
        await _run_step_job(payload)
        _persist_runtime()
    finally:
        SESSION_ID_CTX.reset(token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Background step worker for simulation.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--payload-path", required=True)
    args = parser.parse_args()
    payload = _read_payload(Path(args.payload_path))
    asyncio.run(_run_worker(session_id=args.session_id, payload=payload))


if __name__ == "__main__":
    main()
