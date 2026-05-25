from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

# Force UTF-8 on stdout/stderr so LLM reasoning text with non-ASCII characters
# (e.g. ≈, ×, em-dashes) doesn't crash print() under Windows cp1252 and trigger
# the outer auction try/except — which would silently swap the LLM's decision
# for a math-heuristic fallback.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

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
        print(
            f"[simulation.step_worker] start session={session_id} pid={os.getpid()} mode="
            f"{(runtime.env.config.get('mode') if runtime.env else None)}",
            flush=True,
        )
        _persist_runtime()
        await _run_step_job(payload)
        _persist_runtime()
        print(f"[simulation.step_worker] complete session={session_id} pid={os.getpid()}", flush=True)
    except Exception as exc:
        print(f"[simulation.step_worker] crash session={session_id} pid={os.getpid()} error={exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        runtime = _runtime()
        runtime.step_status["error"] = str(exc)
        runtime.step_status["error_detail"] = traceback.format_exc()
        runtime.step_status["running"] = False
        runtime.step_status["done"] = True
        _persist_runtime()
        raise
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
