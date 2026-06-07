from __future__ import annotations

import argparse
import asyncio
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict

from .env import TravelGameEnv
from .server import (
    DEFAULT_BATCH_SEEDS,
    DEFAULT_FIVE_ATTR_MEGA_BATCH_SEEDS,
    FIXED_MAX_ROUNDS,
    MEGA_BATCH_MODELS,
    _mega_batch_models,
    _bind_session,
    _execute_batch,
    _latest_model_status,
    _llm_status_notice,
    _mega_batch_export_path,
    _mega_batch_status_path,
    _runtime,
    _summarize_batch_results,
    _summarize_mega_batch,
    _worker_session_id,
    _write_json_atomic,
    _write_text_atomic,
)


def _write_status(status_path: Path, status: Dict[str, Any]) -> None:
    status["updated_at"] = time.time()
    _write_json_atomic(status_path, status)


def _strict_matchup_failure_summary() -> Dict[str, Any]:
    return {
        "attempted_episodes": 1,
        "failed_episodes": 1,
        "valid_episodes": 0,
        "failure_rate": 1.0,
        "n": 0,
        "booking_rate": 0.0,
        "avg_customer_reward": 0.0,
        "avg_resort_reward": 0.0,
        "avg_agent_reward": 0.0,
        "avg_total_welfare": 0.0,
    }


async def _run_worker(*, session_id: str, payload: Dict[str, Any], job_dir: Path) -> None:
    token = _bind_session(session_id)
    status_path = _mega_batch_status_path(session_id)
    export_path = _mega_batch_export_path(session_id)
    mode = str(payload.get("mode") or "buyer_seller_negotiation")
    scenario = payload.get("scenario") or None
    default_seeds = DEFAULT_FIVE_ATTR_MEGA_BATCH_SEEDS if mode == "five_attr" else DEFAULT_BATCH_SEEDS
    seed_list = payload.get("seed_list") or list(default_seeds)
    max_rounds = int(payload.get("max_rounds") or FIXED_MAX_ROUNDS)
    use_models = bool(payload.get("use_models", True))
    mega_models = _mega_batch_models(payload, mode)
    total_matchups = len(mega_models) * len(mega_models)
    started_at = time.time()
    status: Dict[str, Any] = {
        "run_id": str(payload.get("run_id") or f"mega_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"),
        "started_at": started_at,
        "running": True,
        "done": False,
        "error": None,
        "results": [],
        "summary": {},
        "completed_matchups": 0,
        "total_matchups": total_matchups,
        "mode": mode,
        "current_matchup": 0,
        "current_episode": 0,
        "current_seed": None,
        "current_models": [],
        "current_buyer_model": None,
        "current_seller_model": None,
        "current_buyer_budget": None,
        "current_seller_floor": None,
        "current_seller_ask": None,
        "current_used_models": None,
        "current_llm_error": None,
        "current_conversation": [],
        "current_turns": [],
        "pid": os.getpid(),
    }
    export_sections: list[str] = []

    try:
        matchup_rows: list[Dict[str, Any]] = []
        matchup_index = 0
        for buyer_model in mega_models:
            for seller_model in mega_models:
                matchup_index += 1
                selected_models = (
                    [buyer_model, seller_model, seller_model]
                    if mode == "buyer_seller_negotiation"
                    else [buyer_model, seller_model, seller_model, seller_model, seller_model]
                )
                batch_payload = {
                    "num_episodes": len(seed_list),
                    "mode": mode,
                    "scenario": scenario,
                    "selected_models": selected_models,
                    "seed_list": list(seed_list),
                    "max_rounds": max_rounds,
                    "use_models": use_models,
                    "agent_prompt_style": str(payload.get("agent_prompt_style") or "utilitarian"),
                    "agent_policy": str(payload.get("agent_policy") or "correct_known"),
                    "customer_policy": str(payload.get("customer_policy") or "skeptical"),
                }
                status["current_matchup"] = matchup_index
                status["current_episode"] = 0
                status["current_seed"] = None
                status["current_models"] = list(selected_models)
                status["current_buyer_model"] = buyer_model
                status["current_seller_model"] = seller_model
                status["current_buyer_budget"] = None
                status["current_seller_floor"] = None
                status["current_seller_ask"] = None
                status["current_used_models"] = None
                status["current_llm_error"] = None
                status["current_conversation"] = []
                status["current_turns"] = []
                _write_status(status_path, status)

                try:
                    def on_episode_start(episode_num: int, seed: int, selected_models: list[str], current_mode: str, env: TravelGameEnv | None = None) -> None:
                        status["current_episode"] = episode_num
                        status["current_seed"] = seed
                        status["current_models"] = list(selected_models)
                        status["mode"] = current_mode
                        buyer = env.world.get("buyer_true") if env else None
                        seller = env.world.get("seller_true") if env else None
                        status["current_buyer_budget"] = getattr(buyer, "budget", None)
                        status["current_seller_floor"] = getattr(seller, "baseline_value", None)
                        status["current_seller_ask"] = getattr(seller, "asking_price", None)
                        status["current_used_models"] = None
                        status["current_llm_error"] = None
                        status["current_conversation"] = []
                        status["current_turns"] = []
                        _write_status(status_path, status)

                    def on_progress(completed: int, results: list[Dict[str, Any]]) -> None:
                        partial_rows = list(matchup_rows)
                        if results:
                            partial_rows.append(
                                {
                                    "matchup_index": matchup_index,
                                    "buyer_model": buyer_model,
                                    "seller_model": seller_model,
                                    "summary": _summarize_batch_results(list(results), mode),
                                    "results": list(results),
                                    "partial": True,
                                }
                            )
                        status["results"] = partial_rows
                        status["summary"] = _summarize_mega_batch(partial_rows, mode=mode, models=mega_models)
                        if results:
                            latest = results[-1]
                            status["current_used_models"], status["current_llm_error"] = _latest_model_status(latest)
                            status["current_conversation"] = list(latest.get("conversation") or [])
                            worker_runtime = _runtime(_worker_session_id("batch"))
                            status["current_turns"] = list(worker_runtime.step_status.get("turns", [])) if isinstance(worker_runtime.step_status, dict) else []
                            if status["current_used_models"] is False and status["current_llm_error"] and not status["current_conversation"]:
                                status["current_conversation"] = [
                                    {
                                        "speaker": "System",
                                        "recipient": "",
                                        "channel": "negotiation",
                                        "text": _llm_status_notice(mode, status["current_llm_error"]),
                                    }
                                ]
                        _write_status(status_path, status)

                    results, summary, export_text = await _execute_batch(
                        batch_payload,
                        progress_cb=on_progress,
                        episode_start_cb=on_episode_start,
                        store_export=False,
                    )
                    if results:
                        latest = results[-1]
                        status["current_used_models"], status["current_llm_error"] = _latest_model_status(latest)
                        status["current_conversation"] = list(latest.get("conversation") or status.get("current_conversation") or [])
                        worker_runtime = _runtime(_worker_session_id("batch"))
                        status["current_turns"] = list(worker_runtime.step_status.get("turns", [])) if isinstance(worker_runtime.step_status, dict) else list(status.get("current_turns") or [])
                        if status["current_used_models"] is False and status["current_llm_error"] and not status["current_conversation"]:
                            status["current_conversation"] = [
                                {
                                    "speaker": "System",
                                    "recipient": "",
                                    "channel": "negotiation",
                                    "text": _llm_status_notice(mode, status["current_llm_error"]),
                                }
                            ]
                    matchup_rows.append(
                        {
                            "matchup_index": matchup_index,
                            "buyer_model": buyer_model,
                            "seller_model": seller_model,
                            "summary": summary,
                            "results": results,
                        }
                    )
                    if export_text:
                        export_sections.append(
                            "\n".join(
                                [
                                    f"Matchup {matchup_index}: Buyer={buyer_model} | Seller={seller_model}",
                                    export_text.lstrip(),
                                ]
                            )
                        )
                except Exception as exc:
                    error_row = {
                        "matchup_index": matchup_index,
                        "buyer_model": buyer_model,
                        "seller_model": seller_model,
                        "summary": _strict_matchup_failure_summary(),
                        "results": [],
                        "error": str(exc),
                    }
                    matchup_rows.append(error_row)
                    status["results"] = list(matchup_rows)
                    status["completed_matchups"] = max(0, matchup_index - 1)
                    status["summary"] = _summarize_mega_batch(matchup_rows, mode=mode, models=mega_models)
                    _write_status(status_path, status)
                    raise RuntimeError(f"Strict live-model failure in matchup {matchup_index}; stopping mega-batch. {exc}") from exc
                status["results"] = list(matchup_rows)
                status["completed_matchups"] = matchup_index
                status["summary"] = _summarize_mega_batch(matchup_rows, mode=mode, models=mega_models)
                _write_status(status_path, status)

        if export_sections:
            seed_text = ", ".join(str(seed) for seed in seed_list)
            attempted_episodes = sum(int((row.get("summary") or {}).get("attempted_episodes", 0)) for row in matchup_rows)
            failed_episodes = sum(int((row.get("summary") or {}).get("failed_episodes", 0)) for row in matchup_rows)
            matchup_failures = sum(1 for row in matchup_rows if row.get("error"))
            successful_matchups = sum(1 for row in matchup_rows if not row.get("error"))
            export_text = (
                "\n".join(
                    [
                        "Mega-Batch Export",
                        f"Run ID: {status.get('run_id', '')}",
                        f"Mode: {mode}",
                        f"Models: {', '.join(mega_models)}",
                        f"Seeds: [{seed_text}]",
                        f"Matchups completed: {successful_matchups}/{total_matchups}",
                        f"Episodes: attempted={attempted_episodes} valid={attempted_episodes - failed_episodes} failed={failed_episodes}",
                        f"Matchup failures: {matchup_failures}",
                    ]
                )
                + "\n\n"
                + ("\n\n" + ("=" * 72) + "\n\n").join(export_sections)
            )
            _write_text_atomic(export_path, export_text)
    except Exception as exc:
        status["error"] = f"{exc}\n\n{traceback.format_exc()}".strip()
    finally:
        status["running"] = False
        status["done"] = True
        _write_status(status_path, status)
        _runtime().mega_batch_task = None
        _runtime().mega_batch_status = dict(status)
        from .server import SESSION_ID_CTX

        SESSION_ID_CTX.reset(token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Background mega-batch worker for simulation.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--payload-path", required=True)
    args = parser.parse_args()

    payload_path = Path(args.payload_path)
    payload = _read_payload(payload_path) if payload_path.exists() else {}
    asyncio.run(_run_worker(session_id=args.session_id, payload=payload, job_dir=Path(args.job_dir)))


def _read_payload(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
