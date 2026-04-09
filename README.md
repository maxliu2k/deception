# Research Simulation Arena

This repo currently centers on the `simulation/` app: a FastAPI-driven multi-agent sandbox for negotiation, auction, and mediation experiments with both LLM agents and a deterministic `Mathematical` policy.

## What You Can Run

- `buyer_seller_negotiation`
- `open_painting_auction`
- `mediation`
- `simple_resort_deception`
- `five_attr`
- `repeated_mediation`

Model pool in the UI:

- `4o`, `5.4`, `Flash`, `Pro`, `Haiku`, `Sonnet`, `Opus`, `Grok`, `Kimi`, `DeepSeek`, `Llama`, `GLM`, `Mathematical`

## Key Runtime Features

- Save-slot workflow with one transient session plus 4 persistent slots (`save_slot_1` through `save_slot_4`)
- Slot persistence across reload/restart for filled slots
- Delete and force-clear slot controls from the UI
- Batch and mega-batch runs (including export to `.txt`, `.csv`, `.xlsx`)
- Dedicated worker processes for stepping and batch jobs

## Negotiation And Auction Notes

- Negotiation has a hard cap of **8 total messages** per episode.
- If no agreement is reached by message 8, both sides receive 0 reward.
- Prompts include explicit timer semantics, including the "2 messages left means you do not speak again after this move" case.
- Auction mode provides bidders full public state: current bids, budgets, budget history, paintings won, painting index, and a last-painting warning.
- `Mathematical` in auction mode uses deterministic policy logic (balanced/catchup/endgame) rather than an LLM response.

## Requirements

- Python 3.10+
- `pip install fastapi uvicorn openai anthropic aiosqlite`

Optional but recommended model keys (env var or `keys/*.txt`):

- `OPENROUTER_API_KEY` or `keys/openkey.txt`
- `OPENAI_API_KEY` or `keys/gptkey.txt`
- `ANTHROPIC_API_KEY` or `keys/claudekey.txt`
- `GEMINI_API_KEY` or `keys/geminikey.txt`

## Run

```bash
uvicorn simulation.server:app --reload --port 8010
```

Open:

```text
http://127.0.0.1:8010/
```

## Useful API Endpoints

- `GET /api/state`
- `POST /api/reset`
- `POST /api/step`
- `GET /api/step_status`
- `POST /api/run_batch_start`
- `GET /api/run_batch_status`
- `POST /api/run_mega_batch_start`
- `GET /api/run_mega_batch_status`
- `GET /api/save_slots`
- `POST /api/save_slot_delete`
- `POST /api/save_slot_force_clear`

## Important Paths

- `simulation/server.py` FastAPI app, slot/session runtime, stepping and batch orchestration
- `simulation/env.py` environment rules and transitions
- `simulation/policies.py` deterministic policies (including `Mathematical`)
- `simulation/state.py` dataclasses and action/state objects
- `simulation/static_sim/index.html` UI
- `simulation/.runtime/` persisted runtime/session/slot data
- `simulation/arena.sqlite` local sqlite file for simulation DB helpers

## Tests

```bash
pytest simulation/test_env.py
```
