import os
from typing import Dict, List

# Project paths (resolve relative to repository root, not current shell directory)
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BENCH_DIR, ".."))

# API Configuration
EVALUATED_MODEL_API_KEY = os.getenv("EVALUATED_MODEL_API_KEY", "")
EVALUATED_MODEL_API_BASE = os.getenv("EVALUATED_MODEL_API_BASE", "")
GPT_API_KEY = os.getenv("GPT_API_KEY", "")
GPT_API_BASE = os.getenv("GPT_API_BASE", "")

# Model Parameters
SINGLE_TURN_MODEL_PARAMS = {
    "max_tokens": 500,
    "temperature": 0.0,
    "top_p": 0.95,
    "do_sample": False
}

MULTI_TURN_TARGET_MODEL_PARAMS = {
    "max_tokens": 500,
    "temperature": 0.0,
    "top_p": 0.95,
    "do_sample": False
}

GPT_MODEL_PARAMS = {
    "model": "gpt-3.5-turbo",
    "max_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.95
}

EVAL_MODEL_PARAMS = {
    "max_tokens": 500,
    "temperature": 0.0,
    "top_p": 0.95,
    "do_sample": False
}

# Processing Parameters
MAX_WORKERS = 50
MAX_TURNS = 5
RETRIES = 3

# Optional listener prior-belief generation (single/multi generation only)
LISTENER_PRIOR_MODEL = "gpt-4o"
LISTENER_PRIOR_KNOWN_FACTS = 2

# Data and Output Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SINGLE_TURN_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result", "generation", "single_turn")  # Single-turn generation (L1, L2): output_L{level}_{role}_{inducement}_{model_id}_{domain}.jsonl
MULTI_TURN_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "result", "generation", "multi_turn")  # Multi-turn generation (L3, using L2 prompts): multi_turn_output_L3_{domain}_{role}_{inducement}_{model_id}.jsonl
SINGLE_TURN_EVAL_DIR = os.path.join(PROJECT_ROOT, "result", "eval", "single_turn")  # Single-turn evaluation (L1, L2): eval_L{level}_{role}_{inducement}_{model_id}_{domain}.jsonl
MULTI_TURN_EVAL_DIR = os.path.join(PROJECT_ROOT, "result", "eval", "multi_turn")  # Multi-turn evaluation (L3, using L2 prompts): multi_turn_eval_L3_{domain}_{role}_{inducement}_{model_id}.jsonl

# Domains and Models
DOMAINS = ["economy", "education", "healthcare", "social", "entertainment"]
MODEL_ID_ALIASES: Dict[str, str] = {
    "Flash": "gemini-3.0-flash",
}
MODEL_IDS = [
    "gpt-3.5-turbo", "gpt-4o", "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219", "gemini-1.5-pro", "gemini-2.0-flash",
    "gemini-3.0-flash", "Flash", "qwen-max-latest", "grok-3", "deepseek-r1"
]
EVAL_MODEL_ID = "gpt-4o"

# Prompt Levels and Dimensions
PROMPT_LEVELS = ["L1", "L2", "L3"]
PROMPT_ROLES = ["self"]
PROMPT_INDUCEMENTS = ["pressure", "reward"]
