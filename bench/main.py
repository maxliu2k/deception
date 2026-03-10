import argparse
import os
import json
from typing import List
from config import (
    DOMAINS, MODEL_IDS, DATA_DIR, MULTI_TURN_OUTPUT_DIR,
    MULTI_TURN_EVAL_DIR, MULTI_TURN_TARGET_MODEL_PARAMS, GPT_MODEL_PARAMS, EVAL_MODEL_PARAMS,
    EVALUATED_MODEL_API_KEY, EVALUATED_MODEL_API_BASE, GPT_API_KEY, GPT_API_BASE,
    MAX_WORKERS, MAX_TURNS, PROMPT_LEVELS, PROMPT_ROLES, PROMPT_INDUCEMENTS, EVAL_MODEL_ID,
    LISTENER_PRIOR_MODEL, LISTENER_PRIOR_KNOWN_FACTS
)
from logger import setup_logger
from processor.multi_turn import generate_dataset_responses as multi_turn_generate
from processor.eval_multi_turn import generate_dataset_responses as eval_multi_turn_generate

logger = setup_logger(__name__)

def validate_prompt_keys(json_path: str, levels: List[str], roles: List[str], inducements: List[str]) -> bool:
    """Validate that at least one L2 prompt key exists in the JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
            prompt_keys = []
            for level in levels:
                for role in roles:
                    for inducement in inducements:
                        prompt_keys.append(f"L2-{role}-{inducement}")
            for sample in data:
                if any(key in sample for key in prompt_keys):
                    return True
            logger.error(f"No valid prompt keys {prompt_keys} found in {json_path}")
            return False
    except Exception as e:
        logger.error(f"Error validating prompt keys in {json_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process dataset with multi-turn generation or evaluation.")
    parser.add_argument(
        "--type",
        choices=["multi", "eval_multi"],
        required=True,
        help="Operation type: 'multi' for generation, 'eval_multi' for evaluation"
    )
    parser.add_argument(
        "--domain",
        choices=DOMAINS,
        help="Domain to process (default: all domains)",
        default=None
    )
    parser.add_argument(
        "--model",
        choices=MODEL_IDS,
        help="Model ID to use for generation (default: all models)",
        default=None
    )
    parser.add_argument(
        "--eval_model",
        choices=MODEL_IDS,
        default=EVAL_MODEL_ID,
        help=f"Model ID for evaluation (default: {EVAL_MODEL_ID})"
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=PROMPT_ROLES,
        default=["self"],
        help="Prompt roles to process"
    )
    parser.add_argument(
        "--inducements",
        nargs="+",
        choices=PROMPT_INDUCEMENTS,
        default=["pressure", "reward"],
        help="Prompt inducements to process"
    )
    parser.add_argument(
        "--listener_prior",
        action="store_true",
        help="Generate listener prior beliefs and prepend to generation prompts"
    )
    parser.add_argument(
        "--listener_prior_known_facts",
        type=int,
        default=LISTENER_PRIOR_KNOWN_FACTS,
        help=f"Number of known listener facts to seed when --listener_prior is enabled (default: {LISTENER_PRIOR_KNOWN_FACTS})"
    )
    args = parser.parse_args()

    if not args.inducements:
        raise ValueError("Inducements (pressure, reward) are required.")

    domains = [args.domain] if args.domain else DOMAINS
    model_ids = [args.model] if args.model else MODEL_IDS

    for domain in domains:
        for model_id in model_ids:
            for role in args.roles:
                for inducement in args.inducements:
                    json_path = os.path.join(DATA_DIR, f"{domain}.json") if args.type == "multi" else os.path.join(MULTI_TURN_OUTPUT_DIR, f"multi_turn_output_L3_{domain}_{role}_{inducement}_{model_id}.jsonl")
                    if args.type == "multi" and not validate_prompt_keys(json_path, ["L3"], [role], [inducement]):
                        logger.error(f"Skipping multi-turn processing for domain={domain}, model={model_id}, role={role}, inducement={inducement} due to missing L2 prompt keys")
                        continue
                    output_dir = MULTI_TURN_OUTPUT_DIR if args.type == "multi" else MULTI_TURN_EVAL_DIR
                    output_filename = f"multi_turn_{'eval' if args.type == 'eval_multi' else 'output'}_L3_{domain}_{role}_{inducement}_{model_id}.jsonl"
                    output_path = os.path.join(output_dir, output_filename)
                    logger.info(f"{'Evaluating' if args.type == 'eval_multi' else 'Processing'} multi-turn: domain={domain}, model={model_id}, role={role}, inducement={inducement}")
                    if args.type == "multi":
                        multi_turn_generate(
                            json_path=json_path,
                            output_path=output_path,
                            model_id=model_id,
                            target_model_params=MULTI_TURN_TARGET_MODEL_PARAMS,
                            gpt_model_params=GPT_MODEL_PARAMS,
                            target_api_key=EVALUATED_MODEL_API_KEY,
                            target_api_base=EVALUATED_MODEL_API_BASE,
                            gpt_api_key=GPT_API_KEY,
                            gpt_api_base=GPT_API_BASE,
                            max_workers=MAX_WORKERS,
                            max_turns=MAX_TURNS,
                            levels=["L3"],
                            roles=[role],
                            inducements=[inducement],
                            listener_prior_enabled=args.listener_prior,
                            listener_prior_model=LISTENER_PRIOR_MODEL,
                            listener_prior_known_facts=args.listener_prior_known_facts,
                        )
                    else:
                        eval_multi_turn_generate(
                            json_path=json_path,
                            output_path=output_path,
                            model_id=args.eval_model,
                            model_params=EVAL_MODEL_PARAMS,
                            api_key=EVALUATED_MODEL_API_KEY,
                            api_base=EVALUATED_MODEL_API_BASE,
                            max_workers=MAX_WORKERS,
                            levels=["L3"],
                            roles=[role],
                            inducements=[inducement]
                        )
                    logger.info(f"Completed processing: domain={domain}, model={model_id}, role={role}, inducement={inducement}")

if __name__ == "__main__":
    main()
