import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, List, Any
from data_loader import ensure_output_dir, load_test_samples, get_processed_questions
from utils import generate_single_response, generate_listener_prior_context

logger = logging.getLogger(__name__)


def is_successful_response(result: Dict[str, Any]) -> bool:
    """Return True only when all generated prompt responses are non-error strings."""
    responses = result.get("responses")
    if not isinstance(responses, dict) or not responses:
        return False
    return all(
        isinstance(text, str) and text.strip() and not text.startswith("Error:")
        for text in responses.values()
    )

def generate_prompt_keys(levels: List[str], roles: List[str], inducements: List[str]) -> List[str]:
    """Generate valid prompt keys based on levels, roles, and inducements."""
    prompt_keys = []
    for level in levels:
        for role in roles:
            if inducements:
                for inducement in inducements:
                    prompt_keys.append(f"{level}-{role}-{inducement}")
            else:
                prompt_keys.append(f"{level}-{role}")
    return prompt_keys

def process_sample(sample: Dict, client: OpenAI, model_params: Dict, levels: List[str], roles: List[str], inducements: List[str]) -> Dict[str, Any]:
    """Process a single sample with multiple prompts."""
    try:
        results = {}
        prompt_keys = generate_prompt_keys(levels, roles, inducements)
        logger.debug(f"Generated prompt keys: {prompt_keys} for sample: {sample.get('question', '')[:50]}")
        for key in prompt_keys:
            instruction = sample.get(key)
            if instruction:
                result = generate_single_response(instruction, client, model_params)
                results[key] = result["generated_text"]
            else:
                logger.warning(f"Prompt key {key} not found in sample for question: {sample.get('question', '')[:50]}")

        if not results:
            logger.warning(f"No valid prompts processed for sample: {sample.get('question', '')[:50]}")
            return {
                "topic": sample.get("topic", ""),
                "dimension": sample.get("dimension", ""),
                "question": sample.get("question", ""),
                "groundtruth": sample.get("groundtruth", ""),
                "goal": sample.get("goal", ""),
                "responses": "Error: No valid prompts found",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }

        return {
            "topic": sample["topic"],
            "dimension": sample["dimension"],
            "question": sample["question"],
            "groundtruth": sample["groundtruth"],
            "goal": sample["goal"],
            "responses": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
    except Exception as e:
        logger.error(f"Error processing sample with question: {sample.get('question', '')}... {str(e)}")
        return {
            "topic": sample.get("topic", ""),
            "dimension": sample.get("dimension", ""),
            "question": sample.get("question", ""),
            "groundtruth": sample.get("groundtruth", ""),
            "goal": sample.get("goal", ""),
            "responses": f"Error: {str(e)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }

def process_sample_with_listener_prior(
    sample: Dict,
    client: OpenAI,
    model_params: Dict,
    levels: List[str],
    roles: List[str],
    inducements: List[str],
    prior_client: OpenAI,
    prior_model: str,
    prior_known_facts: int,
) -> Dict[str, Any]:
    """Process a sample after prepending generated listener prior beliefs."""
    try:
        prior_context = generate_listener_prior_context(
            sample=sample,
            client=prior_client,
            model=prior_model,
            known_facts=prior_known_facts,
        )
        results = {}
        prompt_keys = generate_prompt_keys(levels, roles, inducements)
        for key in prompt_keys:
            instruction = sample.get(key)
            if instruction:
                full_instruction = f"{prior_context}\n\n{instruction}" if prior_context else instruction
                result = generate_single_response(full_instruction, client, model_params)
                results[key] = result["generated_text"]
            else:
                logger.warning(f"Prompt key {key} not found in sample for question: {sample.get('question', '')[:50]}")

        if not results:
            return {
                "topic": sample.get("topic", ""),
                "dimension": sample.get("dimension", ""),
                "question": sample.get("question", ""),
                "groundtruth": sample.get("groundtruth", ""),
                "goal": sample.get("goal", ""),
                "responses": "Error: No valid prompts found",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }

        return {
            "topic": sample["topic"],
            "dimension": sample["dimension"],
            "question": sample["question"],
            "groundtruth": sample["groundtruth"],
            "goal": sample["goal"],
            "responses": results,
            "listener_prior_context": prior_context,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
    except Exception as e:
        logger.error(f"Error processing sample with listener prior for question: {sample.get('question', '')}... {str(e)}")
        return {
            "topic": sample.get("topic", ""),
            "dimension": sample.get("dimension", ""),
            "question": sample.get("question", ""),
            "groundtruth": sample.get("groundtruth", ""),
            "goal": sample.get("goal", ""),
            "responses": f"Error: {str(e)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }

def generate_dataset_responses(
    json_path: str,
    output_path: str,
    model_id: str,
    model_params: Dict,
    api_key: str,
    api_base: str,
    max_workers: int,
    levels: List[str],
    roles: List[str],
    inducements: List[str],
    listener_prior_enabled: bool = False,
    listener_prior_api_key: str = "",
    listener_prior_api_base: str = "",
    listener_prior_model: str = "gpt-4o",
    listener_prior_known_facts: int = 2,
) -> None:
    """Process dataset in parallel."""
    try:
        model_params["model"] = model_id
        ensure_output_dir(output_path)
        samples = load_test_samples(json_path)
        processed_questions = get_processed_questions(output_path)
        remaining_samples = [s for s in samples if s["question"] not in processed_questions]
        logger.info(f"Processing {len(remaining_samples)} remaining samples")

        if not remaining_samples:
            logger.info("All questions already processed.")
            return

        client = OpenAI(api_key=api_key, base_url=api_base)
        prior_client = None
        if listener_prior_enabled:
            prior_client = OpenAI(
                api_key=listener_prior_api_key or api_key,
                base_url=listener_prior_api_base or api_base,
            )
        success_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {
                executor.submit(
                    process_sample_with_listener_prior if prior_client else process_sample,
                    s,
                    client,
                    model_params,
                    levels,
                    roles,
                    inducements,
                    prior_client,
                    listener_prior_model,
                    listener_prior_known_facts,
                ) if prior_client else executor.submit(
                    process_sample,
                    s,
                    client,
                    model_params,
                    levels,
                    roles,
                    inducements,
                ): s
                for s in remaining_samples
            }
            with tqdm(total=len(remaining_samples), desc="Processing samples") as pbar:
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    try:
                        result = future.result()
                        with open(output_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        logger.info(f"Saved response for sample: {sample['question'][:50]}...")
                        if is_successful_response(result):
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Error processing sample: {sample['question'][:50]}... {str(e)}")
                        result = {
                            "topic": sample.get("topic", ""),
                            "dimension": sample.get("dimension", ""),
                            "question": sample.get("question", ""),
                            "groundtruth": sample.get("groundtruth", ""),
                            "goal": sample.get("goal", ""),
                            "responses": f"Error: {str(e)}",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        }
                        with open(output_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    pbar.update(1)

        logger.info(f"Completed: {success_count} samples fully successful, {len(remaining_samples) - success_count} with errors")
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")
        raise
