from openai import OpenAI


MODEL = "gpt-5.2"
AGENT_COUNT = 10
MAX_TURNS = 30


def run_shared_conversation():
    client = OpenAI()
    transcript = []

    seed_prompt = (
        "Shared conversation space with 10 agents. "
        "Begin the discussion with a clear, short topic proposal."
    )
    transcript.append(f"System: {seed_prompt}")

    for turn in range(MAX_TURNS):
        agent_id = (turn % AGENT_COUNT) + 1
        system_msg = (
            f"You are Agent {agent_id} in a 10-agent shared conversation. "
            "Respond in 1-3 sentences. Address the most recent point. "
            "Prefix your response with 'Agent {agent_id}:'."
        )

        convo_text = "\n".join(transcript)
        response = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": convo_text},
            ],
        )
        text = response.output_text.strip()
        if not text.startswith(f"Agent {agent_id}:"):
            text = f"Agent {agent_id}: {text}"
        transcript.append(text)

    print("\n".join(transcript))


if __name__ == "__main__":
    run_shared_conversation()
