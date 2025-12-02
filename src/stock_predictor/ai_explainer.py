import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)


def get_client(api_key=None):
    # Prefer explicit key, else environment
    key = api_key or os.getenv("OPENAI_API_KEY")

    if not key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Set it in your environment or pass api_key explicitly."
        )

    # Ensure the environment variable is set for OpenAI()
    os.environ["OPENAI_API_KEY"] = key

    # Create the client (it reads key from environment)
    return OpenAI()


def explain_trend(ticker, trend, indicators, api_key=None):
    """
    Ask the OpenAI API to explain why the model thinks the stock
    will go up or down, in simple terms.
    """
    client = get_client(api_key)

    direction_text = "UP" if trend == 1 else "DOWN"
    indicators_text = "\n".join(f"- {k}: {v}" for k, v in indicators.items())

    prompt = (
        f"Stock: {ticker}\n"
        f"Model prediction: {direction_text} tomorrow.\n"
        f"Key indicators:\n{indicators_text}\n\n"
        "Explain in simple terms why the model might predict this, "
        "what these indicators generally mean. Also, mention what an investor "
        "should watch out for in a line."
        "Keep response brief and to the point and under 500 token limit."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7,
    )

    choice = response.choices[0]
    text = choice.message.content
    finish_reason = choice.finish_reason

    if finish_reason != "stop":
        logger.warning(
            f"Explanation finish_reason was '{finish_reason}', text may be truncated."
        )
        text += (
            "\n\n(Note: Explanation may be truncated due to token or safety limits.)"
        )

    return text
