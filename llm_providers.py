import urllib.request
import urllib.error
import os
import json
import time

from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
from google.genai.errors import ServerError

load_dotenv()

# Load configuration
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)


# -----------------------------
# OpenAI Provider
# -----------------------------
def run_openai(messages):

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from .env")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=CONFIG["openai"]["model"],
        messages=messages,
        max_tokens=CONFIG["openai"].get("max_tokens", 200),
        temperature=CONFIG["openai"].get("temperature", 0.7),
        top_p=CONFIG["openai"].get("top_p", 0.9),
    )

    return (response.choices[0].message.content or "").strip()


# -----------------------------
# Gemini Provider
# -----------------------------
def run_gemini(history):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from .env")

    client = genai.Client(api_key=api_key)

    system_instruction = []
    contents = []
    started = False

    for message in history:

        role = message.get("role")
        content = message.get("content", "")

        if not isinstance(content, str):
            content = str(content)

        if role == "system":
            system_instruction.append(content)

        elif role == "user":

            contents.append({
                "role": "user",
                "parts": [{"text": content}]
            })

            started = True

        elif role == "assistant" and started:

            contents.append({
                "role": "model",
                "parts": [{"text": content}]
            })

    if not contents:
        raise ValueError("No user message found in conversation history.")

    generation_config = types.GenerateContentConfig(
        temperature=CONFIG["gemini"].get("temperature", 0.6),
        top_p=CONFIG["gemini"].get("top_p", 0.8),
        max_output_tokens=CONFIG["gemini"].get(
            "max_output_tokens", 300
        ),
        system_instruction="\n".join(system_instruction) or None,
    )

    # Retry temporary Gemini server errors
    for attempt in range(3):

        try:

            print(f"Gemini API attempt: {attempt + 1}/3")

            response = client.models.generate_content(
                model=CONFIG["gemini"]["model"],
                contents=contents,
                config=generation_config,
            )

            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            return response.text.strip()

        except ServerError:

            if attempt == 2:
                raise

            wait_time = 2 ** (attempt + 1)

            print(
                f"Gemini temporarily unavailable. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

# -----------------------------
# Ollama Provider
# -----------------------------
def run_ollama(history):

    url = "http://localhost:11434/api/chat"

    messages = []

    for message in history:

        role = message.get("role")
        content = message.get("content", "")

        if role in ["system", "user", "assistant"]:

            messages.append({
                "role": role,
                "content": str(content)
            })

    if not messages:
        raise ValueError("No conversation history found.")

    payload = {
        "model": CONFIG["ollama"].get("model", "qwen2.5:1.5b"),
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": CONFIG["ollama"].get("temperature", 0.7),
            "num_predict": CONFIG["ollama"].get("num_predict", 300)
        }
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))

        answer = result.get("message", {}).get("content", "")

        if not answer:
            raise RuntimeError("Ollama returned an empty response.")

        return answer.strip()

    except urllib.error.URLError as e:
        raise RuntimeError(
            "Could not connect to Ollama. "
            "Please make sure Ollama is running."
        ) from e

# -----------------------------
# Provider Router
# -----------------------------
# -----------------------------
# Provider Router
# -----------------------------
def run_llm(history):

    provider = CONFIG.get("provider", "ollama").lower()

    print(f"DEBUG: Selected provider = {provider}")

    if provider == "ollama":
        return run_ollama(history)

    elif provider == "openai":
        return run_openai(history)

    elif provider == "gemini":

        try:
            return run_gemini(history)

        except ServerError as e:

            print("Gemini failed after retries.")

            if CONFIG.get("fallback_to_openai", False):

                print("Switching to OpenAI fallback...")

                return run_openai(history)

            raise RuntimeError(
                "Gemini is temporarily unavailable. "
                "Please try again later."
            ) from e

    else:
        raise ValueError(f"Invalid provider: {provider}")