import asyncio
import logging
import os
import random as _random
import traceback

from openai import APITimeoutError, AsyncOpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_CLOUD_BASE_URL = "http://localhost:11434/v1/"
ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4/"

MAX_RETRIES = 5
RETRY_DELAY = 10
REQUEST_TIMEOUT = 120

# Models that support OpenRouter reasoning config.
REASONING_MODELS = {
    "z-ai/glm-5",
    "minimax/minimax-m2.7",
    "deepseek/deepseek-v3.2",
    "minimax/minimax-m2.5:free",
    "stepfun/step-3.5-flash:free",
}

# Model -> official provider mapping for OpenRouter provider pinning.
PROVIDER_MAP = {
    "z-ai/glm-5": ["deepinfra/fp4"],
    "z-ai/glm-5:online": ["deepinfra/fp4"],
    "minimax/minimax-m2.7": ["minimax/fp8"],
    "deepseek/deepseek-v3.2": ["parasail/fp8"],
}


def get_client(api_key: str | None = None) -> AsyncOpenAI:
    resolved_api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not resolved_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set.\n"
            "Set it in .env or export it."
        )
    return AsyncOpenAI(api_key=resolved_api_key, base_url=OPENROUTER_BASE_URL)


def get_zai_client(api_key: str | None = None) -> AsyncOpenAI:
    if not api_key:
        raise ValueError(
            "ZAI_API_KEY environment variable not set.\n"
            "Set it in .env or export it."
        )
    return AsyncOpenAI(api_key=api_key, base_url=ZAI_BASE_URL)


def get_ollama_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key="ollama", base_url=OLLAMA_CLOUD_BASE_URL)


zai_client = get_zai_client(os.environ.get("ZAI_API_KEY", ""))
ollama_client = get_ollama_client()


def resolve_openai_client_and_model(
    client: AsyncOpenAI,
    model: str,
) -> tuple[AsyncOpenAI, str, str]:
    if model.startswith("ollama:"):
        return ollama_client, model.split(":", 1)[1], "Ollama"
    if model.startswith("zai:"):
        return zai_client, model.split(":", 1)[1], "ZAI"
    return client, model, "OpenRouter"


def build_extra_body(model: str, reasoning_effort: str = "high") -> dict | None:
    extra = {}
    if model in REASONING_MODELS:
        extra["reasoning"] = {"effort": reasoning_effort}
    if model in PROVIDER_MAP:
        extra["provider"] = {"only": PROVIDER_MAP[model]}
    if not extra:
        return None
    return extra


def extract_cost(response) -> float:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0.0
    cost = getattr(usage, "cost", None)
    if cost is not None:
        return float(cost)
    if isinstance(usage, dict):
        return float(usage.get("cost", 0.0))
    return 0.0


async def call_openai(
    client: AsyncOpenAI,
    name: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
    error_logger: logging.Logger,
) -> tuple[str, float]:
    resolved_client, resolved_model, provider_name = resolve_openai_client_and_model(client, model)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {
                "model": resolved_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "timeout": REQUEST_TIMEOUT,
            }
            if provider_name == "OpenRouter":
                extra = build_extra_body(resolved_model, reasoning_effort="medium")
                if extra:
                    kwargs["extra_body"] = extra
            response = await resolved_client.chat.completions.create(**kwargs)
            result = response.choices[0].message.content or ""
            cost = 0.0 if provider_name == "Ollama" else extract_cost(response)
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            output_tokens = getattr(usage, "completion_tokens", None) if usage else None
            if input_tokens is not None and output_tokens is not None:
                tokens = f"{input_tokens}in/{output_tokens}out"
            else:
                tokens = "n/a"
            if not result.strip():
                if attempt < MAX_RETRIES:
                    error_logger.error(f"[{name}] empty response (attempt {attempt}/{MAX_RETRIES}), model={model}")
                    print(f"  [{name}] empty response (attempt {attempt}/{MAX_RETRIES}), retrying ...")
                    await asyncio.sleep(RETRY_DELAY + _random.uniform(0, 5))
                    continue
                error_logger.error(f"[{name}] empty response after {MAX_RETRIES} attempts, model={model}")
                raise ValueError(f"[{name}] empty response after {MAX_RETRIES} attempts")
            print(f"  [{name}] done — {model} ({provider_name}) — {tokens} tokens — ${cost:.4f}")
            return result, cost
        except APITimeoutError:
            error_logger.error(
                f"[{name}] timeout (attempt {attempt}/{MAX_RETRIES}), model={model}\n{traceback.format_exc()}"
            )
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [{name}] timeout (attempt {attempt}/{MAX_RETRIES}), waiting {wait}s ...")
                await asyncio.sleep(wait)
                continue
            raise
        except Exception as error:
            error_logger.error(
                f"[{name}] error (attempt {attempt}/{MAX_RETRIES}), model={model}: {error}\n{traceback.format_exc()}"
            )
            err_str = str(error).lower()
            is_retryable = any(
                keyword in err_str
                for keyword in ["rate_limit", "overloaded", "429", "529", "timeout", "gateway", "502", "503", "504"]
            )
            if is_retryable and attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [{name}] transient error (attempt {attempt}/{MAX_RETRIES}), waiting {wait}s ...", error)
                await asyncio.sleep(wait)
                continue
            raise
    raise RuntimeError(f"[{name}] failed after {MAX_RETRIES} attempts")
