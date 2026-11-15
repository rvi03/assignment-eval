"""Calling the judge model through any OpenAI-compatible endpoint.

Settings come only from environment variables, so no key is ever stored
in the repository. The endpoint is required, so a run never goes to a
provider nobody chose:

| Variable                | Required | Meaning                                   |
|-------------------------|----------|-------------------------------------------|
| `OPENAI_API_KEY`        | yes      | API key for the endpoint.                 |
| `EVALUATOR_MODEL`       | yes      | Model name; there is no default model.    |
| `OPENAI_BASE_URL`       | yes      | Endpoint URL; there is no default.        |
| `EVALUATOR_TEMPERATURE` | no       | Sampling temperature; not sent if unset.  |
| `EVALUATOR_TIMEOUT`     | no       | Seconds per request; default 60.          |

The SDK is the only retry owner: it retries connection errors, timeouts,
rate limits and server errors up to `MAX_RETRIES` times with backoff.
"""

import os
from dataclasses import dataclass
from typing import Protocol

MAX_RETRIES = 2
REQUIRED_VARIABLES = ("OPENAI_API_KEY", "EVALUATOR_MODEL", "OPENAI_BASE_URL")
DEFAULT_TIMEOUT = 60.0


class ConfigError(Exception):
    """The judge cannot be called because a setting is missing or invalid."""


class CallError(Exception):
    """A request to the judge failed after the SDK's retries."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    base_url: str
    temperature: float | None = None
    timeout: float = DEFAULT_TIMEOUT

    @classmethod
    def from_env(cls, env: dict | None = None) -> "Settings":
        env = os.environ if env is None else env
        missing = [n for n in REQUIRED_VARIABLES if not env.get(n)]
        if missing:
            raise ConfigError(
                f"Missing environment variable(s): {', '.join(missing)}. "
                "Set them before running the evaluator; validate and report work without them."
            )
        return cls(
            api_key=env["OPENAI_API_KEY"],
            model=env["EVALUATOR_MODEL"],
            base_url=env["OPENAI_BASE_URL"],
            temperature=_optional_float(env, "EVALUATOR_TEMPERATURE"),
            timeout=_optional_float(env, "EVALUATOR_TIMEOUT") or DEFAULT_TIMEOUT,
        )

    def describe(self) -> dict:
        """The effective settings without the key, for the run record."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "max_retries": MAX_RETRIES,
        }


@dataclass(frozen=True)
class Reply:
    """What the judge sent back, before any parsing."""

    content: str | None
    finish_reason: str | None
    refusal: str | None = None


class JudgeClient(Protocol):
    def complete(
        self, messages: list[dict], schema: dict, max_tokens: int | None = None
    ) -> Reply: ...


class OpenAIJudgeClient:
    """A JudgeClient backed by the `openai` SDK's Chat Completions API."""

    def __init__(self, settings: Settings):
        import openai

        self._openai = openai
        self._settings = settings
        self._client = openai.OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            max_retries=MAX_RETRIES,
        )

    def complete(
        self, messages: list[dict], schema: dict, max_tokens: int | None = None
    ) -> Reply:
        """One structured-output request; `max_tokens` bounds the reply length."""
        options = {}
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        if self._settings.temperature is not None:
            options["temperature"] = self._settings.temperature
        try:
            response = self._client.chat.completions.create(
                model=self._settings.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "case_assessment", "strict": True, "schema": schema},
                },
                **options,
            )
        except self._openai.APIError as exc:
            raise CallError(f"{type(exc).__name__}: {exc}") from exc
        if not response.choices:
            return Reply(content=None, finish_reason=None)
        choice = response.choices[0]
        return Reply(
            content=choice.message.content,
            finish_reason=choice.finish_reason,
            refusal=getattr(choice.message, "refusal", None),
        )


def _optional_float(env, name: str) -> float | None:
    value = env.get(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {value!r}") from exc
