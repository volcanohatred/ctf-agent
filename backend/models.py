"""Model resolution — Claude SDK, Bedrock, Gateway (OpenAI-compatible), Azure, Google."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import boto3
from pydantic_ai.models import Model
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.bedrock import BedrockProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

if TYPE_CHECKING:
    from backend.config import Settings

# Effort levels accepted by ClaudeAgentOptions(effort=...). Kept as our own alias
# rather than importing the SDK's, so this module stays importable without the
# claude-agent-sdk installed (only the claude-sdk backend needs it).
EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

# Default model lineup — all via Claude Code CLI routed through the unified
# gateway (ANTHROPIC_BASE_URL). The gateway fronts multiple model families
# behind the Anthropic protocol; model names are gateway-side identifiers.
# Override with --models.
DEFAULT_MODELS: list[str] = [
    "claude-sdk/qwen3.8-max",
    "claude-sdk/deepseek-v4-flash",
    "claude-sdk/kimi-k3",
    "claude-sdk/glm-5.3",
]

# Context window sizes (tokens). Unknown models fall back to 200k.
CONTEXT_WINDOWS: dict[str, int] = {
    "us.anthropic.claude-opus-4-6-v1": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-5": 200_000,
    "gpt-5.4": 1_000_000,
    "gpt-5.4-mini": 400_000,
    "gemini-3-flash-preview": 1_000_000,
    # Gateway models (conservative estimates — adjust to your gateway's specs)
    "kimi-k3": 256_000,
    "deepseek-v4-pro": 256_000,
    "deepseek-v4-flash": 256_000,
    "doubao-seed-2.1-pro": 256_000,
    "glm-5.2": 200_000,
    "glm-5.3": 200_000,
    "qwen3.8-max": 256_000,
    "MiniMax-M3": 200_000,
}

# Models that support vision
VISION_MODELS: set[str] = {
    "us.anthropic.claude-opus-4-6-v1",
    "claude-opus-4-6",
    "claude-sonnet-4-5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gemini-3-flash-preview",
}


def resolve_model(spec: str, settings: Settings) -> Model:
    """Resolve a 'provider/model_id' spec to a Pydantic AI Model."""
    provider = provider_from_spec(spec)
    model_id = model_id_from_spec(spec)
    match provider:
        case "bedrock":
            if settings.aws_bearer_token:
                return BedrockConverseModel(
                    model_id,
                    provider=BedrockProvider(
                        api_key=settings.aws_bearer_token,
                        region_name=settings.aws_region,
                    ),
                )
            session = boto3.Session()
            client = session.client("bedrock-runtime", region_name=settings.aws_region)
            return BedrockConverseModel(
                model_id,
                provider=BedrockProvider(bedrock_client=client),
            )
        case "azure":
            return OpenAIChatModel(
                model_id,
                provider=OpenAIProvider(
                    base_url=settings.azure_openai_endpoint,
                    api_key=settings.azure_openai_api_key,
                ),
            )
        case "zen":
            return OpenAIChatModel(
                model_id,
                provider=OpenAIProvider(
                    base_url="https://opencode.ai/zen/v1",
                    api_key=settings.opencode_zen_api_key,
                ),
            )
        case "gateway":
            if not settings.gateway_base_url or not settings.gateway_api_key:
                raise ValueError(
                    "Provider 'gateway' requires GATEWAY_BASE_URL and "
                    "GATEWAY_API_KEY to be set (see .env.example)."
                )
            return OpenAIChatModel(
                model_id,
                provider=OpenAIProvider(
                    base_url=settings.gateway_base_url,
                    api_key=settings.gateway_api_key,
                ),
            )
        case "google":
            return GoogleModel(
                model_id,
                provider=GoogleProvider(api_key=settings.gemini_api_key),
            )
        case "claude-sdk":
            raise ValueError(
                f"Provider '{provider}' uses its own solver backend, not Pydantic AI. "
                f"resolve_model() should not be called for {spec}."
            )
        case _:
            raise ValueError(f"Unknown provider: {provider}")


def _cast_thinking_level(level: str):
    """Narrow a plain string to google-genai's ThinkingLevel alias.

    Kept as a helper so the provider-specific type never appears in a signature
    of this provider-agnostic module.
    """
    from google.genai.types import ThinkingLevel

    return cast(ThinkingLevel, level)


def resolve_model_settings(spec: str) -> ModelSettings:
    """Get provider-specific model settings with caching enabled."""
    provider = spec.split("/", 1)[0]
    match provider:
        case "bedrock":
            return BedrockModelSettings(
                max_tokens=128_000,
                bedrock_cache_instructions=True,
                bedrock_cache_tool_definitions=True,
                bedrock_cache_messages=True,
            )
        case "azure" | "zen":
            # Azure/Zen use OpenAI chat completions — server-side prompt caching
            # is automatic, no explicit config needed. Set max_tokens to avoid
            # reserving the full context window.
            return OpenAIChatModelSettings(
                max_tokens=128_000,
            )
        case "gateway":
            # Gateway (OpenAI-compatible) models vary widely in max output
            # limits — 16k is broadly safe. No provider-specific caching config.
            return OpenAIChatModelSettings(
                max_tokens=16_000,
            )
        case "google":
            from pydantic_ai.models.google import ThinkingConfigDict

            # google-genai types `thinking_level` as its own ThinkingLevel alias;
            # the wire value is the plain string, so the dict is constructed
            # through the provider's own TypedDict rather than importing a
            # provider-specific enum into this provider-agnostic module.
            return GoogleModelSettings(
                max_tokens=64_000,
                google_thinking_config=ThinkingConfigDict(
                    thinking_level=_cast_thinking_level("high"),
                    include_thoughts=True,
                ),
            )
        case _:
            return ModelSettings(max_tokens=128_000)


def model_id_from_spec(spec: str) -> str:
    """Extract just the model ID from a spec (strips effort suffix)."""
    parts = spec.split("/")
    return parts[1] if len(parts) >= 2 else spec


def provider_from_spec(spec: str) -> str:
    """Extract the provider from a spec."""
    return spec.split("/", 1)[0]


def effort_from_spec(spec: str) -> EffortLevel | None:
    """Extract effort level from a spec like 'claude-sdk/claude-opus-4-6/max'.

    Typed as the SDK's Literal so passing it to ClaudeAgentOptions(effort=...)
    does not widen to `str`.
    """
    parts = spec.split("/")
    if len(parts) >= 3 and parts[2] in EFFORT_LEVELS:
        return cast("EffortLevel", parts[2])
    return None


def claude_cli_env(settings: Settings) -> dict[str, str]:
    """Env vars for the Claude Code CLI subprocess (claude-agent-sdk).

    When anthropic_base_url is set, the CLI talks to the unified gateway
    (Anthropic protocol) instead of api.anthropic.com, and authenticates with
    anthropic_auth_token (falling back to the gateway key). CLAUDECODE is
    cleared to avoid nested-session rejection when launched from a coordinator
    that itself runs under Claude Code.
    """
    env: dict[str, str] = {"CLAUDECODE": ""}
    # The CLI refuses --dangerously-skip-permissions when running as root
    # (our agent container runs as root); IS_SANDBOX=1 marks the container
    # as an intentional sandbox and unlocks it.
    env["IS_SANDBOX"] = "1"

    # Bound the CLI's own HTTP call. Without this a half-open gateway connection
    # can stall a turn indefinitely — the wall-clock budget then interrupts a
    # turn that was never making progress, and the reason is invisible.
    api_timeout_ms = int(getattr(settings, "cli_api_timeout_ms", 0) or 0)
    if api_timeout_ms > 0:
        env["API_TIMEOUT_MS"] = str(api_timeout_ms)

    # Bash tool timeouts. The CLI's default cap (~2 min) is far too short for
    # CTF work (compiles, sage/cado-nfs runs, volatility passes), so raise it to
    # just above the in-container `timeout` wrapper: the container must kill the
    # process first so the model sees a real exit code and stderr, rather than a
    # CLI-side kill with no explanation.
    cmd_s = int(getattr(settings, "sandbox_command_timeout_s", 0) or 0)
    if cmd_s > 0:
        bash_ms = str((cmd_s + 30) * 1000)
        env["BASH_DEFAULT_TIMEOUT_MS"] = str(min(cmd_s, 120) * 1000)
        env["BASH_MAX_TIMEOUT_MS"] = bash_ms

    # Stop the CLI from phoning home: with a gateway base URL these calls either
    # fail slowly or leak usage data, and neither is wanted here.
    env["DISABLE_TELEMETRY"] = "1"
    env["DISABLE_ERROR_REPORTING"] = "1"
    env["DISABLE_AUTOUPDATER"] = "1"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

    base_url = getattr(settings, "anthropic_base_url", "") or ""
    token = (
        getattr(settings, "anthropic_auth_token", "")
        or getattr(settings, "gateway_api_key", "")
        or ""
    )
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url
        if token:
            env["ANTHROPIC_AUTH_TOKEN"] = token
    return env


def supports_vision(spec: str) -> bool:
    """Check if a model spec supports vision."""
    return model_id_from_spec(spec) in VISION_MODELS


def context_window(spec: str) -> int:
    """Get context window size for a model spec."""
    return CONTEXT_WINDOWS.get(model_id_from_spec(spec), 200_000)


# Judgement calls — the Strategist's PIVOT/CONTINUE and the contradiction audit —
# are reasoning tasks, not summarization. Measured against a live gateway on the
# same input, the flash summarizer model answered "NONE" to a plain contradiction
# (finish_reason=stop, so not a truncation: it genuinely could not see it) while a
# reasoning model found it in 8.7s with a correct explanation. Routing these
# through SUMMARIZER_MODEL made both mechanisms decorative, and both fail SILENTLY,
# so nothing downstream looked wrong.
DEFAULT_REASONING_MODEL = "qwen3.8-max"


def resolve_reasoning_model(settings: object, override: str = "") -> str:
    """Model for judgement calls, as opposed to `summarizer_model` for condensing.

    Resolution: an explicit per-call override → REASONING_MODEL →
    COORDINATOR_MODEL → the built-in default. Never the summarizer: see above.
    """
    return (
        (override or "").strip()
        or str(getattr(settings, "reasoning_model", "") or "").strip()
        or str(getattr(settings, "coordinator_model", "") or "").strip()
        or DEFAULT_REASONING_MODEL
    )


# Preference order for the escalation ladder: strongest/most-capable first.
# Only used for entries not already in the running lineup.
ESCALATION_CANDIDATES: tuple[str, ...] = (
    "claude-opus-4-6",
    "claude-sonnet-4-5",
    "gpt-5.4",
    "gemini-3-flash-preview",
    "qwen3.8-max",
    "deepseek-v4-pro",
    "kimi-k3",
    "glm-5.3",
    "deepseek-v4-flash",
)


def escalation_spec(current_specs: list[str], settings: Settings) -> str | None:
    """Pick a stronger model that is NOT already in the running lineup.

    Returning None means "there is nobody left to escalate to" — the caller must
    then stop trying. The previous implementation always resolved to
    `claude-sdk/<coordinator_model or qwen3.8-max>`, which collided with the
    default lineup's first entry and silently made the whole escalation ladder a
    no-op.
    """
    running = {"/".join(s.split("/")[:2]) for s in current_specs}
    preferred = [
        getattr(settings, "coordinator_model", "") or "",
        *ESCALATION_CANDIDATES,
    ]
    for candidate in preferred:
        candidate = candidate.strip()
        if not candidate:
            continue
        spec = candidate if "/" in candidate else f"claude-sdk/{candidate}"
        base = "/".join(spec.split("/")[:2])
        if base not in running:
            return spec
    return None
