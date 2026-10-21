"""Model resolution — Claude SDK, Bedrock, Gateway (OpenAI-compatible), Azure, Google."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
            else:
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
            return GoogleModelSettings(
                max_tokens=64_000,
                google_thinking_config={
                    "thinking_level": "high",
                    "include_thoughts": True,
                },
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


def effort_from_spec(spec: str) -> str | None:
    """Extract effort level from a spec like 'claude-sdk/claude-opus-4-6/max'."""
    parts = spec.split("/")
    if len(parts) >= 3 and parts[2] in ("low", "medium", "high", "max"):
        return parts[2]
    return None


def claude_cli_env(settings: "Settings") -> dict[str, str]:
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
