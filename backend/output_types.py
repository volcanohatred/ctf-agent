"""Structured output types for solver agents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FlagFound(BaseModel):
    """A flag the solver has actually VERIFIED (submitted and accepted)."""

    type: Literal["flag_found"] = "flag_found"
    flag: str = Field(description="The verified flag string, exactly as accepted.")
    method: str = Field(description="Brief description of how it was obtained.")


class StillWorking(BaseModel):
    """Turn ended without a verified flag.

    This branch exists on purpose: when `flag_found` was the ONLY allowed
    structured output, a model that had not solved the challenge still had to
    emit something to end its turn — so it invented a flag. That fabricated
    flag then short-circuited the whole swarm (the winner path cancels every
    sibling), which silently killed runs that would otherwise have succeeded.
    """

    type: Literal["still_working"] = "still_working"
    confirmed: str = Field(
        default="",
        description="Facts confirmed so far: paths, offsets, keys, decoded content.",
    )
    excluded: str = Field(
        default="", description="Directions already tried and ruled out."
    )
    next_step: str = Field(default="", description="The single most promising next direction.")


SolverOutput = FlagFound | StillWorking


def solver_output_json_schema() -> dict:
    """JSON schema for solver structured output — shared by Claude SDK solvers.

    Mirrors `SolverOutput`. Only `type` is required at the schema level; the
    flag_found branch is validated in Python so a model cannot end a turn by
    claiming a flag it never verified.
    """
    return {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["flag_found", "still_working"]},
            "flag": {
                "type": "string",
                "description": "Only for type=flag_found: the VERIFIED flag string.",
            },
            "method": {
                "type": "string",
                "description": "Only for type=flag_found: how the flag was obtained.",
            },
            "confirmed": {
                "type": "string",
                "description": "Only for type=still_working: facts confirmed so far.",
            },
            "excluded": {
                "type": "string",
                "description": "Only for type=still_working: directions ruled out.",
            },
            "next_step": {
                "type": "string",
                "description": "Only for type=still_working: most promising next direction.",
            },
        },
        "required": ["type"],
        "additionalProperties": False,
    }
