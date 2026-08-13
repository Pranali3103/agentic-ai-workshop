"""Section 2 - the career mentor with no role and no rules. This one is meant to disappoint."""

import os

from google.adk.agents import LlmAgent

MODEL = os.getenv("MODEL", "gemini-2.5-flash")

# Section 3 is this same agent with a longer instruction. Nothing else changes.
root_agent = LlmAgent(
    name="career_mentor",
    model=MODEL,
    description="A career mentor with a vague instruction.",
    instruction="You are a career mentor. Help the student with their career.",
)
