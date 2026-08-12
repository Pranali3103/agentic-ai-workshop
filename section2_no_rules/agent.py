"""Section 2 - the career mentor with no role and no rules. This one is meant to disappoint."""

import os

from google.adk.agents import LlmAgent

MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")

# The whole agent is fine. The instruction is the problem, and that is the point.
# Give it a resume and watch what comes back: praise, generic advice that would fit any
# student, "learn DSA and build projects", and no idea what it is actually looking at.
# Section 3 changes nothing except these words.
root_agent = LlmAgent(
    name="career_mentor",
    model=MODEL,
    description="A career mentor with a vague instruction.",
    instruction="You are a career mentor. Help the student with their career.",
)
