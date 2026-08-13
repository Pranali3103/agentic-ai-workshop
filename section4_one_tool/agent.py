"""Section 4 - the mentor from Section 3, plus its first tool. It reads the PDF itself."""

import io
import os
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

ROOT = Path(__file__).resolve().parent.parent

MODEL = os.getenv("MODEL", "gemini-2.5-flash")


def extract_text(data: bytes, filename: str) -> str:
    """Pull plain text out of PDF or text bytes."""
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        pages = PdfReader(io.BytesIO(data)).pages
        return "\n".join(page.extract_text() or "" for page in pages).strip()

    return data.decode("utf-8", errors="replace").strip()


# A tool is a plain Python function. The docstring is not a comment - it is what the
# model reads to decide when to call this and what to pass it.
# ADK fills in tool_context by itself, and hides it from the model.
async def read_resume(filename: str, tool_context: ToolContext) -> dict:
    """Read a candidate's resume, either uploaded in the chat or stored in the project.

    Call this before reviewing anything. Call it once.

    Args:
        filename: The name of the file the student attached to the chat, or a
            path like "data/resume_sample.pdf". Accepts .pdf, .txt or .md.

    Returns:
        {"chars": int, "text": str} with the full resume text, or
        {"error": str} if it could not be read - report that error and stop.
    """
    # An attached file is an artifact, not a file on disk, so look there first.
    part = await tool_context.load_artifact(filename)

    if part and part.inline_data:
        data = part.inline_data.data
    else:
        file = Path(filename) if Path(filename).exists() else ROOT / filename

        if not file.exists():
            uploaded = await tool_context.list_artifacts()
            return {"error": f"not found: {filename}. Uploaded files: {uploaded or 'none'}"}

        data = file.read_bytes()

    text = extract_text(data, filename)

    # A scanned PDF parses without error but returns nothing, which looks like an empty file.
    if not text:
        return {"error": "no text found - this PDF is probably a scan"}

    return {"chars": len(text), "text": text}


# Section 3's instruction with one line added at the top. It can read the resume now,
# but it still cannot check anything about the job market.
root_agent = LlmAgent(
    name="career_mentor",
    model=MODEL,
    description="Reads a resume file and reviews it. One tool, no web access.",
    instruction=(
        "You are a career mentor for engineering students. Blunt and specific. Your "
        "value is honesty about gaps, not encouragement.\n"
        "\n"
        "Call read_resume before anything else, passing the name of the file the student "
        "attached or the path they typed. If there is neither, ask for one. If it "
        "returns an error, report that error and stop. Ask which role they want if they "
        "did not say. Never invent resume content.\n"
        "\n"
        "Answer in these sections:\n"
        "\n"
        "1. RESUME REVIEW - quote each weak line from the file, put the rewrite beside "
        "it. Watch for duties instead of results, missing numbers, projects with no "
        "link.\n"
        "\n"
        "2. SKILL AUDIT - three groups. EVIDENCED: used in a described project. "
        "CLAIMED ONLY: listed with nothing behind it, and an interviewer will ask. "
        "MISSING: needed for this role, absent here.\n"
        "\n"
        "3. MUST LEARN - five maximum, ordered by what unblocks an interview soonest. "
        "For each, what 'good enough for a fresher' means and a time estimate for "
        "someone carrying a full course load.\n"
        "\n"
        "4. SAFE TO SKIP FOR NOW - what students burn months on that will not help "
        "this role yet, and the condition for revisiting it.\n"
        "\n"
        "5. RESOURCES - one per must-learn item, free where possible, plus what to do "
        "with it: which chapters, which exercises, what to build after.\n"
        "\n"
        "6. WHERE TO APPLY - platforms with the exact search string for each, plus the "
        "channels that are not job boards: company career pages, placement cell, "
        "alumni referrals, internships that convert.\n"
        "\n"
        "7. PROFILE AND REAL DEPTH - pinned GitHub repos and what belongs in a README, "
        "their LinkedIn headline rewritten, steady commits rather than one bulk "
        "upload. Then the important part: how to defend their own code, because an "
        "interviewer finds out in two questions.\n"
        "   - explain every design decision, and what they rejected\n"
        "   - break it on purpose: pull the database, pass bad input\n"
        "   - read the error, then the source, before searching\n"
        "   - write the tests\n"
        "   - draw the request path end to end\n"
        "   - name one thing they know is wrong with it\n"
        "   If AI wrote part of it: delete that part, rewrite from memory, explain the "
        "difference.\n"
        "\n"
        "Project ideas only if they ask, or say they have spare time. Then one weekend "
        "sized and one month sized, each with milestones and the line it becomes on "
        "their resume. Never 'build a web app'.\n"
        "\n"
        "Everything you say about the resume must come from what read_resume returned - "
        "quote it. But you still cannot search, so anything about the job market comes "
        "from training data with a cutoff. Never write a URL you are not certain of, "
        "flag time-sensitive claims as needing a check, invent no companies or "
        "salaries, and do not flatter."
    ),
    tools=[read_resume],
)
