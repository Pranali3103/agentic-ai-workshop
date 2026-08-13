"""Section 6 - three specialists instead of one generalist, run in a fixed order."""

import io
import os
from pathlib import Path

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import ToolContext, google_search
from google.adk.tools.agent_tool import AgentTool

ROOT = Path(__file__).resolve().parent.parent

MODEL = os.getenv("MODEL", "gemini-2.5-flash")


def extract_text(data: bytes, filename: str) -> str:
    """Pull plain text out of PDF or text bytes."""
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        pages = PdfReader(io.BytesIO(data)).pages
        return "\n".join(page.extract_text() or "" for page in pages).strip()

    return data.decode("utf-8", errors="replace").strip()


async def read_resume(filename: str, tool_context: ToolContext) -> dict:
    """Read a candidate's resume, either uploaded in the chat or stored in the project.

    Args:
        filename: The name of the file the student attached to the chat, or a
            path like "data/resume_sample.pdf". Accepts .pdf, .txt or .md.

    Returns:
        {"chars": int, "text": str}, or {"error": str} if it could not be read.
    """
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

    if not text:
        return {"error": "no text found - this PDF is probably a scan"}

    return {"chars": len(text), "text": text}


web_scout = LlmAgent(
    name="web_scout",
    model=MODEL,
    description="Searches the live web for job postings, required skills and learning resources.",
    instruction=(
        "You are a search specialist. Run the search and report only what you found, "
        "with the source URL for every claim. Answer everything in one reply. Give real "
        "company names, job titles, application links and the skills each posting asks "
        "for. Never invent a URL, a company or a salary. If you find nothing, say so."
    ),
    tools=[google_search],
)

# AGENT 1 - one job, one tool. It extracts, it does not advise.
resume_analyst = LlmAgent(
    name="resume_analyst",
    model=MODEL,
    description="Reads the resume file and extracts a structured profile.",
    instruction=(
        "Call read_resume with the name of the file the student attached, or the path "
        "they typed, then extract a profile. Give no advice - a later agent does that. "
        "On error, report the error and nothing else.\n"
        "\n"
        "Output exactly this:\n"
        "- TARGET ROLE: what they asked for, or UNKNOWN\n"
        "- LOCATION: what they said, or UNKNOWN\n"
        "- CURRENT LEVEL: year, degree, CGPA if present\n"
        "- EVIDENCED SKILLS: skills used in a described project, with the project name\n"
        "- CLAIMED ONLY: skills listed with nothing behind them\n"
        "- PROJECTS: name, stack, what it actually does\n"
        "- WEAKNESSES: quote the weak lines verbatim\n"
        "\n"
        "Every line traces to the file. Invent nothing."
    ),
    tools=[read_resume],
    output_key="resume_profile",
)

# AGENT 2 - never sees the PDF. {resume_profile} is Agent 1's output, read from session state.
market_researcher = LlmAgent(
    name="market_researcher",
    model=MODEL,
    description="Finds what the live job market asks for, based on the extracted profile.",
    instruction=(
        "You research the job market for this candidate:\n"
        "\n"
        "=== PROFILE ===\n"
        "{resume_profile}\n"
        "=== END ===\n"
        "\n"
        "Call web_scout exactly once. In that single request ask for current openings "
        "for the target role and location, the skills those postings require, and the "
        "application links. Do not call it twice - free-tier quota is limited.\n"
        "\n"
        "Then report:\n"
        "- OPENINGS: company, title, link\n"
        "- REQUIRED SKILLS: each skill, and how many postings asked for it\n"
        "- NOT ASKED FOR: things students assume matter that no posting mentioned\n"
        "\n"
        "Only what the search returned. Nothing from memory."
    ),
    tools=[AgentTool(agent=web_scout)],
    output_key="market_data",
)

# AGENT 3 - no tools at all. It only reasons over what the first two produced.
plan_writer = LlmAgent(
    name="plan_writer",
    model=MODEL,
    description="Turns the profile and market data into the final report.",
    instruction=(
        "You are a career mentor. Blunt and specific. Write the final report from the "
        "two inputs below. You have no tools, so anything not in them, you do not "
        "know.\n"
        "\n"
        "=== PROFILE ===\n"
        "{resume_profile}\n"
        "=== MARKET DATA ===\n"
        "{market_data}\n"
        "=== END ===\n"
        "\n"
        "1. RESUME REVIEW - quote each weak line, put the rewrite beside it.\n"
        "\n"
        "2. GAP ANALYSIS - a table: Skill | In resume? | Postings asking for it | "
        "Priority. A row is only allowed if the skill appears in the market data.\n"
        "\n"
        "3. MUST LEARN - top three gaps. Each with what 'good enough for a fresher' "
        "means and a time estimate for someone carrying a full course load.\n"
        "\n"
        "4. SAFE TO SKIP FOR NOW - from the NOT ASKED FOR list, with the condition for "
        "revisiting each one.\n"
        "\n"
        "5. WHERE TO APPLY - the openings from the market data, links exactly as given.\n"
        "\n"
        "6. PROFILE AND REAL DEPTH - pinned GitHub repos and README contents, their "
        "LinkedIn headline rewritten, steady commits rather than one bulk upload. Then "
        "how to defend their own code, because an interviewer finds out in two "
        "questions.\n"
        "   - explain every design decision, and what they rejected\n"
        "   - break it on purpose: pull the database, pass bad input\n"
        "   - read the error, then the source, before searching\n"
        "   - write the tests\n"
        "   - name one thing they know is wrong with it\n"
        "   If AI wrote part of it: delete that part, rewrite from memory, explain the "
        "difference.\n"
        "\n"
        "Project ideas only if the student asked.\n"
        "\n"
        "Only URLs that appear in the market data. If it is thin, say so and write "
        "less. Do not flatter."
    ),
    output_key="career_plan",
)

# The order is fixed by this list, not chosen by a model, which makes it a workflow.
# An LlmAgent with sub_agents would let the model pick who runs next.
root_agent = SequentialAgent(
    name="career_team",
    description="Runs the career mentoring team: analyst, then researcher, then planner.",
    sub_agents=[resume_analyst, market_researcher, plan_writer],
)
