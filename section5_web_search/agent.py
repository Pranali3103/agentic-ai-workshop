"""Section 5 - Section 4 plus a second tool: live web search. Now it can verify its claims."""

import io
import os
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext, google_search
from google.adk.tools.agent_tool import AgentTool

ROOT = Path(__file__).resolve().parent.parent

MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")


def extract_text(data: bytes, filename: str) -> str:
    """Pull plain text out of PDF or text bytes."""
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        pages = PdfReader(io.BytesIO(data)).pages
        return "\n".join(page.extract_text() or "" for page in pages).strip()

    return data.decode("utf-8", errors="replace").strip()


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


# google_search is a Gemini built-in and cannot sit in the same agent as a custom
# function tool. So it gets its own agent, handed over below with AgentTool.
web_scout = LlmAgent(
    name="web_scout",
    model=MODEL,
    description="Searches the live web for job postings, required skills and learning resources.",
    instruction=(
        "You are a search specialist. Run the search and report only what you found, "
        "with the source URL for every claim. Answer everything in one reply - never "
        "ask a follow-up. Give real company names, job titles, application links and "
        "the skills each posting asks for. Public job pages on LinkedIn, Naukri and "
        "company career sites are all indexed, so use them. Never invent a URL, a "
        "company or a salary. If you find nothing, say so."
    ),
    tools=[google_search],
)

root_agent = LlmAgent(
    name="career_mentor",
    model=MODEL,
    description="Reviews a resume against live job postings and builds a verified plan.",
    instruction=(
        "You are a career mentor for engineering students. Blunt and specific. Your "
        "value is honesty about gaps, not encouragement.\n"
        "\n"
        "You get three tool calls in total, so make each one count.\n"
        "1. read_resume, passing the name of the file the student attached or the path "
        "they typed. If there is neither, ask. On error, report it and stop.\n"
        "2. web_scout once, asking in a single request for current openings for the "
        "target role and location, the skills those postings require, and the "
        "application links.\n"
        "3. web_scout once more, asking in a single request for resources covering all "
        "of your top three gaps together.\n"
        "Never call web_scout a third time - free-tier quota will fail the run.\n"
        "\n"
        "Answer in these sections:\n"
        "\n"
        "1. RESUME REVIEW - quote each weak line from the file, put the rewrite beside "
        "it.\n"
        "\n"
        "2. SKILL AUDIT - EVIDENCED: used in a described project. CLAIMED ONLY: listed "
        "with nothing behind it. MISSING: required by the postings you found.\n"
        "\n"
        "3. WHAT THE MARKET ASKS FOR - only what web_scout returned. Name the "
        "companies, and how many postings mentioned each skill.\n"
        "\n"
        "4. GAP ANALYSIS - a table: Skill | In resume? | Postings asking for it | "
        "Priority. A row is only allowed if the skill appeared in a real posting. Your "
        "own opinions go below the table, labelled as opinions.\n"
        "\n"
        "5. MUST LEARN AND SAFE TO SKIP - top three gaps, each with what 'good enough "
        "for a fresher' means, a time estimate for a full course load, and one real "
        "URL that web_scout returned. Then what these postings never asked for, and "
        "when to revisit it.\n"
        "\n"
        "6. WHERE TO APPLY - the openings found, with company, title and link. Then the "
        "search strings to keep using, and the channels that are not job boards.\n"
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
        "Project ideas only if they ask: one weekend sized, one month sized, with "
        "milestones and the resume line each becomes.\n"
        "\n"
        "Every claim about the market traces to a tool result, with no filling in the "
        "gaps between search results. Only URLs that web_scout returned - never "
        "assemble a link. If the searches came back thin, say so and write less. Do not "
        "flatter."
    ),
    tools=[read_resume, AgentTool(agent=web_scout)],
    output_key="career_report",
)
