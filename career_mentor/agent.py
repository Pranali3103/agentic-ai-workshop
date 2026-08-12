"""Autonomous Career Mentor - a single tool-calling agent.

One agent, two tools. It reads a resume, finds what the market is asking for,
works out the gap, and proposes a way to close it.

    read_resume(path)      local PDF/text -> the resume text (custom tool)
    web_scout (AgentTool)  google_search -> live postings, demand, resources

Why no LinkedIn API and no local job fixtures: automating a logged-in LinkedIn
session breaks their User Agreement and risks the account, and canned JSON
makes for a visibly fake demo. Public postings surface through web search
anyway - sanctioned, no credentials, and every number on screen is real.

Run standalone:
    python career_mentor/agent.py data/resume_sample.pdf
    python career_mentor/agent.py data/resume_sample.pdf --role "data analyst" --location Pune

Test the resume tool with zero API calls (costs no quota):
    python career_mentor/agent.py data/resume_sample.pdf --tools-only

Or in the ADK web UI, from the repo root:
    adk web
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

REPO_ROOT = Path(__file__).resolve().parent.parent

# ADK loads <agent_dir>/.env itself. This also picks up the key from the
# existing my_agent/.env so you don't have to duplicate it.
for candidate in (Path(__file__).parent / ".env", REPO_ROOT / "my_agent" / ".env"):
    if candidate.exists():
        load_dotenv(candidate)

# Standing rule for this repo: the cheapest "lite" tier, never anything bigger.
# Free-tier quota is the constraint, not capability. The whole gemini-2.5 family
# now 404s ("no longer available to new users"), so 3.5-flash-lite is the floor.
# Pinned rather than the -latest alias so every student gets identical behaviour.
MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")
APP_NAME = "career_mentor"


# ==========================================================================
# TOOL - read the resume. Deterministic work belongs in code, not the model.
# ==========================================================================
def read_resume(path: str) -> dict:
    """Read a candidate's resume from a local PDF or text file.

    Always call this first - you cannot analyse a resume you have not read.
    Call it once only; the text stays in your context afterwards.

    Args:
        path: Path to the resume file, e.g. "data/resume_sample.pdf".
            Accepts .pdf, .txt or .md, relative to the project root.

    Returns:
        A dict with `chars`, `pages` and `text`: the full extracted resume
        text. Read the text yourself to identify skills, projects and
        education - do not assume anything that is not written there.
    """
    target = Path(path)
    if not target.is_absolute() and not target.exists():
        target = REPO_ROOT / path

    if not target.exists():
        return {"error": f"file not found: {path}", "chars": 0, "pages": 0, "text": ""}

    suffix = target.suffix.lower()

    if suffix in {".txt", ".md"}:
        text = target.read_text(encoding="utf-8", errors="replace")
        return {"chars": len(text), "pages": 1, "text": text}

    if suffix != ".pdf":
        return {"error": f"unsupported file type '{suffix}'", "chars": 0, "pages": 0, "text": ""}

    try:
        from pypdf import PdfReader
    except ImportError:
        return {"error": "pypdf not installed - run: pip install pypdf", "chars": 0, "pages": 0, "text": ""}

    try:
        reader = PdfReader(str(target))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - the model should see why
        return {"error": f"{type(exc).__name__}: {exc}", "chars": 0, "pages": 0, "text": ""}

    text = "\n".join(pages).strip()
    if not text:
        return {
            "error": "no extractable text - this PDF is probably a scan. "
                     "Ask the candidate for a text-based PDF.",
            "chars": 0,
            "pages": len(pages),
            "text": "",
        }
    return {"chars": len(text), "pages": len(pages), "text": text}


# ==========================================================================
# AGENTS
# ==========================================================================

# google_search is a Gemini built-in, and built-in tools have restrictions on
# sharing an agent with custom function tools. So it lives in its own agent and
# gets handed over as an AgentTool - the agents-as-tools pattern.
web_scout = LlmAgent(
    name="web_scout",
    model=MODEL,
    description="Searches the live web for job postings, market demand and learning resources.",
    instruction=(
        "You are a search specialist. Run the requested web search and report "
        "only what you actually found, with the source URL for every claim.\n"
        "\n"
        "A request may bundle several things - handle all of them in one reply "
        "rather than asking for a follow-up. Typical bundles:\n"
        "- Current postings for a role in a location: report real company "
        "  names, titles, and the specific skills each posting asks for. "
        "  Public job pages on LinkedIn, Naukri, Indeed and company career "
        "  sites all appear in search results; use them.\n"
        "- Learning resources for several named skills at once: one specific "
        "  free-where-possible course, doc or tutorial per skill.\n"
        "\n"
        "Never invent a URL, company name or salary figure. Prefer a stable "
        "official docs or course landing page over a video link, because video "
        "IDs are easy to get wrong. If you cannot find something, say so "
        "plainly - a short honest answer beats a padded one."
    ),
    tools=[google_search],
)

root_agent = LlmAgent(
    name="career_mentor",
    model=MODEL,
    description="Reads a resume, finds skill gaps against the live job market, and plans how to close them.",
    instruction=(
        "You are a career mentor for university students. Blunt, specific, and "
        "supportive. Your value is honesty about gaps, not encouragement.\n"
        "\n"
        "WORKFLOW - follow it exactly. You get THREE tool calls in total, so "
        "make each one count:\n"
        "1. read_resume on the path the user gives. If they gave no path, ask "
        "   for one. Never guess resume contents.\n"
        "2. web_scout ONCE, asking in a single request for current postings for "
        "   the target role and location AND which skills those postings "
        "   require. Do not split this across multiple calls.\n"
        "3. web_scout ONCE more, asking in a single request for learning "
        "   resources covering ALL of your top three gaps together.\n"
        "\n"
        "Do not call web_scout more than twice. Free-tier quota is limited and "
        "a third call risks failing the whole run.\n"
        "\n"
        "Then produce exactly these sections:\n"
        "\n"
        "## CANDIDATE SNAPSHOT\n"
        "Current level, and skills evidenced in the resume. Distinguish "
        "'used in a project' from 'listed in a skills section' - they are not "
        "the same, and students routinely overclaim.\n"
        "\n"
        "## WHAT THE MARKET ASKS FOR\n"
        "Requirements you actually retrieved. Name the companies or postings "
        "they came from, and say how many mentioned each skill.\n"
        "\n"
        "## GAP ANALYSIS\n"
        "A table: Skill | In resume? | In demand (n postings) | Priority.\n"
        "Only list a gap that appeared in a retrieved posting. A skill you "
        "personally think matters but did not appear in the search results "
        "does not go in this table - mention it separately, labelled as your "
        "own opinion.\n"
        "\n"
        "## LEARNING PLAN\n"
        "Top 3 gaps only, by priority. For each: what to learn, one specific "
        "resource with a real URL from web_scout, and a realistic time "
        "estimate for a student carrying a full course load.\n"
        "\n"
        "## PORTFOLIO PROJECT\n"
        "One project that exercises several gaps at once. Give the problem it "
        "solves, the stack, 4-5 concrete milestones, and the line the student "
        "would put on their resume once it is done. Make it specific enough to "
        "start this weekend - not 'build a web app'.\n"
        "\n"
        "RULES:\n"
        "- Every market claim traces to a tool result. No claim from memory.\n"
        "- No invented URLs, companies or salary figures.\n"
        "- If search returns little, say the evidence is thin and give a "
        "  shorter analysis. A short grounded answer beats a long invented one.\n"
        "- Do not flatter. If the resume is weak for the target role, say so "
        "  and say exactly what would fix it."
    ),
    tools=[read_resume, AgentTool(agent=web_scout)],
    output_key="career_report",
)


# ==========================================================================
# RUNNER
# ==========================================================================
async def run(resume: str, role: str, location: str) -> None:
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id="student")

    prompt = (
        f"My resume is at {resume}. I am targeting {role} roles"
        + (f" in {location}." if location else ".")
        + " Analyse my gaps against the current market and tell me what to build."
    )

    print("=" * 72)
    print(f"RESUME   : {resume}")
    print(f"TARGET   : {role}" + (f" in {location}" if location else ""))
    print(f"MODEL    : {MODEL}")
    print("=" * 72)

    calls = 0
    last_author = None
    async for event in runner.run_async(
        user_id="student",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
        if event.author != last_author:
            last_author = event.author
            print(f"\n\n{'-' * 72}\n[{event.author}]\n{'-' * 72}")

        if not event.content or not event.content.parts:
            continue

        for part in event.content.parts:
            if part.function_call:
                calls += 1
                args = json.dumps(part.function_call.args, default=str)
                print(f"\n  >> TOOL CALL {calls}  {part.function_call.name}({args[:130]})")
            elif part.function_response:
                blob = json.dumps(part.function_response.response, default=str)
                print(f"  << TOOL RESULT  {len(blob):,} chars")
            elif part.text and not event.partial:
                print(part.text, end="", flush=True)

    print("\n" + "=" * 72)
    print(f"{calls} tool calls total")
    print("=" * 72)


def tools_only(resume: str) -> None:
    """Exercise the custom tool with no model calls at all - costs no quota."""
    print("=" * 72)
    print("TOOLS-ONLY MODE - no API calls, no quota used")
    print("=" * 72)

    r = read_resume(resume)
    print(f"\nread_resume({resume})")
    print(f"  error   : {r.get('error', '-')}")
    print(f"  pages   : {r['pages']}   chars: {r['chars']}")
    if r["text"]:
        print(f"  preview : {' '.join(r['text'].split())[:200]}...")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Autonomous Career Mentor")
    ap.add_argument("resume", nargs="?", default="data/resume_sample.pdf")
    ap.add_argument("--role", default="python developer")
    ap.add_argument("--location", default="")
    ap.add_argument("--tools-only", action="store_true", help="test tools without model calls")
    args = ap.parse_args()

    if args.tools_only:
        tools_only(args.resume)
    else:
        asyncio.run(run(args.resume, args.role, args.location))
