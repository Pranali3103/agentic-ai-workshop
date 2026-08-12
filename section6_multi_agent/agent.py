"""Section 6 - three specialists instead of one generalist, run in a fixed order."""

import os
from pathlib import Path

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool

ROOT = Path(__file__).resolve().parent.parent

MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")


def read_resume(path: str) -> dict:
    """Read a candidate's resume from a local PDF or text file.

    Args:
        path: Path to the resume file, e.g. "data/resume_sample.pdf".
            Accepts .pdf, .txt or .md, relative to the project root.

    Returns:
        A dict with `chars`, `pages` and `text`. On failure it contains
        `error` instead - report that error and stop.
    """
    target = Path(path)

    if not target.is_absolute() and not target.exists():
        target = ROOT / path

    if not target.exists():
        return {"error": f"file not found: {path}", "chars": 0, "pages": 0, "text": ""}

    suffix = target.suffix.lower()

    if suffix in (".txt", ".md"):
        text = target.read_text(encoding="utf-8", errors="replace")
        return {"chars": len(text), "pages": 1, "text": text}

    if suffix != ".pdf":
        return {"error": f"unsupported file type: {suffix}", "chars": 0, "pages": 0, "text": ""}

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(target))
        pages = [page.extract_text() or "" for page in reader.pages]
    except ImportError:
        return {"error": "pypdf not installed - run: pip install pypdf", "chars": 0, "pages": 0, "text": ""}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "chars": 0, "pages": 0, "text": ""}

    text = "\n".join(pages).strip()

    if not text:
        return {
            "error": "no extractable text - this PDF is probably a scan, ask for a text-based PDF",
            "chars": 0,
            "pages": len(pages),
            "text": "",
        }

    return {"chars": len(text), "pages": len(pages), "text": text}


# Same search specialist as Section 5. google_search is a Gemini built-in and cannot sit in
# the same agent as a custom function tool, so it lives here and is handed over as a tool.
web_scout = LlmAgent(
    name="web_scout",
    model=MODEL,
    description="Searches the live web for job postings, in-demand skills and learning resources.",
    instruction=(
        "You are a search specialist. Run the search and report only what you actually "
        "found, with the source URL for every claim. Answer everything the request asks "
        "for in one reply - never ask for a follow-up. Give real company names, job "
        "titles, application links and the skills each posting requires. Never invent a "
        "URL, a company or a salary. If a search finds nothing, say so."
    ),
    tools=[google_search],
)

# AGENT 1 - reads the file and does nothing else. One job, one tool.
resume_analyst = LlmAgent(
    name="resume_analyst",
    model=MODEL,
    description="Reads the resume file and extracts a structured profile.",
    instruction=(
        "Call read_resume on the path the student gives, then extract a profile. Do "
        "not give advice - a later agent does that. If the tool returns an error, "
        "report that error and nothing else.\n"
        "\n"
        "Output exactly this:\n"
        "- TARGET ROLE: what they said they want. If they did not say, write UNKNOWN.\n"
        "- LOCATION: what they said. If they did not say, write UNKNOWN.\n"
        "- CURRENT LEVEL: year of study, degree, CGPA if present.\n"
        "- EVIDENCED SKILLS: skills used in a described project, with the project name.\n"
        "- CLAIMED ONLY: skills listed with nothing behind them.\n"
        "- PROJECTS: name, stack, and what it actually does.\n"
        "- RESUME WEAKNESSES: quote the weak lines verbatim.\n"
        "\n"
        "Every line must trace to the file. Invent nothing."
    ),
    tools=[read_resume],
    output_key="resume_profile",
)

# AGENT 2 - never sees the PDF. It reads Agent 1's output from session state via {resume_profile}.
market_researcher = LlmAgent(
    name="market_researcher",
    model=MODEL,
    description="Finds what the live job market asks for, based on the extracted profile.",
    instruction=(
        "You research the job market. Here is the candidate profile another agent "
        "extracted:\n"
        "\n"
        "=== PROFILE ===\n"
        "{resume_profile}\n"
        "=== END PROFILE ===\n"
        "\n"
        "Call web_scout exactly once. In that single request ask for current openings "
        "for the target role and location, the skills those postings require, and the "
        "application links. Do not call it twice - free-tier quota is limited.\n"
        "\n"
        "Then report:\n"
        "- OPENINGS: company, title, link, for each posting found.\n"
        "- REQUIRED SKILLS: each skill, and how many postings asked for it.\n"
        "- NOT ASKED FOR: things students assume matter that these postings never "
        "  mentioned.\n"
        "\n"
        "Only what the search returned. No market knowledge from memory."
    ),
    tools=[AgentTool(agent=web_scout)],
    output_key="market_data",
)

# AGENT 3 - has no tools at all. It only reasons over what the first two produced.
plan_writer = LlmAgent(
    name="plan_writer",
    model=MODEL,
    description="Turns the profile and market data into the final mentoring report.",
    instruction=(
        "You are a career mentor. Blunt, specific, useful. Write the final report using "
        "only the two inputs below. You have no tools, so if something is not in them, "
        "you do not know it.\n"
        "\n"
        "=== PROFILE ===\n"
        "{resume_profile}\n"
        "=== MARKET DATA ===\n"
        "{market_data}\n"
        "=== END ===\n"
        "\n"
        "1. RESUME REVIEW - quote each weak line from the profile and put the rewrite "
        "beside it.\n"
        "\n"
        "2. GAP ANALYSIS - a table: Skill | In resume? | Postings asking for it | "
        "Priority. A row is only allowed if the skill appears in the market data.\n"
        "\n"
        "3. MUST LEARN - the top three gaps by priority. For each: what 'good enough "
        "for a fresher' looks like, and a realistic time estimate for someone carrying "
        "a full course load.\n"
        "\n"
        "4. SAFE TO SKIP FOR NOW - use the NOT ASKED FOR list. Give a condition for "
        "revisiting each one.\n"
        "\n"
        "5. WHERE TO APPLY - the openings from the market data, with company, title and "
        "link exactly as given.\n"
        "\n"
        "6. PROFILE UPGRADES AND REAL DEPTH - GitHub pinned repositories and README "
        "contents, LinkedIn headline and About rewritten for them, commit history that "
        "shows steady work. Then how to have real understanding instead of generated "
        "code they cannot defend: explain every design decision and what they rejected; "
        "break it on purpose by pulling the database and passing bad input; read the "
        "error and then the source before searching; write the tests; draw the request "
        "path end to end; name one thing they know is wrong with it. Where they used AI "
        "to write code, keep it and earn it - delete a part, rewrite from memory, "
        "explain the difference.\n"
        "\n"
        "Add project ideas only if the student asked for them.\n"
        "\n"
        "RULES\n"
        "- Only URLs that appear in the market data. Never assemble a link.\n"
        "- If the market data is thin, say so and write a shorter report.\n"
        "- Do not flatter."
    ),
    output_key="career_plan",
)

# The order is fixed by this line, not chosen by a model. That makes it a workflow.
# Compare with an LlmAgent that has sub_agents, where the model decides who runs next -
# that contrast is the whole definition of "agentic".
root_agent = SequentialAgent(
    name="career_team",
    description="Runs the career mentoring team end to end: analyst, then researcher, then planner.",
    sub_agents=[resume_analyst, market_researcher, plan_writer],
)


# Checks the tool without calling the model at all, so it uses zero API quota.
# Run: python section6_multi_agent/agent.py
if __name__ == "__main__":
    import sys

    resume = sys.argv[1] if len(sys.argv) > 1 else "data/resume_sample.pdf"
    result = read_resume(resume)

    print(f"read_resume({resume})")
    print(f"  error : {result.get('error', '-')}")
    print(f"  pages : {result['pages']}   chars: {result['chars']}")
