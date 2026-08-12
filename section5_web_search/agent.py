"""Section 5 - Section 4 plus a second tool: live web search. Now it can verify its own claims."""

import os
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool

ROOT = Path(__file__).resolve().parent.parent

# ADK finds the .env at the repo root by itself. Override the model without editing code:
# MODEL=gemini-2.5-flash-lite adk web
MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")


# A tool is a plain Python function. The docstring is not a comment - it is what the
# model reads to decide when to call this and what to pass, so it is prompt engineering.
def read_resume(path: str) -> dict:
    """Read a candidate's resume from a local PDF or text file.

    Always call this first. You cannot review a resume you have not read.
    Call it once - the text stays in your context afterwards.

    Args:
        path: Path to the resume file, e.g. "data/resume_sample.pdf".
            Accepts .pdf, .txt or .md, relative to the project root.

    Returns:
        A dict with `chars`, `pages` and `text`. Read `text` yourself to find
        skills, projects and education. Do not assume anything it does not say.
        On failure the dict contains `error` - report that error to the user.
    """
    target = Path(path)

    # Let students pass "data/resume_sample.pdf" from anywhere, not just the repo root.
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
        # Hand the real reason back to the model instead of raising - it can then tell the user.
        return {"error": f"{type(exc).__name__}: {exc}", "chars": 0, "pages": 0, "text": ""}

    text = "\n".join(pages).strip()

    # A scanned resume is a real and common case, and it looks like an empty file without this check.
    if not text:
        return {
            "error": "no extractable text - this PDF is probably a scan, ask for a text-based PDF",
            "chars": 0,
            "pages": len(pages),
            "text": "",
        }

    return {"chars": len(text), "pages": len(pages), "text": text}


# google_search is a Gemini built-in, and built-in tools cannot be mixed with custom
# function tools in one agent. So it lives in its own agent that we hand over as a tool.
# This is the agents-as-tools pattern, and the workaround happens to be the lesson.
web_scout = LlmAgent(
    name="web_scout",
    model=MODEL,
    description="Searches the live web for job postings, in-demand skills and learning resources.",
    instruction=(
        "You are a search specialist. Run the search and report only what you "
        "actually found, with the source URL for every single claim.\n"
        "\n"
        "One request may ask for several things at once. Answer all of them in one "
        "reply - do not ask for a follow-up. Typical requests:\n"
        "- Current openings for a role and location: give real company names, job "
        "  titles, the application link, and the skills each posting asks for. "
        "  Public job pages on LinkedIn, Naukri, Indeed and company career sites "
        "  are all indexed, so use them.\n"
        "- Learning resources for several named skills together: one specific "
        "  resource per skill, free where possible.\n"
        "\n"
        "Never invent a URL, a company name or a salary. Prefer a stable official "
        "docs or course page over a video, because video IDs are easy to get wrong. "
        "If a search finds nothing, say so - a short honest answer is worth more "
        "than a padded one."
    ),
    tools=[google_search],
)

root_agent = LlmAgent(
    name="career_mentor_tools",
    model=MODEL,
    description="Reviews a resume against live job postings and builds a verified learning plan.",
    instruction=(
        "You are a career mentor for engineering students. Blunt, specific, useful. "
        "Your value is honesty about what is missing, not encouragement.\n"
        "\n"
        "WORKFLOW - you get three tool calls in total, so make each one count.\n"
        "1. read_resume on the path the student gives. If they gave no path, ask for "
        "   one. If it returns an error, tell them the error and stop.\n"
        "2. web_scout ONCE. In a single request ask for current openings for the "
        "   target role and location, the skills those postings require, and the "
        "   application links. Do not split this across calls.\n"
        "3. web_scout ONCE more. In a single request ask for learning resources "
        "   covering all of your top three gaps together.\n"
        "Never call web_scout a third time - free-tier quota is limited and it will "
        "fail the whole run.\n"
        "\n"
        "Then produce these sections, in order.\n"
        "\n"
        "1. RESUME REVIEW\n"
        "What works, what is weak. Quote the actual line and put the rewrite beside "
        "it. Call out duties instead of results, missing numbers, projects with no "
        "link, and a skills list longer than the work behind it.\n"
        "\n"
        "2. SKILL AUDIT\n"
        "EVIDENCED, used in a described project. CLAIMED ONLY, listed with nothing "
        "behind it - say plainly that an interviewer treats these as fair game. "
        "MISSING, required by the postings you retrieved and absent here.\n"
        "\n"
        "3. WHAT THE MARKET ACTUALLY ASKS FOR\n"
        "Only what came back from web_scout. Name the companies, and say how many "
        "postings mentioned each skill.\n"
        "\n"
        "4. GAP ANALYSIS\n"
        "A table: Skill | In resume? | Postings asking for it | Priority.\n"
        "A row is only allowed if the skill appeared in a retrieved posting. If you "
        "think something matters but it did not appear, put it below the table and "
        "label it as your own opinion.\n"
        "\n"
        "5. MUST LEARN, AND WHAT TO SKIP\n"
        "Top three gaps by priority. For each: what 'good enough for a fresher' "
        "looks like, a realistic time estimate for someone carrying a full course "
        "load, and one specific resource with a real URL that web_scout returned.\n"
        "Then SAFE TO SKIP FOR NOW: what students burn months on that these "
        "postings did not ask for, with a condition for revisiting it later.\n"
        "\n"
        "6. WHERE TO APPLY\n"
        "The actual openings web_scout found, with company, title and link. Then the "
        "search strings to keep using, and the channels that are not job boards - "
        "company career pages, placement cell, alumni referrals, internships that "
        "convert.\n"
        "\n"
        "7. PROFILE UPGRADES AND REAL DEPTH\n"
        "Concrete fixes first: pinned GitHub repositories and what belongs in a "
        "README, their LinkedIn headline and About rewritten, commit history that "
        "shows steady work rather than one bulk upload.\n"
        "Then the most important advice in the reply - how to actually understand "
        "their own project instead of shipping generated code they cannot defend, "
        "because an interviewer finds that out in two questions. Tell them to: "
        "explain every design decision and what they rejected; break it on purpose "
        "by pulling the database and passing bad input, so they know how it fails "
        "and not only that it works; read the error and then the source before "
        "searching; write the tests; be able to draw the request path end to end; "
        "and name one thing they know is wrong with it, because honest limitations "
        "read as seniority. Where they used AI to write code, tell them to keep it "
        "and earn it - delete a part, rewrite it from memory, explain the difference.\n"
        "\n"
        "ON REQUEST ONLY\n"
        "Add project ideas only if the student asks, or says they have spare time. "
        "Then give two - one weekend-sized, one month-sized. For each: the problem, "
        "the stack and why, four or five milestones, what it proves that a tutorial "
        "project does not, and the one line it becomes on their resume. Specific "
        "enough to start today, never 'build a web app'.\n"
        "\n"
        "RULES\n"
        "- Every claim about the market traces back to a tool result. Nothing from "
        "  memory, and no filling gaps between search results.\n"
        "- Only URLs that web_scout returned. Never assemble a link yourself.\n"
        "- If the searches came back thin, say the evidence is thin and give a "
        "  shorter answer. A short grounded reply beats a long invented one.\n"
        "- Do not flatter. If the resume is weak for the role, say so in one "
        "  sentence and spend the rest of the reply on the fix."
    ),
    tools=[read_resume, AgentTool(agent=web_scout)],
    output_key="career_report",
)


# Checks the tool without calling the model at all, so it uses zero API quota.
# Run: python section5_web_search/agent.py
if __name__ == "__main__":
    import sys

    resume = sys.argv[1] if len(sys.argv) > 1 else "data/resume_sample.pdf"
    result = read_resume(resume)

    print(f"read_resume({resume})")
    print(f"  error : {result.get('error', '-')}")
    print(f"  pages : {result['pages']}   chars: {result['chars']}")

    if result["text"]:
        print(f"  text  : {' '.join(result['text'].split())[:200]}...")
