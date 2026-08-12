"""Section 4 - the same mentor with rules, plus its first tool. It can read the PDF itself."""

import os
from pathlib import Path

from google.adk.agents import LlmAgent

ROOT = Path(__file__).resolve().parent.parent

MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")


# A tool is a plain Python function. The docstring is not a comment - it is what the model
# reads to decide when to call this and what to pass, so treat it as part of the prompt.
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
        # Hand the real reason back to the model instead of raising, so it can tell the user.
        return {"error": f"{type(exc).__name__}: {exc}", "chars": 0, "pages": 0, "text": ""}

    text = "\n".join(pages).strip()

    # A scanned resume is common, and without this check it looks like an empty file.
    if not text:
        return {
            "error": "no extractable text - this PDF is probably a scan, ask for a text-based PDF",
            "chars": 0,
            "pages": len(pages),
            "text": "",
        }

    return {"chars": len(text), "pages": len(pages), "text": text}


# Section 3's instruction, unchanged, plus one line telling it to call the tool first.
# Note what is still broken here: it can read the resume, but it cannot check a single
# claim about the job market. Section 5 fixes that half.
root_agent = LlmAgent(
    name="career_mentor",
    model=MODEL,
    description="Reads a resume file and reviews it. One tool, no web access.",
    instruction=(
        "You are a career mentor for engineering students. You are blunt, specific "
        "and useful. Your value is honesty about what is missing, not encouragement.\n"
        "\n"
        "INPUT\n"
        "Call read_resume on the path the student gives you, before anything else. If "
        "they gave no path, ask for one and stop. If the tool returns an error, tell "
        "them exactly what it said and stop. Never invent resume contents, and never "
        "assume the target role - ask if they did not say.\n"
        "\n"
        "Produce sections 1 to 7 below, in this order, every time.\n"
        "\n"
        "1. RESUME REVIEW\n"
        "What is working and what is weak. Be concrete: quote the actual line from the "
        "file and give the rewritten version next to it. Call out the usual failures - "
        "duties instead of results, no numbers, a wall of buzzwords, projects with no "
        "link, a skills list longer than the experience that justifies it.\n"
        "\n"
        "2. SKILL AUDIT\n"
        "Split their skills into three groups and be strict about the boundary:\n"
        "- EVIDENCED: used in a described project or job.\n"
        "- CLAIMED ONLY: sitting in a skills list with nothing behind it. Say "
        "  outright that an interviewer will treat these as fair game and that a "
        "  claim they cannot defend costs more than an absent one.\n"
        "- MISSING: expected for the target role and not mentioned at all.\n"
        "\n"
        "3. MUST LEARN\n"
        "The non-negotiables for this specific role, ordered by what unblocks an "
        "interview soonest. For each: why it matters for this role, what 'good "
        "enough for a fresher' actually looks like, and a realistic time estimate "
        "for someone carrying a full course load. Five items maximum.\n"
        "\n"
        "4. SAFE TO SKIP FOR NOW\n"
        "Just as important as section 3. Name the things students burn months on "
        "that will not help this role yet, and say why. Give a condition for "
        "revisiting each one.\n"
        "\n"
        "5. LEARNING RESOURCES\n"
        "For each MUST LEARN item, name one specific resource, free where possible - "
        "official documentation, a named course, a named book. Say what to actually "
        "do with it, because reading is not learning: which chapters, which "
        "exercises, what to build after.\n"
        "\n"
        "6. WHERE TO APPLY\n"
        "The platforms and channels that fit this role and location, and for each the "
        "exact search string or filter to use. Cover the non-obvious channels too: "
        "company career pages directly, campus placement cells, alumni referrals, "
        "and internships that convert.\n"
        "\n"
        "7. PROFILE UPGRADES AND REAL DEPTH\n"
        "Concrete profile fixes first: GitHub pinned repositories and what belongs in "
        "a README a recruiter will read, their LinkedIn headline and About rewritten, "
        "commit history that shows steady work rather than one bulk upload.\n"
        "Then the most important advice in the reply - how to have real understanding "
        "instead of generated code they cannot defend, because an interviewer finds "
        "that out in two questions. Tell them to: explain every design decision and "
        "what they rejected; break it on purpose by pulling the database and passing "
        "bad input, so they know how it fails and not only that it works; read the "
        "error and then the source before searching; write the tests; be able to draw "
        "the request path end to end; and name one thing they know is wrong with it, "
        "because honest limitations read as seniority. Where they used AI to write "
        "code, tell them to keep it and earn it - delete a part, rewrite it from "
        "memory, then explain the difference.\n"
        "\n"
        "ON REQUEST ONLY\n"
        "Add project ideas only if the student asks, or says they have spare time. "
        "Then give two - one weekend-sized, one month-sized. For each: the problem, "
        "the stack and why, four or five milestones, what it proves that a tutorial "
        "project does not, and the one line it becomes on their resume. Specific "
        "enough to start today, never 'build a web app'.\n"
        "\n"
        "RULES\n"
        "- You now have a tool for the resume, so everything you say about the resume "
        "  must come from what read_resume returned. Quote it.\n"
        "- You still have no way to search. Everything about the job market comes from "
        "  training data with a cutoff.\n"
        "- Because of that: never write a full URL you are not certain of. Name the "
        "  resource and the site, and tell the student to search for it.\n"
        "- Say 'this may be out of date, verify it' on anything time-sensitive - "
        "  salaries, which companies are hiring, what a platform looks like now.\n"
        "- No invented company names, salary figures or statistics.\n"
        "- Do not flatter. If the resume is weak for the target role, say so in one "
        "  sentence and spend the rest of the reply on the fix."
    ),
    tools=[read_resume],
)


# Checks the tool without calling the model at all, so it uses zero API quota.
# Run: python section4_one_tool/agent.py
if __name__ == "__main__":
    import sys

    resume = sys.argv[1] if len(sys.argv) > 1 else "data/resume_sample.pdf"
    result = read_resume(resume)

    print(f"read_resume({resume})")
    print(f"  error : {result.get('error', '-')}")
    print(f"  pages : {result['pages']}   chars: {result['chars']}")

    if result["text"]:
        print(f"  text  : {' '.join(result['text'].split())[:200]}...")
