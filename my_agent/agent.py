"""back.py - Autonomous Academic Research Team (ADK multi-agent demo).

Four specialists, run in order by a SequentialAgent:

    researcher  -> searches OpenAlex (custom tool) + the web (google_search
                   via a sub-agent) and writes RESEARCH NOTES
    writer      -> turns those notes into a structured journal draft
    critic      -> checks every claim in the draft against the notes and
                   flags anything unsupported (this is what catches
                   fabricated citations)

Data moves between them through ADK session state via `output_key`: the
researcher writes `research_notes`, and the writer's instruction reads
{research_notes}. No manual plumbing.

Run:
    python back.py "graph neural networks for drug discovery"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

# The ADK scaffold put the key here. Nothing secret lives in this file.
load_dotenv(Path(__file__).parent / "my_agent" / ".env")

MODEL = os.getenv("MODEL", "gemini-2.5-flash")
APP_NAME = "research_team"
OPENALEX = "https://api.openalex.org/works"

# OpenAlex asks for a contact address in the User-Agent - that puts you in
# their "polite pool" with much better rate limits. Be a good API citizen.
UA = "agentic-ai-workshop/0.1 (educational; mailto:instructor@example.com)"


# ==========================================================================
# TOOL - a plain Python function. The docstring below is what the model reads
# to decide when and how to call it, so it is prompt engineering, not comments.
# ==========================================================================
def search_papers(topic: str, limit: int = 5) -> dict:
    """Search real peer-reviewed academic literature on a topic via OpenAlex.

    Use this for any factual claim about published research. Results are
    ranked by relevance, and every paper returned is real - do not invent
    citations, call this instead.

    Args:
        topic: The research topic or question, e.g. "graph neural networks
            for drug discovery". Use natural language, not boolean operators.
        limit: How many papers to return, 1-10. Defaults to 5.

    Returns:
        A dict with `count` and `papers`. Each paper has title, authors,
        year, venue, citations, doi, oa_url (open-access PDF if available)
        and abstract.
    """
    limit = max(1, min(int(limit), 10))
    params = {
        "search": topic,
        "per-page": limit,
        # NOTE: deliberately NOT sorting by cited_by_count. Doing that returns
        # the most-cited paper that merely mentions the topic - a RAG query
        # comes back with "SciPy 1.0" (39k citations). Relevance != popularity.
        "select": ",".join(
            [
                "id", "doi", "title", "publication_year", "cited_by_count",
                "authorships", "abstract_inverted_index", "open_access",
                "primary_location", "type",
            ]
        ),
    }
    url = f"{OPENALEX}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})

    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            data = json.load(resp)
    except Exception as exc:  # noqa: BLE001 - give the model a readable reason
        return {"error": f"{type(exc).__name__}: {exc}", "count": 0, "papers": []}

    papers = []
    for work in data.get("results", []):
        source = (work.get("primary_location") or {}).get("source") or {}
        papers.append(
            {
                "title": work.get("title") or "untitled",
                "authors": [
                    a["author"]["display_name"] for a in work.get("authorships", [])[:5]
                ],
                "year": work.get("publication_year"),
                "venue": source.get("display_name") or "unknown venue",
                "citations": work.get("cited_by_count", 0),
                "doi": work.get("doi") or "",
                "oa_url": (work.get("open_access") or {}).get("oa_url") or "",
                "abstract": _rebuild_abstract(work.get("abstract_inverted_index")),
            }
        )

    return {"count": len(papers), "papers": papers}


def _rebuild_abstract(inverted: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]}, not as text.

    Real APIs are messy. Reassembling this is four lines - and a good
    reminder that tool code, not the LLM, should do deterministic work.
    """
    if not inverted:
        return ""
    positions = {i: word for word, idxs in inverted.items() for i in idxs}
    return " ".join(positions[i] for i in sorted(positions))[:1200]


# ==========================================================================
# AGENTS
# ==========================================================================

# google_search is a Gemini built-in. Built-in tools have restrictions on
# being mixed with custom function tools in the same agent, so it lives alone
# here and gets handed to the researcher as an AgentTool. Isolating it is the
# standard workaround - and a neat illustration of agents-as-tools.
web_scout = LlmAgent(
    name="web_scout",
    model=MODEL,
    description="Searches the live web for context, news and grey literature.",
    instruction=(
        "Search the web for the requested topic. Report only what you actually "
        "found, with source URLs. If results are thin, say so - never fill gaps "
        "with your own knowledge."
    ),
    tools=[google_search],
)

researcher = LlmAgent(
    name="researcher",
    model=MODEL,
    description="Gathers real literature and web context on a topic.",
    instruction=(
        "You are a research assistant preparing material for a journal article.\n"
        "\n"
        "Steps, in order:\n"
        "1. Call search_papers on the user's topic. Call it 2-3 times with "
        "   different phrasings to widen coverage.\n"
        "2. Call web_scout once for recent context the literature may miss.\n"
        "3. Write RESEARCH NOTES.\n"
        "\n"
        "Notes format:\n"
        "- TOPIC: one line\n"
        "- KEY PAPERS: for each, 'Author (Year), Title, Venue, Citations, DOI' "
        "  then two sentences on what it actually shows\n"
        "- THEMES: 3-5 recurring findings across papers\n"
        "- GAPS: what the literature does not yet answer\n"
        "- WEB CONTEXT: anything from web_scout, with URLs\n"
        "\n"
        "Hard rule: every paper you list must have come back from a tool call. "
        "You may not add a citation from memory. If you found nothing, write "
        "'NO RESULTS' - that is a valid and useful answer."
    ),
    tools=[search_papers, AgentTool(agent=web_scout)],
    output_key="research_notes",  # -> session state
)

writer = LlmAgent(
    name="writer",
    model=MODEL,
    description="Turns research notes into a structured journal draft.",
    instruction=(
        "You are an academic writer. Draft a short journal article using ONLY "
        "the research notes below.\n"
        "\n"
        "=== RESEARCH NOTES ===\n"
        "{research_notes}\n"
        "=== END NOTES ===\n"
        "\n"
        "Sections: Title, Abstract (120 words), Introduction, Related Work, "
        "Open Challenges, Conclusion, References.\n"
        "\n"
        "Cite as (Author, Year) inline. Every reference must appear in the "
        "notes above with its DOI. Inventing a plausible-sounding citation is "
        "the single worst thing you can do here - if the notes are thin, write "
        "a shorter article instead of padding it."
    ),
    output_key="draft",
)

critic = LlmAgent(
    name="critic",
    model=MODEL,
    description="Verifies the draft against the source notes.",
    instruction=(
        "You are a peer reviewer checking for fabrication and overreach.\n"
        "\n"
        "=== RESEARCH NOTES (ground truth) ===\n"
        "{research_notes}\n"
        "=== DRAFT UNDER REVIEW ===\n"
        "{draft}\n"
        "=== END ===\n"
        "\n"
        "Report:\n"
        "1. FABRICATED CITATIONS - any reference in the draft that is absent "
        "   from the notes. Quote it. This is the highest-priority check.\n"
        "2. UNSUPPORTED CLAIMS - statements the notes do not back. Quote them.\n"
        "3. MISREPRESENTATION - where the draft overstates a paper's finding.\n"
        "4. VERDICT - ACCEPT, MINOR REVISION, or MAJOR REVISION, one line why.\n"
        "\n"
        "Be specific and quote text. 'Looks good' is not a review. If the "
        "draft is genuinely clean, say so and explain what you checked."
    ),
    output_key="review",
)

# Fixed order: researcher -> writer -> critic.
# This is a WORKFLOW - the order is decided by code, not by a model. Compare
# with an LlmAgent supervisor using sub_agents, where the model chooses who
# runs next. That contrast is the definition of "agentic".
root_agent = SequentialAgent(
    name="research_supervisor",
    description="Runs the academic research team end to end.",
    sub_agents=[researcher, writer, critic],
)


# ==========================================================================
# RUNNER - prints every tool call so the delegation is visible
# ==========================================================================
async def main(topic: str) -> None:
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id="student"
    )

    print("=" * 72)
    print(f"TOPIC : {topic}")
    print(f"MODEL : {MODEL}")
    print("=" * 72)

    message = types.Content(role="user", parts=[types.Part(text=topic)])
    seen_author = None

    async for event in runner.run_async(
        user_id="student", session_id=session.id, new_message=message
    ):
        if event.author != seen_author:
            seen_author = event.author
            print(f"\n\n{'-' * 72}\n[{event.author}]\n{'-' * 72}")

        if not event.content or not event.content.parts:
            continue

        for part in event.content.parts:
            if part.function_call:
                args = json.dumps(part.function_call.args, default=str)[:120]
                print(f"\n  >> TOOL CALL {part.function_call.name}({args})")
            elif part.function_response:
                raw = json.dumps(part.function_response.response, default=str)
                print(f"  << TOOL RESULT {len(raw)} chars")
            elif part.text and not event.partial:
                print(part.text, end="", flush=True)

    final = await runner.session_service.get_session(
        app_name=APP_NAME, user_id="student", session_id=session.id
    )
    print("\n\n" + "=" * 72)
    print("SESSION STATE (how the agents passed work to each other)")
    for key in ("research_notes", "draft", "review"):
        value = final.state.get(key) or ""
        print(f"  {key:16} {len(value):>6} chars")
    print("=" * 72)


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "retrieval augmented generation for education"
    asyncio.run(main(topic))
