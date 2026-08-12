# Agentic AI Workshop

- google adk: Online documentation link: https://adk.dev/get-started/python/
- google ai studio: https://aistudio.google.com/

Commands I have ran to setup:

1. create python virtual env: python -m venv .venv
2. get into/ start using virtual env: .venv\Scripts\Activate.ps1
3. start with / install google-adk: pip install google-adk
4. for reading PDF resumes from section 4 onwards: pip install pypdf
5. move the API key to the repo root so every section finds it: mv my_agent/.env .env

ADK looks for a `.env` starting at the agent folder and walks up to the drive root, so
one file at the repo root covers all sections. It is in `.gitignore` and never commits.

## Running any section

Start the web UI from the repo root, then pick the section from the dropdown:

    adk web

Or run one directly in the terminal:

    adk run section1_basic_agent

## The arc

Each section changes exactly one thing from the section before it. That is the whole
teaching design - so when the output improves, there is no doubt about what caused it.

| Section | What changes | Tools |
|---|---|---|
| 1 | nothing yet - the raw scaffold | 0 |
| 2 | give it a real job, with a vague instruction | 0 |
| 3 | add the role and the rules | 0 |
| 4 | add one tool | 1 |
| 5 | add web search | 2 |
| 6 | split one agent into three | 2, across agents |

## Section 1 - a basic agent

`section1_basic_agent` - exactly what `adk create` generates, untouched. Eight lines: an
agent is a model, a name, a description and an instruction. Nothing else.

No tools, so it can only answer from what the model already knows. Ask it something
current, like today's news or whether a link works, and watch what happens.

## Section 2 - no role, no rules

`section2_no_rules` - the career mentor task, with a one-line instruction:
*"You are a career mentor. Help the student with their career."*

Paste in a resume and ask for a review. What comes back is the failure mode students
need to see: praise, advice that would fit literally any student, "learn DSA and build
projects", and no sign it looked closely at anything.

Nothing here is broken. The model is fine, the code is fine. The instruction is thin.

## Section 3 - the same agent, with a role and rules

`section3_with_rules` - identical to Section 2. Same class, same model, no tools. Only
the words in the instruction changed.

Now it produces seven fixed sections: a resume review that quotes the weak line and puts
the rewrite beside it, a skill audit that separates *evidenced* from *claimed only*, must
learn ordered by what unblocks an interview soonest, **safe to skip for now** with a
condition for revisiting, resources with what to actually do with them, where to apply
with the exact search strings, and profile upgrades plus the habits that produce real
depth instead of generated code you cannot defend. Project ideas only if you ask.

Diff the two instructions side by side. That diff is the lesson.

What is still wrong: every fact comes from training data with a cutoff, and it cannot
open a single link to check itself. So it is told not to write URLs it is unsure of.

Sample resume: [data/resume_sample.pdf](data/resume_sample.pdf) - open it, copy the text.

You can also attach the PDF with the paperclip instead of pasting. That works here with no
code at all, because Gemini reads PDFs natively as part of the message. It is worth
showing: not everything needs a tool.

## Section 4 - the first tool

`section4_one_tool` - one tool, `read_resume`. No more pasting.

A tool is a plain Python function - about fifteen lines here. Read its docstring: that
text is not a comment, it is what the model reads to decide when to call the function and
what to pass it. Tool docstrings are prompt engineering.

    My resume is at data/resume_sample.pdf. I am targeting python developer roles in Pune.

Students can also attach their own resume with the paperclip and say "review my resume".
An attached file never lands on disk - ADK stores it as an artifact - so the tool takes a
`tool_context` parameter and calls `load_artifact` first, falling back to the filesystem.
ADK fills that parameter in itself and hides it from the model, which only ever sees
`filename`.

Two more things to point out. Deterministic work - finding the file, parsing the PDF,
noticing a scan with no text in it - belongs in Python, not in the model. And the function
returns its errors as data rather than raising, so the agent can tell the student what
broke.

Still half broken: it reads the real resume now, but it cannot check one claim about the
job market.

## Section 5 - web search

`section5_web_search` - Section 4 plus `web_scout`, a sub-agent holding `google_search`,
handed over with `AgentTool`.

Why a sub-agent instead of just adding the tool: `google_search` is a Gemini built-in,
and built-ins cannot sit in the same agent as a custom function tool. The workaround is
the lesson - an agent can be someone else's tool.

Same prompt as Section 4. Watch the Events tab for each tool call and result. Now the gap
analysis only lists skills that appeared in a real posting, the openings are real with
real links, and every URL came back from a search instead of from memory.

## Section 6 - one agent becomes three

`section6_multi_agent` - a `SequentialAgent` running three specialists in a fixed order:

    resume_analyst   reads the file, extracts a profile, gives no advice
    market_researcher never sees the PDF, only the profile, searches the market
    plan_writer      has no tools at all, only reasons over the other two

They pass work through session state, not through function arguments. `resume_analyst`
declares `output_key="resume_profile"`, and `market_researcher`'s instruction contains
`{resume_profile}`, which ADK fills in. No plumbing code.

Two things to point out. Each agent has a narrow job and only the tools it needs, which
is why the output is easier to debug than one agent doing everything. And the order is
fixed in Python, by the `sub_agents` list - the model does not choose who runs next. That
makes it a workflow. An `LlmAgent` with `sub_agents` would let the model decide, and that
difference is the definition of "agentic".

## Section 7

To be added.

## Notes

- The model is pinned to `gemini-3.5-flash-lite` everywhere. Section 1 hardcodes it
  because it is the untouched scaffold; the rest read a `MODEL` variable, so you can swap
  without editing code: `MODEL=gemini-2.5-flash-lite adk web`
- Free-tier quota is the real constraint, and it gets worse as the sections progress -
  roughly 1 request for sections 1 to 3, 3 for section 4, 5 for section 5, and 7 for
  section 6, since every sub-agent is its own request. Sections 5 and 6 are capped at one
  search call each for this reason.
- A 429 `RESOURCE_EXHAUSTED` is either the per-minute limit, which clears in a minute, or
  the per-day limit, which clears at midnight Pacific. Check which at
  https://ai.dev/rate-limit
- Test the resume tool without spending any quota: `python section4_one_tool/agent.py`
