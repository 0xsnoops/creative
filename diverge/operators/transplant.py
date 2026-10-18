"""transplant: mechanisms from other fields (biology, logistics, markets, physics, networking)
that solve a structurally similar problem. The labeler states each mechanism in the
problem's own terms, without the metaphor; the implementer builds it. Sources are kept."""
from ..llm import ask_json

QUERY_PROMPT = """Here is a computational problem:

<task>
{task}
</task>

Describe the abstract problem class in one line (e.g. "online assignment of arriving
items to limited-capacity containers"). Then, for each of these fields: {fields}, write
one web search query that would find how that field solves a structurally similar
problem, such as a natural, economic or engineered mechanism with the same shape.

Return JSON: {{"problem_class": "...", "queries": [{{"field": "...", "query": "..."}}]}}"""

EXTRACT_PROMPT = """Problem:

<task>
{task}
</task>

Below are search results about mechanisms from other fields. For each result that
describes a mechanism that could transfer to this problem, extract its STRUCTURAL
mechanism: what state it keeps and what rule decides each step, stated in this problem's
own terms. Leave the metaphor behind (say "items" and "bins", not "ants" and "food").
Skip results with nothing transferable. Cite only URLs that appear below.

Return JSON:
{{"mechanisms": [{{"family": "short name", "field": "...",
                  "one_line_mechanism": "the structural rule in this problem's terms",
                  "source_summary": "1-2 sentences on the source mechanism",
                  "source_urls": ["..."]}}]}}

<results>
{results}
</results>"""


def jobs(ctx):
    labeler, fields = ctx.cfg["models"]["labeler"], ctx.cfg["bench"]["transplant_fields"]
    plan = ask_json(ctx.llm, labeler, QUERY_PROMPT.format(task=ctx.task.prompt, fields=", ".join(fields)),
                    key=ctx.key("queries"), temperature=1.0, required=("queries",))
    results = {}
    for q in plan["queries"]:
        for r in ctx.exa.search(q["query"], num_results=6):
            results.setdefault(r["url"], {**r, "field": q["field"]})
    listing = "\n\n".join(f"[{r['field']}] {r['title']}\n{r['url']}\n" + " ... ".join(r["highlights"])
                          for r in results.values())
    reply = ask_json(ctx.llm, labeler, EXTRACT_PROMPT.format(task=ctx.task.prompt, results=listing),
                     key=ctx.key("extract"), required=("mechanisms",))
    mechanisms, rejected = [], []
    for m in reply.get("mechanisms", []):
        urls = [u for u in m.get("source_urls", []) if u in results]
        (mechanisms if urls else rejected).append({**m, "source_urls": urls})
    ctx.notes.update(problem_class=plan.get("problem_class"), queries=plan["queries"],
                     mechanisms=mechanisms, rejected_unsourced=rejected)
    while mechanisms:
        for m in mechanisms:
            yield ctx.family_job(m)
