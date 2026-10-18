"""LITERATURE: Exa searches -> a sourced taxonomy of solution families."""
import re
from concurrent.futures import ThreadPoolExecutor

from .llm import ask_json

QUERY_PROMPT = """You are surveying the literature on this problem:

<task>
{task}
</task>

Write {n} web search queries that together find the distinct families of algorithmic
approaches to this problem: surveys, classic textbook treatments, seminal papers
(including work published before 2022) and recent research. Each query should target
a different angle or family of approaches.

Return JSON: {{"queries": ["...", "..."]}}"""

TAXONOMY_PROMPT = """You are building a taxonomy of solution families for this problem:

<task>
{task}
</task>

Below are web search results. Identify the distinct families of approaches to the
problem. Define families by MECHANISM (what actually drives each decision), not by
name or wording, so that families do not overlap. Include classic baselines and
less common approaches alike, as long as a result below supports them.

Only list a family under "families" if it can actually be used under this task's rules
and interface. For example, when items must be handled one at a time without seeing
future ones, an algorithm that first sorts all items cannot be used. Put families that the
sources describe but that break the task's rules under "inapplicable", with the reason.

Every family must cite at least one URL, and you may only cite URLs that appear in
the results below.

Return JSON:
{{"families": [{{"family": "short name",
                "one_line_mechanism": "one sentence: how it decides",
                "source_summary": "1-2 sentences on what the cited sources say about it",
                "source_urls": ["..."]}}],
 "inapplicable": [{{"family": "short name", "reason": "which task rule it breaks",
                   "source_urls": ["..."]}}]}}

<results>
{results}
</results>"""


def normalize(name):
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def search_literature(task, llm, exa, cfg, progress=print):
    lit = cfg["literature"]
    queries = ask_json(llm, cfg["models"]["labeler"],
                       QUERY_PROMPT.format(task=task.prompt, n=lit["n_queries"]))["queries"]
    queries = queries[:lit["n_queries"]]
    # Every query once unrestricted, and the first few again limited to pre-2022 work.
    searches = [(q, None) for q in queries] + [(q, "2022-01-01") for q in queries[:lit["n_pre_2022"]]]
    progress(f"literature: {len(searches)} searches")
    with ThreadPoolExecutor(4) as pool:
        batches = list(pool.map(lambda s: exa.search(s[0], lit["results_per_query"], s[1]), searches))
    results = {}
    for (query, before), batch in zip(searches, batches):
        for r in batch:
            results.setdefault(r["url"], {**r, "queries": []})["queries"].append(
                query + (f" (before {before})" if before else ""))
    return {"queries": queries, "searches": [list(s) for s in searches], "results": list(results.values())}


def build_taxonomy(task, literature, llm, cfg):
    listing = "\n\n".join(
        f"[{i}] {r['title']} ({r['published'] or 'undated'})\n{r['url']}\n" + " ... ".join(r["highlights"])
        for i, r in enumerate(literature["results"]))
    reply = ask_json(llm, cfg["models"]["labeler"],
                     TAXONOMY_PROMPT.format(task=task.prompt, results=listing))
    known_urls = {r["url"] for r in literature["results"]}
    families = {}
    rejected = [{**f, "reason": f"breaks the task's rules: {f.get('reason', 'unspecified')}"}
                for f in reply.get("inapplicable", [])]
    for f in reply["families"]:
        urls = [u for u in f.get("source_urls", []) if u in known_urls]
        if not urls:  # a family needs at least one real, retrieved source
            rejected.append({**f, "reason": "no source URL from the search results"})
            continue
        key = normalize(f["family"])
        if key in families:  # duplicate name: merge sources
            families[key]["source_urls"] = sorted(set(families[key]["source_urls"]) | set(urls))
        else:
            families[key] = {"family": f["family"], "one_line_mechanism": f["one_line_mechanism"],
                             "source_summary": f.get("source_summary", ""), "source_urls": urls}
    return {"families": list(families.values()), "rejected": rejected}
