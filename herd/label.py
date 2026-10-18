"""LABEL: assign each sample to one family by mechanism; write a human audit sheet."""
import csv
import random
from concurrent.futures import ThreadPoolExecutor

from .llm import ask_json

EMERGENT = "emergent:"
UNLABELED = "unlabeled"

LABEL_PROMPT = """Assign each candidate solution below to exactly one family of approaches.

Judge by the MECHANISM the code actually implements (what drives each decision), not by
names, comments or claims in the text. If no family fits, use "{emergent}<short-name>"
naming the mechanism.

Problem:
<task>
{task}
</task>

Families:
{families}

Solutions:
{solutions}

Return JSON:
{{"labels": [{{"sample_id": "...", "family": "exact family name or {emergent}<name>",
              "justification": "one sentence on the mechanism you saw"}}]}}"""

MERGE_PROMPT = """Different labelers proposed these new solution families, each with example
justifications. Merge names that describe the same mechanism, and give each final family
a one-sentence mechanism.

{emergent}

Return JSON:
{{"families": [{{"family": "{prefix}<final-name>", "one_line_mechanism": "...",
                "merged_from": ["{prefix}<name>", "..."]}}]}}"""

CLASSIFY_PROMPT = """Which family of approaches does this approach belong to? Judge by its
MECHANISM, not its wording. If none fits, answer "{emergent}<short-name>".

Problem:
<task>
{task}
</task>

Families:
{families}

Approach:
<approach>
{approach}
</approach>

Return JSON: {{"family": "exact family name or {emergent}<name>", "justification": "one sentence"}}"""


def family_listing(families):
    return "\n".join(f"- {f['family']}: {f['one_line_mechanism']}" for f in families)


def sample_body(out_dir, record, limit=8000):
    path = record.get("code_path") or record["text_path"]
    return (out_dir / path).read_text()[:limit]


def label_samples(task, records, taxonomy, llm, cfg, out_dir, progress=print):
    """Return ({sample_id: {family, justification}}, emergent families)."""
    model, batch_size = cfg["models"]["labeler"], cfg["labeling"]["batch_size"]
    names = {f["family"] for f in taxonomy["families"]}
    listing = family_listing(taxonomy["families"])
    answered = [r for r in records if r.get("text_path")]
    # Batches are per model and fixed index ranges, so raising N leaves earlier batches cached.
    batches = {}
    for r in answered:
        batches.setdefault((r["model"], r["idx"] // batch_size), []).append(r)
    done = []

    def run(batch):
        solutions = "\n\n".join(f'<solution id="{r["sample_id"]}">\n{sample_body(out_dir, r)}\n</solution>'
                                for r in batch)
        try:
            reply = ask_json(llm, model, LABEL_PROMPT.format(
                task=task.prompt, families=listing, solutions=solutions, emergent=EMERGENT))
        except ValueError:
            reply = {}  # its samples end up "unlabeled" and are reported as such
        done.append(batch)
        progress(f"label batch {len(done)}/{len(batches)}")
        return {l["sample_id"]: l for l in reply.get("labels", [])}

    with ThreadPoolExecutor(cfg["sampling"]["concurrency"]) as pool:
        replies = list(pool.map(run, batches.values()))
    raw = {k: v for reply in replies for k, v in reply.items()}

    labels = {}
    for r in answered:
        l = raw.get(r["sample_id"])
        valid = l and (l["family"] in names or l["family"].startswith(EMERGENT))
        labels[r["sample_id"]] = ({"family": l["family"], "justification": l.get("justification", "")}
                                  if valid else {"family": UNLABELED, "justification": "labeler gave no valid family"})
    emergent = merge_emergent(labels, llm, model)
    return labels, emergent


def merge_emergent(labels, llm, model):
    """Batches were labeled in parallel, so the same new mechanism may carry several names."""
    examples = {}
    for l in labels.values():
        if l["family"].startswith(EMERGENT):
            examples.setdefault(l["family"], []).append(l["justification"])
    if not examples:
        return []
    listing = "\n".join(f"- {name}: " + " | ".join(js[:3]) for name, js in sorted(examples.items()))
    reply = ask_json(llm, model, MERGE_PROMPT.format(emergent=listing, prefix=EMERGENT))
    rename, families = {}, []
    for f in reply["families"]:
        name = f["family"] if f["family"].startswith(EMERGENT) else EMERGENT + f["family"]
        families.append({"family": name, "one_line_mechanism": f["one_line_mechanism"],
                         "source_summary": "No literature source; this family was observed in model samples.",
                         "source_urls": []})
        for old in f.get("merged_from", []):
            rename[old] = name
    for l in labels.values():
        if l["family"].startswith(EMERGENT):
            l["family"] = rename.get(l["family"], l["family"])
    known = {f["family"] for f in families}
    for name in {l["family"] for l in labels.values() if l["family"].startswith(EMERGENT)} - known:
        families.append({"family": name, "one_line_mechanism": " | ".join(examples.get(name, [""])[:1]),
                         "source_summary": "No literature source; this family was observed in model samples.",
                         "source_urls": []})
    return families


def classify(task, approach, families, llm, cfg):
    names = {f["family"] for f in families}
    reply = ask_json(llm, cfg["models"]["labeler"], CLASSIFY_PROMPT.format(
        task=task.prompt, families=family_listing(families), approach=approach, emergent=EMERGENT))
    if reply.get("family") not in names and not str(reply.get("family", "")).startswith(EMERGENT):
        reply["family"] = UNLABELED
    return reply


def write_audit(path, records, labels, out_dir, n=20, seed=0):
    labeled = [r for r in records if r["sample_id"] in labels]
    picked = random.Random(seed).sample(labeled, min(n, len(labeled)))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "model", "family", "justification", "file", "excerpt",
                    "human_agrees (y/n)", "human_family"])
        for r in picked:
            l = labels[r["sample_id"]]
            w.writerow([r["sample_id"], r["model"], l["family"], l["justification"],
                        r.get("code_path") or r["text_path"], sample_body(out_dir, r, 1500), "", ""])
