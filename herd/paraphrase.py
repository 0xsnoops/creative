"""Step 0 of the overlooked-ideas phase: is the herd a property of the problem or of the prompt?

Meaning-preserving paraphrases of a task prompt are sampled on two models and labeled with the
map's taxonomy. Their family shares are compared, by total variation distance (TVD), with the
spread between groups of the original prompt's existing samples (resampling noise). Protocol and
decision rule: docs/PREREGISTRATION-overlook.md."""
import itertools
import json
import re
import statistics
from collections import Counter

from .label import EMERGENT, UNLABELED, label_samples
from .llm import ask_json, finish_reason, message_text
from .pipeline import read_json, write_json
from .sample import extract_code

MODELS = ["anthropic/claude-sonnet-5.5", "openai/gpt-6.1-sol"]
N_PARAPHRASES, PER_GROUP = 3, 5

PARAPHRASE_PROMPT = """Rewrite the task statement below in {n} different ways, for a study of whether
wording changes which solutions programmers choose.

Each rewrite must keep, exactly:
- every code block, character for character (interfaces, signatures);
- every number, distribution, limit and unit;
- the scoring rule, every constraint and every rule about what the solution may use;
- the final instruction about the reply format.

Change only the wording, the order in which things are explained, and the emphasis. Do not add
hints, examples or advice, and do not remove information.

Return JSON: {{"paraphrases": ["...", "...", "..."]}}

<task>
{task}
</task>"""


def code_blocks(text):
    return re.findall(r"```.*?```", text, re.S)


def numbers(text):
    return sorted(re.findall(r"\d+(?:[.,]\d+)*", text))


def check(original, paraphrase):
    """Automatic meaning checks: code blocks identical and the same numbers present."""
    problems = []
    if code_blocks(paraphrase) != code_blocks(original):
        problems.append("code blocks differ")
    if numbers(paraphrase) != numbers(original):
        problems.append(f"numbers differ: {sorted(set(numbers(original)) ^ set(numbers(paraphrase)))}")
    return problems


def make_paraphrases(herd, task):
    """Three paraphrases; one regeneration if any fails the automatic checks."""
    record = {"attempts": []}
    for attempt in range(2):
        reply = ask_json(herd.llm, herd.cfg["models"]["labeler"],
                         PARAPHRASE_PROMPT.format(n=N_PARAPHRASES, task=task.prompt),
                         key=f"paraphrase:{attempt}", required=("paraphrases",))
        texts = [p.strip() for p in reply["paraphrases"]][:N_PARAPHRASES]
        checks = [check(task.prompt, t) for t in texts]
        record["attempts"].append({"paraphrases": texts, "checks": checks})
        if len(texts) == N_PARAPHRASES and not any(checks):
            break
    record["paraphrases"], record["checks"] = texts, checks
    return record


def family_bucket(family):
    return "emergent" if family.startswith(EMERGENT) else family


def distribution(families):
    counted = Counter(family_bucket(f) for f in families if f != UNLABELED)
    total = sum(counted.values())
    return {f: c / total for f, c in counted.items()} if total else {}


def tvd(p, q):
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def original_groups(out, models=MODELS, per_group=PER_GROUP, n=20):
    """The map's existing original-prompt labels, split by sample index into groups of
    per_group per model (pooled over the models)."""
    samples = {json.loads(l)["sample_id"]: json.loads(l) for l in (out / "samples.jsonl").read_text().splitlines()}
    labels = {json.loads(l)["sample_id"]: json.loads(l)["family"] for l in (out / "labels.jsonl").read_text().splitlines()}
    groups = []
    for start in range(0, n, per_group):
        fams = [labels[sid] for sid, s in samples.items()
                if s["model"] in models and start <= s["idx"] < start + per_group and sid in labels]
        groups.append(fams)
    return groups


def run(herd, task):
    cfg = herd.cfg
    out = herd.out_dir(task)
    pdir = out / "paraphrase"
    pdir.mkdir(parents=True, exist_ok=True)
    para = make_paraphrases(herd, task)
    write_json(pdir / "paraphrases.json", para)

    records = []
    jobs = list(itertools.product(range(len(para["paraphrases"])), MODELS, range(PER_GROUP)))

    def sample(job):
        i, model, idx = job
        sid = f"p{i}-{model.replace('/', '_')}-{idx:03d}"
        record = {"sample_id": sid, "model": f"{model}#p{i}", "true_model": model, "paraphrase": i, "idx": idx}
        try:
            entry = herd.llm.chat(model, [{"role": "user", "content": para["paraphrases"][i]}],
                                  temperature=cfg["sampling"]["temperature"], max_tokens=cfg["sampling"]["max_tokens"],
                                  sample_idx=f"paraphrase:{i}:{idx}")
        except Exception as e:
            return {**record, "status": "api_error", "error": str(e)}
        text = message_text(entry)
        record.update(text_path=f"paraphrase/{sid}.md", finish_reason=finish_reason(entry), cost_usd=entry["cost_usd"])
        (out / record["text_path"]).write_text(text)
        code = extract_code(text, task.entrypoint)
        if code is not None:
            record["code_path"] = f"paraphrase/{sid}.py"
            (out / record["code_path"]).write_text(code)
        return record

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(cfg["sampling"]["concurrency"]) as pool:
        records = list(pool.map(sample, jobs))
    herd.progress(f"paraphrase samples: {sum('text_path' in r for r in records)}/{len(records)}")
    taxonomy = read_json(out / "taxonomy.json")
    labels, emergent = label_samples(task, [r for r in records if "text_path" in r], taxonomy, herd.llm, cfg, out,
                                     herd.progress)
    for r in records:
        r["family"] = labels.get(r["sample_id"], {}).get("family", UNLABELED)
        r["justification"] = labels.get(r["sample_id"], {}).get("justification")
    write_json(pdir / "samples.json", records)
    result = analyze(original_groups(out), [[r["family"] for r in records if r["paraphrase"] == i]
                                            for i in range(len(para["paraphrases"]))])
    result.update(models=MODELS, per_group_per_model=PER_GROUP, paraphrase_checks=para["checks"],
                  emergent_in_paraphrase_labels=sum(r["family"].startswith(EMERGENT) for r in records),
                  unlabeled=sum(r["family"] == UNLABELED for r in records))
    write_json(out / "paraphrase.json", result)
    return result


def analyze(original, paraphrased):
    """R: TVD among original groups; P: TVD between each paraphrase group and each original group.
    Decision rule (pre-registered): mean(P) <= max(R) -> robust to wording, else prompt effect."""
    o = [distribution(g) for g in original]
    p = [distribution(g) for g in paraphrased]
    R = [tvd(a, b) for a, b in itertools.combinations(o, 2)]
    P = [tvd(a, b) for a in p for b in o]
    among = [tvd(a, b) for a, b in itertools.combinations(p, 2)]
    verdict = "robust to wording" if statistics.mean(P) <= max(R) else "partly a prompt effect"
    return {"R_resampling": [round(x, 3) for x in R], "P_paraphrase_vs_original": [round(x, 3) for x in P],
            "among_paraphrases": [round(x, 3) for x in among],
            "mean_R": round(statistics.mean(R), 3), "max_R": round(max(R), 3), "mean_P": round(statistics.mean(P), 3),
            "verdict": verdict,
            "original_group_sizes": [sum(1 for f in g if f != UNLABELED) for g in original],
            "paraphrase_group_sizes": [sum(1 for f in g if f != UNLABELED) for g in paraphrased],
            "original_distributions": o, "paraphrase_distributions": p,
            "caveat": "groups of 10 samples: only large shifts are detectable"}
