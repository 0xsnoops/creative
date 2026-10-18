"""verbalized: Verbalized Sampling (Zhang et al., 2025). Ask for 5 approaches with
probabilities, sampled from the tails (each < 0.10), then implement each one."""
import itertools
import re

from ..llm import message_text

VS_PROMPT = """{task}

---
Generate 5 responses to the task above. Each response is a different approach to solving
it, described in 2-4 sentences without code. Put each response within <response> tags,
containing a short <name>, the <text> of the approach, and a numeric <probability>.
Randomly sample the responses from the tails of the distribution, such that the
probability of each response is less than 0.10."""

RESPONSE = re.compile(r"<response>.*?<name>(.*?)</name>.*?<text>(.*?)</text>.*?"
                      r"<probability>\s*([0-9.]+)\s*</probability>.*?</response>", re.S)


def parse(text):
    return [{"name": n.strip(), "text": t.strip(), "probability": float(p)}
            for n, t, p in RESPONSE.findall(text)]


def jobs(ctx):
    empty_rounds, ctx.notes["responses"] = 0, []
    for round_ in itertools.count():
        entry = ctx.llm.chat(ctx.cfg["models"]["implementer"],
                             [{"role": "user", "content": VS_PROMPT.format(task=ctx.task.prompt)}],
                             temperature=1.0, max_tokens=ctx.cfg["reality_check"]["max_tokens"],
                             sample_idx=ctx.key("vs", round_))
        responses = parse(message_text(entry))
        ctx.notes["responses"] += responses
        empty_rounds = 0 if responses else empty_rounds + 1
        if empty_rounds == 3:
            return
        for r in responses:
            yield ctx.family_job({"family": r["name"], "one_line_mechanism": r["text"],
                                  "source_summary": f"Proposed by verbalized sampling with stated "
                                                    f"probability {r['probability']}.",
                                  "source_urls": []})
