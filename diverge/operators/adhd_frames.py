"""adhd_frames: ADHD's divergent frames as a benchmark operator. Per round, pick 5 frames with
ADHD's rule (4 engineering-tagged frames plus 1 "wild" frame), ask the implementer model for
ideas under each frame with ADHD's divergent-mode prompt, then implement and evaluate each idea
with HERD's shared implementation template. All planning calls are billed to this operator.

The frames, the selection rule and the DIVERGE prompts are copied from ADHD
(https://github.com/UditAkhourii/adhd, src/packs/core.ts, src/frames.ts, src/engine.ts at
commit 7b91ce6), MIT License, Copyright (c) 2026 ADHD contributors. The full license text is in
herd/operators/ADHD_LICENSE. ADHD's later stages (anchor-stripping reframe, critic scoring,
clustering, deepening) are not used: this operator measures frames -> implement -> evaluate."""
import itertools
import json
import random

from ..llm import message_text, parse_json

# Verbatim from ADHD src/packs/core.ts (id, label, prompt, tags).
FRAMES = [
    ("hardware-eyes", "Hardware engineer",
     "You think in latency, memory layout, and physical constraints. Re-ask this problem as if it were a hardware/firmware problem. What does the bus topology, the cache, the timing budget tell you?",
     ["code", "wild"]),
    ("regulator", "Regulator / auditor",
     "You audit systems for compliance and failure modes. What ideas surface when you ask: what must be provable, traceable, or refusable here?",
     ["design", "general"]),
    ("ten-year-old", "10-year-old",
     "You are a curious 10-year-old who has never seen software before. Describe naive but unencumbered approaches. Ignore convention.",
     ["general", "wild"]),
    ("adversary", "Competitor trying to break it",
     "You are a hostile competitor or attacker. Generate approaches that exploit, fail, or sabotage the obvious solution. Then invert into ideas.",
     ["code", "design"]),
    ("biology", "Cross-domain: biology",
     "Transplant a mechanism from biology — immune systems, neural plasticity, cell signaling, evolution, gut flora — and force-fit it onto this engineering problem.",
     ["code", "wild"]),
    ("logistics", "Cross-domain: logistics / supply chain",
     "Steal mechanisms from logistics: queues, batching, just-in-time, hub-and-spoke, returns, last-mile. Apply them literally to this problem.",
     ["code", "design"]),
    ("game-design", "Cross-domain: game design",
     "Approach this as a game designer. What are the loops, rewards, friction, save-states, speedrun tricks? Treat the user/system as a player.",
     ["design", "general"]),
    ("markets", "Cross-domain: markets",
     "Treat the problem as a market. Who are the buyers, sellers, market-makers? What does an auction, a futures contract, a clearing house look like here?",
     ["design", "wild"]),
    ("inversion", "Inversion",
     "Ask the OPPOSITE question. If the goal is X, brainstorm 'how would we guarantee NOT-X' — then negate each answer back into an idea.",
     ["code", "design", "general"]),
    ("extreme-zero", "Extreme: $0 budget, 1 hour",
     "You have no money, no team, one hour. What's the crudest version that still does the load-bearing thing? Hacks, hardcoded values, manual loops welcome.",
     ["code", "general"]),
    ("extreme-infinite", "Extreme: infinite budget, 10 years",
     "You have infinite compute, infinite engineers, a decade. What does the maximalist version look like? What would only be possible at that scale?",
     ["design", "wild"]),
    ("remove-assumption", "Remove the load-bearing assumption",
     "Name the thing everyone treats as fixed in this problem (the framework, the database, the request/response model, the file system, the network). Imagine it's gone. Generate ideas that only exist in that world.",
     ["code", "design", "wild"]),
    ("speedrunner", "Speedrunner",
     "You're a speedrunner. Find glitches, skips, out-of-bounds tricks, frame-perfect shortcuts. What's the abusive-but-legal path through this problem?",
     ["code", "wild"]),
    ("ant-colony", "Ant colony / swarm",
     "No central planner. Many dumb agents, local rules, pheromone trails. How does the problem solve itself emergently?",
     ["code", "wild"]),
    ("ops-3am", "On-call at 3am",
     "You're the on-call engineer woken at 3am when this thing breaks. What design would let you not get paged? What's the runbook-shaped solution?",
     ["code", "design"]),
]

# Verbatim from ADHD src/engine.ts.
DIVERGE_SYSTEM = """You are in DIVERGENT mode. You are a generator, not a critic.
Rules:
- Output a JSON array only. No prose before/after.
- Generate the requested number of distinct ideas.
- Each idea is a SHORT phrase or single sentence. No paragraphs.
- Push past the obvious. The first 3 ideas you'd think of are banned —
  assume the reader already had those. Aim for the awkward middle.
- Bad, weird, and absurd ideas are welcome; they seed better ones.
- Do not evaluate, hedge, or rank. Just generate."""

DIVERGE_USER = """PROBLEM:
{problem}

FRAME — {label}:
{prompt}

Generate {n} ideas under this frame.
Output JSON array: [{{"text": "...", "rationale": "..."}}]
- text: one phrase/sentence, the idea itself
- rationale: 1 short clause on why this frame surfaces it (optional)"""

N_FRAMES = 5         # ADHD's skill: "Pick 5 per run"
IDEAS_PER_FRAME = 6  # ADHD's default: 5 frames x 6 ideas


def select_frames(rng, n=N_FRAMES):
    """ADHD's selection for code-shaped problems: n-1 engineering (code/design) frames plus one
    wild frame. ADHD's selectFrames code can return n-1 frames when its random wild pick is
    already among the engineering picks; we follow its skill ("pick 5": 4 engineering + 1 wild)
    and draw the wild frame from those not already chosen."""
    pool = [f for f in FRAMES if "code" in f[3] or "design" in f[3]]
    shuffled = pool[:]
    rng.shuffle(shuffled)
    picked = shuffled[:max(1, n - 1)]
    wild = [f for f in FRAMES if "wild" in f[3] and f[0] not in {p[0] for p in picked}]
    if wild:
        picked.append(rng.choice(wild))
    return picked[:n]


def parse_ideas(text):
    try:
        rows = parse_json(text) if text.strip().startswith("{") else json.loads(
            text[text.find("["):text.rfind("]") + 1])
    except ValueError:
        return []
    rows = rows if isinstance(rows, list) else []
    return [r for r in rows if isinstance(r, dict) and isinstance(r.get("text"), str) and r["text"].strip()]


def jobs(ctx):
    ctx.notes["attribution"] = "ADHD (github.com/UditAkhourii/adhd @7b91ce6), MIT License"
    ctx.notes["rounds"] = []
    for round_ in itertools.count():
        frames = select_frames(random.Random(f"adhd:{ctx.seed}:{round_}"))
        ideas = {}
        record = {"round": round_, "frames": [f[0] for f in frames], "ideas": {}}
        ctx.notes["rounds"].append(record)
        for position in range(IDEAS_PER_FRAME):
            for fid, label, prompt, _ in frames:
                if fid not in ideas:  # diverge lazily, so a small budget still covers several frames
                    entry = ctx.llm.chat(
                        ctx.cfg["models"]["implementer"],
                        [{"role": "system", "content": DIVERGE_SYSTEM},
                         {"role": "user", "content": DIVERGE_USER.format(problem=ctx.task.prompt, label=label,
                                                                         prompt=prompt, n=IDEAS_PER_FRAME)}],
                        temperature=1.0, max_tokens=ctx.cfg["reality_check"]["max_tokens"],
                        sample_idx=ctx.key("diverge", round_, fid))
                    ideas[fid] = parse_ideas(message_text(entry))[:IDEAS_PER_FRAME]
                    record["ideas"][fid] = [i["text"] for i in ideas[fid]]
                if position >= len(ideas[fid]):
                    continue
                idea = ideas[fid][position]
                family = {"family": f"{label}: {idea['text'][:80]}", "one_line_mechanism": idea["text"],
                          "source_summary": f"Idea generated under the '{label}' frame"
                                            + (f" ({idea['rationale']})." if idea.get("rationale") else "."),
                          "source_urls": []}
                yield ctx.family_job(family, meta={"frame": fid, "adhd_round": round_})
        if all(not v for v in ideas.values()):
            return  # nothing parseable this round; stop rather than loop on planning calls
