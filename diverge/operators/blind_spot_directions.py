"""blind_spot_directions: the blind-spot families, round-robin as blind_spot picks them, but each
successive attempt at a family gets a DIFFERENT design direction appended to the shared template
instead of resampling the same prompt. Motivation: 14 of 18 panel models flagged that blind_spot's
retries resample one prompt and often return the same program (several attempts scored exactly
alike). The directions are a fixed list written here and cycled per retry; the direction and its
round are recorded on the candidate. Budget, cache keys and accounting are blind_spot's.

blind_spot_directions_hybrid is the same schedule on top of HYBRID_TEMPLATE (see blind_spot_hybrid),
so the two levers (hybrid allowance, direction per retry) can be measured apart and together."""
from .blind_spot_hybrid import hybrid_job

# (tag, text appended right before the template's final instruction)
DIRECTIONS = [
    ("state", "Design direction for this attempt: concentrate on WHAT STATE you keep. Choose the bookkeeping "
              "(counters, timestamps, histories, sketches) deliberately and let every decision follow from it."),
    ("signal", "Design direction for this attempt: concentrate on the DECISION SIGNAL. Make the score or ranking "
               "rule behind each decision as sharp as you can and keep everything else simple."),
    ("tie_break", "Design direction for this attempt: concentrate on ADMISSION and TIE-BREAKING. Decide what gets "
                  "let in, and what is kept when candidates look equal, and make that rule do real work."),
    ("adaptivity", "Design direction for this attempt: concentrate on ADAPTIVITY. Let parameters or behaviour "
                   "shift with what the workload has been doing recently instead of staying fixed."),
    ("bookkeeping", "Design direction for this attempt: concentrate on CHEAP VERSUS RICH BOOKKEEPING. Pick one "
                    "end on purpose, minimal fast state or rich state that buys better decisions, and make the "
                    "trade-off pay."),
]


def _jobs(ctx, make_job, template):
    spots = ctx.study["blind_spots"]
    ctx.notes["families"] = [b["family"] for b in spots]
    ctx.notes["template"] = template
    ctx.notes["directions"] = [tag for tag, _ in DIRECTIONS]
    rounds = {b["family"]: 0 for b in spots}
    while spots:
        for family in spots:
            n = rounds[family["family"]]
            rounds[family["family"]] = n + 1
            tag, text = DIRECTIONS[n % len(DIRECTIONS)]
            yield make_job(family, text, {"direction": tag, "direction_round": n})


def jobs(ctx):
    return _jobs(ctx, lambda family, extra, meta: ctx.family_job(family, extra, meta=meta), "shared")


def hybrid_jobs(ctx):
    return _jobs(ctx, lambda family, extra, meta: hybrid_job(ctx, family, extra, meta), "hybrid")
