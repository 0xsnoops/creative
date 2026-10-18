"""blind_spot_hybrid: the blind-spot families, round-robin, exactly as blind_spot picks them, but
implemented with HYBRID_TEMPLATE: the family is the core mechanism and the implementer may add
components from other mechanisms. The matched control is blind_spot itself (same families, the
shared "faithfully ... within the family" template). Motivation: plain sampling's winning cache
policy in the operator benchmark was a hybrid of two mechanisms, which the shared template blocks."""
from ..reality import hybrid_prompt


def hybrid_job(ctx, family, extra="", meta=None):
    """Same budget, cache keys and accounting as ctx.family_job; only the template differs."""
    return ctx.job(hybrid_prompt(ctx.task, family, extra), 1.0, family, {"template": "hybrid", **(meta or {})})


def jobs(ctx):
    spots = ctx.study["blind_spots"]
    ctx.notes["families"] = [b["family"] for b in spots]
    ctx.notes["template"] = "hybrid"
    while spots:
        for family in spots:
            yield hybrid_job(ctx, family)
