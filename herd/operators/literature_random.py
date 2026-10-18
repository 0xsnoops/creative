"""literature_random: the count-matched control for blind_spot. Implements a random set of
sourced literature families of the same size as the blind-spot list (seeded by the repeat), so
blind_spot's advantage cannot come from simply concentrating the budget on fewer families."""
import random

from ..label import EMERGENT


def jobs(ctx):
    sourced = sorted((f for f in ctx.study["families"] if f["source_urls"] and not f["family"].startswith(EMERGENT)),
                     key=lambda f: f["family"])
    n = min(len(sourced), max(1, len(ctx.study["blind_spots"])))
    families = random.Random(f"literature_random:{ctx.seed}").sample(sourced, n)
    ctx.notes["families"] = [f["family"] for f in families]
    while families:
        for family in families:
            yield ctx.family_job(family)
