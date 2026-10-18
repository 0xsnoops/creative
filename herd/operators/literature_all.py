"""literature_all: the ablation for blind_spot. Implements every sourced family from the
literature taxonomy, round-robin, ignoring the herd map. Same implementer, template, evaluator
and budget as blind_spot, so if blind_spot does better, knowing where models concentrate chose
better experiments; if not, the gain came from literature retrieval alone."""
from ..label import EMERGENT


def jobs(ctx):
    families = [f for f in ctx.study["families"] if f["source_urls"] and not f["family"].startswith(EMERGENT)]
    ctx.notes["families"] = [f["family"] for f in families]
    while families:
        for family in families:
            yield ctx.family_job(family)
