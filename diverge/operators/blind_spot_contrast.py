"""blind_spot_contrast: implement the blind-spot families (as blind_spot does) but also tell the
implementer which mechanisms to differ from: the consensus top families, i.e. what most model
samples actually did. This is the only operator that puts consensus-map information into the
generation prompt itself.

blind_spot_contrast_random is its control: identical wording and the same number of listed
mechanisms, drawn at random from the taxonomy families that are not consensus families (other blind
spots included; never the target itself). If both help equally,
the gain comes from "be different" pressure or prompt length, not from knowing the consensus.
(Proposed in the model review as cluster C5 by deepseek, x-ai, google and moonshotai; the
random control follows qwen's critique.)"""
import random

from ..contrast import consensus_families, section


def _jobs(ctx, pick_contrast, kind):
    spots = ctx.study["blind_spots"]
    ctx.notes["families"] = [b["family"] for b in spots]
    ctx.notes["contrast_kind"] = kind
    ctx.notes["contrast"] = {}
    while spots:
        for family in spots:
            contrast = pick_contrast(family)
            ctx.notes["contrast"][family["family"]] = [f["family"] for f in contrast]
            extra = section(contrast)
            yield ctx.family_job(family, extra, meta={"contrast_kind": kind,
                                                      "contrast_families": [f["family"] for f in contrast]})


def jobs(ctx):
    consensus = consensus_families(ctx.study)
    return _jobs(ctx, lambda family: [f for f in consensus if f["family"] != family["family"]], "herd")


def random_jobs(ctx):
    consensus = consensus_families(ctx.study)
    consensus_names = {f["family"] for f in consensus}
    pool = sorted((f for f in ctx.study["families"] if f["family"] not in consensus_names), key=lambda f: f["family"])
    shortfalls = ctx.notes.setdefault("control_shortfall", {})

    def pick(family):
        count = len([f for f in consensus if f["family"] != family["family"]])  # what the consensus list would hold
        rng = random.Random(f"contrast_random:{ctx.seed}:{family['family']}")
        candidates = [f for f in pool if f["family"] != family["family"]]
        if len(candidates) < count:
            shortfalls[family["family"]] = f"{len(candidates)} non-consensus families for {count} slots"
        return rng.sample(candidates, min(count, len(candidates)))
    return _jobs(ctx, pick, "random")
