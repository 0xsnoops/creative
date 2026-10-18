"""blind_spot: implement the families in blind_spots.json, round-robin, with the shared template.
The herd map that found them is built beforehand and is not billed to this operator."""


def jobs(ctx):
    spots = ctx.study["blind_spots"]
    ctx.notes["families"] = [b["family"] for b in spots]
    while spots:
        for family in spots:
            yield ctx.family_job(family)
