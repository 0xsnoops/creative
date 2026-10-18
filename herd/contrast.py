"""The herd-contrast section (model-review cluster C5), shared by the blind_spot_contrast
operator and the optional `contrast` flag of the MCP tools, so both send identical text."""

CONTRAST = """Mechanisms your solution should differ from (do not fall back to them; keep the task's
interface and rules):
{listing}

Implement the family above with a decision rule that is clearly different from these."""


def listing(families):
    return "\n".join(f"- {f['family']}: {f['one_line_mechanism']}" for f in families)


def herd_families(study):
    by_name = {f["family"]: f for f in study["families"]}
    return [by_name[n] for n in study["herd_map"]["top3"] if n in by_name]


def section(families):
    return CONTRAST.format(listing=listing(families)) if families else ""


def herd_section(study, family):
    """Contrast against the herd's top families (excluding the target itself)."""
    return section([f for f in herd_families(study) if f["family"] != family["family"]])
