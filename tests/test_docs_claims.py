"""The agent-facing docs must not overclaim: no sentence may say DIVERGE beats a baseline, another
agent or another system, or that an improvement is proven, unless the sentence negates it
("has not been shown to beat ..."). Also pins the README's top lines and agent vocabulary."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ["README.md", "docs/TRACKS.md", "skills/diverge/SKILL.md"]

SUPERIOR = re.compile(r"\b(beat|beats|beating|outperform(?:s|ed|ing)?|better than|superior to|surpass(?:es|ed)?|"
                      r"wins? (?:against|over)|dominates?)\b", re.I)
RIVAL = re.compile(r"(fresh[_ -]start|\bclaude\b|\bcodex\b|alphaevolve|shinkaevolve|\badhd\b|funsearch|\bgpt|"
                   r"\bgemini\b|\bcursor\b|another model|other models?\b|other agents?\b|continuing|"
                   r"state of the art|\bsota\b|\bhumans?\b)", re.I)
PROVEN = re.compile(r"\b(proven|proves|proved|guarantee[sd]?|guaranteed)\b", re.I)
NEGATION = re.compile(r"(\bnot\b|\bnever\b|\bno\b|\bnor\b|n't\b|\bcannot\b|\bunproven\b|\buntested\b|"
                      r"\bwhether\b|\bif\b|\bwithout\b|\bnone\b)", re.I)


def sentences(text):
    """Table rows and list items are units; other paragraphs are joined and split into sentences."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    out = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.splitlines()
        units, cur = [], []
        for line in lines:
            if line.lstrip().startswith(("|", "- ", "* ", "#")) or re.match(r"\s*\d+\. ", line):
                if cur:
                    units.append(" ".join(cur))
                cur = [line.strip()]
                if line.lstrip().startswith(("|", "#")):
                    units.append(cur.pop())
            else:
                cur.append(line.strip())
        if cur:
            units.append(" ".join(cur))
        for u in units:
            out += [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z`*\"(])", u) if s.strip()]
    return out


def overclaims(text):
    bad = []
    for s in sentences(text):
        if NEGATION.search(s):
            continue
        if (SUPERIOR.search(s) and RIVAL.search(s)) or PROVEN.search(s):
            bad.append(s)
    return bad


@pytest.mark.parametrize("bad", [
    "DIVERGE beats fresh_start on bin packing.",
    "Escape outperforms Claude and Codex.",
    "DIVERGE is better than AlphaEvolve.",
    "Our operator outperforms ADHD.",
    "DIVERGE gives a proven improvement.",
    "In our runs, escape wins against a fresh start.",
])
def test_detector_flags_overclaims(bad):
    assert overclaims(bad) == [bad]


@pytest.mark.parametrize("ok", [
    "Escape has not been shown to beat fresh_start.",
    "It has not been shown to beat continuing, a fresh start or another model.",
    "These runs are no evidence either way about escape vs fresh_start at scale.",
    "It scored 1.096 / 0.973 and lost to fresh_start.",
    "We do not claim that DIVERGE outperforms Claude.",
    "Treat it as one experiment, not a guaranteed fix.",
])
def test_detector_allows_negations_and_neutral_text(ok):
    assert overclaims(ok) == []


def test_sentence_split_joins_wrapped_lines():
    text = "DIVERGE clearly\nbeats fresh_start. Fine.\n\n- item one\n  continues here\n| a | b |"
    assert "DIVERGE clearly beats fresh_start." in sentences(text)
    assert "- item one continues here" in sentences(text)


@pytest.mark.parametrize("doc", DOCS)
def test_docs_do_not_overclaim(doc):
    path = ROOT / doc
    assert path.exists(), doc
    assert overclaims(path.read_text()) == []


def _plain(line):
    return re.sub(r"[*_>`]", "", line).strip()


def test_readme_keeps_top_lines():
    lines = [_plain(l) for l in (ROOT / "README.md").read_text().splitlines() if l.strip()]
    assert lines[:3] == ["# DIVERGE", "Escape strategy loops in AI agents.",
                         "Different code isn't always different thinking."]


def test_readme_agent_section_is_high_and_complete():
    text = (ROOT / "README.md").read_text()
    for word in ("diverge_escape", "diverge_status", "next_action", "preliminary"):
        assert word in text, word
    headings = [l for l in text.splitlines() if l.startswith("## ")]
    assert headings[0] == "## For coding agents"
    for verdict in ("keep", "repair", "escape", "escalate", "inconclusive"):
        assert f"| `{verdict}` |" in text, verdict


def test_readme_verdict_table_matches_contract():
    from diverge.contract import NEXT_ACTION
    text = (ROOT / "README.md").read_text()
    for verdict, action in NEXT_ACTION.items():
        assert action in text, verdict
