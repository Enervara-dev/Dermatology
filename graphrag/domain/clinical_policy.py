"""
graphrag.domain.clinical_policy
─────────────────────────────────
Structured clinical decision policy for the triage + answer layers — the
guideline-aligned rules that shape ranking, urgency, questioning, escalation,
and patient-facing tone.

⭐ EDIT FOR A NEW SPECIALTY/USE CASE. This is where triage behaviour is tuned:
which symptoms carry the most weight, which red flags force escalation, how many
clarifying questions are allowed, and the prose policies woven into the answer
prompt. The code (pipeline + answer_prompt) reads from here; nothing clinical is
hardcoded in the logic.
"""

from __future__ import annotations

import re

# ── Follow-up questioning budget ──────────────────────────────────────────────
# The triage layer may ask up to this many clarifying questions in one turn when
# severity or ambiguity warrants it (was effectively 1 before).
MAX_FOLLOWUP_QUESTIONS: int = 3

# ── Diagnostic loop termination ───────────────────────────────────────────────
# Terminal state for the diagnostic process. The session already carries a turn
# counter (SessionMemory.turn_count / WorkingMemory.turn_count); once the user
# has taken more than MAX_DIAGNOSTIC_TURNS turns — OR the gatekeeper stops needing
# follow-ups — the pipeline forces the intent to ASSESSMENT_READY so Stage 4
# concludes with a final assessment instead of looping on more questions.
MAX_DIAGNOSTIC_TURNS: int = 2
ASSESSMENT_READY_INTENT: str = "assessment_ready"

# Appended to the answer system prompt (Stage 4) when the diagnostic process is
# terminal — exact wording per the loop-prevention contract.
ASSESSMENT_READY_INSTRUCTION: str = (
    "CRITICAL INSTRUCTION: You have collected enough symptoms. Do NOT ask any "
    "further follow-up questions. Provide your final assessment and recommendations "
    "strictly based on the provided context."
)

# Appended when routing falls to NO_RETRIEVAL during a medical interaction — the
# model must wrap up from memory instead of defaulting to open-ended chat.
NO_RETRIEVAL_CONCLUDE_INSTRUCTION: str = (
    "CRITICAL INSTRUCTION: No new clinical information is being retrieved. "
    "Summarize the findings already gathered in this conversation, give your best "
    "assessment and clear next-step recommendations, and conclude — do NOT ask "
    "further follow-up questions or prolong the interaction."
)


def closure_directive(
    *,
    intent: str,
    needs_followup: bool,
    memory_only: bool,
    has_findings: bool,
) -> str | None:
    """
    Resolve the terminal/closure constraint to append at Stage 4, or None.

    - NO_RETRIEVAL during a medical interaction → conclude from memory.
    - assessment_ready (or the gatekeeper needing no more follow-ups) → final
      assessment, no further questions.
    Gated on `has_findings` so greetings / non-clinical turns are never forced
    to "conclude".
    """
    if not has_findings:
        return None
    if memory_only:
        return NO_RETRIEVAL_CONCLUDE_INSTRUCTION
    if intent == ASSESSMENT_READY_INTENT or not needs_followup:
        return ASSESSMENT_READY_INSTRUCTION
    return None

# ── High-signal symptoms (drive ranking + urgency) ────────────────────────────
# Prose, for prompt injection. Mirrors the canonical risk keys in the memory
# layer (session_memory/domain/risk_rules.py) but is phrased for the LLM.
HIGH_SIGNAL_SYMPTOMS_TEXT = (
    "rapidly spreading rash, skin blistering or peeling, severe skin pain, fever with a "
    "rash, signs of skin infection (pus, warmth, spreading redness, swelling), changing or "
    "bleeding moles, severe or persistent pruritus (itching), and rapid hair or nail loss"
)

# ── Emergency red flags (dermatological) ──────────────────────────────────────
# Prose list for the gatekeeper emergency section.
RED_FLAGS_TEXT = (
    "swelling of the face, lips, tongue, or throat; difficulty breathing or swallowing; "
    "blistering, peeling, or sloughing of the skin over large areas; skin turning black, "
    "purple, or necrotic; rapidly spreading redness, warmth, and severe pain; "
    "or a new skin rash accompanied by high fever, chills, or confusion; "
    "or melanoma/skin cancer warning signs (a changing, bleeding, or rapidly growing mole; "
    "asymmetry, irregular borders, multiple colors, diameter greater than 6 mm (size of a pencil eraser), "
    "or an evolving or ulcerated dark/pigmented lesion)"
)

# Deterministic backstop — STRONG, present-tense red-flag phrases. The pipeline
# escalates to the emergency message when any of these match the user's message,
# even if the LLM gatekeeper missed it. Patterns are intentionally conservative
# (they require explicit severity) so ordinary complaints like "dry skin"
# do NOT trip them.
RED_FLAG_PATTERNS: dict[str, re.Pattern[str]] = {
    "anaphylaxis_angioedema": re.compile(
        r"\b(anaphylaxis|anaphylactic|angioedema)\b"
        r"|\bswell\w*\s+(of\s+)?(my\s+)?(face|lips?|tongue|throat)\b"
        r"|\bswollen\s+(face|lips?|tongue|throat)\b"
        r"|\b(difficulty|unable\s+to)\s+(swallow\w*|breathe?\w*)\b",
        re.IGNORECASE,
    ),
    "skin_peeling_sloughing": re.compile(
        r"\bskin\s+(peeling\s+off|peeling\s+in\s+sheets|sloughing\s+off|shedding)\b"
        r"|\bskin\s+is\s+coming\s+off\b"
        r"|\b(blistering|blisters)\s+(over\s+large\s+areas|all\s+over\s+body|widespread)\b"
        r"|\b(stevens[- ]johnson|sjs|toxic\s+epidermal\s+necrolysis|ten)\b",
        re.IGNORECASE,
    ),
    "necrotic_skin": re.compile(
        r"\bskin\s+(turning\s+black|is\s+black|necrotic|dying|dead)\b"
        r"|\bblack\s+(lesion|spot|patch|skin)\s+(turning\s+black|necrotic)\b"
        r"|\bgangrene\b",
        re.IGNORECASE,
    ),
    "severe_spreading_infection": re.compile(
        r"\b(necrotizing\s+fasciitis|flesh[- ]eating|cellulitis)\b"
        r"|\b(rapidly\s+spreading|spreading\s+fast)\s+(redness|warmth|swelling|rash)\b"
        r"|\bsevere\s+(skin\s+)?pain\s+and\s+(redness|warmth|swelling)\b",
        re.IGNORECASE,
    ),
    "systemic_rash_fever": re.compile(
        r"\b(rash|hives|spots?)\b.*\b(high\s+fever|chills|confusion|drowsy|drowsiness)\b"
        r"|\b(high\s+fever|chills|confusion|drowsy|drowsiness)\b.*\b(rash|hives|spots?)\b",
        re.IGNORECASE,
    ),
    "changing_mole": re.compile(
        r"\b(mole|spot|lesion|freckle)s?\b.*\b(chang(e|es|ed|ing)|evolv(e|es|ed|ing))\b"
        r"|\b(chang(e|es|ed|ing)|evolv(e|es|ed|ing))\b.*\b(mole|spot|lesion|freckle)s?\b",
        re.IGNORECASE,
    ),
    "bleeding_mole": re.compile(
        r"\b(mole|spot|lesion|freckle)s?\b.*\b(bleed(s|ing)?|bled|ooz(e|es|ing)|weep(s|ing)?)\b"
        r"|\b(bleed(s|ing)?|bled|ooz(e|es|ing)|weep(s|ing)?)\b.*\b(mole|spot|lesion|freckle)s?\b",
        re.IGNORECASE,
    ),
    "rapidly_growing_mole": re.compile(
        r"\b(mole|spot|lesion|freckle)s?\b.*\b(rapid(ly)?|fast|quick(ly)?)\b.*\b(grow(s|th|ing)?|enlarg(e|es|ed|ing)|spread(s|ing)?)\b"
        r"|\b(rapid(ly)?|fast|quick(ly)?)\b.*\b(grow(s|th|ing)?|enlarg(e|es|ed|ing)|spread(s|ing)?)\b.*\b(mole|spot|lesion|freckle)s?\b",
        re.IGNORECASE,
    ),
    "asymmetrical_mole": re.compile(
        r"\b(mole|spot|lesion|freckle)s?\b.*\b(asymmetric(al|y)?|not\s+symmetric(al)?|lopsided)\b"
        r"|\b(asymmetric(al|y)?|not\s+symmetric(al)?|lopsided)\b.*\b(mole|spot|lesion|freckle)s?\b",
        re.IGNORECASE,
    ),
    "irregular_borders": re.compile(
        r"\b(irregular|jagged|blurred|notched|scalloped|uneven)\b.*\b(border|edge|margin)s?\b",
        re.IGNORECASE,
    ),
    "multiple_colors": re.compile(
        r"\b(mole|spot|lesion|freckle)s?\b.*\b(multi\w*[- ]color\w*|different\s+color\w*|color\s+variation|shades?\s+of)\b"
        r"|\b(multi\w*[- ]color\w*|different\s+color\w*|color\s+variation|shades?\s+of)\b.*\b(mole|spot|lesion|freckle)s?\b",
        re.IGNORECASE,
    ),
    "new_dark_lesion": re.compile(
        r"\bnew\b.*\b(dark|black|brown)\b.*\b(lesion|spot|mole|freckle|mark|growth)s?\b",
        re.IGNORECASE,
    ),
    "evolving_pigmented_lesion": re.compile(
        r"\b(pigment(ed)?|dark(er)?|colored)\b.*\b(lesion|spot|mole|freckle)s?\b.*\b(evolv(e|es|ed|ing)|chang(e|es|ed|ing))\b"
        r"|\b(evolv(e|es|ed|ing)|chang(e|es|ed|ing))\b.*\b(pigment(ed)?|dark(er)?|colored)\b.*\b(lesion|spot|mole|freckle)s?\b",
        re.IGNORECASE,
    ),
    "ulcerated_pigmented_lesion": re.compile(
        r"\b(ulcerat(e|ed|ing)?|open\s+sore|non[- ]healing)\b.*\b(pigment(ed)?|dark(er)?|colored|mole|spot)\b.*\b(lesion|spot|mole)s?\b",
        re.IGNORECASE,
    ),
    "diameter_greater_than_6mm": re.compile(
        r"\b(diameter|size|width|wider|larger)\b.*\b(6\s*mm|6\s*millimeter|six\s*mm|pencil\s+eraser)\b"
        r"|\b(6\s*mm|6\s*millimeter|six\s*mm|pencil\s+eraser)\b.*\b(diameter|size|width)\b",
        re.IGNORECASE,
    ),
}


def detect_red_flags(text: str) -> list[str]:
    """Return the names of any emergency red flags present in `text`."""
    if not text:
        return []
    return [name for name, pat in RED_FLAG_PATTERNS.items() if pat.search(text)]


# ── Answer-layer policy blocks (woven into the answer system prompt) ───────────

DIFFERENTIAL_POLICY = f"""CLINICAL REASONING & DIFFERENTIAL
- Lead with the 1–3 MOST CLINICALLY LIKELY explanations for THIS patient, each with a \
one-line rationale tied to their specific features. Do not enumerate long lists of \
low-probability possibilities.
- Weight high-signal features heavily when ranking and when judging urgency: \
{HIGH_SIGNAL_SYMPTOMS_TEXT}.
- Do NOT surface rare or exotic conditions unless the symptoms strongly support them \
or the patient explicitly asks. Mention a "can't-miss" serious cause only when its \
red flags are plausibly present — and then say what would confirm or exclude it.
- Synthesise the retrieved context into coherent clinical reasoning (why these causes, \
what links the findings) — do not just summarise the source text."""

UNCERTAINTY_POLICY = """HANDLING UNCERTAINTY
- If the picture is uncertain but LOW risk: say so plainly, give sensible self-care and \
clear "see a clinician if…" criteria, and offer to narrow it down with one or two questions.
- If the picture is uncertain AND any severe/high-signal feature is present: do NOT \
reassure. Err toward caution — recommend timely or urgent assessment and state the \
specific red flags that mean "seek care now"."""

QUESTIONING_POLICY = f"""TRIAGE QUESTIONING
- Actively ask the clinically important triage questions when symptoms are ambiguous \
or potentially serious — do not default to "no questions". Good triage questions probe \
onset/duration, progression, severity, triggers/relievers, associated red-flag symptoms, \
and relevant history (e.g. skin cancer history, allergies, topical treatments used).
- Ask only what changes management. Ask at most {MAX_FOLLOWUP_QUESTIONS} questions, \
fewest possible, the most decision-relevant first. If you already have enough to answer \
safely, don't ask."""

SAFEGUARDS = """GENERATION SAFEGUARDS
- No FALSE REASSURANCE: never imply something is harmless when red flags or high-signal \
features are present.
- No PANIC: stay calm and measured; avoid alarming language for low-risk situations.
- No DIAGNOSTIC DUMPING: don't overwhelm with exhaustive lists, jargon, or encyclopedic \
detail. Keep it concise and readable.
- ALWAYS end with clear, concrete next steps (self-care, what to monitor, and exactly \
when/where to seek care)."""
