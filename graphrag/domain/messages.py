"""
graphrag.domain.messages
──────────────────────────
User-facing canned responses emitted directly by the pipeline (no LLM call).

⭐ EDIT FOR A NEW SPECIALTY/USE CASE. These are returned when the gatekeeper
refuses a non-medical query or redirects a detected emergency.
"""

# Returned when the gatekeeper classifies the query as out-of-domain.
REFUSAL_MESSAGE: str = (
    "❌ I can only answer healthcare-related questions. "
    "Please ask a medical question."
)

# Returned when a query is medical but falls below the dermatology relevance
# threshold (see vocabulary.PULMONOLOGY_RELEVANCE_THRESHOLD).
OUT_OF_SCOPE_MESSAGE: str = (
    "🩹 I'm focused on dermatology and skin, hair, and nail disorders, so I can't help "
    "with that one. Please ask about a skin, hair, or nail-related concern "
    "(e.g. rash, acne, eczema, psoriasis, atypical moles, or skin infections)."
)

# Returned when the gatekeeper detects an emergency red-flag.
EMERGENCY_MESSAGE: str = (
    "🚨 EMERGENCY: Your symptoms sound like a serious emergency. "
    "Please call emergency services (112 / 911) immediately or go to the "
    "nearest hospital."
)


def refusal_blocks() -> list[dict]:
    return [
        {"type": "summary", "data": {"text": REFUSAL_MESSAGE}}
    ]


def out_of_scope_blocks() -> list[dict]:
    return [
        {"type": "summary", "data": {"text": OUT_OF_SCOPE_MESSAGE}}
    ]


def emergency_blocks() -> list[dict]:
    return [
        {"type": "warning", "data": {"text": "seek emergency care now", "severity": "critical"}},
        {"type": "next_steps", "data": {"steps": [
            "Call emergency services (112 / 911) immediately.",
            "Go to the nearest hospital emergency department."
        ]}}
    ]
