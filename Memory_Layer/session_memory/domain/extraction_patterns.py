"""
session_memory.domain.extraction_patterns
───────────────────────────────────────────
Regex/keyword patterns the heuristic state extractor uses to pull medical
context out of a user message.

⭐ EDIT FOR A NEW SPECIALTY/USE CASE. The extractor logic
(`state_extractor.py`) is domain-agnostic — it iterates EXTRACTION_PATTERNS and
applies the demographic/name patterns below. To retarget, change the patterns
here; leave the code alone.

Each pattern dict maps a canonical term → list of regex alternatives. The first
alternative that matches (case-insensitive) records the canonical term.
"""

from __future__ import annotations

import re

# ── Clinical entity patterns ──────────────────────────────────────────────────

SYMPTOM_PATTERNS: dict[str, list[str]] = {
    "fever":              [r"\bfever\b", r"\bhigh temperature\b", r"\btemperature\b"],
    "chills":             [r"\bchill(s|ing)?\b", r"\bshiver(ing)?\b"],
    "headache":           [r"\bheadache\b", r"\bhead pain\b"],
    "fatigue":            [r"\bfatigue\b", r"\btired\b", r"\bexhausted\b"],
    "nausea":             [r"\bnausea\b", r"\bfeeling sick\b"],
    "dizziness":          [r"\bdizzy\b", r"\bdizziness\b"],
    # ── Dermatology-specific ──
    "itching":            [r"\bitch(y|ing|es)?\b", r"\bpruritus\b", r"\bpruritic\b"],
    "rash":               [r"\brash(es)?\b", r"\berupt(ion|ions)\b", r"\bhives\b", r"\burticaria\b"],
    "dry_skin":           [r"\bdry(ness)? skin\b", r"\bxerosis\b", r"\bdry patch(es)?\b"],
    "peeling_skin":       [r"\bpeel(ing|s)? skin\b", r"\bskin peeling\b", r"\bshedding skin\b", r"\bslough(ing)?\b"],
    "blistering":         [r"\bblister(s|ing|ed)?\b", r"\bbulla\b", r"\bbullae\b", r"\bvesicle(s)?\b"],
    "redness":            [r"\bredness\b", r"\bred skin\b", r"\berythema\b", r"\beruptive redness\b", r"\breddened\b"],
    "swelling":           [r"\bswell(ing|s|ed)?\b", r"\bedema\b", r"\bpuffiness\b", r"\bpuffy\b"],
    "skin_pain":          [r"\bskin pain\b", r"\bpainful skin\b", r"\bhurt(s)? to touch\b", r"\btender(ness)?\b"],
    "burning_sensation":  [r"\bburn(ing|s)?\b", r"\bburning sensation\b", r"\bsting(ing|s)?\b"],
    "scaling_flaking":    [r"\bscal(ing|es|y)?\b", r"\bflak(ing|es|y)?\b", r"\bpeeling scales\b"],
    "bleeding_lesion":    [r"\bbleed(ing|s)?\b", r"\bbled\b", r"\booz(ing|e)?\b", r"\bweep(ing|s)?\b"],
    "hair_loss":          [r"\bhair loss\b", r"\bhair shedding\b", r"\balopecia\b", r"\bhair thinning\b", r"\blosing hair\b"],
    "nail_changes":       [r"\bnail(s)? (disorder|change|dystrophy|split|brittle|color)\b", r"\bonycholysis\b", r"\bnail loss\b"],
    "pus_discharge":      [r"\bpus\b", r"\bdischarg(e|ing)\b", r"\boozing pus\b", r"\bpustule(s)?\b", r"\bpurulent\b"],
    "skin_warmth":        [r"\b(skin )?warmth\b", r"\b(skin )?feels hot\b", r"\bhot to (the )?touch\b"],
}

# Trigger / pattern recognition — when does the symptom occur or worsen? Captured
# into StructuredState.triggers for cross-turn triage continuity.
TRIGGER_PATTERNS: dict[str, list[str]] = {
    "sunlight":             [r"\bsun(light|ny|exposure)?\b", r"\bUV (rays|exposure)?\b", r"\bout in the sun\b"],
    "heat_sweating":        [r"\bheat\b", r"\bsweat(ing|y)?\b", r"\bhot weather\b", r"\bhot shower(s)?\b"],
    "cold_dry":             [r"\bcold (weather|air)\b", r"\bwinter\b", r"\bdry air\b"],
    "stress":               [r"\bstress(ed|ful)?\b", r"\banxiety\b"],
    "irritants_substances": [r"\bsoap(s)?\b", r"\bcosmetic(s)?\b", r"\bdetergent(s)?\b", r"\bperfume(s)?\b", r"\bfragrance(s)?\b", r"\bmetal(s)?\b", r"\bnickel\b", r"\bchemical(s)?\b"],
    "scratching_friction":  [r"\bscratch(ing)?\b", r"\brub(bing)?\b", r"\bfriction\b", r"\btight clothing\b"],
    "night":                [r"\bat night\b", r"\bnight ?time\b", r"\bbedtime\b"],
    "after_eating":         [r"\bafter (eating|meals?|food)\b"],
}

# Distinguishing between acute conditions and chronic conditions
CHRONIC_PATTERNS: dict[str, list[str]] = {
    "eczema":             [r"\beczema\b", r"\batopic dermatitis\b"],
    "psoriasis":          [r"\bpsoriasis\b", r"\bpsoriatic\b"],
    "rosacea":            [r"\brosacea\b"],
    "acne":               [r"\bacne\b", r"\bacne vulgaris\b", r"\bpimples\b"],
    "vitiligo":           [r"\bvitiligo\b"],
    "alopecia":           [r"\balopecia\b", r"\bhair loss condition\b"],
    "lupus_autoimmune":   [r"\blupus\b", r"\bsle\b", r"\bautoimmune\b"],
    "diabetes":           [r"\bdiabetes\b", r"\bdiabetic\b"],
    "thyroid":            [r"\bthyroid\b"],
}

CONDITION_PATTERNS: dict[str, list[str]] = {
    "cellulitis":         [r"\bcellulitis\b"],
    "impetigo":           [r"\bimpetigo\b"],
    "shingles":           [r"\bshingles\b", r"\bherpes zoster\b"],
    "fungal_infection":   [r"\bfungal infection\b", r"\bringworm\b", r"\btinea\b", r"\bathlete'?s foot\b", r"\bjock itch\b"],
    "contact_dermatitis": [r"\bcontact dermatitis\b", r"\bskin allergy reaction\b"],
    "urticaria_hives":    [r"\bhives\b", r"\burticaria\b", r"\bwelts\b"],
    "scabies":            [r"\bscabies\b", r"\bmite infection\b"],
    "warts":              [r"\bwart(s)?\b", r"\bverruca\b"],
    "skin_cancer":        [r"\bskin cancer\b", r"\bmelanoma\b", r"\bcarcinoma\b", r"\bbcc\b", r"\bscc\b"],
    "infection":          [r"\binfection\b", r"\binfected\b"],
}

ALLERGY_PATTERNS: dict[str, list[str]] = {
    "latex":              [r"\blatex allergy\b", r"\ballergic to latex\b"],
    "nickel_metal":       [r"\bnickel allergy\b", r"\ballergic to nickel\b", r"\bmetal allergy\b"],
    "fragrance_cosmetics": [r"\bfragrance allergy\b", r"\ballergic to fragrance\b", r"\bperfume allergy\b", r"\bcosmetic allergy\b"],
    "poison_ivy":         [r"\bpoison ivy\b", r"\bpoison oak\b", r"\bpoison sumac\b"],
    "penicillin":         [r"\ballergic to penicillin\b", r"\bpenicillin allergy\b"],
}

DRUG_PATTERNS: dict[str, list[str]] = {
    "hydrocortisone":     [r"\bhydrocortisone\b", r"\btopical steroid\b", r"\bcortisone cream\b"],
    "clobetasol":         [r"\bclobetasol\b"],
    "salicylic_acid":     [r"\bsalicylic acid\b"],
    "benzoyl_peroxide":   [r"\bbenzoyl peroxide\b"],
    "tretinoin_retinoids": [r"\btretinoin\b", r"\bretin-a\b", r"\bretinoid\b"],
    "isotretinoin":       [r"\bisotretinoin\b", r"\baccutane\b"],
    "ketoconazole_antifungals": [r"\bketoconazole\b", r"\bclotrimazole\b", r"\bantifungal cream\b"],
    "clindamycin":        [r"\bclindamycin\b"],
    "doxycycline_minocycline": [r"\bdoxycycline\b", r"\bminocycline\b"],
    "antihistamines":     [r"\bdiphenhydramine\b", r"\bbenadryl\b", r"\bcetirizine\b", r"\bzyrtec\b", r"\bantihistamine(s)?\b"],
    "tacrolimus":         [r"\btacrolimus\b", r"\bprotopic\b", r"\belidel\b"],
}

SEVERITY_PATTERNS: dict[str, list[str]] = {
    "mild":     [r"\bmild\b", r"\bslight\b"],
    "moderate": [r"\bmoderate\b", r"\bmedium\b"],
    "severe":   [r"\bsevere\b", r"\bextreme\b", r"\bintense\b", r"\bvery bad\b"],
}

# ── Demographics ──────────────────────────────────────────────────────────────

DURATION_RE = re.compile(
    r"(?:for|since|over|past|last)\s+"
    r"(\d+\s+(?:second|minute|hour|day|week|month|year)s?|yesterday|this morning)",
    re.IGNORECASE,
)

AGE_RE  = re.compile(r"\b(\d{1,3})\s*(?:year(?:s)?\s*old|y\.?o\.?)\b", re.IGNORECASE)
SEX_RE  = re.compile(r"\b(male|female|man|woman)\b", re.IGNORECASE)

SEX_NORMALISE = {"man": "male", "woman": "female"}

# ── Name extraction ───────────────────────────────────────────────────────────
# We pull a first name when the patient explicitly introduces themselves. The
# answer prompt uses it to address the user naturally instead of as "patient".
#
# High-confidence patterns first — these are explicit declarations and almost
# never produce false positives.
NAME_EXPLICIT_RES: list[re.Pattern[str]] = [
    re.compile(r"\bmy name is ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bthe name'?s ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bname'?s ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bcall me ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bthis is ([A-Z][a-zA-Z'\-]{1,30})(?:\s+speaking|\s+here|\s*[,.])"),
    re.compile(r"\bi go by ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
]

# Lower-confidence: "I am X" / "I'm X". Only trust if X looks like a name —
# capitalised in the original text AND not a common adjective / state word.
NAME_SOFT_RE = re.compile(r"\b[Ii]\s*'?\s*[am]{1,2}\s+([A-Z][a-zA-Z'\-]{1,30})\b")

# Common words that can follow "I'm" / "I am" but are NOT names. Lowercased.
NAME_STOPWORDS: frozenset[str] = frozenset({
    # states / feelings
    "sick", "tired", "fine", "ok", "okay", "good", "bad", "well", "great",
    "happy", "sad", "worried", "scared", "confused", "anxious", "depressed",
    "stressed", "exhausted", "hungry", "thirsty", "dizzy", "nauseous", "dying",
    "fasting", "bleeding", "burning", "shaking", "freezing",
    # statuses / dermatology symptoms acting as states
    "married", "single", "pregnant", "diabetic", "allergic", "eczematous",
    "psoriatic", "vegetarian", "vegan", "lost", "ready", "back", "done",
    "late", "early", "here", "there", "home", "outside", "indoors",
    "itchy", "flaky", "peeling", "scaly", "scabby", "red", "swollen", "rashy",
    # progressive verbs after "I'm"
    "having", "feeling", "going", "trying", "looking", "doing", "taking",
    "thinking", "wondering", "asking", "calling", "writing", "experiencing",
    "suffering", "noticing", "starting", "ending", "drinking", "eating",
    # other common
    "afraid", "unsure", "unable", "old", "young", "new", "sorry", "sure",
    "really", "always", "never", "still", "just", "also", "very",
})
