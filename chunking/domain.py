"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  THE ONE FILE TO EDIT FOR A NEW USE CASE.                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Everything domain-specific lives here: entity/relation vocabularies, the      ║
║  specialty taxonomy, the extraction prompt, segmentation patterns, and the     ║
║  validation thresholds. The rest of the pipeline (loaders, cleaner, LLM        ║
║  plumbing, storage, the MicroChunk shape) is domain-agnostic. Edit the values  ║
║  below to retarget the chunker — nothing else.                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import re

# ── 1. Entity vocabulary ──────────────────────────────────────────────────────
# Precise types so the graph can reason: a lab value is NOT a disease, a metabolic
# state is NOT a symptom. Anything emitted outside this set is mapped via
# ENTITY_TYPE_SYNONYMS, else coerced to ENTITY_TYPE_FALLBACK.
ENTITY_TYPES = [
    "disease", "syndrome", "symptom", "lesion_morphology", "lesion_distribution",
    "clinical_finding", "lab_finding", "metabolic_state", "physiological_state",
    "biomarker", "risk_factor", "anatomical_entity", "drug", "drug_class",
    "procedure", "test", "intervention", "mechanism", "pathogen", "gene",
    "protein", "clinical_process", "historical_person", "historical_text",
    "medical_system", "location",
]
ENTITY_TYPE_SET = set(ENTITY_TYPES)

# Common synonyms / mislabels the model emits → canonical type.
ENTITY_TYPE_SYNONYMS = {
    "medication": "drug", "medicine": "drug", "pharmaceutical": "drug",
    "surgery": "procedure", "operation": "procedure",
    "lab": "lab_finding", "laboratory_test": "test", "diagnostic_test": "test",
    "lab_value": "lab_finding", "laboratory_finding": "lab_finding",
    "sign": "clinical_finding", "finding": "clinical_finding",
    "physical_sign": "clinical_finding",
    "metabolic_disturbance": "metabolic_state", "acid_base_disorder": "metabolic_state",
    "electrolyte_abnormality": "lab_finding",
    "condition": "disease", "disorder": "disease", "illness": "disease",
    "bacteria": "pathogen", "virus": "pathogen", "organism": "pathogen",
    "anatomical_structure": "anatomical_entity", "organ": "anatomical_entity",
    "drug_category": "drug_class", "medication_class": "drug_class",
    # ── Dermatology-specific synonyms ──
    "lesion": "clinical_finding", "rash": "clinical_finding", "eruption": "clinical_finding",
    "macule": "lesion_morphology", "papule": "lesion_morphology", "plaque": "lesion_morphology",
    "vesicle": "lesion_morphology", "nodule": "lesion_morphology", "pustule": "lesion_morphology",
    "scale": "lesion_morphology", "crust": "lesion_morphology", "patch": "lesion_morphology",
    "wheal": "lesion_morphology", "bullae": "lesion_morphology", "erosion": "lesion_morphology",
    "ulcer": "lesion_morphology", "fissure": "lesion_morphology", "telangiectasia": "lesion_morphology",
    "lichenification": "lesion_morphology", "atrophy": "lesion_morphology", "erythema": "clinical_finding",
    "itch": "symptom", "itching": "symptom", "pruritus": "symptom", "burning": "symptom",
    "stinging": "symptom", "pain": "symptom", "tenderness": "symptom",
    "corticosteroid": "drug_class", "topical_steroid": "drug_class", "hydrocortisone": "drug",
    "clobetasol": "drug", "triamcinolone": "drug", "retinoid": "drug_class", "tretinoin": "drug",
    "adapalene": "drug", "calcineurin_inhibitor": "drug_class", "tacrolimus": "drug",
    "pimecrolimus": "drug", "antifungal": "drug_class", "ketoconazole": "drug",
    "clotrimazole": "drug", "terbinafine": "drug", "mupirocin": "drug", "dupilumab": "drug",
    "methotrexate": "drug", "cyclosporine": "drug", "biologic": "drug_class",
    "biopsy": "procedure", "skin_biopsy": "procedure", "shave_biopsy": "procedure",
    "punch_biopsy": "procedure", "excision": "procedure", "cryotherapy": "procedure",
    "curettage": "procedure", "phototherapy": "intervention",
    "dermoscopy": "test", "wood_lamp": "test", "koh_prep": "test", "patch_test": "test",
    "dermatophyte": "pathogen", "fungus": "pathogen", "yeast": "pathogen", "scabies": "pathogen",
    "candida": "pathogen", "malassezia": "pathogen", "herpes": "pathogen", "hsv": "pathogen",
    "hpv": "pathogen", "varicella": "pathogen", "shingles": "pathogen",
    "skin": "anatomical_entity", "epidermis": "anatomical_entity", "dermis": "anatomical_entity",
    "scalp": "anatomical_entity", "nail": "anatomical_entity", "mucosa": "anatomical_entity",
    "hair": "anatomical_entity", "follicle": "anatomical_entity",
    "annular": "lesion_distribution", "linear": "lesion_distribution",
    "dermatomal": "lesion_distribution", "flexural": "lesion_distribution",
    "symmetric": "lesion_distribution", "acral": "lesion_distribution",
    "intertriginous": "lesion_distribution",
    # ── Traditional/Historical synonyms ──
    "kustha": "disease", "kushtha": "disease",
    "vata": "metabolic_state", "pitta": "metabolic_state", "kapha": "metabolic_state",
    "dosha": "metabolic_state", "rasayana": "intervention",
    "sushruta": "historical_person", "charaka": "historical_person", "vagbhata": "historical_person",
    "robert_willan": "historical_person", "willan": "historical_person",
    "ayurveda": "medical_system", "siddha": "medical_system",
    "unani": "medical_system", "tibb": "medical_system",
    "india": "location", "baluchistan": "location", "mehrgarh": "location", "balathal": "location", "rajasthan": "location",
    "charaka_samhita": "historical_text", "sushruta_samhita": "historical_text",
    "vagbhata_samhita": "historical_text", "atreya_samhita": "historical_text",
    "agnivesha_samhita": "historical_text", "vinaya_pitaka": "historical_text",
    # ── Subspecialties and medical systems synonyms ──
    "pediatric_dermatology": "clinical_process", "dermatopathology": "clinical_process",
    "trichology": "clinical_process", "venereology": "clinical_process",
    "leprology": "clinical_process", "dermatosurgery": "clinical_process",
    "cosmetic_dermatology": "clinical_process",
    "western_medicine": "clinical_process", "western_modern_medicine": "clinical_process",
    "modern_medicine": "clinical_process",
}
ENTITY_TYPE_FALLBACK = "clinical_finding"

# ── 2. Relation vocabulary + qualifiers ───────────────────────────────────────
RELATION_TYPES = [
    "causes", "leads_to", "contributes_to", "manifests_as", "mimics", "complicates",
    "increases_risk_of", "reduces_risk_of", "predisposes_to", "protects_against",
    "treats", "alleviates", "mitigates", "reduces", "improves", "worsens", "used_for",
    "indicated_for", "prevents", "contraindicated_with", "metabolized_by", "mediated_by",
    "diagnosed_by", "detected_by", "screens_for", "assesses", "evaluates", "monitors",
    "measures", "classifies", "stages", "requires", "includes", "affects",
    "correlates_with", "alternative_to", "increases_likelihood_of",
    "reduces_likelihood_of", "associated_with",
]
RELATION_TYPE_SET = set(RELATION_TYPES)
RELATION_TYPE_FALLBACK = "associated_with"

# Allowed relation qualifier keys (graph edge properties). The clinical AXIS — esp.
# onset/speed — must survive as an edge property, not be flattened away.
RELATION_QUALIFIER_KEYS = ["onset", "temporality", "severity", "certainty", "context"]
ONSET_VALUES = ["instantaneous", "acute", "subacute", "chronic"]

# ── 3. Specialty taxonomy ─────────────────────────────────────────────────────
# Only dermatology is included.
SPECIALTIES = [
    "dermatology",
]
SPECIALTY_SET = set(SPECIALTIES)

# Variants the model emits → canonical specialty.
SPECIALTY_SYNONYMS = {
    "derm": "dermatology", "skin_medicine": "dermatology",
    "cutaneous": "dermatology",
}

# ── 4. Segmentation patterns ──────────────────────────────────────────────────
SECTION_HEADER_PATTERN = re.compile(
    r'^(Symptoms|Diagnosis|Treatment|Introduction|Pathophysiology|Morphology|Differential Diagnosis|Management|Prevention)\s*$',
    re.IGNORECASE,
)
CONCEPT_HEADER_PATTERN = re.compile(
    r'^(treatment|diagnosis|pathophysiology|etiology|clinical presentation|clinical features|'
    r'management|epidemiology|prognosis|pathogenesis|history|physical examination|'
    r'morphology|histopathology|differential diagnosis|complications|prevention|'
    r'indications|contraindications|ayurveda|siddha|traditional medicine|historical outline)\b',
    re.IGNORECASE,
)

# ── 5. Chunk validation thresholds ────────────────────────────────────────────
MIN_ENTITIES = 3
MAX_CHUNK_TOKENS = 500

# ── 6. Extraction prompt ──────────────────────────────────────────────────────
EXTRACTOR_ROLE = "clinical knowledge extraction engine"
SOURCE_DESCRIPTION = "medical reference text"
KNOWLEDGE_NOUN = "clinical knowledge"
CONCEPT_EXAMPLES = "a skin disease, a topical or systemic drug/drug class, a diagnostic approach (e.g. dermoscopy, biopsy), a pathophysiology mechanism, a lesion morphology, or a management strategy"
TOPIC_EXAMPLE = "Diagnosis of plaque psoriasis"
PROSE_NOUN = "clinical prose"
EXPERT_NOUN = "a clinician"
SKIP_EXAMPLES = '"skin", "patient", "water", "body site", "history", "age", "period", "century", "era", "dermatology", "dermatologist", "skin disease", "skin diseases", "medicine", "beauty", "identity", "culture", "science", "social custom", "environment"'
TARGET_ENTITIES_HINT = "~4–8"

_ENTITY_TYPES_STR = ", ".join(ENTITY_TYPES)
_RELATION_TYPES_STR = ", ".join(RELATION_TYPES)
_SPECIALTIES_STR = ", ".join(SPECIALTIES)

RELATION_GUIDANCE = """    • disease  manifests_as  symptom/lesion (e.g., eczema manifests_as vesicle)
    • risk_factor  increases_risk_of  disease (e.g., UV increases_risk_of melanoma)
    • drug  treats  disease/symptom (e.g., steroid treats eczema)
    • disease  diagnosed_by  test/procedure (e.g., melanoma diagnosed_by skin_biopsy)"""

SYSTEM_PROMPT = f"""You are a {EXTRACTOR_ROLE} that converts {SOURCE_DESCRIPTION} into strict JSON chunks of {KNOWLEDGE_NOUN}.

OUTPUT FORMAT:
- Valid JSON only. No markdown, no comments. Follow the provided schema exactly.

WHAT A GOOD CHUNK IS:
- Captures ONE concept ({CONCEPT_EXAMPLES}). Split if multiple.
- ~150–350 tokens of clean text. No OCR/bullet noise.
- source.topic: specific concept title (e.g., "{TOPIC_EXAMPLE}").

SPECIALTIES:
- Tag `specialties` using only: {_SPECIALTIES_STR}.

ENTITIES (Extract {TARGET_ENTITIES_HINT} most salient; skip generic words like {SKIP_EXAMPLES}):
- CRITICAL: DO NOT extract generic disease categories (e.g., "skin disease", "skin diseases", "dermatological disease", "dermatological disorders") or generic root concepts (e.g., "dermatology", "dermatologist", "medicine") unless they are the primary topic.
- CRITICAL: Every single entity in this list MUST participate in at least one relation (as source or target). If an entity cannot be connected to any other, do not extract it.
- REDUCE OVER-EXTRACTION: For historical content, prefer extracting only 4–8 highly informative entities (avoid extracting every noun).
- HISTORICAL GUIDANCE: When historical medical literature is discussed, prefer extracting medical systems, historical texts, historical persons, and disease classifications. Avoid extracting cultural concepts, philosophical ideas, or generic societal descriptions.
- `name`: canonical lowercase singular name (expand abbreviations: "AD" -> "atopic dermatitis", "BCC" -> "basal cell carcinoma"). Use the same canonical name every time.
- `aliases`: list of synonyms.
- `type`: choose from: {_ENTITY_TYPES_STR}. Precise typing:
    • named condition -> disease (e.g. psoriasis)
    • complaint -> symptom (e.g. pruritus)
    • lesion feature -> lesion_morphology (e.g. plaque, vesicle)
    • spatial pattern -> lesion_distribution (e.g. linear)
    • exam sign -> clinical_finding
    • lab/biopsy result -> lab_finding
    • traditional medicine systems -> medical_system (e.g. ayurveda, siddha, unani)
    • historical books/compendia/scriptures -> historical_text (e.g. charaka samhita, sushruta samhita)
    • historical figures/persons -> historical_person (e.g. robert willan, charaka, sushruta)
    • geographical places/countries -> location (e.g. india, baluchistan)

RELATIONS (Source and target must match entity names in this chunk - EVERY single entity must be connected):
- `type`: choose from: {_RELATION_TYPES_STR}. Use the specific directional relation:
{RELATION_GUIDANCE}
- ALL ENTITIES MUST BE CONNECTED (CRITICAL): Every single entity in the 'entities' list MUST participate in at least one relation (as source or target). If there is no causal/clinical relation, connect it using the fallback "associated_with" (e.g., connect an 'historical_text' like 'charaka samhita' to its 'medical_system' like 'ayurveda'). Zero unconnected entities allowed.
- `qualifiers`: onset (one of: {", ".join(ONSET_VALUES)}). E.g. {{"onset": "acute"}}.

SUMMARY FIELDS:
- `summary`: 1–2 key sentences. `clinical_significance`: 1 sentence.

SELF-CHECK:
- Entities canonical and precisely typed?
- CRITICAL: Is every single entity connected to at least one relation in the list? If any entity is unconnected, either add an "associated_with" relation linking it, or remove the entity from the 'entities' list!
- Relations id-resolvable, directional, and onset/qualifiers preserved?
- specialties cover every relevant field? JSON strictly valid?
"""
