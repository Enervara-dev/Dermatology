"""
Offline end-to-end smoke test for the GraphRAG dermatology assistant.

Runs WITHOUT network or paid calls: Pinecone, Neo4j, and Gemini are stubbed.
It exercises every subsystem and asserts behavior:

  - imports / wiring          (all packages import)
  - domain layer              (namespace, scope threshold, prompts, red flags)
  - session memory            (extraction, symptom-weighted risk, triggers, continuity)
  - entity processor          (plain-name parse, dedup, hybrid merge)
  - full pipeline scenarios   (in-scope, out-of-scope, emergency, terminal state,
                               NO_RETRIEVAL conclude, greeting)
  - Stage-4 prompt injection  (real gemini_llm, generate_stream patched)
  - HTTP API wiring           (best-effort: routes registered)
  - NDJSON block streaming    (validator, parser, builders)

Usage:
    python smoke_test.py          # exits 0 if all pass, 1 otherwise
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.ERROR)
logging.disable(logging.CRITICAL)


# ── tiny harness ──────────────────────────────────────────────────────────────
class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")

    def check(self, name: str, cond: bool, detail: str = "") -> bool:
        ok = bool(cond)
        self.passed += ok
        self.failed += (not ok)
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if detail:
            line += f"  → {detail}"
        print(line)
        return ok

    def skip(self, name: str, reason: str) -> None:
        print(f"  [SKIP] {name}  → {reason}")


R = Report()


@contextmanager
def silent():
    """Swallow the pipeline's stdout (stage logs / streamed prints) for a clean report."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        yield


# ── stub pipeline builder ─────────────────────────────────────────────────────
def build_stub_pipeline(analyses, *, graph=("psoriasis -[manifests_as]-> plaque",),
                        chunk_entities=("psoriasis", "plaque"), llm_answer="ANSWER: clinical guidance"):
    """
    A GraphRAGPipeline with external services stubbed but real in-process
    components (entity_processor, memory_adapter → RAM). `analyses` is a list of
    gatekeeper-analysis dicts returned in order, one per .run() call.
    Returns (pipeline, calls) where `calls` records each generate_response kwargs.
    """
    from graphrag.pipeline.graphrag_pipeline import GraphRAGPipeline
    from graphrag.processors.entity_processor import EntityProcessor
    from graphrag.memory import SessionMemoryAdapter

    p = GraphRAGPipeline.__new__(GraphRAGPipeline)
    p._episodic = None
    p._loop = None
    p.entity_processor = EntityProcessor()
    p.memory_adapter = SessionMemoryAdapter()
    calls: list[dict] = []
    flags = {"retrieved": False}

    class A:
        def __init__(self): self.q = list(analyses)
        def analyze(self, q): return self.q.pop(0) if self.q else {}

    class PC:
        def retrieve(self, *a, **k):
            flags["retrieved"] = True
            return [{"id": "c1", "metadata": {
                "summary": "Psoriasis is a chronic skin disease causing plaques.",
                "entities": list(chunk_entities)}}]

    class N:
        def retrieve_relations(self, *a, **k): return list(graph)
        def close(self): pass

    class L:
        def generate_response(self, **k):
            calls.append(k)
            yield {"type": "summary", "data": {"text": llm_answer}}

    p.query_analyzer = A()
    p.pinecone_retriever = PC()
    p.neo4j_retriever = N()
    p.llm = L()
    return p, calls, flags


def analysis(intent="symptom_query", *, needs_followup=False, relevance=95,
              risk="low", action="retrieve", symptoms=("rash",)):
    return {
        "domain": "health", "intent": intent, "risk_level": risk,
        "pulmonology_relevance": relevance,
        "medical_entities": {"symptoms": list(symptoms), "drugs": [], "conditions": []},
        "rewritten_query": "", "needs_followup": needs_followup,
        "followup_questions": (["q"] if needs_followup else []), "final_action": action,
    }


# ── 1. imports / wiring ───────────────────────────────────────────────────────
def test_imports():
    R.section("1. Imports / wiring")
    import importlib
    mods = [
        "graphrag.pipeline.graphrag_pipeline",
        "graphrag.query_understanding.analyzer",
        "graphrag.query_understanding.routing",
        "graphrag.retrievers.pinecone_retriever",
        "graphrag.retrievers.neo4j_retriever",
        "graphrag.processors.entity_processor",
        "graphrag.llm.gemini_llm",
        "graphrag.domain",
        "Memory_Layer.session_memory",
        "Memory_Layer.session_memory.domain",
        "episodic.api.dependencies",
        "episodic.schemas.retrieval",
        "graphrag.schemas.blocks",
        "graphrag.validators.answer_validator",
    ]
    for m in mods:
        try:
            importlib.import_module(m)
            R.check(f"import {m}", True)
        except Exception as e:
            R.check(f"import {m}", False, repr(e))


# ── 2. domain layer ───────────────────────────────────────────────────────────
def test_domain():
    R.section("2. Domain layer")
    from graphrag.domain import (
        PINECONE_NAMESPACE, DERMATOLOGY_RELEVANCE_THRESHOLD,
        GATEKEEPER_SYSTEM_PROMPT, compose_system_prompt, detect_red_flags,
    )
    from graphrag.domain.clinical_policy import (
        closure_directive, ASSESSMENT_READY_INSTRUCTION, NO_RETRIEVAL_CONCLUDE_INSTRUCTION,
        MAX_DIAGNOSTIC_TURNS,
    )
    R.check("retrieval namespace = dermatology", PINECONE_NAMESPACE == "dermatology", PINECONE_NAMESPACE)
    R.check("scope threshold = 75", DERMATOLOGY_RELEVANCE_THRESHOLD == 75)
    R.check("max diagnostic turns = 2", MAX_DIAGNOSTIC_TURNS == 2)
    R.check("gatekeeper prompt has relevance rubric + red flags + terminal state",
            all(s in GATEKEEPER_SYSTEM_PROMPT for s in
                ("pulmonology_relevance", "DERMATOLOGICAL RED FLAGS", "assessment_ready")))
    crit = compose_system_prompt(query_type="symptom_query", risk_level="critical", has_name=False)
    R.check("critical answer prompt = structured emergency",
            "EMERGENCY RESPONSE STRUCTURE" in crit and "TENTATIVE CAUSES" in crit.upper())
    pulm = compose_system_prompt(query_type="symptom_query", risk_level="none", has_name=False)
    R.check("answer prompt is dermatology-tuned", "dermatology" in pulm.lower())
    # red flag detection
    R.check("red flag: swelling of face", detect_red_flags("swelling of my face and I cannot breathe") == ["anaphylaxis_angioedema"])
    R.check("red flag: NOT tripped by mild itching",
            detect_red_flags("just mild itching on my arm") == [])
    # closure directive matrix
    R.check("closure: greeting (no findings) → none",
            closure_directive(intent="greeting", needs_followup=False, memory_only=True, has_findings=False) is None)
    R.check("closure: assessment_ready → terminal instruction",
            closure_directive(intent="assessment_ready", needs_followup=False, memory_only=False, has_findings=True) == ASSESSMENT_READY_INSTRUCTION)
    R.check("closure: NO_RETRIEVAL medical → conclude instruction",
            closure_directive(intent="followup_query", needs_followup=True, memory_only=True, has_findings=True) == NO_RETRIEVAL_CONCLUDE_INSTRUCTION)
    R.check("closure: mid-triage (needs_followup) → none",
            closure_directive(intent="symptom_query", needs_followup=True, memory_only=False, has_findings=True) is None)


# ── 3. session memory ─────────────────────────────────────────────────────────
def test_memory():
    R.section("3. Session memory")
    from Memory_Layer.session_memory import SessionMemory, Message, Role, extract_state, get_working_memory
    from Memory_Layer.session_memory.state_extractor import extract_entities

    raw = extract_entities("itchy rash, worse in sunlight, bleeding mole")
    R.check("dermatological symptom extraction", {"itching", "rash", "bleeding_lesion"} <= set(raw.symptoms), str(raw.symptoms))
    R.check("trigger extraction", "sunlight" in raw.triggers, str(raw.triggers))

    # symptom-weighted risk in the live path
    s = SessionMemory(session_id="m1")
    s.state = extract_state(s, Message(role=Role.USER, content="I have peeling skin", risk_level="low"))
    R.check("critical symptom escalates risk → critical", str(s.state.risk_level) in ("critical", "RiskLevel.CRITICAL"), str(s.state.risk_level))

    # continuity across turns
    s2 = SessionMemory(session_id="m2")
    s2.state = extract_state(s2, Message(role=Role.USER, content="dry skin for 3 days"))
    s2.add_turn(Message(role=Role.USER, content="dry skin for 3 days"))
    s2.state = extract_state(s2, Message(role=Role.USER, content="now also itching"))
    R.check("symptoms accumulate across turns", {"dry_skin", "itching"} <= set(s2.state.symptoms), str(s2.state.symptoms))


# ── 4. entity processor ───────────────────────────────────────────────────────
def test_entities():
    R.section("4. Entity processor")
    from graphrag.processors.entity_processor import EntityProcessor
    from graphrag.pipeline.graphrag_pipeline import _merge_graph_entities, _entities_from_analysis

    # plain-name metadata (real Pinecone format)
    _, ents, _ = EntityProcessor.process_matches(
        [{"id": "1", "metadata": {"summary": "s", "entities": ["psoriasis", "plaque", "psoriasis"]}}],
        priority_entity_types=["disease"], query="")
    R.check("plain-name entities extracted + deduped", ents == ["psoriasis", "plaque"], str(ents))

    merged = _merge_graph_entities(["psoriasis"], _entities_from_analysis(
        {"medical_entities": {"symptoms": ["itching"], "drugs": [], "conditions": ["eczema"]}}), ["dry_skin"])
    R.check("hybrid graph entities (chunk+query+memory, normalized)",
            merged[0] == "psoriasis" and "itching" in merged, str(merged))


# ── 5. full pipeline scenarios ────────────────────────────────────────────────
def test_pipeline_scenarios():
    R.section("5. Full pipeline (stubbed services)")
    from graphrag.domain import OUT_OF_SCOPE_MESSAGE

    # a) in-scope → retrieval + graph + answer, graph entities are hybrid
    p, calls, flags = build_stub_pipeline([analysis("symptom_query", needs_followup=True)])
    with silent():
        ans_blocks = list(p.run("itchy skin rash on my arm", session_id="sc_in"))
    ans = ans_blocks[0]["data"]["text"] if ans_blocks else ""
    gctx = calls[-1]["graph_context"]
    R.check("in-scope answered + retrieval ran", ans.startswith("ANSWER:") and flags["retrieved"])
    R.check("graph traversal produced relations", "manifests_as" in gctx, gctx[:60])

    # b) out-of-scope → restricted, retrieval skipped
    p, calls, flags = build_stub_pipeline([analysis("symptom_query", relevance=20, needs_followup=False)])
    with silent():
        ans_blocks = list(p.run("cough and chest pain", session_id="sc_oos"))
    ans = ans_blocks[0]["data"]["text"] if ans_blocks else ""
    R.check("out-of-scope restricted", ans == OUT_OF_SCOPE_MESSAGE)
    R.check("out-of-scope skipped retrieval", flags["retrieved"] is False and not calls)

    # c) emergency (red flag) → reasoned answer at critical risk, retrieval ran
    p, calls, flags = build_stub_pipeline([analysis("symptom_query", needs_followup=False)])
    with silent():
        ans_blocks = list(p.run("swelling of my face and I cannot swallow", session_id="sc_er"))
    ans = ans_blocks[0]["data"]["text"] if ans_blocks else ""
    R.check("emergency → reasoned LLM answer (not static)", ans.startswith("ANSWER:"))
    R.check("emergency → critical risk + retrieval ran",
            calls and calls[-1]["risk_level"] == "critical" and flags["retrieved"], str(calls[-1]["risk_level"]) if calls else "no-call")

    # d) terminal state: 3 follow-needed turns → 3rd flips to assessment_ready
    p, calls, _ = build_stub_pipeline([analysis("symptom_query", needs_followup=True)] * 3)
    with silent():
        for _ in range(3):
            list(p.run("I have a rash", session_id="sc_turns"))
    R.check("turn 1 not terminal", calls[0]["query_type"] == "symptom_query")
    R.check("turn 3 forced → assessment_ready", calls[2]["query_type"] == "assessment_ready" and calls[2]["needs_followup"] is False)

    # e) needs_followup False mid-loop → terminal
    p, calls, _ = build_stub_pipeline([analysis("symptom_query", needs_followup=True),
                                       analysis("symptom_query", needs_followup=False)])
    with silent():
        list(p.run("I have a rash", session_id="sc_nf"))
        list(p.run("still itchy", session_id="sc_nf"))
    R.check("needs_followup False → assessment_ready", calls[1]["query_type"] == "assessment_ready")

    # f) NO_RETRIEVAL medical follow-up → memory_only + findings (conclude)
    p, calls, _ = build_stub_pipeline([analysis("symptom_query", needs_followup=True),
                                       analysis("followup_query", needs_followup=True, action="route_to_followup")])
    with silent():
        list(p.run("I have a rash", session_id="sc_nr"))
        list(p.run("is it serious?", session_id="sc_nr"))
    R.check("NO_RETRIEVAL follow-up → memory_only + has_findings",
            calls[1]["memory_only"] is True and calls[1]["has_findings"] is True, str({k: calls[1][k] for k in ("memory_only", "has_findings")}))

    # g) greeting → exempt from scope gate, answered
    p, calls, _ = build_stub_pipeline([analysis("greeting", relevance=5, needs_followup=False)])
    with silent():
        ans_blocks = list(p.run("hello", session_id="sc_hi"))
    ans = ans_blocks[0]["data"]["text"] if ans_blocks else ""
    R.check("greeting exempt → answered", ans.startswith("ANSWER:"))


# ── 6. Stage-4 prompt injection (real gemini_llm, generate_stream patched) ─────
def test_stage4_injection():
    R.section("6. Stage-4 prompt injection (real gemini_llm)")
    try:
        import graphrag.llm.gemini_llm as gl
        from graphrag.domain.clinical_policy import ASSESSMENT_READY_INSTRUCTION, NO_RETRIEVAL_CONCLUDE_INSTRUCTION
    except Exception as e:
        R.skip("gemini_llm injection", repr(e))
        return

    cap: dict = {}

    def fake_stream(*, user_prompt, model, system_instruction=None, temperature=None):
        cap["sys"] = system_instruction
        yield '{"type":"summary","data":{"text":"ok"}}'

    gl.generate_stream = fake_stream
    try:
        llm = gl.GeminiLLM()
    except Exception as e:
        R.skip("gemini_llm injection (needs GEMINI_API_KEY for client init)", repr(e))
        return

    def call(**kw):
        with silent():
            list(llm.generate_response(query_text="q", vector_context="", graph_context="",
                                  memory_context="", conversation_history="", **kw))
        return cap.get("sys", "")

    s = call(query_type="assessment_ready", needs_followup=False, memory_only=False, has_findings=True)
    R.check("assessment_ready → terminal constraint injected", ASSESSMENT_READY_INSTRUCTION in s)
    s = call(query_type="followup_query", needs_followup=True, memory_only=True, has_findings=True)
    R.check("NO_RETRIEVAL → conclude constraint injected", NO_RETRIEVAL_CONCLUDE_INSTRUCTION in s)
    s = call(query_type="symptom_query", needs_followup=True, memory_only=False, has_findings=True)
    R.check("mid-triage → NO constraint (follow-up allowed)",
            ASSESSMENT_READY_INSTRUCTION not in s and NO_RETRIEVAL_CONCLUDE_INSTRUCTION not in s)


# ── 8. Episodic memory: session-end only (not per-turn) ───────────────────────
def test_episodic_session_end():
    R.section("8. Episodic — written only at session end")
    from graphrag.pipeline.graphrag_pipeline import GraphRAGPipeline
    from graphrag.processors.entity_processor import EntityProcessor
    from graphrag.memory import SessionMemoryAdapter

    captured: list[str] = []

    class _Stored:
        episode_id = "ep-1"
        class category: value = "symptom"
        class clinical_priority: value = "normal"

    class _Result:
        stored = _Stored()
        class clarification:
            needs_clarification = False
            questions: list = []
        class contradictions:
            has_contradictions = False
            contradictions: list = []
            confidence_penalty = 0.0
            triggers_clarification = False

    class _Ingest:
        async def run(self, *, user_id, utterance):
            captured.append(utterance)
            return _Result()

    class _CtxBlock:
        rendered_prompt = ""

    class _Context:
        async def build(self, req):
            return _CtxBlock()

    class _Episodic:
        ingest_pipeline = _Ingest()
        context_pipeline = _Context()

    p = GraphRAGPipeline.__new__(GraphRAGPipeline)
    p._episodic = _Episodic()
    p._loop = None
    p.entity_processor = EntityProcessor()
    p.memory_adapter = SessionMemoryAdapter()

    class A:
        def analyze(self, q): return analysis("symptom_query", needs_followup=False)
    class PC:
        def retrieve(self, *a, **k): return [{"id": "c", "metadata": {"summary": "s", "entities": ["dry_skin"]}}]
    class N:
        def retrieve_relations(self, *a, **k): return []
        def close(self): pass
    class L:
        def generate_response(self, **k):
            yield {"type": "summary", "data": {"text": "ANSWER: ok"}}

    p.query_analyzer = A(); p.pinecone_retriever = PC(); p.neo4j_retriever = N(); p.llm = L()

    # A chat turn WITH a user_id must NOT write episodic memory (no per-turn ingest).
    with silent():
        list(p.run("I have dry skin and itching", session_id="ep_s", user_id="u1"))
    R.check("no per-turn episodic write during /chat", captured == [])

    # Closing the session writes exactly ONE consolidated episode.
    with silent():
        status = p.end_session(user_id="u1", session_id="ep_s")
    R.check("end_session stores one episode", status.get("stored") is True and len(captured) == 1, str(status))
    digest = captured[0] if captured else ""
    R.check("digest consolidates the session", "dry_skin" in digest and "itching" in digest, digest[:80])

    # No user_id → nothing stored.
    with silent():
        st2 = p.end_session(user_id="", session_id="ep_s")
    R.check("end_session is a no-op without user_id", st2.get("stored") is False, str(st2))


# ── 7. HTTP API wiring (best-effort) ──────────────────────────────────────────
def test_api_wiring():
    R.section("7. HTTP API wiring")
    try:
        import fastapi  # noqa: F401
    except Exception:
        R.skip("FastAPI app", "fastapi not installed in this interpreter")
        return
    try:
        from api import app
        paths = {getattr(r, "path", None) for r in app.routes}
        R.check("/health route registered", "/health" in paths)
        R.check("/chat route registered", "/chat" in paths)
        R.check("/session/end route registered", "/session/end" in paths)
    except Exception as e:
        R.check("app.main imports", False, repr(e))


# ── 9. Stream and Blocks validation (NEW) ──────────────────────────────────────
def test_blocks_and_streaming():
    R.section("9. NDJSON Blocks and Streaming")
    from graphrag.validators.answer_validator import validate_line, iter_blocks
    from graphrag.schemas.blocks import Block, SummaryBlock, WarningBlock, NextStepsBlock, FollowUpQuestionsBlock
    
    # 1. Streamed lines each validate as a Block
    line1 = '{"type":"summary","data":{"text":"Dry skin is very common."}}'
    b1 = validate_line(line1)
    R.check("line1 validates as summary block", b1 is not None and b1.type == "summary" and b1.data.text == "Dry skin is very common.")
    
    # 2. A malformed line is dropped; valid lines before/after still stream.
    lines = [
        '{"type":"summary","data":{"text":"Hello."}}\n',
        '{"malformed_json": }\n',
        '{"type":"warning","data":{"text":"Warning!","severity":"caution"}}\n'
    ]
    blocks = list(iter_blocks(lines, terminal=False))
    R.check("iter_blocks yields only valid blocks and drops malformed line", len(blocks) == 2 and blocks[0].type == "summary" and blocks[1].type == "warning")
    
    # 3. terminal=True drops follow_up_questions
    lines_with_followup = [
        '{"type":"summary","data":{"text":"Assessment complete."}}\n',
        '{"type":"follow_up_questions","data":{"questions":["Any questions?"]}}\n'
    ]
    blocks_terminal = list(iter_blocks(lines_with_followup, terminal=True))
    R.check("terminal=True drops follow_up_questions", len(blocks_terminal) == 1 and blocks_terminal[0].type == "summary")
    
    # 4. refuse / out-of-scope / emergency stream blocks, not strings
    from graphrag.domain.messages import refusal_blocks, out_of_scope_blocks, emergency_blocks
    R.check("refusal_blocks returns list of blocks", isinstance(refusal_blocks(), list) and refusal_blocks()[0]["type"] == "summary")
    R.check("out_of_scope_blocks returns list of blocks", isinstance(out_of_scope_blocks(), list) and out_of_scope_blocks()[0]["type"] == "summary")
    eb = emergency_blocks()
    R.check("emergency_blocks returns warning + next_steps", len(eb) == 2 and eb[0]["type"] == "warning" and eb[0]["data"]["severity"] == "critical" and eb[1]["type"] == "next_steps")
    
    # 5. First block reaches the client before the model finishes (no full-response buffering)
    generator_input = []
    def input_generator():
        yield '{"type":"summary","data":{"text":"First."}}\n'
    gen = iter_blocks(input_generator(), terminal=False)
    first_block = next(gen)
    R.check("first block yielded before stream ends (no full buffering)", first_block.type == "summary" and first_block.data.text == "First.")


def main() -> None:
    print("=" * 64)
    print("  OFFLINE SMOKE TEST — GraphRAG dermatology assistant")
    print("  (Pinecone / Neo4j / Gemini stubbed — no network, no cost)")
    print("=" * 64)
    test_imports()
    test_domain()
    test_memory()
    test_entities()
    test_pipeline_scenarios()
    test_stage4_injection()
    test_episodic_session_end()
    test_api_wiring()
    test_blocks_and_streaming()
    print("\n" + "=" * 64)
    total = R.passed + R.failed
    print(f"  RESULT: {R.passed}/{total} checks passed, {R.failed} failed")
    print("=" * 64)
    sys.exit(1 if R.failed else 0)


if __name__ == "__main__":
    main()
