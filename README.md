# Knowledge Chunker & GraphRAG Assistant (Dermatology template)

Converts reference PDFs into validated, graph-ready **micro-chunks** — each with extracted entities, directional relations, a summary, and a significance note — for a Hybrid GraphRAG pipeline, and provides an end-to-end medical assistant API and CLI. Ships configured for dermatology (skin, hair, and nail disorders), but **retargets to any domain by editing configuration files**.

## ⭐ Retarget to a new use case: edit domain files

The domain knowledge is localized in specific configurations, making the system easy to retarget:

- **Chunking Pipeline**: Edit `chunking/domain.py` to customize entity types, synonym maps, relation types, extraction system prompts, section/concept segmentation patterns, and validation thresholds.
- **GraphRAG Pipeline**: Edit `graphrag/domain/` to configure the gatekeeper prompt (`prompts.py`), answering persona/focus (`answer_prompt.py`), clinical/triage safety policies (`clinical_policy.py`), query types and retrieval tuning (`query_taxonomy.py`), and namespace vocabulary (`vocabulary.py`).
- **Session Memory**: Edit `Memory_Layer/session_memory/domain/` to adjust regex extraction patterns (`extraction_patterns.py`), clinical risk weighting/triage rules (`risk_rules.py`), and state display mapping (`render_fields.py`).

Everything else (loaders, cleaner, LLM clients, retrievers, storage, orchestrator, and schemas) is domain-agnostic and stays the same.

## Layout

```
├── api.py                    # ⭐ FastAPI HTTP API entrypoint (uvicorn api:app)
├── run_graphrag.py           # CLI entrypoint for interactive chat (REPL or one-shot)
├── chunker.py                # generic entry point — processes every PDF in dataset/
├── chunk_pages_27_845.py     # example: plan/smoke/full run over a page range
├── ingest_pinecone.py        # ingests processed micro-chunks into Pinecone index
├── ingest_neo4j.py           # ingests processed micro-chunks into Neo4j graph database
├── live_check.py             # live diagnostics check for Pinecone and Neo4j connection/counts
├── smoke_test.py             # offline end-to-end regression test (stubs external services)
├── test_pipeline.py          # tests for the chunking pipeline
├── .env.example              # copy to .env and configure API keys and database passwords
├── dataset/                  # ← drop your source PDFs here
├── chunking/                 # offline PDF chunking and LLM entity/relation extraction logic
├── graphrag/                 # retrieval, gatekeeper routing, and response generation pipeline
├── Memory_Layer/             # session-state extraction, triage risk weighting, context building
└── episodic/                 # episodic / long-term per-user memory logic
```

`chunks/` and `logs/` are generated at the project root on first run.

## Setup

```powershell
# uv (matches pyproject.toml / .python-version)
uv sync
# or plain pip
python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt

Copy-Item .env.example .env   # then edit .env and set GEMINI_API_KEY, PINECONE_API_KEY, and NEO4J_PASSWORD
```

## Run

1. **Document Processing (Chunking)**:
   Put your reference PDFs in `dataset/`. The filename becomes the `doc_id` / `book_type`.
   ```powershell
   python chunker.py
   python chunker.py --version v2                  # tag a different output version
   python chunker.py --start-page 27 --end-page 845  # inclusive page range
   ```
   Results land under `chunks/`.

2. **Ingest to Data Stores**:
   Populate vector search and graph database:
   ```powershell
   python ingest_pinecone.py --namespace dermatology
   python ingest_neo4j.py --version v1
   ```
   Verify status with `python live_check.py`.

3. **Run Assistant CLI**:
   Interact with the GraphRAG clinical assistant:
   ```powershell
   python run_graphrag.py                          # interactive chat (REPL)
   python run_graphrag.py --query "itchy skin rash on arm" # one-shot query
   python run_graphrag.py --session-id alice       # session memory key
   python run_graphrag.py --user-id user-123       # enable episodic long-term memory
   ```

4. **Serve HTTP API**:
   Deploy / run the FastAPI server:
   ```powershell
   python api.py                                   # runs server on port 8000
   ```
   Or deploy on Render using `render.yaml`.

5. **Test / Validate**:
   Run the offline suite to verify logic/wiring without API costs:
   ```powershell
   python smoke_test.py
   ```

## How it works

- **Offline Chunking Pipeline**:
  PDF → clean text → segment sections (page-tracked) → ~350-token semantic blocks → batched LLM extraction (5 blocks/call, 4 parallel workers, jittered backoff) → strict validation (≥3 entities, relations ≥ entities/2, ≤650 tokens) → authoritative provenance stamping → versioned JSON.
  
- **Online GraphRAG & Memory Assistant**:
  User query → Session memory load (symptom accumulation) → Red-flag detection (anaphylaxis, necrotic skin, SJS/TEN, etc.) → Gatekeeper/analyzer (checks dermatology relevance, intent, risk, and extracts entities) → Routing (HYBRID_RAG, MEMORY_FIRST, or NO_RETRIEVAL) → Vector search (Pinecone) + Graph traversal (Neo4j) → Episodic memory recall → Response generation (Gemini stream with risk-driven formatting) → Session/episodic memory update.

Re-runs of the chunker are **resumable**: completed blocks are marked in `logs/processed_blocks/` and skipped; failures are written to a per-run manifest under `logs/`.

## Cost notes

Output tokens dominate cost. The pipeline keeps output lean (entities emit only `name`+`type`; `normalized_name`/`properties` are derived locally) and sends a compact schema per call.
For the online assistant, cost is determined by Gemini API calls during the gatekeeper analysis stage, response generation stage, and episodic consolidation/ingest stage at session end. See `.env.example` to select models.
