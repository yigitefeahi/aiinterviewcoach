# Quality and Reliability Notes

## Automated tests

- Run tests with:
  - `python3 -m pytest -q`
- Current suite includes:
  - health endpoint checks
  - interview utility behavior
  - rate-limiter guard behavior
  - regression snapshot logic
  - auth/session/answer integration flow
  - submit-answer **retry** path (no crash; `question_index` / `total_questions` present)

## CI checks

- GitHub Actions workflow is configured at `.github/workflows/ci.yml`.
- On push/PR, it runs:
  - backend tests
  - backend mypy checks
  - frontend lint

## Benchmark and regression signals

- `GET /benchmark/health` provides aggregate confidence/score health.
- `GET /benchmark/regression` provides pass/fail style regression snapshot over recent sessions.
- `python scripts/benchmark_regression.py` creates `data/benchmark_snapshot.json` for offline artifact tracking.
- `POST /interview/evaluate/reliability` measures same-answer score consistency.
- `POST /interview/evaluate/rag-compare` compares RAG and no-RAG evaluation modes.
- RAG evidence is also exposed in normal answer responses, final reports, hints, CV suggestions, roadmap/drills, and Story Vault search metadata.
- Retrieval tests use fake Chroma collections so CI can validate reranking, metadata filtering, quality labels, and legacy `retrieve_context` compatibility without live embedding calls.
- `GET /rag/inspector/session/{id}` exposes query context, evidence, graph hits, user memory, quality metrics, and low-confidence warnings for academic defense/debugging.

## Security and operations

- Request throttling is enabled for auth and answer endpoints.
- Rate limits: **Redis** when `REDIS_URL` is set; otherwise **persisted in the database** (`rate_limit_events`) when `RATE_LIMIT_PERSIST=true` (default); else in-memory.
- **CSRF**: stateless `csrf_token` in login/register JSON; clients must send `X-CSRF-Token` on mutating requests when authenticated.
- **Origin** allowlist for `POST`/`PUT`/`PATCH`/`DELETE` with an `Origin` header.
- Audit events are written to `backend/data/audit.log` in JSONL format.
- Login/register set an **HttpOnly** `access_token` cookie (JWT still returned in JSON for API clients). Prefer `AUTH_COOKIE_SECURE=true` behind HTTPS in production.
- Requests receive an **`X-Request-ID`** header for correlation.

## Evaluation methodology (defense-oriented)

- See `docs/EVALUATION_METHODOLOGY.md` and `data/evaluation_reference_scenarios.json` for limits, manual reference scenarios, and how to discuss reliability.

## Evaluation trust (LLM + verification)

- Interview scores and feedback are **LLM-generated** against a fixed rubric in code; they are **not** human-labeled ground truth.
- Use **`POST /interview/evaluate/reliability`** to run the same answer multiple times and inspect score variance (consistency).
- Use **`POST /interview/evaluate/rag-compare`** to compare RAG vs non-RAG evaluation for the same answer.
- For demos/defense: describe limits (non-determinism, no gold dataset in-repo) and point to these endpoints as **sanity checks**, not proof of hiring validity.

## RAG implementation notes

- Core retrieval lives in `app/rag.py` and uses purpose-specific helpers plus a query router. The router selects physical Chroma collections such as `role_kb`, `company_kb`, `question_kb`, `answer_kb`, `cv_kb`, `evaluation_kb`, and `roadmap_kb`; `knowledge_base` remains a compatibility fallback.
- Evidence includes collection, source, layer, doc type, profession/company metadata, semantic score, keyword score, metadata score, layer weight, hybrid score, relevance label, citation preview, and graph/user-memory metadata when applicable.
- Retrieval quality is intentionally shown as a confidence aid, not as proof that the LLM answer is correct.
- Citation-grounded feedback returns `citations`, `citation_notes`, and `rag_evaluation` proxy metrics so the evaluator can show whether feedback overlaps with retrieved evidence.

## CV role suggestions

- Implemented as keyword/embedding ranking plus optional LLM screening and RAG role-fit evidence (see API fields `method`, `retrieval_evidence`, `rag_summary`, `limitations`).
- Still not a hiring decision; honest positioning avoids overselling in jury Q&A.
