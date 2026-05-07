# Evaluation methodology and limits

## What the score is

- Per-answer scores and feedback are produced by an **LLM** using a fixed rubric (`interview_evaluation.py`) and optional **RAG** context.
- Outputs are **not** human hiring decisions and are **not** deterministic across runs (temperature > 0).

## How to argue reliability in defense

1. Run **`POST /interview/evaluate/reliability`** with the same answer text several times; report **variance** and the `consistency_label` from `score_reliability`.
2. Run **`POST /interview/evaluate/rag-compare`** to show whether retrieved context changes scores materially.
3. Use **`GET /benchmark/regression`** (with real session data) as a coarse health check — not a proof of correctness.

## Reference scenarios (manual)

See `data/evaluation_reference_scenarios.json`: hand-written examples with **expected qualitative signals** (strengths/weaknesses). These are **not** automated gold labels; they document how you reason about answers in demos.

## What we did not build

- No large human-annotated benchmark dataset in-repo.
- No statistical calibration of scores to real interview outcomes.
