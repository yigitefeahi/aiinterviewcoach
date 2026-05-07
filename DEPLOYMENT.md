# AI Interview Coach - Production Notes

## 1) Security First

- Rotate any exposed `OPENAI_API_KEY` immediately.
- Do not commit real `.env` files.
- Use `backend/.env.example` and `frontend/.env.example` as templates.

## 2) Local Production-like Run (Docker)

```bash
docker compose up --build
```

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`

## 3) Environment Variables

Backend (`backend/.env`):

- `JWT_SECRET`
- `DATABASE_URL`
- `OPENAI_API_KEY`
- `CORS_ORIGINS` (comma separated frontend URLs in production)

Frontend (`frontend/.env.local`):

- `NEXT_PUBLIC_API_BASE` (backend public URL)

## 4) RAG Pipeline Upgrades Included

- Multi-query retrieval variants per purpose: evaluation, hints, question generation, CV screening, roadmap/drills, and story search.
- Metadata-aware filtering for profession, doc type, company, focus area, difficulty, and sector with safe fallback when filtered retrieval is empty.
- Hybrid reranking: semantic + keyword + metadata boosts, plus source/doc-type diversity penalties to avoid same-source dominance.
- Evidence includes source, doc type, score breakdown, relevance label, preview, and retrieval quality summary.
- Multi-layer RAG uses physical Chroma collections: `role_kb`, `company_kb`, `question_kb`, `answer_kb`, `cv_kb`, `evaluation_kb`, `roadmap_kb`, plus legacy `knowledge_base` fallback and runtime `user_memory_kb` / `graph_kb` evidence.
- Query routing selects purpose-specific collections, then score fusion combines semantic, keyword, metadata, layer weight, collection diversity, user memory, CV facts, and graph relation signals.
- `user_memory_items` stores private personalization signals from answers/CV analysis; keep the database persistent in production if you want long-term coaching memory.
- `GET /rag/inspector/session/{id}` is useful for admin/demo inspection but should remain authenticated because it can expose private answer/memory snippets.
- Static KB files live under `backend/data/kb`; richer RAG docs live under `backend/data/rag/**`.

## 5) Recommended Live Setup

This repository now includes `render.yaml` for one-click Render deployment.

### Render (direct setup)

1. Push project to GitHub (backend + frontend + `render.yaml`).
2. In Render dashboard: **New +** -> **Blueprint** -> select repository.
3. Render will create:
   - `ai-coach-backend` (Python web service)
   - `ai-coach-frontend` (Node web service)
   - `ai-coach-db` (PostgreSQL)
4. Set missing env values in Render:
   - `ai-coach-backend`:
     - `OPENAI_API_KEY` = your new key
     - `CORS_ORIGINS` = `https://<your-frontend-service>.onrender.com`
   - `ai-coach-frontend`:
     - `NEXT_PUBLIC_API_BASE` = `https://<your-backend-service>.onrender.com`
5. Trigger redeploy for both services after env update.

### Post-deploy checks

- Open frontend URL and register/login.
- Start interview and submit at least one text answer.
- Confirm backend `/health` returns `{ "ok": true }`.
- Confirm `/interview/session/{id}/report` returns summary data.

### KB ingest job

Add periodic KB ingest job after deploy:

```bash
python3 scripts/ingest_kb.py
```

Run this after adding or editing markdown KB files. The script ingests both `data/kb/*.md` and `data/rag/**/*.md` into legacy `knowledge_base` and into each layer's physical Chroma collection.
