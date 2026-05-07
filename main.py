import io
import json
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .db import get_db
from .migrations_runner import apply_migrations
from .models import (
    User,
    InterviewSession,
    StoryVaultItem,
    UserMemoryItem,
    UserPreference,
    DrillCompletion,
    InterviewTurn,
    RateLimitEvent,
)
from .schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    ProfessionListResponse,
    SectorListResponse,
    CVRoleSuggestionResponse,
    StartSessionRequest,
    StartSessionResponse,
    SubmitTextRequest,
    PassQuestionRequest,
    SubmitAnswerResponse,
    SessionResultResponse,
    SessionListResponse,
    EvaluationReliabilityRequest,
    RagComparisonRequest,
    UserMeResponse,
    HintRequest,
    HintResponse,
    RoadmapResponse,
    WeeklyDrillsResponse,
    UserPreferenceRequest,
    UserPreferenceResponse,
    DrillCompletionRequest,
    DrillCompletionResponse,
    StoryCreateRequest,
    StoryListResponse,
    StoryItemResponse,
)
from .security import hash_password, verify_password
from .auth import create_access_token, get_current_user
from .openai_client import client
from .config import settings
from .interview import (
    create_session,
    submit_answer,
    pass_current_question,
    evaluate_reliability_for_session,
    compare_rag_modes_for_session,
    load_user_memory,
)
from .rate_limit import RateLimitConfig, enforce_with_backend
from .audit import audit_event
from .cv_extract import extract_text_from_cv_bytes
from .cv_enrich import maybe_enrich_cv_suggestions
from .cv_screening import apply_cv_screening_llm
from .csrf import issue_csrf_token, verify_csrf_token
from .cv_struct import extract_cv_sections
from .cv_embedding import fuse_keyword_and_embedding, profession_embedding_scores
from .reporting import build_session_report, build_regression_snapshot
from .rag import retrieve_for_cv_screening, rank_story_candidates, evaluate_retrieval, graph_context_for_query, RetrievalQuery
from .role_profiles import get_role_profile
from .product_features import (
    build_hint,
    build_roadmap,
    build_weekly_drills,
    get_company_packs,
)


app = FastAPI(title="AI Interview Coach API", version="2.1.0")

default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]
allow_origins = settings.cors_origin_list or default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _origin_allowed(origin: str) -> bool:
    if origin in allow_origins:
        return True
    return bool(re.match(r"^https?://(localhost|127\.0\.0\.1):\d+$", origin))


def _has_auth_credentials(request: Request) -> bool:
    if request.cookies.get("access_token"):
        return True
    auth = (request.headers.get("authorization") or "").strip()
    return auth.lower().startswith("bearer ")


def _csrf_exempt_path(path: str) -> bool:
    for prefix in (
        "/auth/login",
        "/auth/register",
        "/auth/logout",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    ):
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return path.startswith("/openapi")


def _needs_csrf_check(request: Request) -> bool:
    if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return False
    return not _csrf_exempt_path(request.url.path)


@app.middleware("http")
async def security_and_observability_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    origin = request.headers.get("origin")
    if request.method in ("POST", "PUT", "PATCH", "DELETE") and origin:
        if not _origin_allowed(origin):
            return JSONResponse(
                status_code=403,
                content={"detail": "Origin not allowed"},
                headers={"X-Request-ID": request_id},
            )

    if _needs_csrf_check(request) and _has_auth_credentials(request):
        hdr = (request.headers.get("x-csrf-token") or "").strip()
        if not verify_csrf_token(hdr):
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid or missing CSRF token"},
                headers={"X-Request-ID": request_id},
            )

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


apply_migrations()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROF_FILE = DATA_DIR / "professions.json"
SECTOR_FILE = DATA_DIR / "sectors.json"
FILLER_WORDS = {"um", "uh", "like", "you know", "actually", "basically", "so"}
AUTH_RATE_LIMIT = RateLimitConfig(max_requests=12, window_seconds=60)
ANSWER_RATE_LIMIT = RateLimitConfig(max_requests=20, window_seconds=60)


def _client_key(request: Request, user_id: Optional[int] = None, prefix: str = "global") -> str:
    client_host = request.client.host if request.client else "unknown"
    uid = str(user_id) if user_id is not None else "anon"
    return f"{prefix}:{uid}:{client_host}"


def _attach_auth_cookie(response: JSONResponse, token: str) -> JSONResponse:
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=settings.jwt_expires_minutes * 60,
        samesite="lax",
        secure=settings.auth_cookie_secure,
        path="/",
    )
    return response


def enforce_rate_limit(
    request: Request,
    db: Session,
    config: RateLimitConfig,
    user_id: Optional[int],
    scope: str,
) -> None:
    key = _client_key(request=request, user_id=user_id, prefix=scope)
    if enforce_with_backend(
        db,
        use_persistent=settings.rate_limit_persist,
        key=key,
        config=config,
    ):
        raise HTTPException(status_code=429, detail="Too many requests, please slow down.")


def load_professions() -> list[str]:
    with open(PROF_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["professions"]


def load_sectors() -> list[str]:
    with open(SECTOR_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["sectors"]


def _weighted_keyword_score(text: str, signals: dict[str, float]) -> tuple[float, list[str]]:
    score = 0.0
    hits: list[str] = []
    for keyword, weight in signals.items():
        pattern = rf"(?<![a-z0-9]){re.escape(keyword.lower())}(?![a-z0-9])"
        if re.search(pattern, text):
            score += weight
            hits.append(keyword)
    return score, hits


def _build_cv_role_feedback(
    role_scores: list[tuple[str, float, list[str]]],
    suggested_professions: list[str],
) -> dict[str, Any]:
    feedback = []
    for role, score, hits in role_scores[:5]:
        if score <= 0 and role not in suggested_professions:
            continue
        if role in suggested_professions:
            stance = "primary_match" if role == suggested_professions[0] else "secondary_match"
        else:
            stance = "adjacent_signal"
        feedback.append(
            {
                "role": role,
                "stance": stance,
                "score": round(score, 2),
                "evidence": hits[:8],
                "comment": (
                    f"{role} is supported by signals such as {', '.join(hits[:4])}."
                    if hits
                    else f"{role} has limited direct evidence in the extracted CV text."
                ),
            }
        )
    return {
        "primary_role": suggested_professions[0] if suggested_professions else None,
        "secondary_roles": suggested_professions[1:],
        "role_feedback": feedback,
    }


def suggest_roles_from_cv(cv_text: str) -> dict:
    original = cv_text or ""
    text = original.lower()
    professions = load_professions()
    sectors = load_sectors()
    cv_structure = extract_cv_sections(original)
    skill_map: dict[str, dict[str, float]] = {
        "Software Engineer": {
            "software engineer": 4,
            "object oriented": 2,
            "algorithm": 2,
            "java": 1.5,
            "c++": 1.5,
            "api": 1,
            "microservice": 2,
            "system design": 2,
        },
        "Full Stack Developer": {
            "full stack": 5,
            "fullstack": 5,
            "frontend": 2,
            "backend": 2,
            "react": 2,
            "node.js": 2,
            "api": 2,
            "database": 2,
            "typescript": 2,
            "next.js": 2,
        },
        "Backend Developer": {
            "backend": 4,
            "rest": 2,
            "api": 2,
            "database": 2,
            "postgresql": 2,
            "redis": 2,
            "microservice": 2,
            "docker": 1,
        },
        "Frontend Developer": {
            "frontend": 4,
            "react": 3,
            "typescript": 2,
            "javascript": 2,
            "css": 1.5,
            "next.js": 2,
            "ui component": 2,
        },
        "Mobile Developer": {"android": 3, "ios": 3, "flutter": 3, "react native": 3, "kotlin": 2, "swift": 2},
        "Data Scientist": {
            "machine learning": 4,
            "model training": 3,
            "tensorflow": 3,
            "pytorch": 3,
            "scikit-learn": 2,
            "statistics": 2,
            "predictive model": 3,
        },
        "Machine Learning Engineer": {
            "machine learning engineer": 5,
            "ml engineer": 4,
            "model deployment": 3,
            "feature engineering": 3,
            "tensorflow": 2,
            "pytorch": 2,
            "scikit-learn": 2,
            "ml pipeline": 3,
            "model serving": 3,
        },
        "AI Engineer": {
            "ai engineer": 5,
            "generative ai": 4,
            "llm": 4,
            "rag": 4,
            "prompt engineering": 3,
            "openai": 3,
            "langchain": 3,
            "vector database": 3,
            "embeddings": 3,
        },
        "Data Engineer": {
            "data engineer": 5,
            "data pipeline": 4,
            "etl": 4,
            "elt": 3,
            "airflow": 3,
            "spark": 3,
            "kafka": 3,
            "data warehouse": 3,
            "dbt": 3,
            "snowflake": 3,
            "bigquery": 3,
        },
        "Data Analyst": {
            "data analyst": 5,
            "data analysis": 4,
            "sql": 3,
            "excel": 3,
            "power bi": 4,
            "tableau": 4,
            "dashboard": 3,
            "reporting": 3,
            "kpi": 3,
            "data visualization": 3,
            "business intelligence": 4,
            "etl": 2,
            "pandas": 2,
        },
        "Business Analyst": {
            "business analyst": 5,
            "business analysis": 4,
            "requirements": 4,
            "stakeholder": 3,
            "user story": 3,
            "acceptance criteria": 3,
            "process analysis": 4,
            "gap analysis": 3,
            "uat": 3,
            "jira": 2,
            "confluence": 2,
            "kpi": 2,
        },
        "Cloud Engineer": {
            "cloud engineer": 5,
            "aws": 3,
            "azure": 3,
            "gcp": 3,
            "cloudformation": 3,
            "terraform": 3,
            "iam": 2,
            "vpc": 2,
            "load balancer": 2,
            "cloud migration": 3,
        },
        "Site Reliability Engineer": {
            "site reliability": 5,
            "sre": 5,
            "incident response": 3,
            "slo": 3,
            "sla": 2,
            "observability": 3,
            "prometheus": 3,
            "grafana": 3,
            "on-call": 3,
            "reliability": 3,
        },
        "MLOps Engineer": {
            "mlops": 5,
            "ml pipeline": 4,
            "model registry": 3,
            "model monitoring": 3,
            "kubeflow": 3,
            "mlflow": 3,
            "feature store": 3,
            "model deployment": 3,
            "drift": 2,
        },
        "Database Administrator": {
            "database administrator": 5,
            "dba": 5,
            "backup": 3,
            "replication": 3,
            "query tuning": 3,
            "postgresql": 3,
            "mysql": 3,
            "oracle": 3,
            "indexing": 3,
        },
        "System Administrator": {
            "system administrator": 5,
            "sysadmin": 5,
            "linux": 3,
            "windows server": 3,
            "active directory": 3,
            "patch management": 3,
            "shell scripting": 2,
            "server maintenance": 3,
        },
        "Solutions Architect": {
            "solutions architect": 5,
            "solution architecture": 4,
            "architecture design": 3,
            "cloud architecture": 3,
            "stakeholder": 2,
            "technical proposal": 3,
            "integration": 3,
            "enterprise architecture": 3,
        },
        "Software Architect": {
            "software architect": 5,
            "architecture": 3,
            "design patterns": 3,
            "technical leadership": 3,
            "system design": 3,
            "domain driven design": 3,
            "microservice": 2,
            "scalability": 2,
        },
        "Game Developer": {
            "game developer": 5,
            "unity": 4,
            "unreal": 4,
            "c#": 3,
            "gameplay": 3,
            "game engine": 3,
            "multiplayer": 2,
            "3d": 2,
        },
        "Embedded Software Engineer": {
            "embedded": 5,
            "firmware": 4,
            "microcontroller": 4,
            "rtos": 3,
            "c programming": 3,
            "c++": 2,
            "iot": 2,
            "spi": 2,
            "i2c": 2,
        },
        "Blockchain Developer": {
            "blockchain": 5,
            "smart contract": 4,
            "solidity": 4,
            "web3": 3,
            "ethereum": 3,
            "defi": 3,
            "nft": 2,
            "hardhat": 2,
        },
        "Product Manager": {
            "product manager": 5,
            "roadmap": 3,
            "stakeholder": 2,
            "kpi": 2,
            "prioritization": 3,
            "go-to-market": 3,
            "user research": 2,
        },
        "UX/UI Designer": {"figma": 3, "prototype": 2, "ux": 3, "ui": 2, "wireframe": 3, "user research": 3},
        "DevOps Engineer": {"kubernetes": 3, "docker": 2, "ci/cd": 3, "terraform": 3, "cloud": 2, "aws": 2},
        "Cybersecurity Analyst": {"siem": 4, "threat": 3, "security": 2, "incident": 3, "owasp": 3, "soc": 3},
        "Security Engineer": {
            "security engineer": 5,
            "application security": 4,
            "cloud security": 3,
            "iam": 3,
            "vulnerability management": 3,
            "secure coding": 3,
            "threat modeling": 3,
            "owasp": 3,
        },
        "Penetration Tester": {
            "penetration tester": 5,
            "pentest": 5,
            "burp suite": 4,
            "metasploit": 3,
            "kali": 3,
            "vulnerability assessment": 3,
            "exploit": 3,
            "owasp": 2,
        },
        "SOC Analyst": {
            "soc analyst": 5,
            "security operations": 4,
            "siem": 4,
            "splunk": 3,
            "alert triage": 3,
            "incident response": 3,
            "threat hunting": 3,
            "log analysis": 3,
        },
        "QA Engineer": {"test automation": 4, "selenium": 3, "qa": 3, "quality": 1.5, "cypress": 3, "test case": 3},
        "Automation Engineer": {
            "automation engineer": 5,
            "automation": 3,
            "scripting": 3,
            "python": 2,
            "rpa": 3,
            "workflow automation": 4,
            "plc": 3,
            "selenium": 2,
        },
        "Test Automation Engineer": {
            "test automation engineer": 5,
            "test automation": 5,
            "selenium": 4,
            "cypress": 4,
            "playwright": 4,
            "pytest": 3,
            "junit": 3,
            "test framework": 3,
        },
        "Scrum Master": {
            "scrum master": 5,
            "agile": 3,
            "sprint planning": 3,
            "retrospective": 3,
            "facilitation": 3,
            "servant leadership": 3,
            "jira": 2,
        },
        "Technical Project Manager": {
            "technical project manager": 5,
            "project management": 3,
            "delivery": 3,
            "stakeholder": 3,
            "timeline": 2,
            "risk management": 3,
            "technical coordination": 3,
        },
        "Technical Support Engineer": {
            "technical support engineer": 5,
            "technical support": 4,
            "troubleshooting": 4,
            "customer support": 3,
            "ticket": 3,
            "root cause": 3,
            "logs": 2,
        },
        "IT Support Specialist": {
            "it support": 5,
            "helpdesk": 4,
            "hardware": 3,
            "active directory": 3,
            "ticket": 3,
            "troubleshooting": 3,
            "windows": 2,
        },
        "Network Engineer": {
            "network engineer": 5,
            "networking": 4,
            "router": 3,
            "switch": 3,
            "firewall": 3,
            "tcp/ip": 3,
            "vpn": 3,
            "ccna": 3,
        },
        "ERP/CRM Consultant": {
            "erp": 5,
            "crm": 5,
            "consultant": 2,
            "salesforce": 3,
            "dynamics": 3,
            "business process": 3,
            "implementation": 3,
            "requirements": 2,
        },
        "SAP Consultant": {
            "sap": 5,
            "sap consultant": 5,
            "abap": 3,
            "s/4hana": 3,
            "fico": 3,
            "mm": 2,
            "sd": 2,
            "sap implementation": 3,
        },
    }
    sector_map = {
        "Telecommunications": ["telecom", "network", "5g", "lte", "signal"],
        "Fintech": ["bank", "payment", "fraud", "fintech", "ledger"],
        "Healthcare": ["health", "patient", "hipaa", "clinical", "ehr"],
        "E-commerce": ["e-commerce", "checkout", "catalog", "cart", "conversion"],
        "Gaming": ["game", "unity", "unreal", "multiplayer", "latency"],
        "Cybersecurity": ["security", "threat", "vulnerability", "incident", "siem"],
        "Cloud Infrastructure": ["aws", "gcp", "azure", "cloud", "infrastructure"],
        "SaaS": ["subscription", "b2b", "churn", "saas", "tenant"],
        "Public Sector": ["government", "public", "regulation", "compliance", "citizen"],
        "Education Technology": ["learning", "student", "lms", "education", "course"],
    }

    role_scores: list[tuple[str, float, list[str]]] = []
    for p in professions:
        score, hits = _weighted_keyword_score(text, skill_map.get(p, {}))
        profile = get_role_profile(p)
        theme_hits = [theme for theme in profile.get("themes", []) if any(token in text for token in str(theme).lower().split())]
        if theme_hits:
            score += min(2.0, len(theme_hits) * 0.5)
            hits.extend([f"profile:{theme}" for theme in theme_hits[:4]])
        # Generic programming terms should not beat explicit analytics/business evidence.
        if p == "Software Engineer" and any(term in text for term in ["data analyst", "business analyst", "power bi", "tableau", "dashboard", "requirements"]):
            score *= 0.72
        role_scores.append((p, score, hits))
    role_scores.sort(key=lambda x: x[1], reverse=True)
    prof_scores: list[tuple[str, int]] = [(role, int(round(score))) for role, score, _ in role_scores]

    method = "keyword_heuristic"
    embedding_top: list[dict] = []
    suggested_professions: list[str]

    if settings.cv_use_embedding and len(original.strip()) >= 40:
        emb_scores = profession_embedding_scores(original, professions)
        if emb_scores:
            fused = fuse_keyword_and_embedding(prof_scores, emb_scores)
            suggested_professions = [p for p, _ in fused[:3]]
            if role_scores and role_scores[0][1] >= max(3.0, role_scores[1][1] + 1.5 if len(role_scores) > 1 else 3.0):
                suggested_professions = [role_scores[0][0]] + [p for p in suggested_professions if p != role_scores[0][0]]
                suggested_professions = suggested_professions[:3]

            top_emb = sorted(emb_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            embedding_top = [{"profession": p, "similarity_01": round(s, 4)} for p, s in top_emb]
            method = "keyword_embedding_fusion"
            rationale = (
                "Hybrid ranking: keyword overlap with a fixed list plus embedding similarity between your CV "
                "and each profession label (not a hiring guarantee)."
            )
            limitations = (
                "Embeddings capture semantic similarity to role labels; they do not verify employment history. "
                "Poor PDF extraction or non-English text may reduce quality."
            )
        else:
            suggested_professions = [p for p, s, _ in role_scores if s > 0][:3] or professions[:3]
            rationale = (
                "Suggestions are inferred from keyword overlap with a fixed role/sector list (heuristic, not a full CV "
                "understanding model). Adjust manually if you target a different role."
            )
            limitations = (
                "Keyword overlap only (embedding call failed or unavailable). "
                "Upload plain text when possible; raw PDF bytes often decode poorly."
            )
    else:
        suggested_professions = [p for p, s, _ in role_scores if s > 0][:3] or professions[:3]
        rationale = (
            "Suggestions are inferred from keyword overlap with a fixed role/sector list (heuristic, not a full CV "
            "understanding model). Adjust manually if you target a different role."
        )
        limitations = (
            "Keyword overlap only (no embedding model or structured résumé parse). "
            "Upload plain text when possible; raw PDF bytes often decode poorly."
        )

    sec_scores = []
    for s in sectors:
        score = sum(1 for kw in sector_map.get(s, []) if kw in text)
        sec_scores.append((s, score))
    sec_scores.sort(key=lambda x: x[1], reverse=True)
    suggested_sectors = [s for s, v in sec_scores if v > 0][:3] or sectors[:3]

    out = {
        "suggested_professions": suggested_professions,
        "suggested_sectors": suggested_sectors,
        "rationale": rationale,
        "method": method,
        "limitations": limitations,
        "cv_structure": cv_structure,
        "embedding_top_professions": embedding_top or None,
        "role_fit_breakdown": _build_cv_role_feedback(role_scores, suggested_professions),
    }
    try:
        rag_result = retrieve_for_cv_screening(cv_text=original, target_profession=suggested_professions[0] if suggested_professions else "")
        out["retrieval_evidence"] = rag_result.evidence
        out["rag_summary"] = rag_result.summary
        out["retrieval_quality"] = rag_result.quality
        if rag_result.evidence:
            out["rationale"] = f"{out['rationale']} RAG evidence also checked role-fit signals from the interview knowledge base."
            out["limitations"] = f"{out['limitations']} RAG evidence is advisory and depends on KB coverage."
    except Exception:
        out["rag_summary"] = "RAG CV evidence unavailable; keyword/embedding CV pipeline used."
        out["retrieval_quality"] = {"label": "none", "score": 0, "evidence_count": 0}
    return out


def build_speaking_metrics(transcript: str) -> dict:
    words = re.findall(r"\b[\w']+\b", transcript.lower())
    word_count = len(words)
    estimated_duration_seconds = max(10, int(word_count / 2.3))
    wpm = int((word_count / estimated_duration_seconds) * 60) if estimated_duration_seconds else 0
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    filler_ratio = round((filler_count / word_count) * 100, 2) if word_count else 0.0

    if wpm < 110:
        pace_label = "slow"
        pace_comment = "A bit slow. Add a slightly stronger pace for confidence."
    elif wpm > 170:
        pace_label = "fast"
        pace_comment = "A bit fast. Slow down to improve clarity."
    else:
        pace_label = "balanced"
        pace_comment = "Good pace for interview communication."

    return {
        "duration_seconds": estimated_duration_seconds,
        "word_count": word_count,
        "words_per_minute": wpm,
        "pace_label": pace_label,
        "pace_comment": pace_comment,
        "filler_word_count": filler_count,
        "filler_word_ratio_percent": filler_ratio,
    }


def build_dynamic_video_analysis(transcript: str, score: Optional[int], feedback: str) -> dict:
    speaking_metrics = build_speaking_metrics(transcript)
    base = {
        "mode": "transcript-grounded-video-coaching",
        "visual_feedback": {
            "posture": "Posture estimate unavailable from raw video. Keep shoulders open and still.",
            "eye_contact": "Use direct camera gaze for key points and avoid looking down too often.",
            "engagement": "Energy is inferred from transcript clarity and pacing.",
            "presentation": "Use concise story flow and emphasize outcomes.",
            "overall_visual_feedback": "Good baseline. Improve camera presence with steadier rhythm.",
        },
        "speaking_metrics": speaking_metrics,
        "sampled_frame_count": 0,
    }

    prompt = {
        "transcript": transcript,
        "score": score,
        "feedback": feedback,
        "speaking_metrics": speaking_metrics,
        "instructions": (
            "Generate practical video-interview coaching grounded in transcript and speaking metrics. "
            "Avoid generic repeated advice. Return JSON with keys: posture, eye_contact, engagement, presentation, overall_visual_feedback."
        ),
    }
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "You are an executive interview delivery coach."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.55,
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        visual = {
            "posture": str(parsed.get("posture", base["visual_feedback"]["posture"])).strip(),
            "eye_contact": str(parsed.get("eye_contact", base["visual_feedback"]["eye_contact"])).strip(),
            "engagement": str(parsed.get("engagement", base["visual_feedback"]["engagement"])).strip(),
            "presentation": str(parsed.get("presentation", base["visual_feedback"]["presentation"])).strip(),
            "overall_visual_feedback": str(
                parsed.get("overall_visual_feedback", base["visual_feedback"]["overall_visual_feedback"])
            ).strip(),
        }
        base["visual_feedback"] = visual
    except Exception:
        pass
    return base


def analyze_visual_signals(video_bytes: bytes) -> dict:
    result = {
        "sampled_frame_count": 0,
        "face_detect_ratio": 0.0,
        "eye_contact_ratio": 0.0,
        "stability_score": 0.0,
        "visual_confidence_score": 0,
    }
    tmp_path = None
    try:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return result

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return result

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        frame_idx = 0
        sampled = 0
        face_hits = 0
        eye_contact_hits = 0
        centers: list[tuple[float, float]] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % 12 != 0:
                continue
            sampled += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            if len(faces) > 0:
                face_hits += 1
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                cx = (x + (w / 2)) / max(1, frame.shape[1])
                cy = (y + (h / 2)) / max(1, frame.shape[0])
                centers.append((cx, cy))
                if abs(cx - 0.5) < 0.2:
                    eye_contact_hits += 1
            if sampled >= 80:
                break
        cap.release()

        face_ratio = (face_hits / sampled) if sampled else 0.0
        eye_ratio = (eye_contact_hits / sampled) if sampled else 0.0

        stability = 0.0
        if len(centers) >= 3:
            xs = np.array([c[0] for c in centers], dtype=float)
            ys = np.array([c[1] for c in centers], dtype=float)
            motion = float(np.std(xs) + np.std(ys))
            stability = max(0.0, 1.0 - min(1.0, motion * 4.0))

        visual_conf = int(max(0, min(100, (0.45 * face_ratio + 0.35 * eye_ratio + 0.20 * stability) * 100)))

        return {
            "sampled_frame_count": sampled,
            "face_detect_ratio": round(face_ratio, 4),
            "eye_contact_ratio": round(eye_ratio, 4),
            "stability_score": round(stability, 4),
            "visual_confidence_score": visual_conf,
        }
    except Exception:
        return result
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/health/deps")
def health_deps():
    """Non-secret readiness hints (Chroma path, JWT stack, SQLite drift reminder)."""
    chroma_writable = False
    chroma_probe_error: Optional[str] = None
    try:
        p = Path(settings.chroma_dir)
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".health_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        chroma_writable = True
    except OSError as e:
        chroma_probe_error = str(e)

    db_url = (settings.database_url or "").strip()
    scheme = db_url.split(":", 1)[0].lower() if db_url else ""
    hints: list[str] = []
    if scheme == "sqlite":
        hints.append(
            "SQLite: if models gain columns/tables but your existing .db was created earlier, "
            "you may see 'no such column'. Back up and remove the local .db file, then restart "
            "(or introduce migrations)."
        )

    api_key_set = bool((settings.openai_api_key or "").strip())

    payload: dict[str, Any] = {
        "ok": True,
        "jwt_backend": "PyJWT",
        "openai_api_key_configured": api_key_set,
        "chroma_dir": settings.chroma_dir,
        "chroma_dir_writable": chroma_writable,
        "database_scheme": scheme,
        "hints": hints,
    }
    if chroma_probe_error is not None:
        payload["chroma_probe_error"] = chroma_probe_error
    return payload


@app.get("/professions", response_model=ProfessionListResponse)
def professions():
    return {"professions": load_professions()}


@app.get("/sectors", response_model=SectorListResponse)
def sectors():
    return {"sectors": load_sectors()}


@app.get("/interview/company-packs")
def company_packs():
    return {"packs": get_company_packs()}


@app.post("/cv/suggest", response_model=CVRoleSuggestionResponse)
async def cv_suggest(
    cv_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    raw = await cv_file.read()
    text = extract_text_from_cv_bytes(raw)
    base = suggest_roles_from_cv(text)
    if settings.cv_llm_screening:
        base = apply_cv_screening_llm(text, base, user.profession)
    elif settings.cv_llm_enrich:
        base = maybe_enrich_cv_suggestions(text, base)
    memory_bits = []
    if base.get("suggested_professions"):
        memory_bits.append(f"CV suggests target roles: {', '.join(base['suggested_professions'][:3])}.")
    evaluator = base.get("evaluator") if isinstance(base.get("evaluator"), dict) else {}
    if evaluator.get("strengths"):
        memory_bits.append(f"CV strengths: {', '.join(str(x) for x in evaluator.get('strengths', [])[:3])}.")
    if evaluator.get("weaknesses"):
        memory_bits.append(f"CV gaps: {', '.join(str(x) for x in evaluator.get('weaknesses', [])[:3])}.")
    if memory_bits:
        db.add(
            UserMemoryItem(
                user_id=user.id,
                session_id=None,
                memory_type="cv_signal",
                content=" ".join(memory_bits),
                score=0.7,
                meta={
                    "suggested_professions": base.get("suggested_professions", []),
                    "suggested_sectors": base.get("suggested_sectors", []),
                    "retrieval_quality": base.get("retrieval_quality", {}),
                },
            )
        )
        db.commit()
    return base


@app.post("/auth/register", response_model=TokenResponse)
def register(payload: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    try:
        enforce_rate_limit(request, db, AUTH_RATE_LIMIT, user_id=None, scope="auth-register")
        professions = set(load_professions())
        if payload.profession not in professions:
            raise HTTPException(status_code=400, detail="Invalid profession")

        existing = db.query(User).filter(User.email == payload.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        user = User(
            name=payload.name,
            email=payload.email,
            password_hash=hash_password(payload.password),
            profession=payload.profession,
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        token = create_access_token(user.id)
        csrf = issue_csrf_token()
        audit_event("auth.register.success", user_id=user.id, detail={"email": payload.email})

        body = {"access_token": token, "token_type": "bearer", "csrf_token": csrf}
        resp = JSONResponse(content=body)
        return _attach_auth_cookie(resp, token)
    except HTTPException:
        raise
    except Exception as e:
        audit_event("auth.register.error", detail={"error": str(e), "email": payload.email})
        raise HTTPException(status_code=500, detail=f"Register failed: {str(e)}")


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    enforce_rate_limit(request, db, AUTH_RATE_LIMIT, user_id=None, scope="auth-login")
    user = db.query(User).filter(User.email == payload.email).first()

    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(user.id)
    csrf = issue_csrf_token()
    audit_event("auth.login.success", user_id=user.id, detail={"email": payload.email})

    body = {"access_token": token, "token_type": "bearer", "csrf_token": csrf}
    resp = JSONResponse(content=body)
    return _attach_auth_cookie(resp, token)


@app.get("/auth/me", response_model=UserMeResponse)
def auth_me(user: User = Depends(get_current_user)):
    return UserMeResponse(id=user.id, email=user.email, name=user.name, profession=user.profession)


@app.post("/auth/logout")
def auth_logout():
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie("access_token", path="/")
    return resp


@app.post("/interview/start", response_model=StartSessionResponse)
def interview_start(
    payload: StartSessionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    chosen = payload.profession or user.profession
    professions = set(load_professions())

    if chosen not in professions:
        raise HTTPException(status_code=400, detail="Invalid profession")

    config = {
        "difficulty": payload.difficulty,
        "mode": payload.mode,
        "interview_length": payload.interview_length,
        "focus_area": payload.focus_area,
        "sector": payload.sector,
        "target_company": payload.target_company,
        "company_pack": payload.company_pack,
        "instant_mode": payload.instant_mode,
        "interview_date": payload.interview_date,
        "case_type": payload.case_type,
    }

    session, first_q, question_context = create_session(db, user, chosen, config=config)

    return {
        "session_id": session.id,
        "first_question": first_q,
        "question_context": question_context,
        "config": config,
    }


@app.post("/interview/answer/text", response_model=SubmitAnswerResponse)
def answer_text(
    payload: SubmitTextRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        enforce_rate_limit(request, db, ANSWER_RATE_LIMIT, user_id=user.id, scope="answer-text")
        result = submit_answer(db, user, payload.session_id, payload.answer_text)
        audit_event("interview.answer.text", user_id=user.id, session_id=payload.session_id, detail={"score": result.get("score")})
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text evaluation failed: {str(e)}")


@app.post("/interview/pass", response_model=SubmitAnswerResponse)
def pass_question(
    payload: PassQuestionRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        enforce_rate_limit(request, db, ANSWER_RATE_LIMIT, user_id=user.id, scope="answer-pass")
        result = pass_current_question(db, user, payload.session_id)
        audit_event("interview.pass", user_id=user.id, session_id=payload.session_id)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pass failed: {str(e)}")


@app.post("/interview/hint", response_model=HintResponse)
def interview_hint(
    payload: HintRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        enforce_rate_limit(request, db, ANSWER_RATE_LIMIT, user_id=user.id, scope="interview-hint")
        session = db.get(InterviewSession, payload.session_id)
        if not session or session.user_id != user.id:
            raise HTTPException(status_code=404, detail="Session not found")
        result_json = session.result_json or {}
        question = str(result_json.get("current_question") or "").strip()
        if not question:
            turns = sorted(session.turns, key=lambda t: t.id)
            question = turns[-1].question if turns else ""
        if not question:
            raise HTTPException(status_code=400, detail="No active question")
        config = dict(result_json.get("config") or {})
        config["profession"] = session.profession
        memories = load_user_memory(db, user)
        config["user_memory"] = memories
        config["cv_facts"] = [item["content"] for item in memories if item.get("memory_type") == "cv_signal"][:8]
        return build_hint(question, config)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hint failed: {str(e)}")


@app.get("/interview/roadmap", response_model=RoadmapResponse)
def interview_roadmap(
    interview_date: Optional[str] = None,
    target_company: Optional[str] = None,
    focus_area: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    memories = load_user_memory(db, user)
    return build_roadmap(
        profession=user.profession,
        target_company=target_company,
        interview_date=interview_date,
        focus_area=focus_area,
        user_memory=memories,
        cv_facts=[item["content"] for item in memories if item.get("memory_type") == "cv_signal"][:8],
    )


@app.get("/interview/weekly-drills", response_model=WeeklyDrillsResponse)
def interview_weekly_drills(
    interview_date: Optional[str] = None,
    target_company: Optional[str] = None,
    focus_area: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    memories = load_user_memory(db, user)
    return build_weekly_drills(
        profession=user.profession,
        target_company=target_company,
        interview_date=interview_date,
        focus_area=focus_area,
        user_memory=memories,
        cv_facts=[item["content"] for item in memories if item.get("memory_type") == "cv_signal"][:8],
    )


@app.get("/interview/drill-completions", response_model=DrillCompletionResponse)
def get_drill_completions(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = db.query(DrillCompletion).filter(DrillCompletion.user_id == user.id).all()
    return {"completions": {row.drill_key: bool(row.completed) for row in rows}}


@app.put("/interview/drill-completions", response_model=DrillCompletionResponse)
def set_drill_completion(
    payload: DrillCompletionRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    existing = (
        db.query(DrillCompletion)
        .filter(DrillCompletion.user_id == user.id, DrillCompletion.drill_key == payload.drill_key)
        .first()
    )
    if existing:
        existing.completed = 1 if payload.completed else 0
        db.add(existing)
    else:
        db.add(
            DrillCompletion(
                user_id=user.id,
                drill_key=payload.drill_key,
                completed=1 if payload.completed else 0,
            )
        )
    db.commit()
    return get_drill_completions(user=user, db=db)


@app.post("/interview/evaluate/reliability")
def evaluate_reliability(
    payload: EvaluationReliabilityRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        return evaluate_reliability_for_session(
            db=db,
            user=user,
            session_id=payload.session_id,
            answer_text=payload.answer_text,
            runs=payload.runs,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reliability evaluation failed: {str(e)}")


@app.post("/interview/evaluate/rag-compare")
def evaluate_rag_compare(
    payload: RagComparisonRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        return compare_rag_modes_for_session(
            db=db,
            user=user,
            session_id=payload.session_id,
            answer_text=payload.answer_text,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG comparison failed: {str(e)}")


@app.post("/interview/answer/audio", response_model=SubmitAnswerResponse)
async def answer_audio(
    session_id: int = Form(...),
    audio: UploadFile = File(...),
    request: Request = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        if request is not None:
            enforce_rate_limit(request, db, ANSWER_RATE_LIMIT, user_id=user.id, scope="answer-audio")
        audio_bytes = await audio.read()

        buf = io.BytesIO(audio_bytes)
        buf.name = audio.filename or "audio.webm"

        tr = client.audio.transcriptions.create(
            model=settings.transcribe_model,
            file=buf,
        )

        answer_text = (tr.text or "").strip()
        res = submit_answer(db, user, session_id, answer_text)

        return {
            **res,
            "transcript": answer_text,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio evaluation failed: {str(e)}")


@app.post("/interview/answer/video", response_model=SubmitAnswerResponse)
async def answer_video(
    session_id: int = Form(...),
    video: UploadFile = File(...),
    request: Request = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        if request is not None:
            enforce_rate_limit(request, db, ANSWER_RATE_LIMIT, user_id=user.id, scope="answer-video")
        video_bytes = await video.read()

        buf = io.BytesIO(video_bytes)
        buf.name = video.filename or "video.webm"

        tr = client.audio.transcriptions.create(
            model=settings.transcribe_model,
            file=buf,
        )

        transcript = (tr.text or "").strip()
        res = submit_answer(db, user, session_id, transcript)
        visual_signals = analyze_visual_signals(video_bytes)
        dynamic_analysis = build_dynamic_video_analysis(
            transcript=transcript,
            score=res.get("score"),
            feedback=res.get("feedback", ""),
        )
        dynamic_analysis["sampled_frame_count"] = visual_signals.get("sampled_frame_count", 0)
        dynamic_analysis["visual_signals"] = visual_signals
        dynamic_analysis["visual_confidence_score"] = visual_signals.get("visual_confidence_score", 0)

        return {
            **res,
            "transcript": transcript,
            "video_analysis": dynamic_analysis,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video evaluation failed: {str(e)}")


@app.post("/tts")
def tts(text: str, user: User = Depends(get_current_user)):
    import base64

    clean_text = (text or "").strip()
    if not clean_text:
        return {"audio_base64": None, "fallback": True}

    # Natural US English: shimmer works across tts-1 / tts-1-hd; mini-tts also accepts style via instructions.
    tts_instructions = (
        "Speak in clear, natural American English (United States). "
        "Use a calm, professional interviewer tone. "
        "Pronounce English words as a native US English speaker would; do not use a non-English accent."
    )
    models_to_try: list[str] = []
    for m in (settings.tts_model, "gpt-4o-mini-tts", "tts-1", "tts-1-hd"):
        if m and m not in models_to_try:
            models_to_try.append(m)
    try:
        last_error = None
        for model_name in models_to_try:
            try:
                if "mini-tts" in model_name:
                    speech = client.audio.speech.create(
                        model=model_name,
                        voice="shimmer",
                        input=clean_text[:4096],
                        response_format="mp3",
                        speed=1.0,
                        extra_body={"instructions": tts_instructions},
                    )
                else:
                    speech = client.audio.speech.create(
                        model=model_name,
                        voice="shimmer",
                        input=clean_text[:4096],
                        response_format="mp3",
                        speed=1.0,
                    )
                b64 = base64.b64encode(speech.content).decode("utf-8")
                return {"audio_base64": b64, "model": model_name, "fallback": False}
            except Exception as e:
                last_error = e
                continue
        return {
            "audio_base64": None,
            "fallback": True,
            "detail": f"TTS unavailable: {str(last_error)}" if last_error else "TTS unavailable",
        }
    except Exception:
        return {"audio_base64": None, "fallback": True}


@app.get("/interview/session/{session_id}", response_model=SessionResultResponse)
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    session = db.get(InterviewSession, session_id)

    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")

    return {
        "session_id": session.id,
        "profession": session.profession,
        "created_at": str(session.created_at),
        "result_json": session.result_json,
    }


@app.get("/interview/session/{session_id}/report")
def get_session_report(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    session = db.get(InterviewSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    return build_session_report(session)


@app.get("/rag/memory")
def rag_memory(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    memories = load_user_memory(db, user, limit=50)
    grouped: dict[str, list[dict]] = {}
    for item in memories:
        grouped.setdefault(str(item.get("memory_type") or "signal"), []).append(item)
    return {"memories": memories, "grouped": grouped, "count": len(memories)}


@app.get("/rag/inspector/session/{session_id}")
def rag_inspector_session(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    session = db.get(InterviewSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    result_json = session.result_json or {}
    final_summary = result_json.get("final_summary") or {}
    evidence = final_summary.get("retrieval_evidence") or []
    turns = result_json.get("turns") or []
    latest = turns[-1] if turns else {}
    feedback_text = str(final_summary.get("feedback") or latest.get("feedback") or "")
    answer_text = str(latest.get("answer") or "")
    config = result_json.get("config") or {}
    graph_hits = graph_context_for_query(
        RetrievalQuery(
            purpose="inspector",
            profession=session.profession,
            query=" ".join([str(latest.get("question") or ""), answer_text, feedback_text]),
            company=str(config.get("target_company") or config.get("company_pack") or ""),
            focus_area=str(config.get("focus_area") or ""),
            difficulty=str(config.get("difficulty") or ""),
            user_memory=load_user_memory(db, user),
        )
    )
    return {
        "session_id": session.id,
        "profession": session.profession,
        "query_debug": {
            "purpose": "session_report",
            "focus_area": config.get("focus_area"),
            "difficulty": config.get("difficulty"),
            "company": config.get("target_company") or config.get("company_pack"),
            "latest_question": latest.get("question"),
        },
        "retrieval_quality": final_summary.get("retrieval_quality") or {},
        "rag_summary": final_summary.get("rag_summary") or result_json.get("question_rag", {}).get("summary"),
        "retrieval_evaluation": evaluate_retrieval(evidence, answer_text=answer_text, feedback_text=feedback_text),
        "evidence": evidence,
        "question_rag": result_json.get("question_rag") or {},
        "graph_hits": graph_hits,
        "user_memory": load_user_memory(db, user),
        "low_confidence_warning": bool((final_summary.get("retrieval_quality") or {}).get("label") in {"none", "low"}),
    }


def _story_response(item: StoryVaultItem, retrieval_match: Optional[dict] = None) -> StoryItemResponse:
    return StoryItemResponse(
        id=item.id,
        session_id=item.session_id,
        title=item.title,
        tags=item.tags or [],
        question=item.question,
        answer=item.answer,
        score=item.score,
        created_at=str(item.created_at),
        retrieval_match=retrieval_match,
    )


def _preference_response(item: Optional[UserPreference], user: User) -> UserPreferenceResponse:
    return UserPreferenceResponse(
        target_company=item.target_company if item else "",
        interview_date=item.interview_date if item else "",
        default_mode=item.default_mode if item else "text",
        focus_area=item.focus_area if item else "Mixed",
        difficulty=item.difficulty if item else "Junior",
        updated_at=str(item.updated_at) if item and item.updated_at else None,
    )


@app.get("/stories", response_model=StoryListResponse)
def list_stories(
    q: Optional[str] = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    query = db.query(StoryVaultItem).filter(StoryVaultItem.user_id == user.id)
    items = query.order_by(StoryVaultItem.id.desc()).limit(100).all()
    needle = (q or "").strip().lower()
    matches_by_id: dict[int, dict] = {}
    if needle:
        story_docs = [
            {
                "id": item.id,
                "title": item.title,
                "question": item.question,
                "answer": item.answer,
                "tags": item.tags or [],
                "item": item,
            }
            for item in items
        ]
        ranked = rank_story_candidates(needle, story_docs, k=100)
        if ranked:
            items = [entry["item"] for entry in ranked]
            matches_by_id = {int(entry["id"]): entry.get("retrieval_match", {}) for entry in ranked}
        else:
            items = [
                item
                for item in items
                if needle in item.title.lower()
                or needle in (item.answer or "").lower()
                or needle in (item.question or "").lower()
                or any(needle in str(tag).lower() for tag in (item.tags or []))
            ]
    quality = {
        "label": "medium" if needle and matches_by_id else "none" if needle else "not_requested",
        "score": 70 if matches_by_id else 0,
        "evidence_count": len(matches_by_id),
    }
    summary = (
        f"Story Vault semantic search reranked {len(matches_by_id)} private stories for '{needle}'."
        if needle and matches_by_id
        else "Story Vault returned recent stories without semantic rerank."
    )
    return {"stories": [_story_response(item, matches_by_id.get(item.id)) for item in items], "rag_summary": summary, "retrieval_quality": quality}


@app.post("/stories", response_model=StoryItemResponse)
def create_story(
    payload: StoryCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if payload.session_id is not None:
        session = db.get(InterviewSession, payload.session_id)
        if not session or session.user_id != user.id:
            raise HTTPException(status_code=404, detail="Session not found")
    item = StoryVaultItem(
        user_id=user.id,
        session_id=payload.session_id,
        title=payload.title.strip(),
        tags=[str(tag).strip() for tag in payload.tags if str(tag).strip()][:12],
        question=(payload.question or "").strip() or None,
        answer=payload.answer.strip(),
        score=payload.score,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return _story_response(item)


@app.delete("/stories/{story_id}")
def delete_story(
    story_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    item = db.get(StoryVaultItem, story_id)
    if not item or item.user_id != user.id:
        raise HTTPException(status_code=404, detail="Story not found")
    db.delete(item)
    db.commit()
    return {"ok": True}


@app.get("/account/preferences", response_model=UserPreferenceResponse)
def get_account_preferences(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    item = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    return _preference_response(item, user)


@app.put("/account/preferences", response_model=UserPreferenceResponse)
def update_account_preferences(
    payload: UserPreferenceRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    item = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    if not item:
        item = UserPreference(user_id=user.id)
    item.target_company = (payload.target_company or "").strip()
    item.interview_date = (payload.interview_date or "").strip()
    item.default_mode = payload.default_mode
    item.focus_area = payload.focus_area
    item.difficulty = payload.difficulty
    db.add(item)
    db.commit()
    db.refresh(item)
    return _preference_response(item, user)


@app.get("/account/summary")
def account_summary(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sessions = db.query(InterviewSession).filter(InterviewSession.user_id == user.id).all()
    stories = db.query(StoryVaultItem).filter(StoryVaultItem.user_id == user.id).all()
    preferences = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    completed_drills = (
        db.query(DrillCompletion)
        .filter(DrillCompletion.user_id == user.id, DrillCompletion.completed == 1)
        .count()
    )
    completed = 0
    total_turns = 0
    scores: list[int] = []
    for session in sessions:
        result_json = session.result_json or {}
        if result_json.get("status") == "completed":
            completed += 1
        turns = result_json.get("turns") or []
        total_turns += len(turns)
        if isinstance(result_json.get("average_score"), int):
            scores.append(result_json["average_score"])
    return {
        "user": {"id": user.id, "name": user.name, "email": user.email, "profession": user.profession},
        "usage": {
            "sessions": len(sessions),
            "completed_sessions": completed,
            "turns": total_turns,
            "stories": len(stories),
            "completed_drills": completed_drills,
            "average_score": int(sum(scores) / len(scores)) if scores else None,
        },
        "preferences": _preference_response(preferences, user).model_dump(),
        "privacy": {
            "cv_processing": "Uploaded CV files are processed for suggestions and not stored as files by this API.",
            "interview_data": "Interview questions, answers, scores, reports, and saved stories are stored under your account.",
            "delete_data_endpoint": "/account/data",
        },
    }


@app.delete("/account/data")
def delete_account_data(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    stories = db.query(StoryVaultItem).filter(StoryVaultItem.user_id == user.id).all()
    sessions = db.query(InterviewSession).filter(InterviewSession.user_id == user.id).all()
    drills = db.query(DrillCompletion).filter(DrillCompletion.user_id == user.id).all()
    deleted = {"stories": len(stories), "sessions": len(sessions), "drills": len(drills)}
    for item in stories:
        db.delete(item)
    for session in sessions:
        db.delete(session)
    for drill in drills:
        db.delete(drill)
    db.commit()
    audit_event("account.data.delete", user_id=user.id, detail=deleted)
    return {"ok": True, "deleted": deleted}


@app.get("/analytics/progress")
def progress_analytics(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == user.id)
        .order_by(InterviewSession.id.asc())
        .all()
    )
    scores: list[int] = []
    focus_totals: dict[str, list[int]] = {}
    timeline = []
    for session in sessions:
        result_json = session.result_json or {}
        config = result_json.get("config") or {}
        turns = result_json.get("turns") or []
        score = result_json.get("average_score")
        if isinstance(score, int):
            scores.append(score)
            timeline.append(
                {
                    "session_id": session.id,
                    "date": str(session.created_at)[:10],
                    "score": score,
                    "mode": config.get("mode") or "text",
                    "focus_area": config.get("focus_area") or "Mixed",
                }
            )
        for turn in turns:
            turn_score = turn.get("score")
            focus = (turn.get("focus_area") or config.get("focus_area") or "Mixed").strip()
            if isinstance(turn_score, int):
                focus_totals.setdefault(focus, []).append(turn_score)

    focus_breakdown = [
        {"focus": focus, "average_score": int(sum(items) / len(items)), "turns": len(items)}
        for focus, items in sorted(focus_totals.items())
        if items
    ]
    trend = None
    if len(scores) >= 2:
        trend = scores[-1] - scores[0]
    return {
        "summary": {
            "sessions": len(sessions),
            "scored_sessions": len(scores),
            "average_score": int(sum(scores) / len(scores)) if scores else None,
            "best_score": max(scores) if scores else None,
            "trend": trend,
        },
        "timeline": timeline[-12:],
        "focus_breakdown": focus_breakdown,
        "next_best_action": (
            "Run one focused drill on your lowest scoring area."
            if focus_breakdown
            else "Complete a short instant practice to unlock trend analytics."
        ),
    }


def _normalized_question_key(question: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    words = [word for word in text.split() if len(word) > 2]
    return " ".join(words[:18])


@app.get("/quality/questions")
def question_quality_metrics(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == user.id)
        .order_by(InterviewSession.id.desc())
        .limit(20)
        .all()
    )
    questions: list[dict[str, str]] = []
    for session in sessions:
        result_json = session.result_json or {}
        config = result_json.get("config") or {}
        candidates = []
        current = str(result_json.get("current_question") or "").strip()
        if current:
            candidates.append(current)
        for turn in result_json.get("turns") or []:
            if isinstance(turn, dict):
                question = str(turn.get("question") or "").strip()
                if question:
                    candidates.append(question)
        for question in candidates:
            questions.append(
                {
                    "question": question,
                    "focus_area": str(config.get("focus_area") or "Mixed"),
                    "mode": str(config.get("mode") or "text"),
                    "company_pack": str(config.get("company_pack") or "general"),
                }
            )

    keys = [_normalized_question_key(item["question"]) for item in questions if item["question"]]
    unique_keys = set(keys)
    duplicate_count = max(0, len(keys) - len(unique_keys))
    by_focus: dict[str, int] = {}
    by_mode: dict[str, int] = {}
    for item in questions:
        by_focus[item["focus_area"]] = by_focus.get(item["focus_area"], 0) + 1
        by_mode[item["mode"]] = by_mode.get(item["mode"], 0) + 1

    freshness_score = 100 if not keys else int((len(unique_keys) / len(keys)) * 100)
    return {
        "summary": {
            "sessions_scanned": len(sessions),
            "questions_scanned": len(keys),
            "unique_questions": len(unique_keys),
            "duplicate_count": duplicate_count,
            "freshness_score": freshness_score,
            "recent_session_guard": "Avoids questions from the current session and recent sessions.",
        },
        "by_focus": by_focus,
        "by_mode": by_mode,
        "recent_questions": [item["question"] for item in questions[:10]],
        "recommendation": (
            "Question freshness is healthy."
            if freshness_score >= 85
            else "Run more focused/company-specific sessions to refresh the question mix."
        ),
    }


@app.get("/account/usage-guards")
def usage_guards(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    scopes = {
        "answer-text": ANSWER_RATE_LIMIT,
        "answer-audio": ANSWER_RATE_LIMIT,
        "answer-video": ANSWER_RATE_LIMIT,
        "answer-pass": ANSWER_RATE_LIMIT,
        "interview-hint": ANSWER_RATE_LIMIT,
        "auth-login": AUTH_RATE_LIMIT,
    }
    now_ts = time.time()
    rows = []
    for scope, config in scopes.items():
        key_prefix = f"{scope}:{user.id}:"
        used = (
            db.query(RateLimitEvent)
            .filter(
                RateLimitEvent.scope_key.like(f"{key_prefix}%"),
                RateLimitEvent.ts >= now_ts - config.window_seconds,
            )
            .count()
            if settings.rate_limit_persist
            else None
        )
        rows.append(
            {
                "scope": scope,
                "max_requests": config.max_requests,
                "window_seconds": config.window_seconds,
                "used_in_window": used,
                "guarded": True,
            }
        )
    return {
        "limits": rows,
        "cost_controls": [
            "Answer, pass, hint, audio, and video paths are rate limited.",
            "TTS uses configured model fallback and authenticated access.",
            "Roadmap and weekly drills are generated from deterministic product logic.",
        ],
        "status": "active",
    }


@app.post("/demo/seed")
def seed_demo_data(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    existing_sessions = db.query(InterviewSession).filter(InterviewSession.user_id == user.id).count()
    existing_stories = db.query(StoryVaultItem).filter(StoryVaultItem.user_id == user.id).count()
    if existing_sessions or existing_stories:
        return {"ok": True, "created": {"sessions": 0, "stories": 0}, "message": "Demo data already exists."}

    demo_session = InterviewSession(
        user_id=user.id,
        profession=user.profession,
        result_json={
            "status": "completed",
            "average_score": 82,
            "config": {
                "mode": "text",
                "difficulty": "Mid",
                "focus_area": "Behavioral",
                "company_pack": "general",
                "instant_mode": False,
            },
            "turns": [
                {
                    "question": "Tell me about a time you improved a process under pressure.",
                    "answer": "I mapped the bottleneck, aligned the team on one owner, and measured cycle time weekly.",
                    "score": 82,
                    "feedback": "Strong structure and measurable impact. Add one tradeoff you considered.",
                    "focus_area": "Behavioral",
                }
            ],
        },
    )
    db.add(demo_session)
    db.commit()
    db.refresh(demo_session)
    db.add(
        InterviewTurn(
            session_id=demo_session.id,
            question="Tell me about a time you improved a process under pressure.",
            answer_text="I mapped the bottleneck, aligned the team on one owner, and measured cycle time weekly.",
            ai_feedback="Strong structure and measurable impact. Add one tradeoff you considered.",
            score=82,
        )
    )
    db.add(
        StoryVaultItem(
            user_id=user.id,
            session_id=demo_session.id,
            title="Process improvement under pressure",
            tags=["behavioral", "ownership", "impact"],
            question="Tell me about a time you improved a process under pressure.",
            answer="Situation: A release process was blocking weekly delivery.\nTask: I needed to reduce handoff delay.\nAction: I mapped the bottleneck, assigned one clear owner, and introduced a weekly cycle-time review.\nResult: The team shipped more predictably and escalations dropped.",
            score=82,
        )
    )
    db.commit()
    return {"ok": True, "created": {"sessions": 1, "stories": 1}}


@app.get("/interview/sessions", response_model=SessionListResponse)
def list_sessions(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == user.id)
        .order_by(InterviewSession.id.desc())
        .all()
    )

    items = []
    for session in sessions:
        result_json = session.result_json or {}
        turns = result_json.get("turns") or []

        items.append(
            {
                "session_id": session.id,
                "profession": session.profession,
                "created_at": str(session.created_at),
                "average_score": result_json.get("average_score"),
                "done": result_json.get("status") == "completed",
                "completed": result_json.get("status") == "completed",
                "turn_count": len(turns),
                "config": result_json.get("config"),
                "current_question": result_json.get("current_question"),
            }
        )

    return {"sessions": items}


@app.get("/recruiter/compare")
def recruiter_compare(
    session_ids: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    parsed_ids = [int(x) for x in session_ids.split(",") if x.strip().isdigit()]
    if len(parsed_ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 session ids")
    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == user.id, InterviewSession.id.in_(parsed_ids))
        .all()
    )
    rows = []
    for s in sessions:
        rj = s.result_json or {}
        summary = rj.get("final_summary") or {}
        rows.append(
            {
                "session_id": s.id,
                "profession": s.profession,
                "average_score": rj.get("average_score"),
                "final_score": summary.get("score"),
                "confidence_score": summary.get("confidence_score"),
                "red_flags": summary.get("red_flags", []),
                "strengths": summary.get("strengths", []),
                "weaknesses": summary.get("weaknesses", []),
            }
        )
    rows.sort(key=lambda x: (x.get("final_score") or 0), reverse=True)
    return {"items": rows}


@app.get("/benchmark/health")
def benchmark_health(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == user.id)
        .order_by(InterviewSession.id.desc())
        .limit(20)
        .all()
    )
    completed = []
    for s in sessions:
        rj = s.result_json or {}
        if rj.get("status") != "completed":
            continue
        summary = rj.get("final_summary") or {}
        completed.append(
            {
                "score": int(summary.get("score", rj.get("average_score") or 0)),
                "confidence": int(summary.get("confidence_score", 0) or 0),
                "red_flags": len(summary.get("red_flags", []) or []),
            }
        )
    if not completed:
        return {"status": "insufficient_data", "detail": "No completed sessions found"}
    avg_score = int(sum(x["score"] for x in completed) / len(completed))
    avg_conf = int(sum(x["confidence"] for x in completed) / len(completed))
    avg_flags = round(sum(x["red_flags"] for x in completed) / len(completed), 2)
    healthy = avg_conf >= 55 and avg_score >= 50 and avg_flags <= 3
    return {
        "status": "healthy" if healthy else "needs_attention",
        "sample_size": len(completed),
        "avg_score": avg_score,
        "avg_confidence": avg_conf,
        "avg_red_flags": avg_flags,
    }


@app.get("/benchmark/regression")
def benchmark_regression(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == user.id)
        .order_by(InterviewSession.id.desc())
        .limit(40)
        .all()
    )
    return build_regression_snapshot(sessions)