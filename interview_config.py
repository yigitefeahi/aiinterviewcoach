from typing import Any, Optional
import json
import random
import re

from .openai_client import client
from .config import settings
from .product_features import build_case_question_prefix, build_company_prompt_context, normalize_company_pack
from .role_profiles import get_role_profile

QUESTION_BANK: dict[str, dict[str, list[str]]] = {
    "Junior": {
        "Technical": [
            "Explain a recent bug you fixed and how you validated the fix.",
            "Walk through a small feature you built from requirement to release.",
            "How would you investigate a page that loads slowly for some users but not others?",
            "Describe how you would test a form submission flow end to end.",
            "What steps would you take before deploying a small but user-facing change?",
            "Explain how you would debug an API response that sometimes returns incomplete data.",
            "How do you decide whether a bug fix needs a unit test, integration test, or manual check?",
            "Tell me how you would improve the reliability of a feature you recently shipped.",
        ],
        "Behavioral": [
            "Describe a time you received critical feedback. How did you respond?",
            "Tell me about a teammate conflict and how you handled communication.",
            "Tell me about a time you had to learn something quickly to finish a task.",
            "Describe a situation where you missed something important and how you fixed it.",
            "Give an example of when you asked for help effectively.",
            "Tell me about a time you had to explain a technical issue to a non-technical person.",
            "Describe a time you had competing tasks and how you prioritized them.",
            "Tell me about a small improvement you made that helped your team work better.",
        ],
    },
    "Mid": {
        "Technical": [
            "Describe a trade-off you made between performance and maintainability.",
            "How would you debug a production issue with limited logs?",
            "Design a plan to reduce flaky failures in an important workflow.",
            "How would you migrate a feature without disrupting current users?",
            "Describe a technical decision where you rejected a popular approach and why.",
            "How would you measure whether a backend or frontend optimization actually worked?",
            "Walk through how you would split a large feature into safe delivery milestones.",
            "Explain how you would handle a data consistency issue across two services.",
        ],
        "Behavioral": [
            "Share an example where you influenced a technical decision without authority.",
            "Tell me about a project where priorities changed mid-sprint.",
            "Tell me about a time you pushed back on scope to protect quality.",
            "Describe a time you mentored someone or helped unblock a teammate.",
            "Give an example of a project where communication was the main risk.",
            "Tell me about a time you made a wrong technical call and recovered.",
            "Describe how you handled ambiguity in a project with unclear requirements.",
            "Tell me about a time you improved a process without being asked.",
        ],
    },
    "Senior": {
        "Technical": [
            "Design a scalable approach for a feature expected to grow 10x in usage.",
            "How do you decide where to place boundaries between services or modules?",
            "How would you design a migration strategy for a critical system with no downtime?",
            "Describe how you would evaluate build-versus-buy for an important platform capability.",
            "How would you diagnose a reliability issue that spans product, infra, and data layers?",
            "Explain how you would set technical direction when multiple teams depend on the outcome.",
            "Design an observability strategy for a workflow with unclear failure modes.",
            "How would you reduce architectural complexity without slowing product delivery?",
        ],
        "Behavioral": [
            "Describe a time you coached a struggling teammate to a better outcome.",
            "Tell me about a tough stakeholder conversation and how you handled it.",
            "Tell me about a time you changed leadership's mind using evidence.",
            "Describe a time you took ownership of a problem outside your direct scope.",
            "Give an example of when you had to balance speed, quality, and team morale.",
            "Tell me about a time you handled disagreement between senior peers.",
            "Describe a situation where you created alignment across teams with different incentives.",
            "Tell me about a time your decision had long-term consequences and how you managed them.",
        ],
    },
}


def normalize_config(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    config = config or {}
    company_pack = normalize_company_pack(config.get("target_company"), config.get("company_pack"))
    return {
        "difficulty": str(config.get("difficulty", "Junior")).strip() or "Junior",
        "mode": str(config.get("mode", "text")).strip() or "text",
        "interview_length": str(config.get("interview_length", "10 Questions")).strip() or "10 Questions",
        "focus_area": str(config.get("focus_area", "Mixed")).strip() or "Mixed",
        "sector": str(config.get("sector", "")).strip(),
        "target_company": str(config.get("target_company", "")).strip(),
        "company_pack": company_pack,
        "instant_mode": bool(config.get("instant_mode", False)),
        "interview_date": str(config.get("interview_date", "")).strip(),
        "case_type": str(config.get("case_type", "") or "").strip(),
    }


def target_question_count(interview_length: str) -> int:
    value = interview_length.lower().strip()
    if "15" in value:
        return 15
    if "5" in value:
        return 5
    if "10" in value:
        return 10
    if "20 minute" in value:
        return 6
    if "30 minute" in value:
        return 8
    return 5


def build_question_context(profession: str, config: dict[str, Any]) -> str:
    """Single line for UI: role, level, focus, optional sector/company — not part of the spoken question."""
    difficulty = config["difficulty"]
    focus_area = config["focus_area"]
    parts = [profession, difficulty, focus_area]
    if config.get("sector"):
        parts.append(f"Sector: {config['sector']}")
    if config.get("target_company"):
        parts.append(f"Target company: {config['target_company']}")
    if config.get("mode") == "case":
        parts.append(f"Case: {config.get('case_type') or 'product_sense'}")
    return " · ".join(parts)


def _case_questions(case_type: str) -> list[str]:
    return {
        "product_sense": [
            "How would you improve the onboarding experience for a product with declining activation?",
            "Design a success metric framework for a new AI coaching feature.",
            "A mobile app has strong signups but weak week-two retention. How would you diagnose and improve it?",
            "How would you prioritize the next three features for a B2B SaaS dashboard?",
            "Design an experiment to test whether a new recommendation feature improves user outcomes.",
            "A marketplace has a supply-demand imbalance. How would you identify the root cause and respond?",
            "How would you launch a premium tier for a free productivity product?",
            "Define the north-star metric and guardrail metrics for an interview practice platform.",
        ],
        "system_design": [
            "Design a scalable interview practice platform that supports live audio sessions and feedback.",
            "Design a rate-limited question bank service for company-specific interview loops.",
            "Design a notification system that sends personalized prep reminders without spamming users.",
            "Design a story vault search service for saved interview answers.",
            "Design a real-time transcription and feedback pipeline for mock interviews.",
            "Design a reporting service that aggregates interview scorecards across sessions.",
            "Design a multi-tenant company rubric service with versioned question packs.",
            "Design a resilient file upload and CV parsing system.",
        ],
        "market_sizing": [
            "Estimate the annual market size for AI interview preparation tools in the US.",
            "Estimate how many mock interview sessions are completed globally each month.",
            "Estimate the market size for paid technical interview coaching in Europe.",
            "Estimate how many candidates apply to software engineering jobs each year in the US.",
            "Estimate the revenue opportunity for an AI career coach targeting university students.",
            "Estimate how many hours professionals spend preparing for job interviews each year.",
            "Estimate the total addressable market for resume analysis tools.",
            "Estimate the potential usage of peer mock interview rooms among bootcamp graduates.",
        ],
    }.get(case_type, [])


def _flatten_bank_questions(difficulty: str, focus_area: str) -> list[str]:
    bank = QUESTION_BANK.get(difficulty, QUESTION_BANK["Junior"])
    if focus_area.lower() == "mixed":
        return bank.get("Technical", []) + bank.get("Behavioral", [])
    normalized_focus = "Behavioral" if focus_area.lower() == "behavioral" else "Technical"
    return bank.get(normalized_focus, bank["Technical"])


def _role_profile_questions(profession: str, config: dict[str, Any]) -> list[str]:
    profile = get_role_profile(profession)
    themes = [str(item) for item in profile.get("themes", [])[:5]]
    evaluation_focus = [str(item) for item in profile.get("evaluation_focus", [])[:4]]
    questions: list[str] = []
    for theme in themes:
        questions.append(
            f"Tell me about a {profession} challenge involving {theme}. What tradeoff did you make, how did you validate it, and what was the measurable outcome?"
        )
        questions.append(
            f"How would you approach a {theme} problem as a {profession}, including constraints, risks, and success metrics?"
        )
    if evaluation_focus:
        questions.append(
            f"Describe a project where you demonstrated {', '.join(evaluation_focus[:2])} as a {profession}. What evidence shows it worked?"
        )
    return questions


def choose_fresh_question(candidates: list[str], avoid_questions: list[str]) -> str:
    candidates = [q for q in candidates if q and q.strip()]
    if not candidates:
        return "Tell me about a challenging situation you handled and what impact you made."
    shuffled = candidates[:]
    random.shuffle(shuffled)
    for candidate in shuffled:
        if not is_similar_to_any(candidate, avoid_questions, threshold=0.66):
            return candidate
    scored = [
        (
            max((_jaccard_similarity(candidate, previous) for previous in avoid_questions), default=0.0),
            candidate,
        )
        for candidate in shuffled
    ]
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def generate_bank_question(
    profession: str,
    config: dict[str, Any],
    avoid_questions: Optional[list[str]] = None,
) -> str:
    avoid_questions = avoid_questions or []
    if config.get("mode") == "case":
        case_type = config.get("case_type") or "product_sense"
        picked = choose_fresh_question(_case_questions(case_type), avoid_questions)
        return f"{build_case_question_prefix(config)}{picked}"
    difficulty = config["difficulty"]
    focus_area = config["focus_area"]
    candidates = _role_profile_questions(profession, config) + _flatten_bank_questions(difficulty, focus_area)
    return choose_fresh_question(candidates, avoid_questions)


def generate_first_question(
    profession: str,
    config: dict[str, Any],
    avoid_questions: Optional[list[str]] = None,
) -> tuple[str, str]:
    """Returns (context_line, question_text_only) so UI can separate session info from the actual ask."""
    picked = generate_bank_question(profession, config, avoid_questions=avoid_questions)
    context = build_question_context(profession, config)
    return context, picked


def _jaccard_similarity(a: str, b: str) -> float:
    at = set(re.findall(r"[a-zA-Z0-9_]+", (a or "").lower()))
    bt = set(re.findall(r"[a-zA-Z0-9_]+", (b or "").lower()))
    if not at or not bt:
        return 0.0
    return len(at & bt) / max(1, len(at | bt))


def is_similar_to_any(candidate: str, asked_questions: list[str], threshold: float = 0.72) -> bool:
    return any(_jaccard_similarity(candidate, q) >= threshold for q in asked_questions if q)


def extract_topic_hint(question: str) -> str:
    words = re.findall(r"[a-zA-Z0-9_]+", (question or "").lower())
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "about", "your", "you", "are", "how", "what", "when",
        "where", "why", "which", "would", "could", "should", "into", "role", "interview", "question", "please",
    }
    filtered = [w for w in words if len(w) > 4 and w not in stop]
    return " ".join(filtered[:4]) if filtered else "general"


def generate_dynamic_question(
    profession: str,
    config: dict[str, Any],
    asked_topics: list[str],
    asked_questions: list[str],
    retrieval_context: str = "",
    rag_summary: str = "",
) -> Optional[str]:
    payload = {
        "profession": profession,
        "difficulty": config.get("difficulty"),
        "focus_area": config.get("focus_area"),
        "sector": config.get("sector"),
        "target_company": config.get("target_company"),
        "company_context": build_company_prompt_context(config),
        "role_profile": get_role_profile(profession),
        "case_type": config.get("case_type"),
        "asked_topics": asked_topics[-12:],
        "asked_questions": asked_questions[-24:],
        "retrieved_question_context": retrieval_context,
        "rag_summary": rag_summary,
        "user_memory_signals": config.get("user_memory", [])[:8],
        "cv_facts": config.get("cv_facts", [])[:8],
        "instructions": (
            "Generate one fresh interview question only. "
            "Avoid repeating previous topics or any question semantically similar to asked_questions. "
            "Keep it realistic and challenging. "
            "If sector/company/company_context is provided, include domain-relevant constraints. "
            "Use retrieved_question_context when it provides relevant role, rubric, framework, or company evidence. "
            "Use role_profile themes and evaluation_focus to make the question specific to the selected profession. "
            "Use user_memory_signals and cv_facts to target the user's gaps without exposing private details verbatim. "
            "Prefer questions that test missing evidence: metrics, tradeoffs, validation, ownership, or role-specific skills. "
            "If mode is case, ask a case-style prompt with enough context to solve."
        ),
    }
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You output only valid JSON with one key: question. "
                        "The value must be a single interview question only — no preamble, no 'you are interviewing', "
                        "no restatement of role/sector/company (those are shown separately in the UI). "
                        "English only. One clear paragraph at most."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.65,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw) if raw else {}
        q = str(data.get("question", "")).strip()
        if q and len(q) > 15:
            return q
    except Exception:
        return None
    return None


def dedupe_question(candidate: str, asked_questions: list[str], fallback: str) -> str:
    if not candidate:
        return fallback
    if is_similar_to_any(candidate, asked_questions[-30:], threshold=0.68):
        return fallback
    return candidate
