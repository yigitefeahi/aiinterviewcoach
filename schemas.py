from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    profession: str = Field(min_length=2, max_length=80)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserMeResponse(BaseModel):
    id: int
    email: EmailStr
    name: str
    profession: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    csrf_token: str


class ProfessionListResponse(BaseModel):
    professions: List[str]


class SectorListResponse(BaseModel):
    sectors: List[str]


class CVRoleSuggestionResponse(BaseModel):
    suggested_professions: List[str]
    suggested_sectors: List[str]
    rationale: str
    method: Literal[
        "keyword_heuristic",
        "keyword_heuristic_plus_llm_refine",
        "keyword_embedding_fusion",
        "keyword_embedding_fusion_plus_llm_refine",
        "keyword_heuristic_plus_screening",
        "keyword_embedding_fusion_plus_screening",
    ] = "keyword_heuristic"
    limitations: str = Field(
        default="Keyword overlap only (no embedding model or structured résumé parse). "
        "Upload plain text when possible; raw PDF bytes often decode poorly."
    )
    cv_structure: Optional[Dict[str, Any]] = None
    embedding_top_professions: Optional[List[Dict[str, Any]]] = None
    role_fit_breakdown: Optional[Dict[str, Any]] = None
    evaluator: Optional[Dict[str, Any]] = None
    retrieval_evidence: Optional[List[Dict[str, Any]]] = None
    rag_summary: Optional[str] = None
    retrieval_quality: Optional[Dict[str, Any]] = None
    citations: Optional[List[Dict[str, Any]]] = None
    citation_notes: Optional[List[str]] = None
    rag_evaluation: Optional[Dict[str, Any]] = None


class StartSessionRequest(BaseModel):
    profession: Optional[str] = Field(default=None, min_length=2, max_length=80)
    difficulty: Literal["Junior", "Mid", "Senior"] = "Junior"
    mode: Literal["text", "audio", "video", "presence", "case"] = "text"
    interview_length: Literal["5 Questions", "10 Questions", "15 Questions", "20 Minutes", "30 Minutes"] = "10 Questions"
    focus_area: Literal[
        "Mixed",
        "Technical",
        "Behavioral",
        "System Design",
        "Problem Solving",
        "Product Sense",
        "Communication",
        "Market Sizing",
    ] = "Mixed"
    sector: Optional[str] = Field(default=None, max_length=80)
    target_company: Optional[str] = Field(default=None, max_length=120)
    company_pack: Optional[str] = Field(default=None, max_length=80)
    instant_mode: bool = False
    interview_date: Optional[str] = Field(default=None, max_length=20)
    case_type: Optional[Literal["product_sense", "system_design", "market_sizing"]] = None


class StartSessionResponse(BaseModel):
    session_id: int
    first_question: str
    question_context: Optional[str] = None
    config: Dict[str, Any]


class SubmitTextRequest(BaseModel):
    session_id: int
    answer_text: str = Field(min_length=2, max_length=10000)

    @field_validator("answer_text")
    @classmethod
    def strip_answer_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("answer_text cannot be empty")
        return cleaned


class PassQuestionRequest(BaseModel):
    session_id: int


class SubmitAnswerResponse(BaseModel):
    next_question: Optional[str]
    feedback: str
    score: int
    done: bool
    question_context: Optional[str] = None
    sub_scores: Optional[Dict[str, int]] = None
    strengths: Optional[List[str]] = None
    weaknesses: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None
    recommended_next_steps: Optional[List[str]] = None
    retrieval_evidence: Optional[List[Dict[str, Any]]] = None
    transcript: Optional[str] = None
    video_analysis: Optional[Dict[str, Any]] = None
    can_retry: Optional[bool] = None
    attempts_left: Optional[int] = None
    pending_next_question: Optional[str] = None
    confidence_score: Optional[int] = None
    red_flags: Optional[List[str]] = None
    passes_left: Optional[int] = None
    question_index: Optional[int] = None
    total_questions: Optional[int] = None
    score_explanation: Optional[str] = None
    scorecard: Optional[Dict[str, int]] = None
    tone_signals: Optional[Dict[str, Any]] = None
    company_rubric: Optional[Dict[str, Any]] = None
    rag_summary: Optional[str] = None
    retrieval_quality: Optional[Dict[str, Any]] = None


class HintRequest(BaseModel):
    session_id: int


class HintResponse(BaseModel):
    question: str
    hint: str
    bullets: List[str]
    retrieval_evidence: Optional[List[Dict[str, Any]]] = None
    rag_summary: Optional[str] = None
    retrieval_quality: Optional[Dict[str, Any]] = None


class RoadmapResponse(BaseModel):
    profession: str
    target_company: str
    interview_date: str
    days_left: int
    schedule: List[Dict[str, Any]]
    retrieval_evidence: Optional[List[Dict[str, Any]]] = None
    rag_summary: Optional[str] = None
    retrieval_quality: Optional[Dict[str, Any]] = None


class WeeklyDrillsResponse(BaseModel):
    profession: str
    target_company: str
    interview_date: str
    weeks: int
    drills: List[Dict[str, Any]]
    retrieval_evidence: Optional[List[Dict[str, Any]]] = None
    rag_summary: Optional[str] = None
    retrieval_quality: Optional[Dict[str, Any]] = None


class UserPreferenceRequest(BaseModel):
    target_company: Optional[str] = Field(default=None, max_length=120)
    interview_date: Optional[str] = Field(default=None, max_length=20)
    default_mode: Literal["text", "audio", "video", "presence", "case"] = "text"
    focus_area: Literal[
        "Mixed",
        "Technical",
        "Behavioral",
        "System Design",
        "Problem Solving",
        "Product Sense",
        "Communication",
        "Market Sizing",
    ] = "Mixed"
    difficulty: Literal["Junior", "Mid", "Senior"] = "Junior"


class UserPreferenceResponse(UserPreferenceRequest):
    updated_at: Optional[str] = None


class DrillCompletionRequest(BaseModel):
    drill_key: str = Field(min_length=2, max_length=255)
    completed: bool = True


class DrillCompletionResponse(BaseModel):
    completions: Dict[str, bool]


class StoryCreateRequest(BaseModel):
    session_id: Optional[int] = None
    title: str = Field(min_length=2, max_length=160)
    tags: List[str] = Field(default_factory=list)
    question: Optional[str] = None
    answer: str = Field(min_length=2, max_length=10000)
    score: Optional[int] = Field(default=None, ge=0, le=100)


class StoryItemResponse(BaseModel):
    id: int
    session_id: Optional[int] = None
    title: str
    tags: List[str]
    question: Optional[str] = None
    answer: str
    score: Optional[int] = None
    created_at: str
    retrieval_match: Optional[Dict[str, Any]] = None


class StoryListResponse(BaseModel):
    stories: List[StoryItemResponse]
    rag_summary: Optional[str] = None
    retrieval_quality: Optional[Dict[str, Any]] = None


class EvaluationReliabilityRequest(BaseModel):
    session_id: int
    answer_text: str
    runs: int = Field(default=3, ge=2, le=6)


class RagComparisonRequest(BaseModel):
    session_id: int
    answer_text: str


class SessionResultResponse(BaseModel):
    session_id: int
    profession: str
    created_at: str
    result_json: Optional[Dict[str, Any]]


class SessionListItem(BaseModel):
    session_id: int
    profession: str
    created_at: str
    average_score: Optional[int] = None
    done: bool = False
    completed: bool = False
    turn_count: int = 0
    config: Optional[Dict[str, Any]] = None
    current_question: Optional[str] = None


class SessionListResponse(BaseModel):
    sessions: List[SessionListItem]