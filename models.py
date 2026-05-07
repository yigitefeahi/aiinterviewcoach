from typing import Optional, Dict, Any, List

from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, JSON, Float, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))

    profession: Mapped[str] = mapped_column(String(80))

    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())

    sessions: Mapped[List["InterviewSession"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    profession: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Stores full session summary: average_score + turns etc.
    result_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    user: Mapped["User"] = relationship(back_populates="sessions")
    turns: Mapped[List["InterviewTurn"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class RateLimitEvent(Base):
    """Sliding-window rate limit hits (optional persistence; pruned periodically)."""

    __tablename__ = "rate_limit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scope_key: Mapped[str] = mapped_column(String(512), index=True)
    ts: Mapped[float] = mapped_column(Float, index=True)


class InterviewTurn(Base):
    __tablename__ = "interview_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("interview_sessions.id", ondelete="CASCADE"))
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())

    question: Mapped[str] = mapped_column(Text)
    answer_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    ai_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    session: Mapped["InterviewSession"] = relationship(back_populates="turns")


class StoryVaultItem(Base):
    __tablename__ = "story_vault_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[Optional[int]] = mapped_column(ForeignKey("interview_sessions.id", ondelete="SET NULL"), nullable=True)
    title: Mapped[str] = mapped_column(String(160))
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    question: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    answer: Mapped[str] = mapped_column(Text)
    score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserMemoryItem(Base):
    __tablename__ = "user_memory_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[Optional[int]] = mapped_column(ForeignKey("interview_sessions.id", ondelete="SET NULL"), nullable=True)
    memory_type: Mapped[str] = mapped_column(String(80), index=True)
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.55)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True)
    target_company: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    interview_date: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    default_mode: Mapped[str] = mapped_column(String(40), default="text")
    focus_area: Mapped[str] = mapped_column(String(80), default="Mixed")
    difficulty: Mapped[str] = mapped_column(String(40), default="Junior")
    updated_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DrillCompletion(Base):
    __tablename__ = "drill_completions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    drill_key: Mapped[str] = mapped_column(String(255), index=True)
    completed: Mapped[int] = mapped_column(Integer, default=1)
    completed_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())