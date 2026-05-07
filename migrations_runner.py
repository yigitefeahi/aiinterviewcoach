"""Apply Alembic migrations on startup without breaking legacy SQLite DBs."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect

from .config import settings
from .db import engine


def _alembic_ini_path() -> Path:
    return Path(__file__).resolve().parent.parent / "alembic.ini"


def alembic_config() -> Config:
    cfg = Config(str(_alembic_ini_path()))
    cfg.set_main_option("sqlalchemy.url", settings.database_url)
    return cfg


def _revision_head(cfg: Config) -> str:
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    if len(heads) != 1:
        raise RuntimeError(f"Expected a single alembic head, got {heads!r}")
    return heads[0]


def apply_migrations() -> None:
    cfg = alembic_config()
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    with engine.connect() as conn:
        from alembic.runtime.migration import MigrationContext

        ctx = MigrationContext.configure(conn)
        rev = ctx.get_current_revision()

    if rev is not None:
        command.upgrade(cfg, "head")
        return

    if "users" in tables:
        # Created before Alembic: schema matches baseline revision; stamp without re-running DDL.
        command.stamp(cfg, _revision_head(cfg))
        return

    command.upgrade(cfg, "head")
