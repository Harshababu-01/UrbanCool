"""SQLite connection helpers for the UrbanCool API."""

from __future__ import annotations

import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
DATABASE_PATH = ROOT_DIR / "data/processed/urbancool.db"


def get_connection() -> sqlite3.Connection:
    """Open a row-factory SQLite connection at the project database path."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection