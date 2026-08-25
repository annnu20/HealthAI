"""
Persistent storage layer using SQLite (Python's built-in sqlite3 — no
extra service to install or configure).

Adds what the original notebook didn't have: real user accounts and a
prediction history that survives across sessions, instead of living only
in an in-memory Python list (the notebook's `prediction_history`).

Two tables:
  - users        : registered patients (username, email, salted+hashed password)
  - predictions  : one row per prediction, linked to the user who ran it

Passwords are never stored in plain text. We use PBKDF2-HMAC-SHA256 (from
the standard library `hashlib` — no extra dependency like bcrypt needed)
with a random per-user salt and 260,000 iterations, which is the same
scheme Django uses by default.
"""

import hashlib
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "app.db")

PBKDF2_ITERATIONS = 260_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@contextmanager
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Creates tables if they don't exist yet. Safe to call on every startup."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symptoms TEXT NOT NULL,
                predicted_disease TEXT NOT NULL,
                confidence REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    ).hex()


def _make_salt() -> bytes:
    return os.urandom(16)


# ---------------------------------------------------------------------------
# User registration / authentication
# ---------------------------------------------------------------------------
class UserAlreadyExistsError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


class ValidationError(Exception):
    pass


def register_user(username: str, email: str, password: str) -> int:
    """
    Creates a new user account. Returns the new user's id.

    Raises ValidationError for bad input, UserAlreadyExistsError if the
    username or email is already taken.
    """
    username = (username or "").strip()
    email = (email or "").strip().lower()

    if len(username) < 3:
        raise ValidationError("Username must be at least 3 characters.")
    if not EMAIL_RE.match(email):
        raise ValidationError("Please enter a valid email address.")
    if len(password or "") < 6:
        raise ValidationError("Password must be at least 6 characters.")

    salt = _make_salt()
    password_hash = _hash_password(password, salt)

    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ? OR email = ?",
            (username, email),
        ).fetchone()
        if existing:
            raise UserAlreadyExistsError(
                "That username or email is already registered."
            )

        cursor = conn.execute(
            """
            INSERT INTO users (username, email, password_hash, password_salt, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                username,
                email,
                password_hash,
                salt.hex(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cursor.lastrowid


def authenticate_user(username_or_email: str, password: str):
    """
    Returns a dict with the user's id/username/email on success, or None
    if the credentials don't match.
    """
    identifier = (username_or_email or "").strip()

    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (identifier, identifier.lower()),
        ).fetchone()

    if row is None:
        return None

    salt = bytes.fromhex(row["password_salt"])
    candidate_hash = _hash_password(password, salt)

    if candidate_hash != row["password_hash"]:
        return None

    return {"id": row["id"], "username": row["username"], "email": row["email"]}


# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------
def save_prediction(user_id: int, symptoms: str, disease: str, confidence):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (user_id, symptoms, predicted_disease, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                symptoms,
                disease,
                confidence,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_prediction_history(user_id: int):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT symptoms, predicted_disease, confidence, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def clear_prediction_history(user_id: int):
    with get_connection() as conn:
        conn.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
