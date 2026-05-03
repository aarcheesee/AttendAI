"""
db.py — Database layer for AttendAI
All tables live in a single SQLite file (attend.db).

Schema:
  users       — login accounts
  students    — student profile details
  lectures    — lecture sessions
  attendance  — one row per (student, lecture, date)
"""

import sqlite3
import datetime
from werkzeug.security import generate_password_hash

DB_PATH = "attend.db"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # faster concurrent writes
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT UNIQUE NOT NULL,
                password   TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS students (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT UNIQUE NOT NULL,
                class_name TEXT,
                division   TEXT,
                roll_no    TEXT,
                email      TEXT,
                phone      TEXT,
                created_at TEXT DEFAULT (date('now'))
            );

            CREATE TABLE IF NOT EXISTS lectures (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                student    TEXT NOT NULL,
                lecture    TEXT NOT NULL,
                date       TEXT NOT NULL,
                time       TEXT NOT NULL,
                UNIQUE(student, lecture, date)
            );

            CREATE INDEX IF NOT EXISTS idx_att_student  ON attendance(student);
            CREATE INDEX IF NOT EXISTS idx_att_lecture  ON attendance(lecture);
            CREATE INDEX IF NOT EXISTS idx_att_date     ON attendance(date);
        """)

        # Default admin
        conn.execute(
            "INSERT OR IGNORE INTO users (username, password) VALUES (?,?)",
            ("admin", generate_password_hash("admin123"))
        )
        conn.commit()


# ─── Users ────────────────────────────────────
def get_user(username: str):
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()


def get_all_users():
    with get_db() as conn:
        return conn.execute("SELECT id, username FROM users").fetchall()


def add_user(username: str, password: str) -> bool:
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?,?)",
                (username, generate_password_hash(password))
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def delete_user(user_id: int) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT username FROM users WHERE id=?", (user_id,)
        ).fetchone()
        if not row or row["username"] == "admin":
            return False
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
    return True


# ─── Students ─────────────────────────────────
def upsert_student(name, class_name="", division="",
                   roll_no="", email="", phone="") -> None:
    with get_db() as conn:
        conn.execute("""
            INSERT INTO students (name, class_name, division, roll_no, email, phone)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
                class_name = excluded.class_name,
                division   = excluded.division,
                roll_no    = excluded.roll_no,
                email      = excluded.email,
                phone      = excluded.phone
        """, (name, class_name, division, roll_no, email, phone))
        conn.commit()


def get_student(name: str):
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM students WHERE name=?", (name,)
        ).fetchone()


def get_all_students():
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM students ORDER BY created_at DESC"
        ).fetchall()


def delete_student(name: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM students WHERE name=?", (name,))
        conn.execute("DELETE FROM attendance WHERE student=?", (name,))
        conn.commit()


def student_count() -> int:
    with get_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) as c FROM students"
        ).fetchone()["c"]


# ─── Attendance ───────────────────────────────
def mark_attendance(student: str, lecture: str) -> bool:
    """
    Insert one attendance row. Returns True if newly inserted,
    False if already marked today.
    Uses INSERT OR IGNORE so duplicates are silently dropped.
    """
    now    = datetime.datetime.now()
    date_s = now.strftime("%Y-%m-%d")
    time_s = now.strftime("%H:%M:%S")

    try:
        with get_db() as conn:
            cur = conn.execute("""
                INSERT OR IGNORE INTO attendance (student, lecture, date, time)
                VALUES (?,?,?,?)
            """, (student, lecture, date_s, time_s))
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        return False


def get_all_attendance():
    with get_db() as conn:
        return conn.execute("""
            SELECT student AS Name, lecture AS Lecture,
                   date AS Date, time AS Time
            FROM attendance
            ORDER BY date DESC, time DESC
        """).fetchall()


def attendance_count_for(student: str) -> int:
    with get_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) as c FROM attendance WHERE student=?", (student,)
        ).fetchone()["c"]


def total_attendance_count() -> int:
    with get_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) as c FROM attendance"
        ).fetchone()["c"]


def reset_attendance() -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM attendance")
        conn.commit()


def export_attendance_csv(path: str) -> None:
    """Write current attendance table to a CSV file."""
    import csv
    rows = get_all_attendance()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Lecture", "Date", "Time"])
        for r in rows:
            writer.writerow([r["Name"], r["Lecture"], r["Date"], r["Time"]])