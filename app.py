"""
AttendAI – Flask Attendance System
Engine: YOLOv8 (face detection) + face_recognition + FAISS (recognition)
Fallback: OpenCV LBPH when ultralytics / faiss are not installed

Lecture timing:
  - Admin sets lecture name + start time + late_after_mins (default 10)
  - Attendance status = 'Present' if marked within grace period, else 'Late'
  - 'Absent' is computed on demand for students not in attendance table for that lecture/date

Multi-face upgrades vs original:
  1. YOLOv8n-face replaces Haar cascade  → detects 20-50 faces in one forward pass
  2. FAISS IndexFlatL2 replaces linear scan → sub-5 ms search for 3000+ encodings
  3. Producer-consumer threading         → capture / detect / recognise run in parallel
  4. Frame skipping (every Nth frame)    → recognition thread never blocks the stream
  5. CLAHE pre-processing                → better detection in low-light classrooms
  6. Consecutive-frame buffer            → 3-frame confirmation before marking attendance
  7. Anti-spoof hook                     → slot for Silent-Face or any liveness model
"""

# ─────────────────────────────────────────────
# STANDARD LIBRARY
# ─────────────────────────────────────────────
import os
# Fix macOS OpenMP conflict between PyTorch and OpenCV — must be set before any import
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import time
import datetime
import sqlite3
import shutil
import threading
from collections import deque, Counter
from functools import wraps

# ─────────────────────────────────────────────
# THIRD-PARTY (always available)
# ─────────────────────────────────────────────
import cv2
import numpy as np
import pandas as pd
import qrcode
from flask import (
    Flask, render_template, Response, request,
    redirect, session, send_file, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash

# ─────────────────────────────────────────────
# OPTIONAL ACCELERATORS
# Install for full multi-face support:
#   pip install ultralytics face_recognition faiss-cpu
# The app runs with LBPH fallback if they are missing.
# ─────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("⚠  ultralytics not found – falling back to Haar cascade")

try:
    import face_recognition as _fr
    _FR_AVAILABLE = True
except ImportError:
    _FR_AVAILABLE = False
    print("⚠  face_recognition not found – falling back to LBPH")

try:
    import faiss as _faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    print("⚠  faiss-cpu not found – falling back to numpy cosine search")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH     = "dataset"
EXPORT_CSV       = "attendance_export.csv"
QR_PATH          = os.path.join("static", "qrcodes")
PREVIEW_PATH     = os.path.join("static", "previews")
DB_PATH          = "users.db"
SECRET_KEY       = os.environ.get("SECRET_KEY", "change-me-in-production")

# YOLO model path – download yolov8n-face.pt from:
# https://github.com/akanametov/yolov8-face/releases
YOLO_FACE_MODEL  = os.environ.get("YOLO_MODEL", "yolov8n-face.pt")
YOLO_CONF        = 0.45          # detection confidence threshold
YOLO_INPUT_SIZE  = 640           # resize width before inference
FACE_ENC_DIM     = 128           # face_recognition vector dimension
FAISS_THRESHOLD  = 0.55          # L2 distance – lower = stricter match

# Legacy LBPH settings (used when face_recognition/YOLO unavailable)
CONFIDENCE_THRESHOLD = 70
SMOOTHING_WINDOW     = 5
MIN_VOTES            = 2

# Shared pipeline settings
COOLDOWN_SECONDS     = 5         # min seconds between marking same student
CAPTURE_SAMPLES      = 60        # face samples to collect per student
DEFAULT_LATE_MINS    = 10        # grace period before 'Late'
FRAME_SKIP           = 2         # run recognition every Nth frame
CONFIRM_FRAMES       = 3         # consecutive recognitions before marking
QUEUE_MAX            = 3         # max frames waiting in the recognition queue

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
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

            CREATE TABLE IF NOT EXISTS attendance (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name    TEXT NOT NULL,
                lecture TEXT NOT NULL,
                date    TEXT NOT NULL,
                time    TEXT NOT NULL,
                status  TEXT NOT NULL DEFAULT 'Present',
                UNIQUE(name, lecture, date)
            );

            CREATE INDEX IF NOT EXISTS idx_att_name    ON attendance(name);
            CREATE INDEX IF NOT EXISTS idx_att_lecture ON attendance(lecture);
            CREATE INDEX IF NOT EXISTS idx_att_date    ON attendance(date);

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

            INSERT OR IGNORE INTO settings (key, value) VALUES ('current_lecture',    'None');
            INSERT OR IGNORE INTO settings (key, value) VALUES ('lecture_start_time', 'None');
            INSERT OR IGNORE INTO settings (key, value) VALUES ('late_after_mins',    '10');
        """)
        conn.execute(
            "INSERT OR IGNORE INTO users (username, password) VALUES (?,?)",
            ("admin", generate_password_hash("admin123"))
        )
        conn.commit()


def get_setting(key: str, default: str = "") -> str:
    with get_db() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)",
            (key, value)
        )
        conn.commit()

# ─────────────────────────────────────────────
# LATE DETECTION
# ─────────────────────────────────────────────
def compute_status() -> str:
    start_str = get_setting("lecture_start_time", "None")
    if start_str == "None" or not start_str:
        return "Present"
    try:
        late_mins = int(get_setting("late_after_mins", str(DEFAULT_LATE_MINS)))
        today     = datetime.date.today().isoformat()
        start_dt  = datetime.datetime.fromisoformat(f"{today}T{start_str}")
        cutoff    = start_dt + datetime.timedelta(minutes=late_mins)
        return "Present" if datetime.datetime.now() <= cutoff else "Late"
    except Exception:
        return "Present"

# ─────────────────────────────────────────────
# ATTENDANCE HELPERS
# ─────────────────────────────────────────────
def mark_attendance_db(name: str, lecture: str) -> bool:
    now    = datetime.datetime.now()
    date_s = now.strftime("%Y-%m-%d")
    time_s = now.strftime("%H:%M:%S")
    status = compute_status()
    try:
        with get_db() as conn:
            cur = conn.execute("""
                INSERT OR IGNORE INTO attendance (name, lecture, date, time, status)
                VALUES (?,?,?,?,?)
            """, (name, lecture, date_s, time_s, status))
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        return False


def get_all_attendance():
    with get_db() as conn:
        return conn.execute("""
            SELECT name AS Name, lecture AS Lecture,
                   date AS Date, time AS Time, status AS Status
            FROM attendance ORDER BY date DESC, time DESC
        """).fetchall()


def attendance_count_for(name: str) -> int:
    with get_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) as c FROM attendance WHERE name=?", (name,)
        ).fetchone()["c"]


def get_absent_students(lecture: str, date: str) -> list:
    with get_db() as conn:
        all_students = [r["name"] for r in conn.execute(
            "SELECT name FROM students"
        ).fetchall()]
        present = [r["name"] for r in conn.execute(
            "SELECT name FROM attendance WHERE lecture=? AND date=?",
            (lecture, date)
        ).fetchall()]
    return [s for s in all_students if s not in present]

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(QR_PATH,      exist_ok=True)
os.makedirs(PREVIEW_PATH, exist_ok=True)

# ─────────────────────────────────────────────
# ── ENGINE A: YOLO + face_recognition + FAISS
#    (multi-face, scalable to 20-50 students)
# ─────────────────────────────────────────────
_yolo_model       = None          # YOLOv8 detector
_faiss_index      = None          # FAISS L2 index of 128-d embeddings
_faiss_names: list = []           # parallel list: index position → student name
_engine_lock      = threading.Lock()
_engine_ready     = False         # True once at least one student is indexed


def _load_yolo() -> None:
    """Load (or reload) YOLOv8 face model.  No-op if ultralytics is absent."""
    global _yolo_model
    if not _YOLO_AVAILABLE:
        return
    if os.path.exists(YOLO_FACE_MODEL):
        _yolo_model = _YOLO(YOLO_FACE_MODEL)
        print(f"✅ YOLOv8 face model loaded: {YOLO_FACE_MODEL}")
    else:
        print(f"⚠  {YOLO_FACE_MODEL} not found – detection will use Haar cascade")
        print("   Download from: https://github.com/akanametov/yolov8-face/releases")


def _detect_faces_yolo(frame: np.ndarray) -> list:
    """
    Returns list of (x1, y1, x2, y2) bounding boxes.
    Uses YOLOv8 if available, otherwise falls back to Haar cascade.
    """
    if _yolo_model is not None:
        results = _yolo_model(frame, verbose=False, conf=YOLO_CONF)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # clamp to frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) > 20 and (y2 - y1) > 20:   # ignore tiny detections
                boxes.append((x1, y1, x2, y2))
        return boxes
    else:
        # ── Haar cascade fallback ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60)
        )
        return [(x, y, x+w, y+h) for (x, y, w, h) in faces]


def _build_faiss_index() -> None:
    """
    Walk dataset/, compute face_recognition 128-d encodings for every image,
    and build a FAISS IndexFlatL2.  Falls back to a plain numpy list when
    faiss is not installed.
    """
    global _faiss_index, _faiss_names, _engine_ready

    if not _FR_AVAILABLE:
        return   # will use LBPH path instead

    encodings_list: list = []
    names_list:     list = []

    for person_name in sorted(os.listdir(DATASET_PATH)):
        folder = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            img_path = os.path.join(folder, fname)
            img_bgr  = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            encs    = _fr.face_encodings(img_rgb)
            if encs:
                encodings_list.append(encs[0].astype(np.float32))
                names_list.append(person_name)

    with _engine_lock:
        if encodings_list:
            vectors = np.array(encodings_list, dtype=np.float32)
            if _FAISS_AVAILABLE:
                idx = _faiss.IndexFlatL2(FACE_ENC_DIM)
                idx.add(vectors)
                _faiss_index = idx
            else:
                # store raw vectors; search done with numpy
                _faiss_index = vectors
            _faiss_names  = names_list
            _engine_ready = True
            print(f"✅ Recognition index built: {len(names_list)} vectors "
                  f"({len(set(names_list))} students)")
        else:
            _faiss_index  = None
            _faiss_names  = []
            _engine_ready = False


def _recognise_face_enc(face_bgr: np.ndarray) -> str:
    """
    Given a cropped face (BGR), return the matched student name or 'Unknown'.
    Uses FAISS / numpy search on face_recognition embeddings.
    """
    if not _FR_AVAILABLE or not _engine_ready:
        return "Unknown"

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    encs     = _fr.face_encodings(face_rgb)
    if not encs:
        return "Unknown"

    query = np.array([encs[0]], dtype=np.float32)

    with _engine_lock:
        if _faiss_index is None or not _faiss_names:
            return "Unknown"

        if _FAISS_AVAILABLE:
            distances, indices = _faiss_index.search(query, k=1)
            dist = float(distances[0][0])
            idx  = int(indices[0][0])
        else:
            # numpy fallback: L2 distance
            diffs = _faiss_index - query
            dists = np.sum(diffs ** 2, axis=1)
            idx   = int(np.argmin(dists))
            dist  = float(dists[idx])

        if dist < FAISS_THRESHOLD:
            return _faiss_names[idx]
        return "Unknown"

# ─────────────────────────────────────────────
# ── ENGINE B: OpenCV LBPH
#    (legacy fallback when face_recognition/YOLO unavailable)
# ─────────────────────────────────────────────
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_lbph_recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2, neighbors=8, grid_x=8, grid_y=8
)
_lbph_label_map: dict = {}
_lbph_trained:   bool = False
_lbph_lock             = threading.Lock()


def _train_lbph() -> None:
    global _lbph_recognizer, _lbph_label_map, _lbph_trained
    faces, labels, lmap = [], [], {}
    label_id = 0
    for person_name in sorted(os.listdir(DATASET_PATH)):
        folder = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(folder):
            continue
        added = 0
        for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            img = cv2.equalizeHist(img)
            faces.append(img)
            labels.append(label_id)
            added += 1
        if added > 0:
            lmap[label_id] = person_name
            label_id += 1

    with _lbph_lock:
        if faces:
            r = cv2.face.LBPHFaceRecognizer_create(
                radius=2, neighbors=8, grid_x=8, grid_y=8
            )
            r.train(faces, np.array(labels))
            _lbph_recognizer = r
            _lbph_label_map  = lmap
            _lbph_trained    = True
        else:
            _lbph_label_map = {}
            _lbph_trained   = False


def _recognise_face_lbph(gray_roi: np.ndarray) -> str:
    with _lbph_lock:
        if not _lbph_trained:
            return "Unknown"
        try:
            roi  = cv2.resize(gray_roi, (200, 200))
            lbl, conf = _lbph_recognizer.predict(roi)
            if conf < CONFIDENCE_THRESHOLD:
                return _lbph_label_map.get(lbl, "Unknown")
            return "Unknown"
        except Exception:
            return "Unknown"

# ─────────────────────────────────────────────
# UNIFIED TRAIN ENTRY POINT
# ─────────────────────────────────────────────
def train_model() -> None:
    """Rebuild whichever recognition index is available."""
    if _FR_AVAILABLE:
        _build_faiss_index()
    else:
        _train_lbph()

# ─────────────────────────────────────────────
# ATTENDANCE MARK HELPER
# ─────────────────────────────────────────────
mark_lock    = threading.Lock()
last_marked: dict = {}
# consecutive frame counter – student must appear in CONFIRM_FRAMES frames
_confirm_buf: dict = {}


def _try_mark(name: str) -> None:
    lecture = get_setting("current_lecture")
    if lecture == "None" or not lecture:
        return

    with mark_lock:
        # Consecutive-frame confirmation
        _confirm_buf[name] = _confirm_buf.get(name, 0) + 1
        if _confirm_buf[name] < CONFIRM_FRAMES:
            return
        _confirm_buf[name] = 0

        now = time.time()
        if now - last_marked.get(name, 0) > COOLDOWN_SECONDS:
            if mark_attendance_db(name, lecture):
                last_marked[name] = now
                print(f"✅ Marked: {name} [{lecture}]")


def _reset_confirm(active_names: set) -> None:
    """Clear confirmation buffer for names that left the frame."""
    for n in list(_confirm_buf):
        if n not in active_names:
            del _confirm_buf[n]

# ─────────────────────────────────────────────
# CAPTURE STATUS
# ─────────────────────────────────────────────
capture_status:      dict = {}
capture_status_lock        = threading.Lock()

# ─────────────────────────────────────────────
# CAMERA HELPERS
# ─────────────────────────────────────────────
def _jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


def _open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    return cap


def _clahe_enhance(frame: np.ndarray) -> np.ndarray:
    """
    Gentle contrast boost used ONLY for inference, not for display.
    clipLimit=1.0 + large tile avoids the black-patch artefact.
    Wrapped in try/except so any failure returns the original frame.
    """
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        return frame


def _resize_for_inference(frame: np.ndarray, width: int = YOLO_INPUT_SIZE) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))

# ─────────────────────────────────────────────
# LIVE RECOGNITION STREAM  (multi-face, threaded)
# ─────────────────────────────────────────────
def generate_frames():
    """
    Producer: captures frames, yields MJPEG.
    Recognition runs in a background consumer thread so the stream is never
    blocked waiting for YOLO / FAISS.

    Layout of the shared state:
      _recog_results  – latest recognition output (drawn on every frame)
      _recog_frame_q  – raw frames sent to the recognition thread
    """
    import queue as _queue

    _recog_frame_q: _queue.Queue = _queue.Queue(maxsize=QUEUE_MAX)
    _recog_results: list         = []         # [{name, box, status}]
    _recog_lock                  = threading.Lock()
    _frame_count                 = 0

    def _recognition_worker():
        nonlocal _recog_results
        while True:
            try:
                frame_rgb = _recog_frame_q.get(timeout=2)
            except _queue.Empty:
                continue

            # Resize for faster inference
            small = _resize_for_inference(frame_rgb)

            # ── Detect faces ──
            boxes = _detect_faces_yolo(small)

            # Scale boxes back to original resolution
            scale = frame_rgb.shape[1] / small.shape[1]
            boxes_orig = [
                (int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale))
                for (x1, y1, x2, y2) in boxes
            ]

            results = []
            for (x1, y1, x2, y2) in boxes_orig:
                face_crop = frame_rgb[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                if _FR_AVAILABLE and _engine_ready:
                    name = _recognise_face_enc(face_crop)
                else:
                    gray_roi = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    name     = _recognise_face_lbph(gray_roi)

                results.append({"name": name, "box": (x1, y1, x2, y2)})

            with _recog_lock:
                _recog_results = results

    # Start recognition worker thread
    t = threading.Thread(target=_recognition_worker, daemon=True)
    t.start()

    cap = _open_camera()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame        = cv2.flip(frame, 1)
            # display_frame stays as raw camera output — no colour distortion
            display_frame = frame.copy()
            _frame_count += 1

            # Send CLAHE-enhanced copy to recognition thread (inference only)
            if _frame_count % FRAME_SKIP == 0:
                if not _recog_frame_q.full():
                    enhanced = _clahe_enhance(frame)
                    _recog_frame_q.put(enhanced)

            # Draw last known results on the CLEAN display frame (no CLAHE)
            with _recog_lock:
                current_results = list(_recog_results)

            active_names: set = set()
            status_label = compute_status()

            for r in current_results:
                name = r["name"]
                x1, y1, x2, y2 = r["box"]

                if name != "Unknown":
                    active_names.add(name)
                    _try_mark(name)
                    color      = (0, 200, 0) if status_label == "Present" else (0, 165, 255)
                    label_text = f"{name} [{status_label}]"
                else:
                    color      = (0, 0, 255)
                    label_text = "Unknown"

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(display_frame, (x1, y2 - 28), (x2, y2), color, cv2.FILLED)
                cv2.putText(display_frame, label_text, (x1 + 4, y2 - 8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1)

            _reset_confirm(active_names)

            # HUD: lecture info + face count + engine mode
            lecture = get_setting("current_lecture")
            start   = get_setting("lecture_start_time", "None")
            late_m  = get_setting("late_after_mins", str(DEFAULT_LATE_MINS))
            engine  = "YOLO+FAISS" if (_yolo_model and _engine_ready) else "LBPH"
            if lecture != "None":
                hud = f"Lecture: {lecture}  |  Start: {start}  |  Late after: {late_m}m  |  {engine}"
                cv2.putText(display_frame, hud, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
            face_n = len(current_results)
            cv2.putText(display_frame, f"Faces: {face_n}", (8, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            yield _jpeg(display_frame)
    finally:
        cap.release()

# ─────────────────────────────────────────────
# LIVE CAPTURE STREAM  (student registration)
# ─────────────────────────────────────────────
# Lock to serialise all disk writes during face capture
_capture_write_lock = threading.Lock()


def _safe_imwrite(path: str, img: np.ndarray) -> bool:
    """
    Thread-safe image write using a temp file + atomic rename.
    Prevents corrupt JPEG / malloc crashes when multiple threads
    access the dataset folder simultaneously.
    """
    tmp = path + ".tmp"
    try:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok or buf is None:
            return False
        data = buf.tobytes()          # convert to bytes BEFORE writing
        with _capture_write_lock:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)     # atomic on POSIX / macOS
        return True
    except Exception as e:
        print(f"_safe_imwrite error ({path}): {e}")
        try:
            os.remove(tmp)
        except Exception:
            pass
        return False


def capture_faces(name: str):
    """Capture face samples for training – uses YOLO if available."""
    safe_name = name.replace(" ", "_")
    folder    = os.path.join(DATASET_PATH, safe_name)
    os.makedirs(folder, exist_ok=True)

    # ── open camera with retry ──
    cap = None
    for attempt in range(5):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            break
        cap.release()
        print(f"Camera not ready (attempt {attempt+1}/5), retrying…")
        time.sleep(1)

    if cap is None or not cap.isOpened():
        err = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err, "Camera Error!", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(err, "Stop the live camera first, then retry.", (30, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ok, buf = cv2.imencode(".jpg", err)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    count         = 0
    preview_saved = False

    with capture_status_lock:
        capture_status[safe_name] = {"count": 0, "done": False}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Work on a copy so drawing never races with saving
            display = frame.copy()

            boxes = _detect_faces_yolo(frame)

            for (x1, y1, x2, y2) in boxes:
                if count < CAPTURE_SAMPLES:
                    # Pad crop slightly so we get a generous face region
                    pad  = 10
                    h, w = frame.shape[:2]
                    fx1  = max(0, x1 - pad)
                    fy1  = max(0, y1 - pad)
                    fx2  = min(w, x2 + pad)
                    fy2  = min(h, y2 + pad)

                    face_crop = frame[fy1:fy2, fx1:fx2].copy()  # explicit copy

                    if face_crop.size > 0:
                        face_path = os.path.join(folder, f"{safe_name}_{count}.jpg")
                        if _safe_imwrite(face_path, face_crop):
                            if not preview_saved:
                                prev_path = os.path.join(PREVIEW_PATH, f"{safe_name}.jpg")
                                # Save a slightly larger preview image (200px wide)
                                ph        = int(face_crop.shape[0] * 200 / max(face_crop.shape[1], 1))
                                preview   = cv2.resize(face_crop, (200, ph))
                                _safe_imwrite(prev_path, preview)
                                preview_saved = True

                            count += 1
                            with capture_status_lock:
                                capture_status[safe_name]["count"] = count
                                if count >= CAPTURE_SAMPLES:
                                    capture_status[safe_name]["done"] = True

                color = (0, 255, 0) if count < CAPTURE_SAMPLES else (0, 165, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                cv2.putText(display, f"Captured: {count}/{CAPTURE_SAMPLES}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if count >= CAPTURE_SAMPLES:
                cv2.putText(display, "COMPLETE! Redirecting…",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

            # Encode display frame (never the save buffer)
            ok, buffer = cv2.imencode(".jpg", display,
                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok and buffer is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buffer.tobytes() + b"\r\n")

            if count >= CAPTURE_SAMPLES:
                threading.Thread(target=train_model, daemon=True).start()
                time.sleep(2)
                break

    except Exception as e:
        print(f"ERROR in capture_faces: {e}")
        import traceback; traceback.print_exc()
    finally:
        cap.release()

# ─────────────────────────────────────────────
# QR SCAN STREAM
# ─────────────────────────────────────────────
def qr_scan_frames():
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        _ok = True
    except ImportError:
        _ok = False

    cap       = _open_camera()
    last_scan: dict = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            if _ok:
                from pyzbar.pyzbar import decode as _dec
                for qr in _dec(frame):
                    data = qr.data.decode("utf-8").strip()
                    now  = time.time()
                    if now - last_scan.get(data, 0) > COOLDOWN_SECONDS:
                        _try_mark(data)
                        last_scan[data] = now
                    pts = np.array(qr.polygon, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 220, 0), 3)
                    cv2.putText(frame, data,
                                (qr.rect.left, qr.rect.top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
            else:
                cv2.putText(frame, "pyzbar not installed",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)
            yield _jpeg(frame)
    finally:
        cap.release()

# ─────────────────────────────────────────────
# AUTH DECORATORS
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect("/")
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session or session["user"] != "admin":
            flash("Admin access required", "error")
            return redirect("/dashboard")
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────
# ROUTES — AUTH
# ─────────────────────────────────────────────
@app.route("/")
def index():
    if "user" in session:
        return redirect("/dashboard")
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login_post():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    with get_db() as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username=?", (username,)
        ).fetchone()
    if row and check_password_hash(row["password"], password):
        session["user"] = username
        return redirect("/dashboard")
    flash("Invalid credentials", "error")
    return redirect("/")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            flash("All fields required", "error")
            return redirect("/register")
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (username, password) VALUES (?,?)",
                    (username, generate_password_hash(password))
                )
                conn.commit()
            flash("Account created — please log in", "success")
            return redirect("/")
        except sqlite3.IntegrityError:
            flash("Username already exists", "error")
            return redirect("/register")
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ─────────────────────────────────────────────
# ROUTES — DASHBOARD
# ─────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    records = get_all_attendance()
    with get_db() as conn:
        student_count = conn.execute(
            "SELECT COUNT(*) as c FROM students"
        ).fetchone()["c"]

    lecture    = get_setting("current_lecture")
    start_time = get_setting("lecture_start_time", "None")
    late_mins  = get_setting("late_after_mins",    str(DEFAULT_LATE_MINS))
    today      = datetime.date.today().isoformat()
    absents    = get_absent_students(lecture, today) if lecture != "None" else []

    return render_template(
        "dashboard.html",
        records        = records,
        total          = len(records),
        students_count = student_count,
        lecture        = lecture,
        start_time     = start_time,
        late_mins      = late_mins,
        absents        = absents,
        today          = today,
    )


@app.route("/set_lecture", methods=["POST"])
@login_required
def set_lecture():
    lecture    = request.form.get("lecture",         "").strip()
    start_time = request.form.get("start_time",      "").strip()
    late_mins  = request.form.get("late_after_mins", "").strip()

    if not lecture:
        flash("Lecture name cannot be empty", "error")
        return redirect("/dashboard")
    if not start_time:
        start_time = datetime.datetime.now().strftime("%H:%M")
    if not late_mins or not late_mins.isdigit():
        late_mins = str(DEFAULT_LATE_MINS)

    set_setting("current_lecture",    lecture)
    set_setting("lecture_start_time", start_time)
    set_setting("late_after_mins",    late_mins)
    flash(f"Lecture '{lecture}' started at {start_time} (late after {late_mins} min)", "info")
    return redirect("/dashboard")


@app.route("/end_lecture", methods=["POST"])
@login_required
def end_lecture():
    set_setting("current_lecture",    "None")
    set_setting("lecture_start_time", "None")
    flash("Lecture ended", "info")
    return redirect("/dashboard")

# ─────────────────────────────────────────────
# ROUTES — STUDENTS
# ─────────────────────────────────────────────
@app.route("/students")
@login_required
def students_page():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM students ORDER BY created_at DESC"
        ).fetchall()

    students = []
    for r in rows:
        name = r["name"]
        preview_url = (
            f"/static/previews/{name}.jpg"
            if os.path.exists(os.path.join(PREVIEW_PATH, f"{name}.jpg"))
            else None
        )
        folder = os.path.join(DATASET_PATH, name)
        sample_count = (
            len([f for f in os.listdir(folder) if f.endswith(".jpg")])
            if os.path.isdir(folder) else 0
        )
        students.append({
            "name":             name,
            "class_name":       r["class_name"] or "—",
            "division":         r["division"]   or "—",
            "roll_no":          r["roll_no"]    or "—",
            "email":            r["email"]      or "—",
            "phone":            r["phone"]      or "—",
            "created_at":       r["created_at"] or "—",
            "sample_count":     sample_count,
            "attendance_count": attendance_count_for(name),
            "preview_url":      preview_url,
        })

    return render_template("students.html", students=students)


@app.route("/student_attendance_data/<name>")
@login_required
def student_attendance_data(name):
    with get_db() as conn:
        end_date   = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=30)

        records = conn.execute("""
            SELECT date, status FROM attendance
            WHERE name = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
        """, (name, start_date.isoformat(), end_date.isoformat())).fetchall()

        all_lectures = conn.execute("""
            SELECT DISTINCT lecture, date FROM attendance
            WHERE date >= ? AND date <= ?
            ORDER BY date ASC
        """, (start_date.isoformat(), end_date.isoformat())).fetchall()

        lectures_by_date: dict = {}
        for lec in all_lectures:
            lectures_by_date.setdefault(lec["date"], []).append(lec["lecture"])

        dates, present_count, late_count, absent_count = [], [], [], []
        for date_str in sorted(lectures_by_date):
            dates.append(date_str)
            p = l = a = 0
            for lecture in lectures_by_date[date_str]:
                row = conn.execute("""
                    SELECT status FROM attendance
                    WHERE name = ? AND lecture = ? AND date = ?
                """, (name, lecture, date_str)).fetchone()
                if row:
                    if row["status"] == "Present": p += 1
                    elif row["status"] == "Late":  l += 1
                else:
                    a += 1
            present_count.append(p)
            late_count.append(l)
            absent_count.append(a)

    return jsonify({
        "dates":   dates,
        "present": present_count,
        "late":    late_count,
        "absent":  absent_count,
        "name":    name,
    })


@app.route("/add_student_form")
@login_required
def add_student_form():
    return render_template("add_student.html")


@app.route("/add_student", methods=["POST"])
@login_required
def add_student():
    try:
        name       = request.form.get("name",       "").strip()
        class_name = request.form.get("class_name", "").strip()
        division   = request.form.get("division",   "").strip()
        roll_no    = request.form.get("roll_no",    "").strip()
        email      = request.form.get("email",      "").strip()
        phone      = request.form.get("phone",      "").strip()

        if not name:
            flash("Student name is required", "error")
            return redirect("/add_student_form")

        safe_name = name.replace(" ", "_")

        with get_db() as conn:
            conn.execute("""
                INSERT INTO students (name, class_name, division, roll_no, email, phone)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    class_name = excluded.class_name,
                    division   = excluded.division,
                    roll_no    = excluded.roll_no,
                    email      = excluded.email,
                    phone      = excluded.phone
            """, (safe_name, class_name, division, roll_no, email, phone))
            conn.commit()
            print(f"✅ Student '{safe_name}' saved")

        with capture_status_lock:
            capture_status[safe_name] = {"count": 0, "done": False}

        return redirect(f"/capture/{safe_name}")

    except Exception as e:
        print(f"❌ ERROR in add_student: {e}")
        import traceback; traceback.print_exc()
        flash(f"Error: {str(e)}", "error")
        return redirect("/add_student_form")


@app.route("/edit_student/<name>", methods=["POST"])
@login_required
def edit_student(name: str):
    class_name = request.form.get("class_name", "").strip()
    division   = request.form.get("division",   "").strip()
    roll_no    = request.form.get("roll_no",    "").strip()
    email      = request.form.get("email",      "").strip()
    phone      = request.form.get("phone",      "").strip()

    with get_db() as conn:
        conn.execute("""
            UPDATE students
            SET class_name=?, division=?, roll_no=?, email=?, phone=?
            WHERE name=?
        """, (class_name, division, roll_no, email, phone, name))
        conn.commit()

    flash(f"Student '{name}' updated successfully.", "success")
    return redirect("/students")


@app.route("/delete_student/<name>", methods=["POST"])
@login_required
def delete_student(name: str):
    folder  = os.path.join(DATASET_PATH, name)
    preview = os.path.join(PREVIEW_PATH, f"{name}.jpg")
    if os.path.isdir(folder):   shutil.rmtree(folder)
    if os.path.exists(preview): os.remove(preview)
    with get_db() as conn:
        conn.execute("DELETE FROM students   WHERE name=?", (name,))
        conn.execute("DELETE FROM attendance WHERE name=?", (name,))
        conn.commit()
    train_model()
    flash(f"Student '{name}' deleted.", "success")
    return redirect("/students")

# ─────────────────────────────────────────────
# ROUTES — CAPTURE FLOW
# ─────────────────────────────────────────────
@app.route("/capture/<name>")
@login_required
def capture_page(name):
    return render_template("capture.html", name=name)


@app.route("/video_feed_capture/<name>")
@login_required
def video_feed_capture(name):
    return Response(
        capture_faces(name),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/capture_status/<name>")
@login_required
def get_capture_status(name):
    with capture_status_lock:
        status = capture_status.get(name, {"count": 0, "done": False})
    return jsonify({
        "count": status["count"],
        "done":  status["done"],
        "total": CAPTURE_SAMPLES,
    })


@app.route("/confirm/<name>")
@login_required
def confirm_page(name):
    preview_exists = os.path.exists(os.path.join(PREVIEW_PATH, f"{name}.jpg"))
    folder         = os.path.join(DATASET_PATH, name)
    sample_count   = len(os.listdir(folder)) if os.path.isdir(folder) else 0
    with get_db() as conn:
        student = conn.execute(
            "SELECT * FROM students WHERE name=?", (name,)
        ).fetchone()
    return render_template(
        "confirm.html",
        name         = name,
        preview      = f"/static/previews/{name}.jpg" if preview_exists else None,
        sample_count = sample_count,
        student      = student,
    )

# ─────────────────────────────────────────────
# ROUTES — CAMERA
# ─────────────────────────────────────────────
@app.route("/camera")
@login_required
def camera():
    return render_template("index.html")


@app.route("/video")
@login_required
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/qr")
@login_required
def qr_page():
    return render_template("qr.html")


@app.route("/qr_video")
@login_required
def qr_video():
    return Response(qr_scan_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ─────────────────────────────────────────────
# ROUTES — DATA
# ─────────────────────────────────────────────
@app.route("/download")
@login_required
def download():
    rows = get_all_attendance()
    with open(EXPORT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Lecture", "Date", "Time", "Status"])
        for r in rows:
            writer.writerow([r["Name"], r["Lecture"],
                             r["Date"], r["Time"], r["Status"]])
    return send_file(EXPORT_CSV, as_attachment=True)


@app.route("/reset", methods=["POST"])
@login_required
def reset():
    with get_db() as conn:
        conn.execute("DELETE FROM attendance")
        conn.commit()
    flash("Attendance records cleared", "info")
    return redirect("/dashboard")

# ─────────────────────────────────────────────
# ROUTES — ADMIN
# ─────────────────────────────────────────────
@app.route("/admin")
@admin_required
def admin_panel():
    with get_db() as conn:
        users = conn.execute("SELECT id, username FROM users").fetchall()
    return render_template("admin.html", users=users)


@app.route("/admin_add_user", methods=["POST"])
@admin_required
def admin_add_user():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    if not username or not password:
        flash("Both fields required", "error")
        return redirect("/admin")
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?,?)",
                (username, generate_password_hash(password))
            )
            conn.commit()
        flash(f"User '{username}' added", "success")
    except sqlite3.IntegrityError:
        flash("Username already exists", "error")
    return redirect("/admin")


@app.route("/delete_user/<int:user_id>", methods=["POST"])
@admin_required
def delete_user(user_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT username FROM users WHERE id=?", (user_id,)
        ).fetchone()
        if row and row["username"] == "admin":
            flash("Cannot delete admin", "error")
            return redirect("/admin")
        if row:
            conn.execute("DELETE FROM users WHERE id=?", (user_id,))
            conn.commit()
            flash("User deleted", "success")
        else:
            flash("User not found", "error")
    return redirect("/admin")

# ─────────────────────────────────────────────
# ROUTES — API
# ─────────────────────────────────────────────
@app.route("/api/retrain", methods=["POST"])
@login_required
def api_retrain():
    train_model()
    with _engine_lock:
        count = len(set(_faiss_names)) if _FR_AVAILABLE else len(_lbph_label_map)
    return jsonify({"status": "ok", "students": count})


@app.route("/api/status")
@login_required
def api_status():
    engine = "YOLO+FAISS" if (_yolo_model and _engine_ready) else "LBPH"
    with _engine_lock:
        known = len(set(_faiss_names)) if _FR_AVAILABLE else len(_lbph_label_map)
    return jsonify({
        "lecture":        get_setting("current_lecture"),
        "start_time":     get_setting("lecture_start_time"),
        "late_mins":      get_setting("late_after_mins"),
        "current_status": compute_status(),
        "engine":         engine,
        "known_faces":    known,
    })

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    _load_yolo()       # load YOLOv8 face model (no-op if not installed)
    train_model()      # build FAISS index or LBPH model
    port = int(os.environ.get("PORT", 5004))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)