"""
face_engine.py — Face recognition engine for AttendAI
Uses: face_recognition (dlib ResNet 128-d embeddings)

Install:
    brew install cmake
    pip install face_recognition opencv-python opencv-contrib-python

Key design:
  - Encodings cached to encodings.pkl (fast startup after first run)
  - Cache rebuilt only when force=True (after new student added)
  - identify_face() returns (name, distance) — caller decides threshold
  - Thread-safe via a single RLock
"""

import os
import pickle
import threading
import logging

import cv2
import numpy as np

try:
    import face_recognition as FR
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logging.warning("face_recognition not installed — falling back to LBPH")

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH   = "dataset"
PREVIEW_PATH   = os.path.join("static", "previews")
ENCODINGS_FILE = "encodings.pkl"

# Recognition tuning
TOLERANCE        = 0.42   # Euclidean distance — lower = stricter
DETECTION_MODEL  = "hog"  # "hog" (CPU) | "cnn" (GPU, more accurate)
CAPTURE_SAMPLES  = 60

# LBPH fallback tuning (used only when dlib unavailable)
LBPH_THRESHOLD   = 58
LBPH_RADIUS      = 2
LBPH_NEIGHBORS   = 8

# ─────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────
_lock            = threading.RLock()
_known_encodings = []     # list of np.ndarray (128-d)
_known_names     = []     # parallel list of str
_model_trained   = False

# LBPH fallback
_lbph_recognizer = None
_lbph_label_map  = {}
_face_cascade    = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def is_trained() -> bool:
    with _lock:
        return _model_trained


def load_model(force: bool = False) -> None:
    """
    Load face encodings from cache (fast) or rebuild from dataset (slow).
    Call with force=True after adding a new student.
    """
    if DLIB_AVAILABLE:
        _load_dlib_model(force)
    else:
        _load_lbph_model()


def identify_face(face_img_bgr: np.ndarray) -> tuple[str, float]:
    """
    Given a BGR crop of a face, return (name, distance).
    distance = 0.0 means perfect match; > TOLERANCE means Unknown.
    """
    if DLIB_AVAILABLE:
        return _identify_dlib(face_img_bgr)
    else:
        return _identify_lbph(face_img_bgr)


def detect_faces_dlib(frame_bgr: np.ndarray) -> tuple[list, list]:
    """
    Run face detection + encoding on a full frame.
    Returns (locations, encodings) where locations are (top,right,bottom,left).
    """
    if not DLIB_AVAILABLE:
        return [], []
    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    locs  = FR.face_locations(rgb, model=DETECTION_MODEL)
    encs  = FR.face_encodings(rgb, locs)
    return locs, encs


def detect_faces_haar(gray: np.ndarray) -> list:
    """Haar cascade detection — used in capture flow and LBPH fallback."""
    return _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100)
    )


def capture_and_build(name: str,
                      frame_bgr: np.ndarray,
                      count: int) -> tuple[bool, np.ndarray]:
    """
    Try to extract a face from frame_bgr and save it.
    Returns (face_found, annotated_frame).
    Saves colour crops (for dlib) AND grayscale (for LBPH fallback).
    """
    path = os.path.join(DATASET_PATH, name)
    os.makedirs(path, exist_ok=True)

    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    boxes = detect_faces_haar(gray)
    frame = frame_bgr.copy()
    found = False

    for (x, y, w, h) in boxes:
        if count >= CAPTURE_SAMPLES:
            break

        # Save colour crop (face_recognition needs colour)
        colour_crop = cv2.resize(frame_bgr[y:y+h, x:x+w], (200, 200))
        cv2.imwrite(os.path.join(path, f"{count}.jpg"), colour_crop)

        # Save preview at midpoint
        if count == CAPTURE_SAMPLES // 2:
            os.makedirs(PREVIEW_PATH, exist_ok=True)
            cv2.imwrite(os.path.join(PREVIEW_PATH, f"{name}.jpg"), colour_crop)

        found = True
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 220, 0), 2)
        cv2.rectangle(frame, (x, y+h-28), (x+w, y+h), (0, 220, 0), cv2.FILLED)
        cv2.putText(frame, f"{count}/{CAPTURE_SAMPLES}",
                    (x+4, y+h-8), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

    return found, frame


# ─────────────────────────────────────────────
# DLIB INTERNALS
# ─────────────────────────────────────────────
def _load_dlib_model(force: bool) -> None:
    global _known_encodings, _known_names, _model_trained

    if not force and os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
            with _lock:
                _known_encodings = data["encodings"]
                _known_names     = data["names"]
                _model_trained   = len(_known_encodings) > 0
            logger.info(f"Loaded {len(_known_encodings)} encodings from cache")
            return
        except Exception as e:
            logger.warning(f"Cache load failed ({e}), rebuilding…")

    _rebuild_dlib_encodings()


def _rebuild_dlib_encodings() -> None:
    global _known_encodings, _known_names, _model_trained

    enc_list:  list = []
    name_list: list = []

    for person_name in sorted(os.listdir(DATASET_PATH)):
        folder = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(folder):
            continue

        for fname in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, fname)
            try:
                img  = FR.load_image_file(fpath)   # returns RGB numpy
                encs = FR.face_encodings(img)
                if encs:
                    enc_list.append(encs[0])
                    name_list.append(person_name)
            except Exception as e:
                logger.debug(f"Skip {fpath}: {e}")

    # Persist cache
    try:
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump({"encodings": enc_list, "names": name_list}, f)
    except Exception as e:
        logger.warning(f"Could not save cache: {e}")

    with _lock:
        _known_encodings = enc_list
        _known_names     = name_list
        _model_trained   = len(enc_list) > 0

    logger.info(f"Rebuilt {len(enc_list)} encodings for "
                f"{len(set(name_list))} students")


def _identify_dlib(face_img_bgr: np.ndarray) -> tuple[str, float]:
    with _lock:
        if not _model_trained or not _known_encodings:
            return "Unknown", 1.0

