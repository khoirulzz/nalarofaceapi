"""
Face verification service.

Strategy:
- Browser (Human.js) detects the face and extracts a 512-dim embedding from
  the live camera frame.  The embedding is sent to this API rather than a raw
  image, so we never store or transmit biometric photos.
- This service fetches the stored profile embedding from Firestore and computes
  cosine similarity.  Everything runs in NumPy — no heavy ML model needed
  server-side.
- Threshold is configurable via FACE_MATCH_THRESHOLD env var (default 0.72).
"""

import os
from typing import Optional

import numpy as np

FACE_MATCH_THRESHOLD: float = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.72"))

# Minimum embedding dimension we accept (Human.js outputs 512)
MIN_EMBEDDING_DIM = 64
MAX_EMBEDDING_DIM = 2048


class FaceVerificationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _validate_embedding(embedding: list, label: str = "embedding") -> np.ndarray:
    """Validate and convert an embedding list to a NumPy float array."""
    if not isinstance(embedding, list) or not MIN_EMBEDDING_DIM <= len(embedding) <= MAX_EMBEDDING_DIM:
        raise FaceVerificationError(
            "INVALID_EMBEDDING",
            f"The {label} must be a list of at least {MIN_EMBEDDING_DIM} numbers.",
        )
    try:
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in embedding):
            raise ValueError("Embedding must contain numbers")
        arr = np.array(embedding, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FaceVerificationError("INVALID_EMBEDDING", f"The {label} must be a numeric vector.") from exc
    if arr.ndim != 1 or not np.all(np.isfinite(arr)) or np.any(np.abs(arr) > 100) or np.linalg.norm(arr) == 0:
        raise FaceVerificationError(
            "INVALID_EMBEDDING",
            f"The {label} contains non-finite values (NaN or Inf).",
        )
    return arr


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return the cosine similarity between two vectors, in [0, 1]."""
    if left.shape != right.shape or left.size == 0:
        return 0.0
    dot = float(np.dot(left, right))
    norm_left = float(np.linalg.norm(left))
    norm_right = float(np.linalg.norm(right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def verify_embedding(
    live_embedding: list,
    profile_embedding: list,
    threshold: Optional[float] = None,
) -> dict:
    """
    Compare a live embedding against a stored profile embedding.

    Returns:
        {
            "verified": bool,
            "similarity": float,   # cosine similarity, -1..1 but typically 0..1
            "threshold": float,
        }
    """
    if threshold is None:
        threshold = FACE_MATCH_THRESHOLD

    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise FaceVerificationError("INVALID_THRESHOLD", "Face match threshold must be in (0, 1].")

    live_arr = _validate_embedding(live_embedding, "live_embedding")
    profile_arr = _validate_embedding(profile_embedding, "profile_embedding")

    if live_arr.shape != profile_arr.shape:
        raise FaceVerificationError(
            "EMBEDDING_DIMENSION_MISMATCH",
            "Live embedding dimension does not match the stored profile. "
            "Please re-enroll your face profile.",
        )

    similarity = cosine_similarity(live_arr, profile_arr)
    verified = similarity >= threshold

    return {
        "verified": verified,
        "similarity": round(similarity, 6),
        "threshold": threshold,
    }

