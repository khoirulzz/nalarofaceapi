"""
JWT verification token service.

Issues short-lived HS256 JWTs that the Cloudflare Worker can verify locally
using a shared secret — no network round-trip required on the Worker side.

Token payload:
    {
      "uid":          "<firebase uid>",
      "attendanceId": "<uuid>",
      "classId":      "<uuid>",
      "similarity":   0.87,
      "iat":          <unix timestamp>,
      "exp":          <iat + TOKEN_TTL_SECONDS>,
      "iss":          "nalaro-face-api"
    }
"""

import os
import time
from typing import Optional

import jwt  # PyJWT

_SECRET: Optional[str] = None
TOKEN_TTL_SECONDS: int = int(os.environ.get("FACE_TOKEN_TTL", "60"))
ISSUER = "nalaro-face-api"
ALGORITHM = "HS256"


def _get_secret() -> str:
    global _SECRET
    if _SECRET is None:
        secret = os.environ.get("FACE_API_SECRET", "")
        if len(secret) < 32:
            raise RuntimeError(
                "FACE_API_SECRET must be set and at least 32 characters long."
            )
        _SECRET = secret
    return _SECRET


def create_verification_token(
    uid: str,
    attendance_id: str,
    class_id: str,
    similarity: float,
) -> dict:
    """
    Create a short-lived JWT that proves server-side face verification passed.

    Returns:
        {"token": str, "expires_at": str (ISO-8601)}
    """
    now = int(time.time())
    exp = now + TOKEN_TTL_SECONDS
    payload = {
        "uid": uid,
        "attendanceId": attendance_id,
        "classId": class_id,
        "similarity": round(float(similarity), 6),
        "iat": now,
        "exp": exp,
        "iss": ISSUER,
    }
    token = jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)
    expires_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(exp))
    return {"token": token, "expires_at": expires_at}


def decode_verification_token(token: str) -> dict:
    """
    Decode and validate a verification token.
    Raises jwt.PyJWTError on failure.
    """
    return jwt.decode(
        token,
        _get_secret(),
        algorithms=[ALGORITHM],
        issuer=ISSUER,
        options={"verify_exp": True},
    )

