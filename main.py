"""
Nalaro Face API — FastAPI application.

Endpoints:
  GET  /health         — health check (keeps Render service warm)
  POST /verify-face    — verify browser embedding against Firestore profile
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.face_verifier import FaceVerificationError, verify_embedding
from services.firebase_service import get_face_profile, verify_id_token
from services.token_service import create_verification_token

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("nalaro-face-api")

# ─── Allowed origins ──────────────────────────────────────────────────────────

_ALLOWED_ORIGINS_ENV = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://nalaroclass.pages.dev,https://class.nalaro.digital,https://nalaro.web.id,http://localhost:5173",
)
ALLOWED_ORIGINS = [o.strip() for o in _ALLOWED_ORIGINS_ENV.split(",") if o.strip()]

# ─── Startup ──────────────────────────────────────────────────────────────────

_startup_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly validate env vars and log startup."""
    global _startup_time
    _startup_time = time.time()

    required = ["FIREBASE_SERVICE_ACCOUNT_JSON", "FACE_API_SECRET"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        logger.error("Missing required environment variables: %s", missing)
        raise RuntimeError(f"Missing env vars: {missing}")

    logger.info(
        "Nalaro Face API starting — FACE_MATCH_THRESHOLD=%s TOKEN_TTL=%s",
        os.environ.get("FACE_MATCH_THRESHOLD", "0.72"),
        os.environ.get("FACE_TOKEN_TTL", "60"),
    )
    yield
    logger.info("Nalaro Face API shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Nalaro Face API",
    description="Server-side face verification for Nalaro Class attendance.",
    version="1.0.0",
    lifespan=lifespan,
    # Hide docs in production for security
    docs_url="/docs" if os.environ.get("ENABLE_DOCS") == "1" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)

# ─── Models ───────────────────────────────────────────────────────────────────


class VerifyFaceRequest(BaseModel):
    idToken: str = Field(..., description="Firebase ID token from the browser client.")
    liveEmbedding: list[float] = Field(
        ...,
        description=(
            "Face embedding extracted in the browser by Human.js "
            "(512-dim float array)."
        ),
    )
    attendanceId: str = Field(..., max_length=64, description="Attendance session UUID.")
    classId: str = Field(..., max_length=64, description="Class UUID.")


class VerifyFaceResponse(BaseModel):
    verified: bool
    similarity: float
    verificationToken: Optional[str] = None
    expiresAt: Optional[str] = None
    reason: Optional[str] = None
    message: Optional[str] = None


# ─── Global error handler ─────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"verified": False, "reason": "INTERNAL_ERROR", "message": "Terjadi kesalahan server. Coba lagi."},
    )


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health", tags=["ops"])
def health():
    """Health check — used by Render and UptimeRobot keep-alive pings."""
    uptime = round(time.time() - _startup_time, 1) if _startup_time else 0
    return {"status": "ok", "uptime_seconds": uptime}


@app.post(
    "/verify-face",
    response_model=VerifyFaceResponse,
    tags=["face"],
    summary="Verify face embedding against stored Firestore profile",
)
async def verify_face(payload: VerifyFaceRequest):
    """
    Verify that the live embedding captured by the browser matches the stored
    face profile for the authenticated user.

    On success returns a short-lived JWT (`verificationToken`) that the browser
    must forward to the Cloudflare Worker check-in endpoint.  The Worker
    validates the JWT locally using the shared `FACE_API_SECRET` — no extra
    network call required.
    """

    # 1. Verify Firebase ID token ─────────────────────────────────────────────
    try:
        claims = verify_id_token(payload.idToken)
    except Exception as exc:
        logger.warning("Invalid Firebase ID token: %s", exc)
        raise HTTPException(
            status_code=401,
            detail={"verified": False, "reason": "AUTH_INVALID", "message": "Token autentikasi tidak valid. Masuk kembali lalu coba lagi."},
        )

    uid: str = claims["uid"]

    # 2. Load face profile from Firestore ──────────────────────────────────────
    try:
        profile = get_face_profile(uid)
    except Exception as exc:
        logger.error("Failed to fetch face profile for uid=%s: %s", uid, exc)
        raise HTTPException(
            status_code=503,
            detail={"verified": False, "reason": "PROFILE_UNAVAILABLE", "message": "Profil wajah tidak dapat diambil. Coba lagi."},
        )

    if profile is None:
        return VerifyFaceResponse(
            verified=False,
            similarity=0.0,
            reason="FACE_PROFILE_NOT_FOUND",
            message="Profil wajah belum didaftarkan. Daftarkan profil wajah terlebih dahulu.",
        )

    stored_embedding = profile.get("embedding")
    if not stored_embedding:
        return VerifyFaceResponse(
            verified=False,
            similarity=0.0,
            reason="FACE_PROFILE_EMPTY",
            message="Data profil wajah tidak lengkap. Daftarkan ulang profil wajah.",
        )

    # 3. Compute cosine similarity ─────────────────────────────────────────────
    try:
        result = verify_embedding(
            live_embedding=payload.liveEmbedding,
            profile_embedding=stored_embedding,
        )
    except FaceVerificationError as exc:
        logger.warning("Embedding validation error for uid=%s: %s", uid, exc)
        return VerifyFaceResponse(
            verified=False,
            similarity=0.0,
            reason=exc.code,
            message=str(exc),
        )

    logger.info(
        "Face verification uid=%s attendanceId=%s similarity=%.4f verified=%s",
        uid,
        payload.attendanceId,
        result["similarity"],
        result["verified"],
    )

    if not result["verified"]:
        return VerifyFaceResponse(
            verified=False,
            similarity=result["similarity"],
            reason="FACE_MISMATCH",
            message="Wajah belum cocok dengan profil yang terdaftar. Atur pencahayaan lalu coba lagi.",
        )

    # 4. Issue verification token ──────────────────────────────────────────────
    try:
        token_data = create_verification_token(
            uid=uid,
            attendance_id=payload.attendanceId,
            class_id=payload.classId,
            similarity=result["similarity"],
        )
    except Exception as exc:
        logger.error("Token creation failed for uid=%s: %s", uid, exc)
        raise HTTPException(
            status_code=500,
            detail={"verified": False, "reason": "TOKEN_ERROR", "message": "Gagal membuat token verifikasi. Coba lagi."},
        )

    return VerifyFaceResponse(
        verified=True,
        similarity=result["similarity"],
        verificationToken=token_data["token"],
        expiresAt=token_data["expires_at"],
    )


# ─── Dev entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

