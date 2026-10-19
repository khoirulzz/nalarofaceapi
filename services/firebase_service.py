"""Firebase Admin SDK initialization and Firestore operations."""

import json
import os
import logging
from typing import Optional

import firebase_admin
from firebase_admin import credentials, auth, firestore

logger = logging.getLogger(__name__)

_app: Optional[firebase_admin.App] = None


def _init_app() -> firebase_admin.App:
    """Initialize Firebase Admin SDK from environment variable."""
    global _app
    if _app is not None:
        return _app

    service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    if not service_account_json:
        raise RuntimeError(
            "FIREBASE_SERVICE_ACCOUNT_JSON environment variable is not set."
        )

    try:
        service_account_info = json.loads(service_account_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON."
        ) from exc

    cred = credentials.Certificate(service_account_info)
    _app = firebase_admin.initialize_app(cred)
    logger.info("Firebase Admin SDK initialized.")
    return _app


def get_firebase_app() -> firebase_admin.App:
    return _init_app()


def verify_id_token(id_token: str) -> dict:
    """
    Verify a Firebase ID token and return its decoded claims.
    Raises firebase_admin.auth.InvalidIdTokenError on failure.
    """
    _init_app()
    decoded = auth.verify_id_token(id_token)
    return decoded


def get_face_profile(uid: str) -> Optional[dict]:
    """
    Fetch the face profile document for a given UID from Firestore.
    Returns None if the document does not exist.
    """
    _init_app()
    db = firestore.client()
    doc_ref = db.collection("faceProfiles").document(uid)
    snapshot = doc_ref.get()
    if not snapshot.exists:
        return None
    return snapshot.to_dict()

