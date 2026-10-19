"""Tests for token_service."""

import os
import time

import pytest

# Set required env var before importing the module under test
os.environ.setdefault("FACE_API_SECRET", "test_secret_that_is_long_enough_for_testing_1234")
os.environ.setdefault("FACE_TOKEN_TTL", "60")

from services.token_service import create_verification_token, decode_verification_token
import jwt as pyjwt


def test_token_round_trip():
    result = create_verification_token(
        uid="user-abc",
        attendance_id="att-123",
        class_id="cls-456",
        similarity=0.87,
    )
    assert "token" in result
    assert "expires_at" in result

    decoded = decode_verification_token(result["token"])
    assert decoded["uid"] == "user-abc"
    assert decoded["attendanceId"] == "att-123"
    assert decoded["classId"] == "cls-456"
    assert abs(decoded["similarity"] - 0.87) < 1e-4
    assert decoded["iss"] == "nalaro-face-api"


def test_token_has_correct_ttl():
    before = int(time.time())
    result = create_verification_token("u", "a", "c", 0.8)
    decoded = decode_verification_token(result["token"])
    after = int(time.time())
    ttl = int(os.environ["FACE_TOKEN_TTL"])
    assert decoded["exp"] - decoded["iat"] == ttl
    assert before <= decoded["iat"] <= after + 1


def test_tampered_token_raises():
    result = create_verification_token("u", "a", "c", 0.8)
    tampered = result["token"][:-4] + "XXXX"
    with pytest.raises(pyjwt.exceptions.PyJWTError):
        decode_verification_token(tampered)


def test_wrong_secret_raises():
    result = create_verification_token("u", "a", "c", 0.8)
    # Temporarily change secret
    original = os.environ["FACE_API_SECRET"]
    try:
        import services.token_service as ts
        ts._SECRET = "completely_different_secret_that_is_long_enough_abc"
        with pytest.raises(pyjwt.exceptions.PyJWTError):
            decode_verification_token(result["token"])
    finally:
        ts._SECRET = original

