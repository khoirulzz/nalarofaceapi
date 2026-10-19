"""Exercise browser-to-Python contracts without credentials or real biometrics."""
import os
from unittest.mock import patch

os.environ.setdefault('FACE_API_SECRET', 'test_secret_that_is_long_enough_for_testing_1234')
os.environ.setdefault('FIREBASE_SERVICE_ACCOUNT_JSON', '{}')

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
BODY = {'idToken': 'synthetic-token', 'liveEmbedding': [1.0] * 512, 'attendanceId': 'att-1', 'classId': 'class-1'}
PROFILE = {'embedding': [1.0] * 512, 'status': 'active', 'modelVersion': 'human-v1', 'schemaVersion': 1}


def test_production_cors_preflight():
    for origin in ['https://class.nalaro.digital', 'https://nalaroclass.pages.dev', 'https://nalaro.web.id']:
        response = client.options('/verify-face', headers={'Origin': origin, 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type'})
        assert response.status_code == 200
        assert response.headers['access-control-allow-origin'] == origin
    denied = client.options('/verify-face', headers={'Origin': 'https://attacker.test', 'Access-Control-Request-Method': 'POST'})
    assert denied.status_code == 400


def test_authentication_error_is_structured_and_has_cors():
    with patch('main.verify_id_token', side_effect=ValueError('invalid')):
        response = client.post('/verify-face', json=BODY, headers={'Origin': 'https://class.nalaro.digital'})
    assert response.status_code == 401
    assert response.json()['detail']['reason'] == 'AUTH_INVALID'
    assert response.headers['access-control-allow-origin'] == 'https://class.nalaro.digital'


def test_valid_profile_issues_worker_compatible_token():
    from services.token_service import decode_verification_token
    with patch('main.verify_id_token', return_value={'uid': 'u-1'}), patch('main.get_face_profile', return_value=PROFILE):
        response = client.post('/verify-face', json=BODY)
    data = response.json()
    assert data['verified'] is True
    claims = decode_verification_token(data['verificationToken'])
    assert claims['uid'] == 'u-1'
    assert claims['attendanceId'] == BODY['attendanceId']
    assert claims['classId'] == BODY['classId']


def test_deleted_or_incompatible_profiles_never_issue_tokens():
    for change in [{'status': 'deleted'}, {'schemaVersion': 2}, {'modelVersion': 'other'}]:
        with patch('main.verify_id_token', return_value={'uid': 'u-1'}), patch('main.get_face_profile', return_value={**PROFILE, **change}):
            response = client.post('/verify-face', json=BODY)
        assert response.json()['reason'] == 'FACE_PROFILE_INVALID'
        assert response.json()['verificationToken'] is None


def test_request_limits_reject_oversized_or_missing_data():
    for change in [{'liveEmbedding': [1.0] * 2049}, {'liveEmbedding': [1.0]}, {'classId': ''}, {'idToken': ''}]:
        response = client.post('/verify-face', json={**BODY, **change})
        assert response.status_code == 422
