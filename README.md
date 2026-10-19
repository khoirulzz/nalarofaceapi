# Nalaro Face API

Server-side face verification service for **Nalaro Class** attendance.

Deployed at: `https://nalarofaceapi.onrender.com`

## Architecture

```
Browser (Human.js)
  └─ extract face embedding from live camera frame
       └─ POST /verify-face  {idToken, liveEmbedding, attendanceId, classId}
            └─ verify Firebase ID Token
            └─ fetch stored embedding from Firestore (faceProfiles/{uid})
            └─ cosine similarity (NumPy) — no ML model on server
            └─ issue short-lived JWT (60 s) signed with FACE_API_SECRET
                 └─ browser sends JWT to Cloudflare Worker check-in endpoint
                      └─ Worker verifies JWT locally (no network call)
                      └─ Worker stores attendance record in D1
```

## Why embedding-only (no raw image)?

- **No biometric photos transmitted or stored** on the Python server.
- **Tiny dependencies** — only `numpy`, `firebase-admin`, and `PyJWT`.
- **Fast** — cosine similarity in NumPy is < 1 ms.
- **Render Free Tier friendly** — service stays well under 512 MB RAM.

## Local development

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

# Copy and fill in the env file
cp .env.example .env
# Edit .env with your Firebase service account JSON and secret

# Run with auto-reload
python main.py
# or
uvicorn main:app --reload
```

Health check: `GET http://localhost:8000/health`

API docs (dev only, set ENABLE_DOCS=1): `http://localhost:8000/docs`

## Deployment (Render)

1. Push this repo to GitHub.
2. Create a new **Web Service** on Render, connect the GitHub repo.
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Set environment variables in Render Dashboard (see `.env.example`).
6. Service URL will be `https://nalarofaceapi.onrender.com`.

### Required Render env vars

| Variable | Description |
|----------|-------------|
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Full Firebase service account JSON (compact/single-line) |
| `FACE_API_SECRET` | Shared secret with Cloudflare Worker (≥ 32 chars) |
| `FACE_MATCH_THRESHOLD` | Cosine similarity threshold (default `0.72`) |
| `FACE_TOKEN_TTL` | JWT validity in seconds (default `60`) |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins |

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | None | Health check |
| `POST` | `/verify-face` | Firebase ID Token in body | Verify embedding, return JWT |

## Security notes

- The `FACE_API_SECRET` must be the **same** value set as `FACE_API_SECRET` in Cloudflare Worker secrets.
- JWT payload includes `uid`, `attendanceId`, `classId` — Worker must verify all three match the request.
- Tokens expire in 60 seconds — replay attacks are limited to that window.
- No images or raw biometric data are stored on this server.

