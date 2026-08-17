from __future__ import annotations

import base64
import ipaddress
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from web.api import router


STATIC_DIR = Path(__file__).parent / "static"
app = FastAPI(title="sdcpp-modal", docs_url="/api/docs")


def _is_loopback_request(request: Request) -> bool:
    host = (request.url.hostname or "").strip().lower()
    if host in {"localhost", "testserver"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _authorized(request: Request) -> bool:
    token = os.environ.get("SDCPP_WEB_TOKEN", "")
    if not token:
        return _is_loopback_request(request)
    authorization = request.headers.get("authorization", "")
    candidate = ""
    if authorization.lower().startswith("bearer "):
        candidate = authorization[7:].strip()
    elif authorization.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(authorization[6:].strip()).decode("utf-8")
            _, candidate = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError):
            candidate = ""
    return bool(candidate) and secrets.compare_digest(candidate, token)


@app.middleware("http")
async def require_web_token(request: Request, call_next) -> Response:
    if _authorized(request):
        return await call_next(request)
    token_configured = bool(os.environ.get("SDCPP_WEB_TOKEN"))
    status = 401 if token_configured else 403
    detail = "authentication required" if token_configured else "non-loopback access requires SDCPP_WEB_TOKEN"
    headers = {"WWW-Authenticate": 'Basic realm="sdcpp-modal"'} if token_configured else {}
    if request.url.path.startswith("/api/") and "text/html" not in request.headers.get("accept", ""):
        return JSONResponse({"detail": detail}, status_code=status, headers=headers)
    return Response(detail, status_code=status, headers=headers)


app.include_router(router)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
