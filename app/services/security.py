import hashlib
import hmac

from fastapi import HTTPException, Request, status

from app.core.config import Settings


async def verify_exotel_request(request: Request, settings: Settings) -> None:
    """Verify shared token and optional HMAC signature.

    Configure the actual header/canonical string to match the Exotel product enabled
    for your account. Keeping this gate configurable avoids accepting public traffic.
    """
    if not settings.exotel_webhook_token and not settings.exotel_signature_secret:
        return  # Convenient locally; set at least one secret in production.

    token = request.headers.get("X-Exotel-Token") or request.query_params.get("token")
    if settings.exotel_webhook_token and not hmac.compare_digest(
        token or "", settings.exotel_webhook_token
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook token")

    if settings.exotel_signature_secret:
        supplied = request.headers.get("X-Exotel-Signature", "")
        body = await request.body()
        canonical = f"{request.method}\n{request.url.path}\n".encode() + body
        expected = hmac.new(
            settings.exotel_signature_secret.encode(), canonical, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

