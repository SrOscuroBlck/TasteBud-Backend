import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp
from typing import Callable, Optional, Tuple, Dict

from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

SKIP_PATHS = {"/", "/docs", "/redoc", "/openapi.json", "/api/v1/health"}
SKIP_PREFIXES = ("/api/v1/early-access/approve", "/api/v1/early-access/deny")


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._redis = None
        if settings.RATE_LIMIT_ENABLED:
            try:
                from utils.redis_client import get_redis
                self._redis = get_redis()
            except Exception as e:
                logger.warning(f"Rate limit middleware: Redis init failed, failing open: {e}")

    def _get_identifier(self, request: Request) -> str:
        token = self._extract_token(request)
        if token and self._redis:
            try:
                cached_email = self._redis.get(f"token_email:{token}")
                if cached_email:
                    return f"user:{cached_email}"
            except Exception:
                pass

        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    def _is_exempt(self, request: Request) -> bool:
        if not settings.EARLY_ACCESS_EXEMPT_EMAIL:
            return False

        token = self._extract_token(request)
        if token and self._redis:
            try:
                cached_email = self._redis.get(f"token_email:{token}")
                if cached_email and cached_email.lower() == settings.EARLY_ACCESS_EXEMPT_EMAIL.lower():
                    return True
            except Exception:
                pass
        return False

    @staticmethod
    def _extract_token(request: Request) -> Optional[str]:
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        return None

    def _check_rate_limit(self, identifier: str) -> Tuple[bool, int, Dict[str, str]]:
        """Returns (allowed, retry_after_seconds, headers)."""
        now = time.time()
        pipe = self._redis.pipeline()

        minute_key = f"rl:{identifier}:min"
        hour_key = f"rl:{identifier}:hr"

        # Clean old entries, add current, count, set TTL
        for key, window in [(minute_key, 60), (hour_key, 3600)]:
            pipe.zremrangebyscore(key, 0, now - window)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, window + 1)

        results = pipe.execute()
        # results: [zrem_min, zadd_min, zcard_min, expire_min, zrem_hr, zadd_hr, zcard_hr, expire_hr]
        minute_count = results[2]
        hour_count = results[6]

        headers = {
            "X-RateLimit-Limit-Minute": str(settings.RATE_LIMIT_REQUESTS_PER_MINUTE),
            "X-RateLimit-Remaining-Minute": str(max(0, settings.RATE_LIMIT_REQUESTS_PER_MINUTE - minute_count)),
            "X-RateLimit-Limit-Hour": str(settings.RATE_LIMIT_REQUESTS_PER_HOUR),
            "X-RateLimit-Remaining-Hour": str(max(0, settings.RATE_LIMIT_REQUESTS_PER_HOUR - hour_count)),
        }

        if minute_count > settings.RATE_LIMIT_REQUESTS_PER_MINUTE:
            return False, 60, headers
        if hour_count > settings.RATE_LIMIT_REQUESTS_PER_HOUR:
            return False, 3600, headers

        return True, 0, headers

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not settings.RATE_LIMIT_ENABLED or not self._redis:
            return await call_next(request)

        path = request.url.path
        if path in SKIP_PATHS or any(path.startswith(p) for p in SKIP_PREFIXES):
            return await call_next(request)

        if self._is_exempt(request):
            return await call_next(request)

        identifier = self._get_identifier(request)

        try:
            allowed, retry_after, headers = self._check_rate_limit(identifier)
        except Exception as e:
            logger.warning(f"Rate limit check failed, allowing request: {e}")
            return await call_next(request)

        if not allowed:
            resp = JSONResponse(
                status_code=429,
                content={"detail": "rate_limit_exceeded", "retry_after": retry_after},
            )
            resp.headers["Retry-After"] = str(retry_after)
            for k, v in headers.items():
                resp.headers[k] = v
            return resp

        response = await call_next(request)
        for k, v in headers.items():
            response.headers[k] = v
        return response
