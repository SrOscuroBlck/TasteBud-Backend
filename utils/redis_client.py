import redis
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

_client = None


def get_redis():
    global _client
    if not settings.RATE_LIMIT_ENABLED:
        return None
    if _client is None:
        try:
            _client = redis.from_url(
                settings.RATE_LIMIT_REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=1,
            )
            _client.ping()
            logger.info("Redis client connected for rate limiting")
        except Exception as e:
            logger.warning(f"Redis unavailable, rate limiting disabled: {e}")
            _client = None
    return _client
