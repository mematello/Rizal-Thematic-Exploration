from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from app.models.database import _db_url
import concurrent.futures

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Dedicated health probe engine with driver-level connect_timeout and statement_timeout.
# Uses NullPool so that connections are immediately opened and closed without holding pooled connections.
_health_engine = create_engine(
    _db_url,
    connect_args={
        "connect_timeout": 5,
        "options": "-c statement_timeout=5000"
    },
    poolclass=NullPool
)

# Bounded executor to isolate health probe concurrency (maximum 2 worker threads)
_health_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="health-probe")

def _check_db():
    with _health_engine.connect() as conn:
        res = conn.execute(text("SELECT 1 FROM sentences LIMIT 1")).scalar()
        if res != 1:
            raise ValueError("Unexpected query result")
        return True

@router.get("/health/db")
@router.get("/api/v1/health/db")
def health_db():
    try:
        future = _health_executor.submit(_check_db)
        future.result(timeout=5.0)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ok", "db": "ok"},
            headers={"Cache-Control": "no-store"}
        )
    except Exception as exc:
        logger.warning("Database health probe failed: %s", type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "degraded", "db": "error"},
            headers={"Cache-Control": "no-store"}
        )
