from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import logging
from sqlalchemy import text
from app.models.database import SessionLocal
import concurrent.futures

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_health_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="health-probe")

def _check_db():
    db = SessionLocal()
    try:
        db.execute(text("SET LOCAL statement_timeout = 5000"))
        res = db.execute(text("SELECT 1")).scalar()
        if res != 1:
            raise ValueError("Unexpected query result")
        return True
    finally:
        db.close()

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

