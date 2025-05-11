# app/routers/ui.py
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, asc # For querying and ordering
from typing import List, Optional

from app.database import get_db
from app.models import OpenRouterRequest
from app.config import settings # For future UI config if needed
from app.logging_config import LOGGER # Import the logger

router = APIRouter(prefix="/ui", tags=["Tracking UI"])

# Configure Jinja2Templates
# Assuming 'templates' directory is at the root of the project, sibling to 'app/'
templates = Jinja2Templates(directory="templates")

@router.get("/tracking", response_class=HTMLResponse)
async def get_tracking_ui(request: Request):
    """Serves the main HTML page for tracking UI."""
    return templates.TemplateResponse("tracking_ui.html", {"request": request, "title": "API Request Tracking"})

@router.get("/tracking_data")
async def get_tracking_data(
    request: Request,
    db: AsyncSession = Depends(get_db),
    page: int = 1,
    page_size: int = 20,
    sort_by: Optional[str] = "timestamp",
    sort_order: Optional[str] = "desc" # 'asc' or 'desc'
):
    """API endpoint to fetch paginated and sorted tracking data."""
    if page < 1: page = 1
    if page_size < 1: page_size = 1
    if page_size > 100: page_size = 100 # Max page size

    offset = (page - 1) * page_size

    # Base query
    query = select(OpenRouterRequest)
    total_query = select(func.count()).select_from(OpenRouterRequest)

    # Sorting
    # Ensure sort_by is a valid column name to prevent SQL injection if it were user-provided without validation
    # Here, it's from a select dropdown, but good practice.
    valid_sort_columns = {
        "timestamp", "internal_request_id", "endpoint_called", 
        "model_requested_by_client", "model_routed_to_openrouter",
        "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd",
        "status_code_returned_to_client", "processing_duration_ms", "openrouter_latency_ms"
    }
    if sort_by not in valid_sort_columns:
        sort_by = "timestamp" # Default to timestamp if invalid

    sort_column = getattr(OpenRouterRequest, sort_by, OpenRouterRequest.timestamp)
    
    if sort_order == "asc":
        query = query.order_by(asc(sort_column))
    else: # Default to desc
        query = query.order_by(desc(sort_column))
    
    # Pagination
    query = query.offset(offset).limit(page_size)

    try:
        total_count_result = await db.execute(total_query)
        total_items = total_count_result.scalar_one()

        results = await db.execute(query)
        items = results.scalars().all()
        
        items_as_dicts = []
        for item in items:
            items_as_dicts.append({
                "id": item.id,
                "timestamp": item.timestamp.isoformat() if item.timestamp else None,
                "internal_request_id": item.internal_request_id,
                "endpoint_called": item.endpoint_called,
                "model_requested_by_client": item.model_requested_by_client,
                "model_routed_to_openrouter": item.model_routed_to_openrouter,
                "prompt_tokens": item.prompt_tokens,
                "completion_tokens": item.completion_tokens,
                "total_tokens": item.total_tokens,
                "cost_usd": item.cost_usd,
                "status_code_returned_to_client": item.status_code_returned_to_client,
                "error_message": item.error_message,
                "processing_duration_ms": item.processing_duration_ms,
                "openrouter_latency_ms": item.openrouter_latency_ms,
                "is_multimodal": item.is_multimodal,
                "media_type_processed": item.media_type_processed
            })

        return {
            "items": items_as_dicts,
            "total_items": total_items,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_items + page_size - 1) // page_size if page_size > 0 else 0
        }

    except Exception as e:
        LOGGER.error(f"Error fetching tracking data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error fetching tracking data.")