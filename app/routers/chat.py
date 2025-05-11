# app/routers/chat.py
import time
import json
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any, Dict, Optional # Added Optional

from app.config import settings
from app.database import get_db
from app.logging_config import LOGGER
from app.models import OpenRouterRequest
from app.openrouter_client import OpenRouterClient
from app.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatCompletionChoice, OpenAIUsage
from app.media_processing import process_messages_for_media # Updated import
# Add stream response schemas if implementing streaming later

router = APIRouter(prefix="/v1/chat", tags=["Chat Completions"])

# Dependency to get OpenRouterClient from app.state
def get_openrouter_client(request: Request) -> OpenRouterClient:
    return request.app.state.openrouter_client

async def save_request_log_to_db(
    db: AsyncSession,
    internal_request_id: str,
    endpoint_called: str,
    client_ip: Optional[str],
    model_requested_by_client: Optional[str],
    model_routed_to_openrouter: str,
    openrouter_response: Dict[str, Any], # The raw dict from OpenRouterClient
    processing_duration_ms: float,
    input_char_length: int,
    output_char_length: int,
    status_code_returned_to_client: int,
    is_multimodal: bool = False,
    media_type_processed: Optional[str] = None,
    error_source: Optional[str] = None,
    error_message: Optional[str] = None
):
    # Extract data from OpenRouter response
    or_data = openrouter_response.get("data", {})
    # or_usage = or_data.get("usage", {}) if isinstance(or_data, dict) else {} # Not used directly
    
    # If OpenRouter itself returned an error, or_data might be the error dict
    # This check is slightly different from instructions to handle cases where 'error' exists but 'data' might also exist (e.g. partial success)
    # The main goal is to log the error structure if it's the primary outcome.
    if openrouter_response.get("error") and not or_data.get("choices"): # Check if choices are missing, indicating a full error
         or_data_log_target = openrouter_response.get("error") # Log the error structure if no 'data.choices'
    else:
         or_data_log_target = or_data # Log the data structure

    db_log = OpenRouterRequest(
        internal_request_id=internal_request_id,
        # timestamp field is now omitted here to use the model's default
        endpoint_called=endpoint_called,
        client_ip=client_ip,
        model_requested_by_client=model_requested_by_client,
        model_routed_to_openrouter=model_routed_to_openrouter,
        openrouter_request_id=openrouter_response.get("openrouter_request_id"), # This comes from our client wrapper
        prompt_tokens=openrouter_response.get("prompt_tokens"), 
        completion_tokens=openrouter_response.get("completion_tokens"),
        total_tokens=openrouter_response.get("total_tokens"),
        cost_usd=openrouter_response.get("cost_usd"),
        is_multimodal=is_multimodal,
        media_type_processed=media_type_processed,
        input_char_length=input_char_length,
        output_char_length=output_char_length,
        processing_duration_ms=int(processing_duration_ms),
        openrouter_latency_ms=int(openrouter_response.get("latency_ms", 0)), # This comes from our client wrapper
        status_code_returned_to_client=status_code_returned_to_client,
        error_source=error_source or ("openrouter" if openrouter_response.get("error") else None),
        error_message=str(error_message or openrouter_response.get("error")) if (error_message or openrouter_response.get("error")) else None,
        # Storing the relevant part of OpenRouter's response (either data or error)
        # openrouter_response_json=json.dumps(or_data_log_target) # This field is not in the model, removing
    )
    db.add(db_log)
    try:
        await db.commit()
        await db.refresh(db_log)
        LOGGER.info("Request log saved to DB", extra={"internal_request_id": internal_request_id, "db_id": db_log.id})
    except Exception as e:
        await db.rollback()
        LOGGER.error("Failed to save request log to DB", exc_info=True, extra={"internal_request_id": internal_request_id})


@router.post("/completions", response_model=ChatCompletionResponse) # Or StreamingResponse if streaming
async def create_chat_completion(
    request_data: ChatCompletionRequest,
    request: Request, # FastAPI request object to access app.state, client IP etc.
    db: AsyncSession = Depends(get_db),
    or_client: OpenRouterClient = Depends(get_openrouter_client)
):
    start_router_processing_time = time.time()
    internal_request_id = request.state.internal_request_id # From middleware
    http_session = request.app.state.http_session # Get aiohttp session

    # Process messages for media (images or video)
    processed_payload_messages, media_detected, media_type, _ = await process_messages_for_media(
        request_data.messages, http_session
    ) # video_data (transcript/frames) is not directly used here, it's embedded in messages by process_messages_for_media

    is_multimodal_request = media_detected
    
    # Determine model
    if is_multimodal_request:
        effective_model = settings.VISION_MODEL
        LOGGER.info(f"Media detected ({media_type}). Routing to vision model: {effective_model}", extra={"internal_request_id": internal_request_id})
    else:
        effective_model = request_data.model or settings.DEFAULT_TEXT_MODEL

    # Prepare payload for OpenRouter
    openrouter_payload = request_data.model_dump(exclude_none=True)
    openrouter_payload["model"] = effective_model
    openrouter_payload["messages"] = processed_payload_messages # Use processed messages

    # Calculate input_char_length based on potentially modified messages
    current_input_char_length = 0
    for msg_dict in processed_payload_messages:
        content = msg_dict.get("content")
        if isinstance(content, str):
            current_input_char_length += len(content)
        elif isinstance(content, list): # List of content parts
            for part in content:
                if part.get("type") == "text":
                    current_input_char_length += len(part.get("text", ""))
    
    LOGGER.info(
        "Prepared payload for OpenRouter",
        extra={
            "internal_request_id": internal_request_id,
            "target_model": effective_model,
            "is_multimodal": is_multimodal_request,
            # "payload_messages": openrouter_payload["messages"] # Careful with logging full payloads
        }
    )

    # Call OpenRouter
    openrouter_response_dict = await or_client.create_chat_completion(openrouter_payload)
    
    # Handle OpenRouter errors
    if openrouter_response_dict.get("error"):
        error_content = openrouter_response_dict.get("error", {}) # Renamed for clarity
        status_code = openrouter_response_dict.get("status_code", 500)
        
        db_error_message = ""
        log_error_source = "openrouter" # Default

        if isinstance(error_content, dict):
            db_error_message = str(error_content.get("message", error_content))
            # log_error_source can remain "openrouter" or be derived if dict has type info
        elif isinstance(error_content, str):
            db_error_message = error_content
            log_error_source = "client_side_error" # Align with HTTPException type
        else:
            db_error_message = str(error_content) # Fallback for other types
            log_error_source = "unknown_error_format" # Align with HTTPException type

        # Log error to DB
        router_processing_duration_ms = (time.time() - start_router_processing_time) * 1000
        await save_request_log_to_db(
            db=db,
            internal_request_id=internal_request_id,
            endpoint_called=str(request.url.path),
            client_ip=request.client.host if request.client else None,
            model_requested_by_client=request_data.model,
            model_routed_to_openrouter=effective_model,
            openrouter_response=openrouter_response_dict, # Pass the raw dict from client
            processing_duration_ms=router_processing_duration_ms,
            input_char_length=current_input_char_length,
            output_char_length=0, # No output on error
            status_code_returned_to_client=status_code,
            is_multimodal=is_multimodal_request,
            media_type_processed=media_type if is_multimodal_request else None,
            error_source=log_error_source, # Updated error source
            error_message=db_error_message # Use the processed db_error_message
        )
        
        # Construct detail for HTTPException
        exception_detail_error = {}
        if isinstance(error_content, dict):
            exception_detail_error = {
                "message": str(error_content.get("message", "Error from OpenRouter")),
                "type": error_content.get("type", "openrouter_error"),
                "param": error_content.get("param"),
                "code": error_content.get("code"),
            }
        elif isinstance(error_content, str): # Error content is a simple string
            exception_detail_error = {
                "message": error_content,
                "type": "client_side_error",
                "param": None,
                "code": None,
            }
        else: # Fallback for unexpected error_content type
             exception_detail_error = {
                "message": "An unexpected error format was received from the upstream service.",
                "type": "unknown_error_format",
                "param": None,
                "code": None,
            }
        
        exception_detail_error["internal_request_id"] = internal_request_id

        # Return OpenAI-compatible error
        raise HTTPException(
            status_code=status_code,
            detail={"error": exception_detail_error}
        )

    # Process successful response from OpenRouter
    or_data = openrouter_response_dict.get("data", {}) # This 'data' is from OpenRouter's actual API response structure
    
    # Transform OpenRouter response to OpenAI ChatCompletionResponse format
    # This assumes OpenRouter's successful chat response structure is OpenAI-like
    # We need to construct our ChatCompletionResponse Pydantic model
    
    response_choices = []
    output_text_char_length = 0
    for idx, choice_data in enumerate(or_data.get("choices", [])):
        message_data = choice_data.get("message", {})
        content = message_data.get("content") # This should be string or list of parts from OpenRouter
        
        # Calculate output_char_length from text parts if content is a list
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    output_text_char_length += len(part.get("text", ""))
        elif isinstance(content, str):
             output_text_char_length += len(content)

        response_choices.append(
            ChatCompletionChoice(
                index=idx,
                message=ChatMessage( # Ensure ChatMessage can handle list content
                    role=message_data.get("role", "assistant"),
                    content=content,
                ),
                finish_reason=choice_data.get("finish_reason")
            )
        )

    usage_data = or_data.get("usage", {})
    openai_usage = None
    if usage_data: # Ensure usage_data is not None or empty
        openai_usage = OpenAIUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )

    final_response_data = ChatCompletionResponse(
        id=or_data.get("id", internal_request_id), # Use OpenRouter's ID if available
        created=or_data.get("created", int(time.time())),
        model=or_data.get("model", effective_model),
        choices=response_choices,
        usage=openai_usage,
        system_fingerprint=or_data.get("system_fingerprint")
    )
    
    # Log successful request to DB
    router_processing_duration_ms = (time.time() - start_router_processing_time) * 1000
    await save_request_log_to_db(
        db=db,
        internal_request_id=internal_request_id,
        endpoint_called=str(request.url.path),
        client_ip=request.client.host if request.client else None,
        model_requested_by_client=request_data.model,
        model_routed_to_openrouter=effective_model,
        openrouter_response=openrouter_response_dict, # Pass the full dict from client wrapper
        processing_duration_ms=router_processing_duration_ms,
        input_char_length=current_input_char_length,
        output_char_length=output_text_char_length,
        status_code_returned_to_client=200, # Assuming 200 OK
        is_multimodal=is_multimodal_request,
        media_type_processed=media_type if is_multimodal_request else None
    )

    LOGGER.info(
        "Successfully processed chat completion request",
        extra={
            "internal_request_id": internal_request_id,
            "openrouter_request_id": openrouter_response_dict.get("openrouter_request_id"),
            "model_used": effective_model,
            "is_multimodal": is_multimodal_request,
            "media_type": media_type
        }
    )
    return final_response_data