# app/routers/completions.py
import time
import json
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any, Dict, List, Union # Added List, Union

from app.config import settings
from app.database import get_db
from app.logging_config import LOGGER
from app.models import OpenRouterRequest
from app.openrouter_client import OpenRouterClient
from app.schemas import (
    CompletionRequest, CompletionResponse, CompletionChoice, OpenAIUsage,
    ChatMessage # For transforming to chat messages if media detected
)
from app.media_processing import process_messages_for_media
# Import the save_request_log_to_db utility
from app.routers.chat import save_request_log_to_db, get_openrouter_client # Re-use if suitable

router = APIRouter(prefix="/v1", tags=["Legacy Completions"])


@router.post("/completions", response_model=CompletionResponse)
async def create_legacy_completion(
    request_data: CompletionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    or_client: OpenRouterClient = Depends(get_openrouter_client)
):
    start_router_processing_time = time.time()
    internal_request_id = request.state.internal_request_id
    http_session = request.app.state.http_session

    # Handle prompt: if list, take first element (OpenAI allows batching, we simplify)
    prompt_text = ""
    if isinstance(request_data.prompt, list):
        if request_data.prompt:
            prompt_text = request_data.prompt[0]
        else: # Empty list of prompts
            raise HTTPException(status_code=400, detail="Prompt cannot be an empty list.")
    else:
        prompt_text = request_data.prompt

    # Transform prompt into a ChatMessage list for media processing
    # This allows reusing the existing media processing logic.
    temp_chat_messages = [ChatMessage(role="user", content=prompt_text)]
    
    processed_payload_messages, media_detected, media_type, _ = await process_messages_for_media(
        temp_chat_messages, http_session
    )

    is_multimodal_request = media_detected
    
    # Determine model and prepare OpenRouter payload
    openrouter_payload = {}
    if is_multimodal_request:
        effective_model = settings.VISION_MODEL
        LOGGER.info(f"Media detected in legacy prompt ({media_type}). Routing to vision model: {effective_model}", 
                    extra={"internal_request_id": internal_request_id})
        # Payload for multimodal needs to be in chat format
        openrouter_payload = {
            "model": effective_model,
            "messages": processed_payload_messages, # Already structured by process_messages_for_media
            "temperature": request_data.temperature,
            "top_p": request_data.top_p,
            "n": request_data.n,
            "stream": request_data.stream, # Note: Streaming for legacy + media needs careful thought
            "stop": request_data.stop,
            "max_tokens": request_data.max_tokens,
            "presence_penalty": request_data.presence_penalty,
            "frequency_penalty": request_data.frequency_penalty,
            "user": request_data.user,
        }
    else: # Text-only legacy completion
        effective_model = request_data.model or settings.DEFAULT_TEXT_MODEL
        LOGGER.info(f"Text-only legacy prompt. Routing to model: {effective_model} via chat format.",
                    extra={"internal_request_id": internal_request_id})
        openrouter_payload = {
            "model": effective_model,
            "messages": [{"role": "user", "content": prompt_text}], # Simple text prompt
            "temperature": request_data.temperature,
            "top_p": request_data.top_p,
            "n": request_data.n,
            "stream": request_data.stream,
            "stop": request_data.stop,
            "max_tokens": request_data.max_tokens,
            "presence_penalty": request_data.presence_penalty,
            "frequency_penalty": request_data.frequency_penalty,
            "user": request_data.user,
            "logit_bias": request_data.logit_bias,
        }
    
    # Remove None values from payload before sending
    openrouter_payload = {k: v for k, v in openrouter_payload.items() if v is not None}

    # Calculate input_char_length
    current_input_char_length = len(prompt_text)
    if is_multimodal_request: # Recalculate if messages were transformed
        current_input_char_length = 0
        for msg_dict_model in processed_payload_messages: # msg_dict_model is ChatMessage model instance
            content = msg_dict_model.content
            if isinstance(content, str): current_input_char_length += len(content)
            elif isinstance(content, list): # content is List[Dict[str, Any]]
                for part in content:
                    if part.get("type") == "text": current_input_char_length += len(part.get("text", ""))
    
    LOGGER.info(
        "Prepared payload for OpenRouter (legacy completions endpoint)",
        extra={
            "internal_request_id": internal_request_id,
            "payload_to_openrouter": openrouter_payload,
            "is_multimodal": is_multimodal_request,
            "media_type": media_type if is_multimodal_request else None,
        }
    )

    # Call OpenRouter (using create_chat_completion method of client)
    openrouter_response_dict = await or_client.create_chat_completion(openrouter_payload)

    # Handle OpenRouter errors
    if openrouter_response_dict.get("error"):
        error_content = openrouter_response_dict.get("error", {}) # Renamed for clarity
        status_code = openrouter_response_dict.get("status_code", 500)
        
        # Ensure status_code is a valid HTTP status code integer
        if not isinstance(status_code, int) or not (100 <= status_code <= 599):
            LOGGER.warning(f"Invalid status_code from OpenRouter error: {status_code}. Defaulting to 500.",
                           extra={"internal_request_id": internal_request_id})
            status_code = 500

        db_error_message = ""
        log_error_source = "openrouter" # Default

        if isinstance(error_content, dict):
            db_error_message = str(error_content.get("message", error_content))
        elif isinstance(error_content, str):
            db_error_message = error_content
            log_error_source = "client_side_error"
        else:
            db_error_message = str(error_content)
            log_error_source = "unknown_error_format"

        router_processing_duration_ms = (time.time() - start_router_processing_time) * 1000
        
        await save_request_log_to_db(
            db=db, internal_request_id=internal_request_id, endpoint_called=str(request.url.path),
            client_ip=request.client.host if request.client else None, # Added check for request.client
            model_requested_by_client=request_data.model,
            model_routed_to_openrouter=effective_model, openrouter_response=openrouter_response_dict,
            processing_duration_ms=router_processing_duration_ms, input_char_length=current_input_char_length,
            output_char_length=0, status_code_returned_to_client=status_code,
            is_multimodal=is_multimodal_request, media_type_processed=media_type if is_multimodal_request else None,
            error_source=log_error_source, # Use the determined error source
            error_message=db_error_message # Use the processed db_error_message
        )
        
        LOGGER.error(
            f"Error from OpenRouter (legacy completions): {db_error_message}", # Use db_error_message for logging
            extra={
                "internal_request_id": internal_request_id,
                "openrouter_response": openrouter_response_dict,
                "status_code_from_openrouter": status_code,
            }
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

        raise HTTPException(
            status_code=status_code,
            detail={"error": exception_detail_error}
        )

    # Process successful response from OpenRouter (which will be in ChatCompletion format)
    or_data = openrouter_response_dict.get("data", {})
    
    # Transform ChatCompletion response from OpenRouter to legacy CompletionResponse format
    response_choices = []
    output_text_char_length = 0
    if or_data.get("choices"):
        for idx, choice_data in enumerate(or_data["choices"]):
            # Extract text content from chat message structure
            chat_message_content = choice_data.get("message", {}).get("content", "")
            text_output = ""
            if isinstance(chat_message_content, str):
                text_output = chat_message_content
            elif isinstance(chat_message_content, list): # Multimodal output, extract text parts
                for part in chat_message_content:
                    if part.get("type") == "text":
                        text_output += part.get("text", "")
            
            output_text_char_length += len(text_output)
            response_choices.append(
                CompletionChoice(
                    text=text_output,
                    index=idx, # OpenAI spec uses choice_data.get("index", idx) but OR might not provide it
                    finish_reason=choice_data.get("finish_reason")
                    # logprobs might need specific handling if OpenRouter provides them differently
                )
            )
    
    usage_data = or_data.get("usage", {})
    openai_usage = OpenAIUsage(
        prompt_tokens=usage_data.get("prompt_tokens", 0),
        completion_tokens=usage_data.get("completion_tokens", 0),
        total_tokens=usage_data.get("total_tokens", 0)
    )

    final_response_data = CompletionResponse(
        id=or_data.get("id", internal_request_id),
        created=or_data.get("created", int(time.time())),
        model=or_data.get("model", effective_model), # Model that OpenRouter actually used
        choices=response_choices,
        usage=openai_usage,
        system_fingerprint=or_data.get("system_fingerprint")
    )
    
    # Log successful request to DB
    router_processing_duration_ms = (time.time() - start_router_processing_time) * 1000
    await save_request_log_to_db(
        db=db, internal_request_id=internal_request_id, endpoint_called=str(request.url.path),
        client_ip=request.client.host, model_requested_by_client=request_data.model,
        model_routed_to_openrouter=effective_model, openrouter_response=openrouter_response_dict,
        processing_duration_ms=router_processing_duration_ms, input_char_length=current_input_char_length,
        output_char_length=output_text_char_length, status_code_returned_to_client=200,
        is_multimodal=is_multimodal_request, media_type_processed=media_type if is_multimodal_request else None
    )

    LOGGER.info(
        "Successfully processed legacy completion request and sent response to client.",
        extra={
            "internal_request_id": internal_request_id,
            "response_to_client": final_response_data.model_dump(exclude_none=True),
            "model_used": final_response_data.model,
            "usage": openai_usage.model_dump(),
        }
    )
    return final_response_data