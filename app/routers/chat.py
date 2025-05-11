# app/routers/chat.py
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    BackgroundTasks,
)  # Added BackgroundTasks
from fastapi.responses import StreamingResponse  # Added
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any, Dict, Optional, AsyncGenerator  # Added AsyncGenerator
import json  # Added
import time  # Added
import datetime  # Added

from app.config import settings
from app.database import get_db, AsyncSessionLocal as SessionLocal  # Corrected import
from app.logging_config import LOGGER
from app.models import OpenRouterRequest
from app.openrouter_client import OpenRouterClient
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionChoice,
    OpenAIUsage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatCompletionChoiceDelta,  # Added Stream Schemas
)
from app.media_processing import process_messages_for_media

router = APIRouter(prefix="/v1/chat", tags=["Chat Completions"])


# Dependency to get OpenRouterClient from app.state
def get_openrouter_client(request: Request) -> OpenRouterClient:
    return request.app.state.openrouter_client


# Wrapper function for background task logging
async def _save_log_wrapper_for_bg(
    internal_request_id: str,
    endpoint_called: str,
    client_ip: Optional[str],
    model_requested_by_client: Optional[str],
    model_routed_to_openrouter: str,
    openrouter_response_summary: Dict[str, Any],
    processing_duration_ms: float,
    input_char_length: int,
    output_char_length: int,
    status_code_returned_to_client: int,
    is_streaming: bool = False,
    is_multimodal: bool = False,
    media_type_processed: Optional[str] = None,
    error_source: Optional[str] = None,
    error_message: Optional[str] = None,
):
    async with SessionLocal() as db_for_bg:
        await save_request_log_to_db(
            db=db_for_bg,
            internal_request_id=internal_request_id,
            endpoint_called=endpoint_called,
            client_ip=client_ip,
            model_requested_by_client=model_requested_by_client,
            model_routed_to_openrouter=model_routed_to_openrouter,
            openrouter_response_summary=openrouter_response_summary,
            processing_duration_ms=processing_duration_ms,
            input_char_length=input_char_length,
            output_char_length=output_char_length,
            status_code_returned_to_client=status_code_returned_to_client,
            is_streaming=is_streaming,
            is_multimodal=is_multimodal,
            media_type_processed=media_type_processed,
            error_source=error_source,
            error_message=error_message,
        )


async def save_request_log_to_db(
    db: AsyncSession,
    internal_request_id: str,
    endpoint_called: str,
    client_ip: Optional[str],
    model_requested_by_client: Optional[str],
    model_routed_to_openrouter: str,
    openrouter_response_summary: Dict[str, Any],
    processing_duration_ms: float,
    input_char_length: int,
    output_char_length: int,
    status_code_returned_to_client: int,
    is_streaming: bool = False,
    is_multimodal: bool = False,
    media_type_processed: Optional[str] = None,
    error_source: Optional[str] = None,
    error_message: Optional[str] = None,
):
    or_data = openrouter_response_summary.get("data", {})
    or_error = openrouter_response_summary.get("error")

    db_log = OpenRouterRequest(
        internal_request_id=internal_request_id,
        timestamp=datetime.datetime.utcnow(),
        endpoint_called=endpoint_called,
        client_ip=client_ip,
        model_requested_by_client=model_requested_by_client,
        model_routed_to_openrouter=model_routed_to_openrouter,
        openrouter_request_id=openrouter_response_summary.get("openrouter_request_id"),
        prompt_tokens=openrouter_response_summary.get("prompt_tokens"),
        completion_tokens=openrouter_response_summary.get("completion_tokens"),
        total_tokens=openrouter_response_summary.get("total_tokens"),
        cost_usd=openrouter_response_summary.get("cost_usd"),
        is_streaming=is_streaming,
        is_multimodal=is_multimodal,
        media_type_processed=media_type_processed,
        input_char_length=input_char_length,
        output_char_length=output_char_length,
        processing_duration_ms=int(processing_duration_ms),
        openrouter_latency_ms=int(openrouter_response_summary.get("latency_ms", 0)),
        status_code_returned_to_client=status_code_returned_to_client,
        error_source=error_source or ("openrouter" if or_error else None),
        error_message=(
            str(error_message or or_error) if (error_message or or_error) else None
        ),
    )
    db.add(db_log)
    try:
        await db.commit()
        await db.refresh(db_log)
        LOGGER.info(
            "Request log saved to DB",
            extra={
                "internal_request_id": internal_request_id,
                "db_id": db_log.id,
                "is_streaming_log": is_streaming,
            },
        )
    except Exception as e:
        await db.rollback()
        LOGGER.error(
            "Failed to save request log to DB",
            exc_info=True,
            extra={"internal_request_id": internal_request_id},
        )


@router.post("/completions")
async def create_chat_completion(
    request_data: ChatCompletionRequest,
    request: Request,
    background_tasks: BackgroundTasks,  # Added
    db: AsyncSession = Depends(
        get_db
    ),  # Remains for now, though direct usage will be removed
    or_client: OpenRouterClient = Depends(get_openrouter_client),
):
    start_router_processing_time = time.time()
    internal_request_id = request.state.internal_request_id
    http_session = request.app.state.http_session

    processed_payload_messages, media_detected, media_type, _ = (
        await process_messages_for_media(request_data.messages, http_session)
    )
    is_multimodal_request = media_detected
    effective_model = request_data.model or (
        settings.VISION_MODEL if is_multimodal_request else settings.DEFAULT_TEXT_MODEL
    )
    if is_multimodal_request and not settings.VISION_MODEL:
        LOGGER.error(
            "Multimodal request received but no VISION_MODEL configured.",
            extra={"internal_request_id": internal_request_id},
        )
        raise HTTPException(
            status_code=501, detail="Server not configured for multimodal requests."
        )
    if (
        is_multimodal_request and effective_model != settings.VISION_MODEL
    ):  # Ensure vision model is used if multimodal
        LOGGER.warning(
            f"Multimodal request for model {request_data.model}, but will be routed to VISION_MODEL {settings.VISION_MODEL}",
            extra={"internal_request_id": internal_request_id},
        )
        effective_model = settings.VISION_MODEL

    openrouter_payload = request_data.model_dump(exclude_none=True)
    openrouter_payload["model"] = effective_model
    openrouter_payload["messages"] = processed_payload_messages
    openrouter_payload["stream"] = request_data.stream

    current_input_char_length = sum(
        (
            len(msg.get("content", ""))
            if isinstance(msg.get("content"), str)
            else sum(
                len(part.get("text", ""))
                for part in msg.get("content", [])
                if part.get("type") == "text"
            )
        )
        for msg in processed_payload_messages
    )

    LOGGER.info(
        "Prepared payload for OpenRouter (Chat)",
        extra={
            "internal_request_id": internal_request_id,
            "target_model": effective_model,
            "is_multimodal": is_multimodal_request,
            "streaming_requested": request_data.stream,
        },
    )

    if request_data.stream:

        async def stream_generator() -> AsyncGenerator[str, None]:
            aggregated_completion_tokens = 0
            aggregated_prompt_tokens = 0
            aggregated_total_tokens = None  # Initialize to None
            aggregated_cost_usd = 0.0
            output_text_char_count = 0
            final_openrouter_request_id = None
            # final_latency_ms = 0 # This will be the total processing time for the request, not just OR latency
            stream_had_errors = False
            error_details_for_log = None
            response_model_name = effective_model
            response_id_from_stream = internal_request_id

            stream_start_time = time.time()

            try:
                idx = 0
                async for or_chunk_dict in or_client.create_chat_completion(
                    openrouter_payload
                ):
                    if or_chunk_dict.get("error"):
                        stream_had_errors = True
                        error_details_for_log = or_chunk_dict.get("error")
                        LOGGER.error(
                            f"Error during OpenRouter stream: {error_details_for_log}",
                            extra={
                                "internal_request_id": internal_request_id,
                                "chunk": or_chunk_dict,
                            },
                        )
                        # Optionally send an error formatted SSE to client
                        # yield f"data: {json.dumps({'error': error_details_for_log})}\n\n"
                        break

                    if or_chunk_dict.get("stream_done") or or_chunk_dict.get(
                        "stream_ended_without_done"
                    ):
                        final_openrouter_request_id = or_chunk_dict.get(
                            "openrouter_request_id", final_openrouter_request_id
                        )
                        if or_chunk_dict.get("stream_ended_without_done"):
                            LOGGER.warning(
                                "OpenRouter stream ended without [DONE] message.",
                                extra={"internal_request_id": internal_request_id},
                            )
                        break

                    chunk_data = or_chunk_dict.get("data")
                    if not chunk_data:
                        LOGGER.warning(
                            "Empty data chunk received in stream.",
                            extra={
                                "internal_request_id": internal_request_id,
                                "chunk": or_chunk_dict,
                            },
                        )
                        continue

                    final_openrouter_request_id = or_chunk_dict.get(
                        "openrouter_request_id", final_openrouter_request_id
                    )  # Update if available in every chunk
                    response_id_from_stream = chunk_data.get(
                        "id", response_id_from_stream
                    )
                    response_model_name = chunk_data.get("model", response_model_name)

                    stream_choices = []
                    for choice_idx, or_choice in enumerate(
                        chunk_data.get("choices", [])
                    ):
                        delta_content = or_choice.get("delta", {}).get("content")
                        if delta_content:
                            output_text_char_count += len(delta_content)

                        stream_choices.append(
                            ChatCompletionStreamChoice(
                                index=or_choice.get("index", choice_idx),
                                delta=ChatMessage(
                                    role=or_choice.get("delta", {}).get("role"),
                                    content=delta_content,
                                    tool_calls=or_choice.get("delta", {}).get(
                                        "tool_calls"
                                    ),
                                ),
                                finish_reason=or_choice.get("finish_reason"),
                                logprobs=or_choice.get("logprobs"),
                            )
                        )

                    # Accumulate usage if OpenRouter provides it in stream chunks (often in the last one or via X-Headers)
                    # This part is speculative for OpenRouter; typically usage is at the end.
                    if chunk_data.get(
                        "usage"
                    ):  # Check if 'usage' is in the chunk_data itself
                        usage_chunk = chunk_data.get("usage")
                        aggregated_prompt_tokens = usage_chunk.get(
                            "prompt_tokens", aggregated_prompt_tokens
                        )
                        aggregated_completion_tokens = usage_chunk.get(
                            "completion_tokens", aggregated_completion_tokens
                        )
                        aggregated_total_tokens = usage_chunk.get(
                            "total_tokens", aggregated_total_tokens
                        )  # Assign here
                        aggregated_cost_usd = usage_chunk.get(
                            "cost", aggregated_cost_usd
                        )

                    sse_event = ChatCompletionStreamResponse(
                        id=response_id_from_stream,
                        created=chunk_data.get("created", int(time.time())),
                        model=response_model_name,
                        choices=stream_choices,
                        system_fingerprint=chunk_data.get("system_fingerprint"),
                    )
                    yield f"data: {sse_event.model_dump_json(exclude_none=True)}\n\n"  # Corrected to \n\n
                    idx += 1

                # Ensure [DONE] is sent if loop finishes without error or explicit break for error
                if not stream_had_errors:
                    yield "data: [DONE]\n\n"
                elif (
                    error_details_for_log
                ):  # If there was an error, and we have details
                    # Optionally send a final error SSE before closing, then [DONE] or just close
                    # For now, we just log and the client might see a broken stream if error was mid-data
                    # If an error was already sent or handled, this [DONE] might be confusing.
                    # Consider if [DONE] should always be sent or only on clean exit.
                    # For now, if stream_had_errors, we assume the stream might be compromised.
                    # However, OpenAI usually sends [DONE] even after errors in some cases.
                    # Let's send [DONE] to see if it helps client termination.
                    yield "data: [DONE]\n\n"

            finally:
                # Log after stream completion
                router_processing_duration_ms = (
                    time.time() - start_router_processing_time
                ) * 1000
                openrouter_stream_duration_ms = (time.time() - stream_start_time) * 1000

                # If prompt/completion tokens were not in stream, they might be None.
                # Cost might also be None if not provided or aggregated.
                total_tokens_for_log = None
                if (
                    aggregated_prompt_tokens is not None
                    and aggregated_completion_tokens is not None
                ):
                    total_tokens_for_log = (
                        aggregated_prompt_tokens + aggregated_completion_tokens
                    )
                elif aggregated_total_tokens is not None:
                    total_tokens_for_log = aggregated_total_tokens

                log_summary = {
                    "openrouter_request_id": final_openrouter_request_id,
                    "prompt_tokens": (
                        aggregated_prompt_tokens if aggregated_prompt_tokens else None
                    ),
                    "completion_tokens": (
                        aggregated_completion_tokens
                        if aggregated_completion_tokens
                        else None
                    ),
                    "total_tokens": total_tokens_for_log,
                    "cost_usd": aggregated_cost_usd if aggregated_cost_usd else None,
                    "latency_ms": openrouter_stream_duration_ms,  # This is duration of OR interaction
                    "error": error_details_for_log if stream_had_errors else None,
                    "data": {
                        "id": response_id_from_stream,
                        "model": response_model_name,
                    },  # Minimal data for context
                }

                background_tasks.add_task(
                    _save_log_wrapper_for_bg,
                    internal_request_id=internal_request_id,
                    endpoint_called=str(request.url.path),
                    client_ip=request.client.host,
                    model_requested_by_client=request_data.model,
                    model_routed_to_openrouter=effective_model,
                    openrouter_response_summary=log_summary,
                    processing_duration_ms=router_processing_duration_ms,
                    input_char_length=current_input_char_length,
                    output_char_length=output_text_char_count,
                    status_code_returned_to_client=(
                        200
                        if not stream_had_errors
                        else 500  # Or a more specific error code
                    ),
                    is_streaming=True,
                    is_multimodal=is_multimodal_request,
                    media_type_processed=media_type if is_multimodal_request else None,
                    error_source=(
                        "openrouter_stream"
                        if stream_had_errors and error_details_for_log
                        else None
                    ),
                    error_message=(
                        str(error_details_for_log)
                        if stream_had_errors and error_details_for_log
                        else None
                    ),
                )
                LOGGER.info(
                    "Finished streaming chat completion to client (logging in background).",
                    extra={
                        "internal_request_id": internal_request_id,
                        "had_errors": stream_had_errors,
                        "total_chunks_sent_to_client": idx,  # Ensure idx is defined in finally's scope
                    },
                )

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:  # Non-streaming
        openrouter_response_dict = None
        # The client's create_chat_completion is an async generator. Consume the single item.
        async for item in or_client.create_chat_completion(openrouter_payload):
            openrouter_response_dict = item
            break  # Expect only one item for non-streaming

        router_processing_duration_ms = (
            time.time() - start_router_processing_time
        ) * 1000

        if not openrouter_response_dict:
            LOGGER.error(
                "No response received from OpenRouter client for non-streaming chat request.",
                extra={
                    "internal_request_id": internal_request_id,
                    "payload_model": openrouter_payload.get("model"),
                },
            )
            background_tasks.add_task(
                _save_log_wrapper_for_bg,
                internal_request_id=internal_request_id,
                endpoint_called=str(request.url.path),
                client_ip=request.client.host,
                model_requested_by_client=request_data.model,
                model_routed_to_openrouter=effective_model,
                openrouter_response_summary={
                    "error": "No response from OpenRouter client",
                    "status_code": 503,  # Assuming this status for the error
                    "latency_ms": router_processing_duration_ms,  # This is router's view of latency
                },
                processing_duration_ms=router_processing_duration_ms,
                input_char_length=current_input_char_length,
                output_char_length=0,
                status_code_returned_to_client=503,
                is_streaming=False,
                is_multimodal=is_multimodal_request,
                media_type_processed=media_type if is_multimodal_request else None,
                error_source="internal_client_chat",
                error_message="No response from OpenRouter client",
            )
            raise HTTPException(
                status_code=503,
                detail="Internal server error: No response from provider.",
            )

        if openrouter_response_dict.get("error"):
            error_data = openrouter_response_dict.get("error")
            # Attempt to get status_code from the error dict, default to 500
            status_code = openrouter_response_dict.get("status_code")
            if (
                isinstance(error_data, dict)
                and "code" in error_data
                and isinstance(error_data["code"], int)
            ):
                status_code = error_data[
                    "code"
                ]  # Prefer code from error object if it's a status
            elif status_code is None:
                status_code = 500

            LOGGER.error(
                f"Error from OpenRouter (Chat Non-Streaming): Status {status_code}, Error: {error_data}",
                extra={
                    "internal_request_id": internal_request_id,
                    "openrouter_error": error_data,
                    "openrouter_status": status_code,
                    "openrouter_full_response": openrouter_response_dict,
                },
            )
            background_tasks.add_task(
                _save_log_wrapper_for_bg,
                internal_request_id=internal_request_id,
                endpoint_called=str(request.url.path),
                client_ip=request.client.host,
                model_requested_by_client=request_data.model,
                model_routed_to_openrouter=effective_model,
                openrouter_response_summary=openrouter_response_dict,  # Pass the whole dict
                processing_duration_ms=router_processing_duration_ms,
                input_char_length=current_input_char_length,
                output_char_length=0,  # No successful output
                status_code_returned_to_client=status_code,
                is_streaming=False,
                is_multimodal=is_multimodal_request,
                media_type_processed=media_type if is_multimodal_request else None,
                error_source="openrouter_chat",
                error_message=str(
                    error_data.get("message")
                    if isinstance(error_data, dict)
                    else error_data
                ),
            )
            detail_to_send = (
                error_data.get("message")
                if isinstance(error_data, dict)
                else str(error_data)
            )
            raise HTTPException(status_code=status_code, detail=detail_to_send)

        # Successful non-streaming response
        or_data = openrouter_response_dict.get("data", {})
        output_text = ""
        if or_data.get("choices") and or_data["choices"][0].get("message"):
            output_text = or_data["choices"][0]["message"].get("content", "")

        current_output_char_length = len(output_text)

        background_tasks.add_task(
            _save_log_wrapper_for_bg,
            internal_request_id=internal_request_id,
            endpoint_called=str(request.url.path),
            client_ip=request.client.host,
            model_requested_by_client=request_data.model,
            model_routed_to_openrouter=or_data.get(
                "model", effective_model
            ),  # Use model from response if available
            openrouter_response_summary=openrouter_response_dict,  # Pass the whole dict
            processing_duration_ms=router_processing_duration_ms,
            input_char_length=current_input_char_length,
            output_char_length=current_output_char_length,
            status_code_returned_to_client=200,
            is_streaming=False,
            is_multimodal=is_multimodal_request,
            media_type_processed=media_type if is_multimodal_request else None,
            error_source=None,
            error_message=None,
        )

        # Construct the ChatCompletionResponse
        # Ensure all fields are correctly populated from or_data
        final_choices = []
        if or_data.get("choices"):
            for choice_data in or_data.get("choices", []):
                message_data = choice_data.get("message", {})
                final_choices.append(
                    ChatCompletionChoice(
                        index=choice_data.get("index", 0),
                        message=ChatMessage(
                            role=message_data.get("role", "assistant"),
                            content=message_data.get("content"),
                            tool_calls=message_data.get("tool_calls"),
                        ),
                        finish_reason=choice_data.get("finish_reason"),
                        logprobs=choice_data.get(
                            "logprobs"
                        ),  # Ensure this is handled if present
                    )
                )

        usage_data = or_data.get("usage")
        final_usage = None
        if usage_data:
            final_usage = OpenAIUsage(**usage_data)

        return ChatCompletionResponse(
            id=or_data.get("id", internal_request_id),
            object=or_data.get("object", "chat.completion"),
            created=or_data.get("created", int(time.time())),
            model=or_data.get("model", effective_model),
            choices=final_choices,
            usage=final_usage,
            system_fingerprint=or_data.get("system_fingerprint"),
        )
