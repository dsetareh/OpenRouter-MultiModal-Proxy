# app/routers/completions.py
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    BackgroundTasks,
)  # Added BackgroundTasks
from fastapi.responses import StreamingResponse  # Added
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any, Dict, List, Union, AsyncGenerator  # Added AsyncGenerator
import json  # Added
import time  # Added
import aiohttp  # Added for specific exception handling if needed by client directly, though client handles its own.

from app.config import settings
from app.database import get_db, AsyncSessionLocal as SessionLocal  # Corrected import
from app.logging_config import LOGGER

# from app.models import OpenRouterRequest # Not directly used here, save_request_log_to_db handles model
from app.openrouter_client import OpenRouterClient
from app.schemas import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    OpenAIUsage,
    ChatMessage,  # For transforming to chat messages
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatCompletionChoiceDelta,  # For streaming legacy as chat
)
from app.media_processing import process_messages_for_media

# Import the save_request_log_to_db utility and the new wrapper from chat router
from app.routers.chat import (
    save_request_log_to_db,
    get_openrouter_client,
    _save_log_wrapper_for_bg,
)  # Added _save_log_wrapper_for_bg

router = APIRouter(prefix="/v1", tags=["Legacy Completions"])


@router.post("/completions")  # Removed response_model for flexibility
async def create_legacy_completion(
    request_data: CompletionRequest,
    request: Request,
    background_tasks: BackgroundTasks,  # Added
    db: AsyncSession = Depends(get_db),  # Remains for now
    or_client: OpenRouterClient = Depends(get_openrouter_client),
):
    start_router_processing_time = time.time()
    internal_request_id = request.state.internal_request_id
    http_session = request.app.state.http_session

    prompt_text = ""
    if isinstance(request_data.prompt, list):
        if request_data.prompt:
            prompt_text = str(request_data.prompt[0])  # Ensure it's a string
        # else: prompt_text remains ""
    elif request_data.prompt is not None:  # Ensure prompt is not None before str()
        prompt_text = str(request_data.prompt)

    temp_chat_messages = [ChatMessage(role="user", content=prompt_text)]

    processed_payload_messages, media_detected, media_type, _ = (
        await process_messages_for_media(
            temp_chat_messages, http_session  # Pass the ChatMessage list
        )
    )

    is_multimodal_request = media_detected

    effective_model = request_data.model
    if is_multimodal_request:
        if not settings.VISION_MODEL:
            LOGGER.error(
                "Multimodal legacy request but no VISION_MODEL configured.",
                extra={"internal_request_id": internal_request_id},
            )
            raise HTTPException(
                status_code=501,
                detail="Server not configured for multimodal legacy requests.",
            )
        effective_model = settings.VISION_MODEL
        LOGGER.info(
            f"Media detected in legacy prompt ({media_type}). Routing to vision model: {effective_model}",
            extra={"internal_request_id": internal_request_id},
        )
    else:  # Text-only legacy completion
        effective_model = request_data.model or settings.DEFAULT_TEXT_MODEL
        LOGGER.info(
            f"Text-only legacy prompt. Routing to model: {effective_model} via chat format.",
            extra={"internal_request_id": internal_request_id},
        )

    # Construct payload for OpenRouter (always as chat completion)
    openrouter_payload = {
        "model": effective_model,
        "messages": processed_payload_messages,  # Use the (potentially modified for media) messages
        "temperature": request_data.temperature,
        "top_p": request_data.top_p,
        "n": request_data.n,
        "stream": request_data.stream,  # Pass stream preference
        "stop": request_data.stop,
        "max_tokens": request_data.max_tokens,
        "presence_penalty": request_data.presence_penalty,
        "frequency_penalty": request_data.frequency_penalty,
        "user": request_data.user,
        "logit_bias": request_data.logit_bias,
    }
    openrouter_payload = {k: v for k, v in openrouter_payload.items() if v is not None}

    current_input_char_length = len(prompt_text)
    # If multimodal, input_char_length might be more complex if we consider image data size, but for now, stick to text part.
    if (
        is_multimodal_request
        and processed_payload_messages
        and isinstance(processed_payload_messages[0].get("content"), list)
    ):
        current_input_char_length = sum(
            len(part.get("text", ""))
            for part in processed_payload_messages[0].get("content", [])
            if part.get("type") == "text"
        )

    LOGGER.info(
        "Prepared payload for OpenRouter (Legacy Completions as Chat)",
        extra={
            "internal_request_id": internal_request_id,
            "target_model": effective_model,
            "is_multimodal": is_multimodal_request,
            "streaming_requested": request_data.stream,
            # "payload_to_openrouter": openrouter_payload # Avoid logging full payload if too large, especially with media
        },
    )

    if request_data.stream:

        async def stream_generator() -> AsyncGenerator[str, None]:
            aggregated_completion_tokens = 0
            aggregated_prompt_tokens = 0
            aggregated_total_tokens = None
            aggregated_cost_usd = 0.0
            output_text_char_count = 0
            final_openrouter_request_id = None
            stream_had_errors = False
            error_details_for_log = None
            response_model_name = effective_model
            response_id_from_stream = internal_request_id  # Default to our internal ID
            legacy_completion_idx = 0  # For legacy stream choice index
            stream_start_time = time.time()
            idx = 0  # for logging total chunks sent

            try:
                async for or_chunk_dict in or_client.create_chat_completion(
                    openrouter_payload
                ):
                    if or_chunk_dict.get("error"):
                        stream_had_errors = True
                        error_details_for_log = or_chunk_dict.get("error")
                        LOGGER.error(
                            f"Error during OpenRouter stream (Legacy): {error_details_for_log}",
                            extra={
                                "internal_request_id": internal_request_id,
                                "chunk": or_chunk_dict,
                            },
                        )
                        break

                    if or_chunk_dict.get("stream_done") or or_chunk_dict.get(
                        "stream_ended_without_done"
                    ):
                        final_openrouter_request_id = or_chunk_dict.get(
                            "openrouter_request_id", final_openrouter_request_id
                        )
                        if or_chunk_dict.get("stream_ended_without_done"):
                            LOGGER.warning(
                                "OpenRouter stream (Legacy) ended without [DONE] message.",
                                extra={"internal_request_id": internal_request_id},
                            )
                        break  # Exit loop

                    chunk_data = or_chunk_dict.get("data")
                    if not chunk_data:
                        LOGGER.warning(
                            "Empty data chunk received in legacy stream.",
                            extra={
                                "internal_request_id": internal_request_id,
                                "chunk": or_chunk_dict,
                            },
                        )
                        continue

                    final_openrouter_request_id = or_chunk_dict.get(
                        "openrouter_request_id", final_openrouter_request_id
                    )
                    response_id_from_stream = chunk_data.get(
                        "id", response_id_from_stream
                    )  # Use OpenRouter's chunk ID
                    response_model_name = chunk_data.get("model", response_model_name)

                    # Transform ChatCompletion chunk from OpenRouter to legacy Completion chunk format
                    # A single legacy completion stream usually has one choice with appended text.
                    if chunk_data.get("choices"):
                        or_choice = chunk_data["choices"][
                            0
                        ]  # Assume first choice for legacy
                        delta_content = or_choice.get("delta", {}).get("content", "")
                        if delta_content:
                            output_text_char_count += len(delta_content)

                        # Legacy stream choice format
                        legacy_stream_choice = {
                            "text": delta_content,
                            "index": legacy_completion_idx,  # OpenAI legacy stream usually has index 0 for the main completion
                            "logprobs": or_choice.get(
                                "logprobs"
                            ),  # Pass through if available
                            "finish_reason": or_choice.get("finish_reason"),
                        }
                        # legacy_completion_idx += 1 # Only increment if we are sending multiple choices, usually not for legacy stream

                        # Legacy stream response format
                        legacy_sse_event = {
                            "id": response_id_from_stream,  # Use the ID from the OR chunk
                            "object": "text_completion",  # Legacy object type
                            "created": chunk_data.get("created", int(time.time())),
                            "model": response_model_name,
                            "choices": [legacy_stream_choice],
                        }
                        yield f"data: {json.dumps(legacy_sse_event, ensure_ascii=False)}\n\n"
                    idx += 1

                # Ensure [DONE] is sent if loop finishes without error or explicit break for error
                if not stream_had_errors:
                    yield "data: [DONE]\n\n"
                elif (
                    error_details_for_log
                ):  # If there was an error, and we have details
                    # Similar to chat.py, consider if [DONE] should always be sent.
                    yield "data: [DONE]\n\n"

            finally:
                router_processing_duration_ms = (
                    time.time() - start_router_processing_time
                ) * 1000
                openrouter_stream_duration_ms = (time.time() - stream_start_time) * 1000

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
                    "latency_ms": openrouter_stream_duration_ms,
                    "error": error_details_for_log if stream_had_errors else None,
                    "data": {
                        "id": response_id_from_stream,
                        "model": response_model_name,
                    },
                }

                background_tasks.add_task(
                    _save_log_wrapper_for_bg,  # Use the wrapper from chat.py
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
                        "openrouter_stream_legacy"
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
                    "Finished streaming legacy completion to client (logging in background).",
                    extra={
                        "internal_request_id": internal_request_id,
                        "had_errors": stream_had_errors,
                        "total_chunks_sent_to_client": idx,
                    },
                )

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:  # Non-streaming legacy completion
        openrouter_response_dict = None
        async for item in or_client.create_chat_completion(openrouter_payload):
            openrouter_response_dict = item
            break  # Expect only one item

        router_processing_duration_ms = (
            time.time() - start_router_processing_time
        ) * 1000

        if not openrouter_response_dict:
            LOGGER.error(
                "No response received from OpenRouter client for non-streaming legacy request.",
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
                    "error": "No response from OpenRouter client (legacy)",
                    "status_code": 503,
                    "latency_ms": router_processing_duration_ms,
                },
                processing_duration_ms=router_processing_duration_ms,
                input_char_length=current_input_char_length,
                output_char_length=0,
                status_code_returned_to_client=503,
                is_streaming=False,
                is_multimodal=is_multimodal_request,
                media_type_processed=media_type if is_multimodal_request else None,
                error_source="internal_client_legacy",
                error_message="No response from OpenRouter client (legacy)",
            )
            raise HTTPException(
                status_code=503,
                detail="Internal server error: No response from provider.",
            )

        if openrouter_response_dict.get("error"):
            error_data = openrouter_response_dict.get("error")
            status_code = openrouter_response_dict.get("status_code")
            if (
                isinstance(error_data, dict)
                and "code" in error_data
                and isinstance(error_data["code"], int)
            ):
                status_code = error_data["code"]
            elif status_code is None:
                status_code = 500

            LOGGER.error(
                f"Error from OpenRouter (Legacy Non-Streaming): Status {status_code}, Error: {error_data}",
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
                openrouter_response_summary=openrouter_response_dict,
                processing_duration_ms=router_processing_duration_ms,
                input_char_length=current_input_char_length,
                output_char_length=0,
                status_code_returned_to_client=status_code,
                is_streaming=False,
                is_multimodal=is_multimodal_request,
                media_type_processed=media_type if is_multimodal_request else None,
                error_source="openrouter_legacy",
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

        # Successful non-streaming legacy response
        or_data = openrouter_response_dict.get("data", {})
        output_text = ""
        if or_data.get("choices") and or_data["choices"][0].get(
            "message"
        ):  # From chat format
            output_text = or_data["choices"][0]["message"].get("content", "")

        current_output_char_length = len(output_text)

        background_tasks.add_task(
            _save_log_wrapper_for_bg,
            internal_request_id=internal_request_id,
            endpoint_called=str(request.url.path),
            client_ip=request.client.host,
            model_requested_by_client=request_data.model,
            model_routed_to_openrouter=or_data.get("model", effective_model),
            openrouter_response_summary=openrouter_response_dict,
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

        # Transform OpenRouter chat response to OpenAI legacy CompletionResponse
        legacy_choices = []
        if or_data.get("choices"):
            for idx, chat_choice in enumerate(or_data.get("choices", [])):
                text_content = ""
                if chat_choice.get("message") and chat_choice["message"].get("content"):
                    text_content = chat_choice["message"]["content"]
                elif chat_choice.get("delta") and chat_choice["delta"].get("content"):
                    # Should not happen here as this is non-streaming, but as a fallback
                    text_content = chat_choice["delta"]["content"]

                legacy_choices.append(
                    CompletionChoice(
                        text=text_content,
                        index=idx,
                        logprobs=chat_choice.get("logprobs"),  # Pass if available
                        finish_reason=chat_choice.get("finish_reason"),
                    )
                )

        openai_usage = OpenAIUsage(
            prompt_tokens=openrouter_response_dict.get("prompt_tokens", 0),
            completion_tokens=openrouter_response_dict.get("completion_tokens", 0),
            total_tokens=openrouter_response_dict.get("total_tokens", 0),
        )

        final_response_data = CompletionResponse(
            id=or_data.get("id", internal_request_id),
            created=or_data.get("created", int(time.time())),
            model=or_data.get("model", effective_model),
            choices=legacy_choices,
            usage=openai_usage,
            system_fingerprint=or_data.get("system_fingerprint"),  # Pass if available
        )

        router_processing_duration_ms = (
            time.time() - start_router_processing_time
        ) * 1000
        await save_request_log_to_db(
            db=db,
            internal_request_id=internal_request_id,
            endpoint_called=str(request.url.path),
            client_ip=request.client.host,
            model_requested_by_client=request_data.model,
            model_routed_to_openrouter=effective_model,
            openrouter_response_summary=openrouter_response_dict,
            processing_duration_ms=router_processing_duration_ms,
            input_char_length=current_input_char_length,
            output_char_length=current_output_char_length,
            status_code_returned_to_client=200,
            is_streaming=False,
            is_multimodal=is_multimodal_request,
            media_type_processed=media_type if is_multimodal_request else None,
        )

        LOGGER.info(
            "Successfully processed legacy completion (non-streaming) and sent response to client.",
            extra={
                "internal_request_id": internal_request_id,
                "response_to_client": final_response_data.model_dump(exclude_none=True),
                "model_used": final_response_data.model,
                "usage": openai_usage.model_dump(),
            },
        )
        return final_response_data
