# app/routers/ollama.py
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import json
import time
import datetime # For created_at
import re # For regex operations
import uuid # For generating tool call IDs if necessary
from typing import List, Dict, Any, AsyncGenerator, Union, Optional
from app.config import settings
from app.logging_config import LOGGER
from app.schemas import (
    OllamaTagsResponse, OllamaGenerateRequest,
    OllamaGenerateResponseStreamChunk, OllamaGenerateResponseFinal,
    OllamaChatRequest, OllamaChatResponseStreamChunk, OllamaChatResponseFinal,
    OllamaTagModel, OllamaTagModelDetails, OllamaChatMessage,
    OllamaToolCallDefinition, # Used by OllamaToolDefinition and for representing an invoked tool's function
    OllamaToolDefinition,   # Used for defining available tools in request, and for representing tool calls in response
    OllamaChatMessagePart
)
from app.openrouter_client import OpenRouterClient

router = APIRouter(prefix="/api", tags=["ollama"])

# Placeholder for /api/tags
@router.get("/tags", response_model=OllamaTagsResponse)
async def list_ollama_tags(request: Request):
    LOGGER.info("Received request for /api/tags")
    try:
        openrouter_client: OpenRouterClient = request.app.state.openrouter_client
    except AttributeError:
        LOGGER.error("OpenRouterClient not found in application state.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: OpenRouter client not initialized.",
        )

    try:
        openrouter_models_data = await openrouter_client.list_openrouter_models()
    except Exception as e: # Catch any unexpected error from the client call itself
        LOGGER.error(f"Failed to call list_openrouter_models: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to fetch models from upstream service.",
        )

    if not openrouter_models_data:
        LOGGER.warning("No models returned from OpenRouter or an error occurred during fetch (empty list received).")
        # The client logs specific errors. If it returns empty, it means an issue.
        # Depending on strictness, could return empty list or error.
        # For Ollama compatibility, an empty list of tags is valid if no models are available.
        # However, if the upstream call failed (which client indicates by returning []), an error is more appropriate.
        # The client returns [] on HTTP errors or parsing issues.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Error fetching models from upstream provider or no models available.",
        )

    transformed_models: List[OllamaTagModel] = []
    current_time_utc = datetime.datetime.now(datetime.timezone.utc)

    for model_data in openrouter_models_data:
        model_id = model_data.get("id", "unknown_model_id")
        architecture = model_data.get("architecture", {})

        # Infer family
        family = "unknown"
        model_id_lower = model_id.lower()
        if "llama" in model_id_lower:
            family = "llama"
        elif "claude" in model_id_lower:
            family = "claude"
        elif "gemini" in model_id_lower:
            family = "gemini"
        elif "gpt" in model_id_lower or "openai/" in model_id_lower: # openai/gpt-4 etc.
            family = "gpt"
        elif "mixtral" in model_id_lower: # check before mistral
            family = "mixtral"
        elif "mistral" in model_id_lower:
            family = "mistral"
        elif "command" in model_id_lower: # Cohere
            family = "command"
        elif "dbrx" in model_id_lower:
            family = "dbrx"
        elif "phi" in model_id_lower:
            family = "phi"
        # Add more families as needed

        families = [family] if family != "unknown" else None

        # Infer parameter size
        parameter_size = "Unknown"
        instruct_type = architecture.get("instruct_type") # e.g., "7B Instruct", "8x7B"
        if instruct_type and isinstance(instruct_type, str):
            # Regex to find patterns like "7B", "13B", "8x7B", "1.5B"
            match = re.search(r"(\d+(\.\d+)?B|\d+x\d+B)", instruct_type, re.IGNORECASE)
            if match:
                parameter_size = match.group(0).upper()
        
        if parameter_size == "Unknown": # Fallback to model ID
            # More robust regex for model ID
            # Looks for common patterns like 7b, 13b, 70b, 8x7b, 34b, 1.5b, etc.
            # (?:^|[\/\-_]) ensures it's a distinct part of the ID
            match_id = re.search(r"(?:^|[\/\-_])(\d+(\.\d+)?[Bb]|\d+x\d+[Bb])(?:$|[\/\-_])", model_id)
            if match_id:
                 # Extract the part like "7b" or "8x7B" from the match group
                param_str = match_id.group(1)
                # Ensure it ends with B, and make it uppercase
                if not param_str.upper().endswith('B'):
                    parameter_size = param_str.upper() + 'B'
                else:
                    parameter_size = param_str.upper()


        # Quantization level
        quantization_level = architecture.get("quantization", "Unknown")
        if not quantization_level or not isinstance(quantization_level, str): # Ensure it's a string
            quantization_level = "Unknown"


        details = OllamaTagModelDetails(
            format="gguf", # Common default for Ollama
            family=family,
            families=families,
            parameter_size=parameter_size,
            quantization_level=quantization_level,
        )

        # Create digest placeholder
        clean_model_id_for_digest = model_id.replace('/', '_').replace(':', '_')
        digest = f"sha256:{clean_model_id_for_digest}_placeholder_digest"

        ollama_model = OllamaTagModel(
            name=model_id, # Ollama uses model_id:tag, but OpenRouter IDs are often full
            model=model_id, # As per schema, model field is also present
            modified_at=current_time_utc,
            size=0, # Placeholder size
            digest=digest,
            details=details,
        )
        transformed_models.append(ollama_model)

    LOGGER.info(f"Successfully transformed {len(transformed_models)} models for /api/tags response.")
    return OllamaTagsResponse(models=transformed_models)


# Placeholder for /api/generate
@router.post("/generate")
async def generate_ollama_completion(ollama_request: OllamaGenerateRequest, request: Request):
    try:
        openrouter_client: OpenRouterClient = request.app.state.openrouter_client
    except AttributeError:
        LOGGER.error("OpenRouterClient not found in application state for /api/generate.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: OpenRouter client not initialized.",
        )

    try:
        LOGGER.info(f"Received /api/generate request for model: {ollama_request.model}, stream: {ollama_request.stream}")

        # Request Mapping
        or_payload: Dict[str, Any] = {
            "model": ollama_request.model,
            "stream": ollama_request.stream,
        }
        
        messages: List[Dict[str, Any]] = []
        if ollama_request.system:
            messages.append({"role": "system", "content": ollama_request.system})

        user_message_content_parts: List[Dict[str, Any]] = [{"type": "text", "text": ollama_request.prompt}]
        if ollama_request.images:
            for img_b64 in ollama_request.images:
                user_message_content_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                )
        messages.append({"role": "user", "content": user_message_content_parts})
        or_payload["messages"] = messages

        if ollama_request.options:
            if ollama_request.options.temperature is not None:
                or_payload["temperature"] = ollama_request.options.temperature
            if ollama_request.options.top_p is not None:
                or_payload["top_p"] = ollama_request.options.top_p
            if ollama_request.options.num_predict is not None:
                or_payload["max_tokens"] = ollama_request.options.num_predict
            if ollama_request.options.stop is not None:
                or_payload["stop"] = ollama_request.options.stop
        
        if ollama_request.format == "json":
            or_payload["response_format"] = {"type": "json_object"}

        # Log warnings for unmapped parameters
        if ollama_request.raw is not None and ollama_request.raw:
            LOGGER.warning("Ollama parameter 'raw' is not mapped to OpenRouter and will be ignored.")
        if ollama_request.template is not None:
            LOGGER.warning("Ollama parameter 'template' is not mapped to OpenRouter and will be ignored.")
        if ollama_request.keep_alive is not None:
            LOGGER.warning("Ollama parameter 'keep_alive' is not mapped to OpenRouter and will be ignored.")
        if ollama_request.context is not None:
            LOGGER.warning("Ollama parameter 'context' is not mapped to OpenRouter and will be ignored.")

        LOGGER.debug(f"OpenRouter payload for /generate: {json.dumps(or_payload, indent=2)}")

        if ollama_request.stream:
            async def stream_generator() -> AsyncGenerator[str, None]:
                start_time = time.time()
                prompt_tokens_count = 0
                completion_tokens_count = 0
                
                try:
                    async for or_chunk_result in openrouter_client.create_chat_completion(or_payload):
                        LOGGER.debug(f"OR Stream chunk: {or_chunk_result}")
                        if "data" in or_chunk_result and isinstance(or_chunk_result["data"], dict): # Actual content chunk
                            delta_content = or_chunk_result["data"].get("choices", [{}])[0].get("delta", {}).get("content")
                            if delta_content:
                                ollama_chunk = OllamaGenerateResponseStreamChunk(
                                    model=ollama_request.model,
                                    created_at=datetime.datetime.now(datetime.timezone.utc),
                                    response=delta_content,
                                    done=False,
                                    context=None # Deprecated
                                )
                                yield f"{ollama_chunk.model_dump_json(exclude_none=True)}\n"
                        
                        # Check for usage/end of stream markers (adjust based on actual client behavior)
                        # The prompt implies these keys might be at the top level of the chunk
                        if or_chunk_result.get("stream_done") or or_chunk_result.get("stream_ended_without_done") or \
                           (isinstance(or_chunk_result.get("data"), dict) and or_chunk_result["data"].get("usage")): # Check for usage in data
                            
                            usage_data = or_chunk_result if (or_chunk_result.get("stream_done") or or_chunk_result.get("stream_ended_without_done")) \
                                else or_chunk_result.get("data", {}).get("usage", {})

                            prompt_tokens_count = usage_data.get("prompt_tokens", prompt_tokens_count)
                            completion_tokens_count = usage_data.get("completion_tokens", completion_tokens_count)
                            if or_chunk_result.get("stream_done") or or_chunk_result.get("stream_ended_without_done"):
                                LOGGER.info("Stream ended or done signal received from OpenRouter.")
                                break
                        
                        if "error" in or_chunk_result:
                            LOGGER.error(f"Error received in stream from OpenRouter: {or_chunk_result.get('error')}")
                            # How to propagate this error within a generator to the StreamingResponse?
                            # One way is to raise an exception that the FastAPI framework might handle or log.
                            # For now, we log and the stream will likely terminate.
                            # A more robust solution might involve a custom exception and handler.
                            # For Ollama, it might just stop sending chunks.
                            # Let's yield a final error-like chunk if possible, or just break.
                            # For now, we break and the final "done" chunk will be sent.
                            break


                except Exception as e_stream:
                    LOGGER.error(f"Error during stream generation: {e_stream}", exc_info=True)
                    # Yield a final error message if possible, or just let it end.
                    # This exception will likely be caught by FastAPI's StreamingResponse handling.
                    # For now, we ensure the final "done" chunk is attempted.
                    pass # Fall through to send final chunk

                total_response_time_ns = int((time.time() - start_time) * 1_000_000_000)
                final_ollama_chunk = OllamaGenerateResponseFinal(
                    model=ollama_request.model,
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                    response="", # Empty for the final streaming message
                    done=True,
                    done_reason="stop", # Default
                    total_duration=total_response_time_ns,
                    prompt_eval_count=prompt_tokens_count if prompt_tokens_count > 0 else None,
                    eval_count=completion_tokens_count if completion_tokens_count > 0 else None,
                    load_duration=None, # Placeholder
                    prompt_eval_duration=None, # Placeholder
                    eval_duration=None, # Placeholder
                    context=None # Deprecated
                )
                yield f"{final_ollama_chunk.model_dump_json(exclude_none=True)}\n"
            
            return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

        else: # Non-streaming response
            start_time = time.time()
            full_response_content = ""
            or_final_data_wrapper = None # To store the entire chunk that contains final data and usage
            
            async for or_chunk in openrouter_client.create_chat_completion(or_payload):
                LOGGER.debug(f"OR Non-stream chunk: {or_chunk}")
                if "data" in or_chunk and isinstance(or_chunk["data"], dict) and not or_chunk.get("stream", False): # Non-streaming data from OR
                    or_final_data_wrapper = or_chunk
                    # Content is typically in choices[0].message.content for non-streaming
                    full_response_content = or_chunk["data"].get("choices", [{}])[0].get("message", {}).get("content", "")
                    break
                elif "error" in or_chunk:
                    err_details = or_chunk.get("error")
                    status_code = 500
                    if isinstance(err_details, dict):
                        status_code = err_details.get("status_code", err_details.get("status", 500)) # some APIs use "status"
                        err_details = str(err_details) # convert dict to string for detail
                    elif isinstance(or_chunk.get("status_code"), int): # top level status_code
                        status_code = or_chunk.get("status_code")

                    LOGGER.error(f"Error from OpenRouter (non-stream): {err_details}")
                    raise HTTPException(status_code=status_code, detail=err_details)

            if or_final_data_wrapper is None or "data" not in or_final_data_wrapper:
                LOGGER.error("No valid non-streaming response received from OpenRouter.")
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to get valid response from upstream provider")

            total_response_time_ns = int((time.time() - start_time) * 1_000_000_000)
            
            # Extract usage from the final data chunk (OpenRouter specific)
            # Prompt assumes prompt_tokens and completion_tokens are top-level in or_final_data,
            # but they are usually in data.usage
            usage_info = or_final_data_wrapper.get("data", {}).get("usage", {})
            prompt_tokens = usage_info.get("prompt_tokens")
            completion_tokens = usage_info.get("completion_tokens")

            response_model = OllamaGenerateResponseFinal(
                model=ollama_request.model,
                created_at=datetime.datetime.now(datetime.timezone.utc),
                response=full_response_content,
                done=True,
                done_reason="stop",
                total_duration=total_response_time_ns,
                prompt_eval_count=prompt_tokens,
                eval_count=completion_tokens,
                load_duration=None, # Placeholder
                prompt_eval_duration=None, # Placeholder
                eval_duration=None, # Placeholder
                context=None # Deprecated
            )
            return JSONResponse(content=response_model.model_dump(mode="json", exclude_none=False)) # Ollama usually includes nulls

    except httpx.HTTPStatusError as e:
        LOGGER.error(f"HTTPStatusError from OpenRouter during generate: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        LOGGER.error(f"RequestError connecting to OpenRouter during generate: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Error connecting to upstream service: {str(e)}")
    except HTTPException: # Re-raise HTTPExceptions raised internally
        raise
    except Exception as e:
        LOGGER.error(f"Unexpected error in generate_ollama_completion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/chat", summary="Ollama Compatible Chat Endpoint")
async def chat_ollama_completion(ollama_request: OllamaChatRequest, request: Request):
    LOGGER.info(f"Received /api/chat request for model: {ollama_request.model}, stream: {ollama_request.stream}")
    try:
        openrouter_client: OpenRouterClient = request.app.state.openrouter_client
    except AttributeError:
        LOGGER.error("OpenRouterClient not found in application state for /api/chat.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: OpenRouter client not initialized.",
        )

    try:
        # --- Request Mapping (Ollama to OpenRouter) ---
        or_payload: Dict[str, Any] = {
            "model": ollama_request.model,
            "stream": ollama_request.stream if ollama_request.stream is not None else False, # Default to False if not set, though Ollama defaults to True
        }

        # Map messages
        or_messages: List[Dict[str, Any]] = []
        for ollama_msg in ollama_request.messages:
            or_msg_content: Union[str, List[Dict[str, Any]]]
            if ollama_msg.images and len(ollama_msg.images) > 0:
                content_parts = [{"type": "text", "text": ollama_msg.content or ""}]
                for img_b64 in ollama_msg.images:
                    content_parts.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    )
                or_msg_content = content_parts
            else:
                or_msg_content = ollama_msg.content or ""

            current_or_msg: Dict[str, Any] = {
                "role": ollama_msg.role,
                "content": or_msg_content,
            }
            # `name` is not directly in OllamaChatMessage, but OpenAI uses it for tool responses.
            # If Ollama adds it or it's conventional for tool roles, this would be the place.
            # For now, OllamaChatMessage doesn't have a 'name' field.

            # Map assistant's tool calls from history (OllamaChatMessage.tool_calls)
            # These are tools the assistant *has decided to call*.
            # OllamaChatMessage.tool_calls is List[OllamaToolDefinition]
            # OllamaToolDefinition is {type, function: {name, arguments: Dict}}
            # OpenRouter/AI tool_calls is List[{id, type, function: {name, arguments: str}}]
            if ollama_msg.role == "assistant" and ollama_msg.tool_calls:
                or_tool_calls_for_msg = []
                for i, otc in enumerate(ollama_msg.tool_calls):
                    if otc.type == "function" and otc.function:
                        try:
                            args_str = json.dumps(otc.function.arguments or {})
                        except TypeError:
                            LOGGER.warning(f"Could not serialize arguments for tool call {otc.function.name}: {otc.function.arguments}")
                            args_str = "{}"
                        
                        or_tool_calls_for_msg.append({
                            "id": f"call_{ollama_msg.role}_{i}_{str(uuid.uuid4())[:8]}", # Generate a unique ID
                            "type": "function",
                            "function": {
                                "name": otc.function.name,
                                "arguments": args_str,
                            },
                        })
                if or_tool_calls_for_msg:
                    current_or_msg["tool_calls"] = or_tool_calls_for_msg
            
            # If role is 'tool', we need tool_call_id. Ollama's schema doesn't explicitly have it.
            # This part requires a convention for how tool responses are formatted in Ollama requests.
            # Assuming if role is 'tool', content is result, and name might be used for tool_call_id if available.
            # For now, this is a gap if strict OpenAI tool role mapping is needed.

            or_messages.append(current_or_msg)
        or_payload["messages"] = or_messages

        # Map available tools (ollama_request.tools)
        # ollama_request.tools is List[OllamaToolDefinition]
        # OllamaToolDefinition is {type, function: {name, arguments: Dict}} (as per schemas.py)
        # The prompt implies function has name, description, parameters.
        # We will use `arguments` from schema as `parameters` for OR, and log missing description.
        if ollama_request.tools:
            or_defined_tools = []
            for ollama_tool_def_item in ollama_request.tools:
                if ollama_tool_def_item.type == "function" and ollama_tool_def_item.function:
                    # `ollama_tool_def_item.function` is OllamaToolCallDefinition {name, arguments}
                    # We assume `arguments` is the JSON schema for parameters.
                    # Description is not in OllamaToolCallDefinition.
                    func_name = ollama_tool_def_item.function.name
                    params_schema = ollama_tool_def_item.function.arguments or {}
                    
                    # Try to get description if it was passed as an extra attribute (non-standard)
                    description = getattr(ollama_tool_def_item.function, 'description', None)
                    if description is None:
                         LOGGER.warning(f"Ollama tool definition for function '{func_name}' is missing 'description'. OpenRouter expects this.")

                    or_defined_tools.append({
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "description": description, # Will be None if not found
                            "parameters": params_schema,
                        },
                    })
            if or_defined_tools:
                or_payload["tools"] = or_defined_tools
                # Default tool_choice to "auto" if tools are provided, common behavior
                or_payload.setdefault("tool_choice", "auto")


        # Map options
        if ollama_request.options:
            options = ollama_request.options
            if options.get("temperature") is not None:
                or_payload["temperature"] = options["temperature"]
            if options.get("top_p") is not None:
                or_payload["top_p"] = options["top_p"]
            if options.get("num_predict") is not None: # Ollama's num_predict maps to max_tokens
                or_payload["max_tokens"] = options["num_predict"]
            if options.get("stop") is not None:
                or_payload["stop"] = options["stop"]
            # Other options like mirostat, top_k, seed etc. are not directly mapped.
            unmapped_options = {k:v for k,v in options.items() if k not in ["temperature", "top_p", "num_predict", "stop"]}
            if unmapped_options:
                LOGGER.warning(f"Unmapped Ollama options: {unmapped_options}")


        if ollama_request.format == "json":
            or_payload["response_format"] = {"type": "json_object"}

        # Log warnings for unmapped top-level Ollama parameters
        if ollama_request.template is not None:
            LOGGER.warning("Ollama parameter 'template' is not mapped to OpenRouter and will be ignored.")
        if ollama_request.keep_alive is not None:
            LOGGER.warning("Ollama parameter 'keep_alive' is not mapped to OpenRouter and will be ignored.")

        LOGGER.debug(f"OpenRouter payload for /chat: {json.dumps(or_payload, indent=2)}")

        # --- Streaming Response ---
        if or_payload["stream"]:
            async def stream_generator() -> AsyncGenerator[str, None]:
                start_time = time.time()
                prompt_tokens_count = 0
                completion_tokens_count = 0
                # To aggregate arguments for tool calls if they are streamed piece by piece
                # {tool_call_index: {"id": ..., "name": ..., "arguments_str": ..., "type": ...}}
                active_tool_calls_aggregator: Dict[int, Dict[str, Any]] = {}
                
                try:
                    async for or_chunk_result in openrouter_client.create_chat_completion(or_payload):
                        LOGGER.debug(f"OR Chat Stream chunk: {or_chunk_result}")
                        
                        ollama_chunk_done = False
                        ollama_done_reason = None
                        
                        # Default message part for this chunk
                        ollama_msg_part = OllamaChatMessagePart(role="assistant", content=None, images=None, tool_calls=None)

                        if "data" in or_chunk_result and isinstance(or_chunk_result["data"], dict):
                            choice = or_chunk_result["data"].get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            
                            delta_role = delta.get("role")
                            if delta_role: # Role usually comes first
                                ollama_msg_part.role = delta_role
                            
                            delta_content = delta.get("content")
                            if delta_content:
                                ollama_msg_part.content = delta_content

                            # Handle streaming tool calls from OpenRouter
                            # OR delta tool_calls: List[{index, id, type, function: {name, arguments}}]
                            # Ollama expects: List[OllamaToolDefinition {type, function: {name, arguments: Dict}}]
                            # We can only form OllamaToolDefinition when arguments are complete.
                            # This usually happens when finish_reason is 'tool_calls'.
                            
                            or_delta_tool_calls = delta.get("tool_calls")
                            if or_delta_tool_calls:
                                # Accumulate tool call data. We'll form the full Ollama tool_calls
                                # when finish_reason indicates they are complete.
                                for tc_delta_item in or_delta_tool_calls:
                                    idx = tc_delta_item.get("index")
                                    if idx is None: continue # Should not happen

                                    if idx not in active_tool_calls_aggregator:
                                        active_tool_calls_aggregator[idx] = {
                                            "id": tc_delta_item.get("id"),
                                            "type": tc_delta_item.get("type", "function"),
                                            "name": "",
                                            "arguments_str": ""
                                        }
                                    
                                    if tc_delta_item.get("function"):
                                        if tc_delta_item["function"].get("name"):
                                            active_tool_calls_aggregator[idx]["name"] = tc_delta_item["function"]["name"]
                                        if tc_delta_item["function"].get("arguments"):
                                            active_tool_calls_aggregator[idx]["arguments_str"] += tc_delta_item["function"]["arguments"]
                            
                            finish_reason = choice.get("finish_reason")
                            if finish_reason:
                                ollama_chunk_done = True # This chunk is the one causing a finish
                                ollama_done_reason = finish_reason

                                if finish_reason == "tool_calls":
                                    # Now form the complete tool_calls for Ollama
                                    final_ollama_tool_calls = []
                                    # The `delta.tool_calls` in a `finish_reason: tool_calls` chunk from OR
                                    # should contain the *full* definitions.
                                    # Or, use our aggregator if OR streams them incrementally.
                                    # Let's rely on the aggregator, as OR might send full calls in one delta or split.
                                    for _idx, aggregated_tc in sorted(active_tool_calls_aggregator.items()):
                                        try:
                                            args_dict = json.loads(aggregated_tc["arguments_str"] or "{}")
                                        except json.JSONDecodeError:
                                            LOGGER.error(f"Failed to parse JSON arguments for tool call {aggregated_tc.get('name')}: {aggregated_tc['arguments_str']}")
                                            args_dict = {}
                                        
                                        final_ollama_tool_calls.append(
                                            OllamaToolDefinition(
                                                type=aggregated_tc.get("type", "function"),
                                                function=OllamaToolCallDefinition(
                                                    name=aggregated_tc.get("name"),
                                                    arguments=args_dict
                                                )
                                            )
                                        )
                                    if final_ollama_tool_calls:
                                        ollama_msg_part.tool_calls = final_ollama_tool_calls
                                    active_tool_calls_aggregator.clear() # Clear for next potential set

                            # Only yield if there's content, a role change, or it's a finishing chunk with tool_calls
                            if ollama_msg_part.content or ollama_msg_part.role or (ollama_chunk_done and ollama_msg_part.tool_calls):
                                 ollama_stream_chunk = OllamaChatResponseStreamChunk(
                                    model=ollama_request.model,
                                    created_at=datetime.datetime.now(datetime.timezone.utc),
                                    message=ollama_msg_part,
                                    done=False, # This individual chunk is not the *final* done=True chunk unless finish_reason is terminal
                                    done_reason=None
                                )
                                 yield f"{ollama_stream_chunk.model_dump_json(exclude_none=True)}\n"

                        # Handle usage/end of stream markers from OpenRouter client
                        # (These are custom signals from our client, not standard OpenAI SSE fields)
                        if or_chunk_result.get("stream_done") or or_chunk_result.get("stream_ended_without_done"):
                            usage_data = or_chunk_result.get("usage", {})
                            prompt_tokens_count = usage_data.get("prompt_tokens", prompt_tokens_count)
                            completion_tokens_count = usage_data.get("completion_tokens", completion_tokens_count)
                            LOGGER.info("OpenRouter stream ended or done signal received.")
                            ollama_chunk_done = True # Mark as done for the final Ollama chunk
                            if not ollama_done_reason: ollama_done_reason = "stop" # Default if not set by finish_reason
                            break
                        
                        if "error" in or_chunk_result:
                            LOGGER.error(f"Error in stream from OpenRouter: {or_chunk_result.get('error')}")
                            # How to propagate? For now, break and send a final done chunk.
                            # Ideally, yield an error structure if Ollama supports it in stream.
                            ollama_chunk_done = True
                            ollama_done_reason = "error" # Custom reason
                            break
                        
                        if ollama_chunk_done and ollama_done_reason not in ["tool_calls"]: # If done for reasons other than tool_calls, break
                            break


                except Exception as e_stream:
                    LOGGER.error(f"Error during chat stream generation: {e_stream}", exc_info=True)
                    ollama_chunk_done = True
                    ollama_done_reason = "error" # Custom reason
                
                # Send the final "done" Ollama chunk
                total_response_time_ns = int((time.time() - start_time) * 1_000_000_000)
                final_message_part = OllamaChatMessagePart(role="assistant", content="", images=None) # Empty content for final
                
                # If the last substantive part was tool_calls, it should have been sent.
                # This final chunk is mainly for done=True and stats.
                # If ollama_done_reason was tool_calls, the tool_calls themselves were in the *previous* chunk.
                
                final_ollama_chunk = OllamaChatResponseStreamChunk(
                    model=ollama_request.model,
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                    message=final_message_part, # Usually empty for the very last one
                    done=True,
                    done_reason=ollama_done_reason if ollama_done_reason else "stop",
                    total_duration=total_response_time_ns,
                    prompt_eval_count=prompt_tokens_count if prompt_tokens_count > 0 else None,
                    eval_count=completion_tokens_count if completion_tokens_count > 0 else None,
                    # load_duration, prompt_eval_duration, eval_duration are not easily available from OR stream
                )
                yield f"{final_ollama_chunk.model_dump_json(exclude_none=True)}\n"

            return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

        # --- Non-Streaming Response ---
        else:
            start_time = time.time()
            full_or_response_message: Optional[Dict[str, Any]] = None
            or_usage_data: Optional[Dict[str, Any]] = None # To store usage from the OR response
            or_response_model_name: Optional[str] = None
            or_finish_reason: Optional[str] = "stop"

            try:
                # For non-streaming, create_chat_completion should yield one final result or error
                async for or_chunk in openrouter_client.create_chat_completion(or_payload):
                    LOGGER.debug(f"OR Chat Non-stream chunk: {or_chunk}")
                    if "data" in or_chunk and isinstance(or_chunk["data"], dict) and not or_chunk.get("stream", False):
                        # This is the complete non-streaming response from OpenRouter
                        full_or_response_message = or_chunk["data"].get("choices", [{}])[0].get("message", {})
                        or_usage_data = or_chunk["data"].get("usage", {}) # Usage is typically here
                        or_response_model_name = or_chunk["data"].get("model", ollama_request.model)
                        or_finish_reason = or_chunk["data"].get("choices", [{}])[0].get("finish_reason", "stop")
                        break # Got the full response
                    elif "error" in or_chunk:
                        err_details = or_chunk.get("error")
                        status_code = 500
                        if isinstance(err_details, dict):
                            status_code = err_details.get("status_code", err_details.get("status", 500))
                            err_details = str(err_details)
                        elif isinstance(or_chunk.get("status_code"), int):
                            status_code = or_chunk.get("status_code")
                        LOGGER.error(f"Error from OpenRouter (non-stream chat): {err_details}")
                        raise HTTPException(status_code=status_code, detail=err_details)
                
                if full_or_response_message is None:
                    LOGGER.error("No valid non-streaming response message received from OpenRouter for /chat.")
                    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to get valid response message from upstream provider")

            except httpx.HTTPStatusError as e:
                LOGGER.error(f"HTTPStatusError from OpenRouter during chat: {e.response.status_code} - {e.response.text}", exc_info=True)
                raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
            except httpx.RequestError as e:
                LOGGER.error(f"RequestError connecting to OpenRouter during chat: {e}", exc_info=True)
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Error connecting to upstream service: {str(e)}")
            except HTTPException:
                raise
            except Exception as e: # Catch-all for other unexpected errors during the call
                LOGGER.error(f"Unexpected error during non-streaming chat call: {str(e)}", exc_info=True)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")


            total_response_time_ns = int((time.time() - start_time) * 1_000_000_000)

            # Map OpenRouter message.tool_calls to OllamaChatMessage.tool_calls
            # OR tool_calls: List[{id, type, function: {name, arguments: str}}]
            # Ollama tool_calls: List[OllamaToolDefinition {type, function: {name, arguments: Dict}}]
            ollama_response_tool_calls: Optional[List[OllamaToolDefinition]] = None
            if full_or_response_message.get("tool_calls"):
                ollama_response_tool_calls = []
                for or_tc in full_or_response_message["tool_calls"]:
                    if or_tc.get("type") == "function" and or_tc.get("function"):
                        func_name = or_tc["function"].get("name")
                        args_str = or_tc["function"].get("arguments", "{}")
                        try:
                            args_dict = json.loads(args_str)
                        except json.JSONDecodeError:
                            LOGGER.error(f"Failed to parse JSON arguments for tool call {func_name} in non-streaming response: {args_str}")
                            args_dict = {}
                        
                        ollama_response_tool_calls.append(
                            OllamaToolDefinition(
                                type="function",
                                function=OllamaToolCallDefinition(name=func_name, arguments=args_dict)
                            )
                        )
            
            response_message_content = full_or_response_message.get("content", "")
            # If tool_calls are present, content might be null. Ollama expects string.
            if response_message_content is None and ollama_response_tool_calls:
                response_message_content = ""


            ollama_final_message = OllamaChatMessage(
                role=full_or_response_message.get("role", "assistant"),
                content=response_message_content,
                images=None, # Images are not part of assistant's response message typically
                tool_calls=ollama_response_tool_calls
            )

            final_response_model = OllamaChatResponseFinal(
                model=or_response_model_name or ollama_request.model,
                created_at=datetime.datetime.now(datetime.timezone.utc),
                message=ollama_final_message,
                done=True,
                done_reason=or_finish_reason,
                total_duration=total_response_time_ns,
                prompt_eval_count=or_usage_data.get("prompt_tokens") if or_usage_data else None,
                eval_count=or_usage_data.get("completion_tokens") if or_usage_data else None,
                # load_duration, prompt_eval_duration, eval_duration not directly available
            )
            return JSONResponse(content=final_response_model.model_dump(mode="json", exclude_none=True))

    # General error handling for setup issues before stream/non-stream logic
    except HTTPException: # Re-raise HTTPExceptions raised internally
        raise # Corrected indentation
    except Exception as e:
        LOGGER.error(f"Unexpected error in chat_ollama_completion setup: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")