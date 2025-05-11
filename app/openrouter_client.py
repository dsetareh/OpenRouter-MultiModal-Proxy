import aiohttp
import json
import time # Added as per instructions
from app.config import settings
from app.logging_config import LOGGER
# from app.schemas import OpenRouterResponse # If using Pydantic models from schemas.py

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions" # Default to chat

class OpenRouterClient:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.api_key = settings.OPENROUTER_API_KEY
        self.referer = settings.OPENROUTER_REFERER
        self.x_title = settings.OPENROUTER_X_TITLE

    async def _request(self, method: str, url: str, payload: dict):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        request_log_extra = {
            "url": url,
            "method": method,
            "model_requested": payload.get("model") # Log which model is in payload
        }
        LOGGER.info("Sending request to OpenRouter", extra=request_log_extra)

        start_time = time.time()
        try:
            async with self.session.request(method, url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as response:
                latency_ms = (time.time() - start_time) * 1000 # Calculate latency early
                openrouter_req_id = response.headers.get("X-Request-ID")
                
                response_log_extra = {
                    "url": url,
                    "status_code": response.status,
                    "openrouter_request_id": openrouter_req_id,
                    "latency_ms": round(latency_ms, 2),
                    "content_type": response.content_type
                }

                if response.status >= 200 and response.status < 300:
                    # Attempt to parse JSON, but be ready for ContentTypeError
                    if response.content_type == 'application/json':
                        response_data = await response.json()
                        LOGGER.info("Received successful JSON response from OpenRouter", extra=response_log_extra)
                        usage = response_data.get("usage", {})
                        return {
                            "data": response_data,
                            "status_code": response.status,
                            "openrouter_request_id": openrouter_req_id,
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                            "total_tokens": usage.get("total_tokens"),
                            "cost_usd": usage.get("cost"),
                            "latency_ms": latency_ms
                        }
                    elif response.content_type == 'text/event-stream':
                        LOGGER.warning(
                            f"OpenRouter returned 'text/event-stream' for a non-streaming request (status: {response.status}). Attempting to reassemble stream.",
                            extra=response_log_extra
                        )
                        
                        accumulated_content = ""
                        accumulated_tool_calls = []
                        final_role = "assistant"
                        final_finish_reason = None
                        response_id, response_model, response_created, response_object = None, None, None, None
                        final_usage = None

                        full_stream_content = await response.text()

                        try:
                            for line in full_stream_content.splitlines():
                                line = line.strip()
                                if not line or line.startswith(":"):
                                    continue
                                if line.startswith("data:"):
                                    json_str = line[len("data:"):].strip()
                                    if json_str == "[DONE]":
                                        break
                                    
                                    try:
                                        chunk_data = json.loads(json_str)
                                    except json.JSONDecodeError as e:
                                        LOGGER.error(f"Failed to parse JSON chunk from unexpected stream. Error: {e}. Chunk: '{json_str}'", extra=response_log_extra)
                                        # Fallback to 502
                                        return {
                                            "error": {
                                                "message": "Failed to reassemble unexpected stream from OpenRouter: Malformed JSON chunk.",
                                                "type": "unexpected_stream_reassembly_failed",
                                                "details": f"Problematic chunk: {json_str[:200]}"
                                            },
                                            "status_code": 502,
                                            "openrouter_request_id": openrouter_req_id,
                                            "latency_ms": latency_ms
                                        }

                                    if response_id is None: # Populate from first valid chunk
                                        response_id = chunk_data.get("id")
                                        response_model = chunk_data.get("model")
                                        response_created = chunk_data.get("created")
                                        response_object = "chat.completion" # Override for non-streaming format

                                    delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                                    
                                    if delta.get("role"):
                                        final_role = delta["role"]
                                    
                                    if delta.get("content"):
                                        accumulated_content += delta["content"]
                                    
                                    if delta.get("tool_calls"):
                                        for tool_call_chunk in delta["tool_calls"]:
                                            idx = tool_call_chunk.get("index")
                                            existing_tool_call = next((tc for tc in accumulated_tool_calls if tc.get("index") == idx), None)
                                            
                                            if existing_tool_call is None:
                                                new_tc = {
                                                    "index": idx,
                                                    "id": tool_call_chunk.get("id"),
                                                    "type": tool_call_chunk.get("type", "function"),
                                                    "function": {
                                                        "name": tool_call_chunk.get("function", {}).get("name", ""),
                                                        "arguments": tool_call_chunk.get("function", {}).get("arguments", "")
                                                    }
                                                }
                                                accumulated_tool_calls.append(new_tc)
                                            else:
                                                if tool_call_chunk.get("id"):
                                                    existing_tool_call["id"] = tool_call_chunk["id"]
                                                if tool_call_chunk.get("type"):
                                                    existing_tool_call["type"] = tool_call_chunk["type"]
                                                
                                                func_chunk = tool_call_chunk.get("function", {})
                                                if "name" in func_chunk:
                                                    existing_tool_call["function"]["name"] += func_chunk["name"]
                                                if "arguments" in func_chunk:
                                                    existing_tool_call["function"]["arguments"] += func_chunk["arguments"]
                                    
                                    finish_reason_chunk = chunk_data.get("choices", [{}])[0].get("finish_reason")
                                    if finish_reason_chunk is not None:
                                        final_finish_reason = finish_reason_chunk
                                    
                                    if chunk_data.get("usage"):
                                        final_usage = chunk_data["usage"]
                            
                            # After loop, check if essential data was populated
                            if response_id is None: # Indicates no valid chunks or essential data missing
                                LOGGER.error("Failed to reassemble stream: Essential metadata (e.g., ID) not found in any chunk.", extra=response_log_extra)
                                return {
                                    "error": {
                                        "message": "Failed to reassemble unexpected stream from OpenRouter: Essential data missing from stream.",
                                        "type": "unexpected_stream_reassembly_failed_critical_data",
                                        "details": f"Full stream content (first 500 chars): {full_stream_content[:500]}"
                                    },
                                    "status_code": 502,
                                    "openrouter_request_id": openrouter_req_id,
                                    "latency_ms": latency_ms
                                }

                            # Construct final reassembled JSON
                            valid_tool_calls = []
                            if accumulated_tool_calls:
                                for tc in accumulated_tool_calls:
                                    if tc.get("id"): # Only include if ID was populated
                                        valid_tool_calls.append({
                                            "id": tc["id"],
                                            "type": tc.get("type", "function"),
                                            "function": {
                                                "name": tc.get("function", {}).get("name", ""),
                                                "arguments": tc.get("function", {}).get("arguments", "")
                                            }
                                        })
                            
                            message_content = accumulated_content if accumulated_content else None
                            
                            reassembled_response_data = {
                                "id": response_id,
                                "object": response_object,
                                "created": response_created,
                                "model": response_model,
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": final_role,
                                        "content": message_content,
                                    },
                                    "finish_reason": final_finish_reason
                                }]
                            }
                            if valid_tool_calls:
                                reassembled_response_data["choices"][0]["message"]["tool_calls"] = valid_tool_calls
                            if final_usage:
                                reassembled_response_data["usage"] = final_usage

                            LOGGER.info("Successfully reassembled unexpected 'text/event-stream' into a single JSON response.", extra=response_log_extra)
                            return {
                                "data": reassembled_response_data,
                                "status_code": response.status, # Original status (e.g. 200)
                                "openrouter_request_id": openrouter_req_id,
                                "prompt_tokens": final_usage.get("prompt_tokens") if final_usage else None,
                                "completion_tokens": final_usage.get("completion_tokens") if final_usage else None,
                                "total_tokens": final_usage.get("total_tokens") if final_usage else None,
                                "cost_usd": final_usage.get("cost") if final_usage else None, # Assuming cost is part of usage
                                "latency_ms": latency_ms
                            }

                        except Exception as e: # Catch-all for unexpected errors during stream processing
                            LOGGER.error(f"Unexpected error during stream reassembly: {e}", extra=response_log_extra, exc_info=True)
                            return {
                                "error": {
                                    "message": f"Failed to reassemble unexpected stream from OpenRouter due to an internal error: {str(e)}",
                                    "type": "unexpected_stream_reassembly_internal_error",
                                    "details": f"Full stream content (first 500 chars): {full_stream_content[:500]}"
                                },
                                "status_code": 502,
                                "openrouter_request_id": openrouter_req_id,
                                "latency_ms": latency_ms
                            }
                    else: # Other unexpected content type with 2xx status
                        unexp_content = await response.text()
                        LOGGER.error(f"OpenRouter API success status but unexpected Content-Type: {response.content_type}. Body: {unexp_content[:200]}", extra=response_log_extra)
                        return {
                            "error": {
                                "message": f"OpenRouter returned an unexpected content type '{response.content_type}' with a success status.",
                                "type": "unexpected_content_type",
                                "details": unexp_content[:500]
                            },
                            "status_code": 502,
                            "openrouter_request_id": openrouter_req_id,
                            "latency_ms": latency_ms
                        }
                else: # Non-2xx status codes
                    try:
                        # Try to parse error as JSON, as OpenRouter often returns JSON errors
                        error_data = await response.json()
                    except aiohttp.client_exceptions.ContentTypeError:
                        # If error is not JSON, read as text
                        error_data = await response.text()
                    
                    LOGGER.error(f"OpenRouter API error: Status {response.status}", extra={**response_log_extra, "response_body": error_data})
                    return {
                        "error": error_data, # This could be a dict or string
                        "status_code": response.status,
                        "openrouter_request_id": openrouter_req_id,
                        "latency_ms": latency_ms
                    }

        except aiohttp.client_exceptions.ContentTypeError as cte: # This might catch cases where response.json() is called on non-2xx non-JSON
            # This specific handler is more for cte during response.json() on non-2xx if not caught by above logic
            latency_ms = (time.time() - start_time) * 1000 # Recalculate if error before first calc
            LOGGER.error(f"aiohttp.ContentTypeError during OpenRouter call: {cte}", exc_info=True, extra={"url": url, "latency_ms": latency_ms, "response_content_type": cte.headers.get('Content-Type') if cte.headers else 'N/A'})
            return {"error": f"ContentTypeError: {cte.message} (Content-Type: {cte.headers.get('Content-Type') if cte.headers else 'N/A'})", "status_code": 502, "latency_ms": latency_ms}

        except aiohttp.ClientError as e: # General client errors (timeout, connection error, etc.)
            latency_ms = (time.time() - start_time) * 1000
            LOGGER.error(f"aiohttp.ClientError calling OpenRouter: {e}", exc_info=True, extra={"url": url, "latency_ms": latency_ms})
            return {"error": str(e), "status_code": 503, "latency_ms": latency_ms} # 503 Service Unavailable
        
        except Exception as e: # Catch any other unexpected errors during the request
            latency_ms = (time.time() - start_time) * 1000
            LOGGER.error(f"Unexpected error calling OpenRouter: {e}", exc_info=True, extra={"url": url, "latency_ms": latency_ms})
            return {"error": str(e), "status_code": 500, "latency_ms": latency_ms}


    async def create_chat_completion(self, payload: dict):
        # Ensure 'stream': False is set if not explicitly handled, or handle streaming if needed
        # The plan doesn't explicitly ask for streaming support from OpenRouter yet.
        payload.setdefault('stream', False)
        # Add 'usage: {include: true}' if OpenRouter supports it this way for detailed cost.
        # Some models/routes might require it in the payload.
        # payload['usage'] = {'include': True} # This might be specific to certain OpenRouter features or models.
                                             # For now, we rely on standard usage fields in response.
        return await self._request("POST", OPENROUTER_API_URL, payload)

# How to manage ClientSession:
# One way is to create it in main.py lifespan and pass it around or use a global/singleton.
# For now, the client expects a session to be passed.