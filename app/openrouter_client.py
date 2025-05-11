import aiohttp
import time
import json
from typing import AsyncGenerator, Dict, List, Any

from app.config import settings
from app.logging_config import LOGGER


class OpenRouterClient:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.api_key = settings.OPENROUTER_API_KEY
        self.referer = settings.OPENROUTER_REFERER
        self.x_title = settings.OPENROUTER_X_TITLE
        self.openrouter_api_url = settings.OPENROUTER_API_URL
        self.openrouter_models_api_url = settings.OPENROUTER_MODELS_API_URL

    async def _request(
        self, method: str, url: str, payload: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:  # Modified
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.x_title:
            headers["X-Title"] = self.x_title
        payload.setdefault("usage", {"include": True})  # Default to False if not set
        request_log_extra = {
            "url": url,
            "method": method,
            "model_requested": payload.get("model"),  # Log which model is in payload
            "streaming_requested": payload.get("stream", False),
        }
        LOGGER.info("Sending request to OpenRouter", extra=request_log_extra)

        start_time = time.time()
        is_streaming_request = payload.get("stream", False)

        try:
            async with self.session.request(
                method,
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:  # Increased timeout for streaming
                latency_ms = (
                    time.time() - start_time
                ) * 1000  # Calculate latency early
                openrouter_req_id = response.headers.get("X-Request-ID")

                response_log_extra = {
                    "url": url,
                    "status_code": response.status,
                    "openrouter_request_id": openrouter_req_id,
                    "latency_ms": round(latency_ms, 2),
                    "content_type": response.content_type,
                    "streaming_requested": is_streaming_request,
                }

                if response.status >= 200 and response.status < 300:
                    if (
                        is_streaming_request
                        and response.content_type == "text/event-stream"
                    ):
                        LOGGER.info(
                            "Receiving streaming response from OpenRouter",
                            extra=response_log_extra,
                        )
                        # Yield chunks as they come
                        async for line in response.content:
                            if line.strip():
                                line_str = line.decode("utf-8").strip()
                                if line_str.startswith("data: "):
                                    data_content = line_str[len("data: ") :]
                                    if data_content == "[DONE]":
                                        LOGGER.info(
                                            "OpenRouter stream finished with [DONE]",
                                            extra=response_log_extra,
                                        )
                                        yield {
                                            "stream_done": True,
                                            "openrouter_request_id": openrouter_req_id,
                                            "latency_ms": (time.time() - start_time)
                                            * 1000,
                                        }
                                        return
                                    try:
                                        chunk_json = json.loads(data_content)
                                        yield {
                                            "data": chunk_json,  # This is the stream chunk
                                            "status_code": response.status,
                                            "openrouter_request_id": openrouter_req_id,
                                            "latency_ms": (time.time() - start_time)
                                            * 1000,  # Current latency
                                        }
                                    except json.JSONDecodeError:
                                        LOGGER.warning(
                                            f"Failed to decode JSON stream chunk: {data_content}",
                                            extra=response_log_extra,
                                        )
                                        # Potentially yield an error chunk or just log and continue
                        LOGGER.info(
                            "OpenRouter stream completed.", extra=response_log_extra
                        )
                        # Ensure a final marker if [DONE] was not received but stream ended.
                        # This might indicate an abrupt end from server.
                        yield {
                            "stream_ended_without_done": True,
                            "openrouter_request_id": openrouter_req_id,
                            "latency_ms": (time.time() - start_time) * 1000,
                        }

                    elif (
                        not is_streaming_request
                        and response.content_type == "application/json"
                    ):
                        response_data = await response.json()
                        LOGGER.info(
                            "Received successful JSON response from OpenRouter",
                            extra=response_log_extra,
                        )
                        usage = response_data.get("usage", {})
                        yield {  # Yield a single item for non-streaming
                            "data": response_data,
                            "status_code": response.status,
                            "openrouter_request_id": openrouter_req_id,
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                            "total_tokens": usage.get("total_tokens"),
                            "cost_usd": response_data.get("usage", {}).get(
                                "cost"
                            ),  # OpenRouter cost is in usage.cost
                            "latency_ms": latency_ms,
                        }
                    elif (
                        is_streaming_request
                        and response.content_type != "text/event-stream"
                    ):
                        error_message = f"OpenRouter returned '{response.content_type}' for a streaming request. Expected 'text/event-stream'."
                        LOGGER.error(error_message, extra=response_log_extra)
                        # Try to read body for more details if possible
                        try:
                            error_body = await response.text()
                            LOGGER.error(
                                f"OpenRouter error response body: {error_body}",
                                extra=response_log_extra,
                            )
                            error_data_to_yield = {
                                "message": error_message,
                                "details": error_body,
                            }
                        except Exception:
                            error_data_to_yield = {"message": error_message}
                        yield {
                            "error": error_data_to_yield,
                            "status_code": response.status,  # Use actual status
                            "openrouter_request_id": openrouter_req_id,
                            "latency_ms": latency_ms,
                        }
                    elif (
                        not is_streaming_request
                        and response.content_type == "text/event-stream"
                    ):
                        # This case was handled before, but now it's an error for non-streaming _request call
                        # The previous logic tried to reassemble. Now, we treat it as an unexpected response type.
                        error_message = f"OpenRouter returned 'text/event-stream' for a non-streaming request. This is unexpected."
                        LOGGER.error(error_message, extra=response_log_extra)
                        # Attempt to read and log the stream content for debugging
                        full_stream_content = await response.text()
                        LOGGER.error(
                            f"Unexpected stream content: {full_stream_content[:1000]}...",
                            extra=response_log_extra,
                        )  # Log first 1KB
                        yield {
                            "error": {
                                "message": error_message,
                                "details": "Received stream for non-stream request.",
                            },
                            "status_code": 502,  # Bad Gateway, as we received an unexpected response type
                            "openrouter_request_id": openrouter_req_id,
                            "latency_ms": latency_ms,
                        }
                    else:  # Other content types or unexpected combinations
                        error_message = f"Unexpected content type from OpenRouter: {response.content_type}. Status: {response.status}"
                        LOGGER.error(error_message, extra=response_log_extra)
                        try:
                            error_body = await response.text()
                            LOGGER.error(
                                f"OpenRouter error response body: {error_body}",
                                extra=response_log_extra,
                            )
                            error_data_to_yield = {
                                "message": error_message,
                                "details": error_body,
                            }
                        except Exception:
                            error_data_to_yield = {"message": error_message}
                        yield {
                            "error": error_data_to_yield,
                            "status_code": response.status,
                            "openrouter_request_id": openrouter_req_id,
                            "latency_ms": latency_ms,
                        }
                else:  # Non-2xx status codes
                    error_data = {}
                    try:
                        # Try to parse error as JSON, OpenRouter usually provides JSON errors
                        error_json = await response.json()
                        error_data = error_json.get(
                            "error", error_json
                        )  # Use the 'error' field if present, else the whole thing
                        LOGGER.error(
                            f"OpenRouter API error (JSON): Status {response.status}",
                            extra={**response_log_extra, "response_body": error_data},
                        )
                    except (
                        aiohttp.client_exceptions.ContentTypeError,
                        json.JSONDecodeError,
                    ):
                        error_text = await response.text()
                        error_data = {
                            "message": f"Non-JSON error response. Status: {response.status}",
                            "details": error_text[:500],
                        }  # Truncate long errors
                        LOGGER.error(
                            f"OpenRouter API error (Non-JSON): Status {response.status}",
                            extra={
                                **response_log_extra,
                                "response_body": error_text[:500],
                            },
                        )

                    yield {
                        "error": error_data,
                        "status_code": response.status,
                        "openrouter_request_id": openrouter_req_id,
                        "latency_ms": latency_ms,
                    }

        except aiohttp.client_exceptions.ContentTypeError as cte:
            latency_ms = (time.time() - start_time) * 1000
            LOGGER.error(
                f"aiohttp.ContentTypeError during OpenRouter call: {cte}",
                exc_info=True,
                extra={
                    "url": url,
                    "latency_ms": latency_ms,
                    "response_content_type": (
                        cte.headers.get("Content-Type") if cte.headers else "N/A"
                    ),
                },
            )
            yield {
                "error": f"ContentTypeError: {cte.message} (Content-Type: {cte.headers.get('Content-Type') if cte.headers else 'N/A'})",
                "status_code": 502,
                "latency_ms": latency_ms,
            }

        except (
            aiohttp.ClientError
        ) as e:  # General client errors (timeout, connection error, etc.)
            latency_ms = (time.time() - start_time) * 1000
            LOGGER.error(
                f"aiohttp.ClientError calling OpenRouter: {e}",
                exc_info=True,
                extra={"url": url, "latency_ms": latency_ms},
            )
            yield {"error": str(e), "status_code": 503, "latency_ms": latency_ms}

        except Exception as e:  # Catch any other unexpected errors during the request
            latency_ms = (time.time() - start_time) * 1000
            LOGGER.error(
                f"Unexpected error calling OpenRouter: {e}",
                exc_info=True,
                extra={"url": url, "latency_ms": latency_ms},
            )
            yield {"error": str(e), "status_code": 500, "latency_ms": latency_ms}

    async def create_chat_completion(
        self, payload: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:  # Modified
        # The 'stream' key in payload now dictates behavior in _request.
        # No need to setdefault('stream', False) here if the caller is responsible.
        # However, if this method is called internally and might not have stream set,
        # it could be a good idea, but for now, assume caller sets it.
        async for item in self._request("POST", self.openrouter_api_url, payload):
            yield item


    async def list_openrouter_models(self) -> List[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        request_log_extra = {
            "url": self.openrouter_models_api_url,
            "method": "GET",
        }
        LOGGER.info("Sending request to OpenRouter for models list", extra=request_log_extra)
        start_time = time.time()

        try:
            async with self.session.get(
                self.openrouter_models_api_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60), # 60 seconds timeout
            ) as response:
                latency_ms = (time.time() - start_time) * 1000
                openrouter_req_id = response.headers.get("X-Request-ID")

                response_log_extra = {
                    "url": self.openrouter_models_api_url,
                    "status_code": response.status,
                    "openrouter_request_id": openrouter_req_id,
                    "latency_ms": round(latency_ms, 2),
                    "content_type": response.content_type,
                }

                if response.status == 200:
                    if response.content_type == "application/json":
                        response_data = await response.json()
                        if "data" in response_data and isinstance(response_data["data"], list):
                            LOGGER.info(
                                "Successfully fetched models list from OpenRouter",
                                extra=response_log_extra,
                            )
                            return response_data["data"]
                        else:
                            LOGGER.error(
                                "OpenRouter models response missing 'data' list or is not a list",
                                extra={**response_log_extra, "response_body": response_data},
                            )
                            return []
                    else:
                        error_text = await response.text()
                        LOGGER.error(
                            f"OpenRouter models response not JSON. Content-Type: {response.content_type}",
                            extra={**response_log_extra, "response_body": error_text[:500]},
                        )
                        return []
                else:
                    error_text = await response.text()
                    LOGGER.error(
                        f"OpenRouter API error fetching models: Status {response.status}",
                        extra={**response_log_extra, "response_body": error_text[:500]},
                    )
                    return []

        except aiohttp.ClientError as e:
            latency_ms = (time.time() - start_time) * 1000
            LOGGER.error(
                f"aiohttp.ClientError calling OpenRouter for models: {e}",
                exc_info=True,
                extra={"url": self.openrouter_models_api_url, "latency_ms": latency_ms},
            )
            return []
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            LOGGER.error(
                f"Unexpected error calling OpenRouter for models: {e}",
                exc_info=True,
                extra={"url": self.openrouter_models_api_url, "latency_ms": latency_ms},
            )
            return []

# How to manage ClientSession:
# One way is to create it in main.py lifespan and pass it around or use a global/singleton.
# For now, the client expects a session to be passed.
