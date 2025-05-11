# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

# OpenAI-compatible Schemas
class ChatMessage(BaseModel):
    role: str # "system", "user", "assistant", "tool"
    content: Union[str, List[Dict[str, Any]]] # content can be string or list of content parts for multimodal
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None # Will be overridden or defaulted by our router
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Add other OpenAI params as needed: functions, function_call, tool_choice etc.
    # For now, keeping it to common ones.

class ChatCompletionChoiceDelta(BaseModel): # For streaming
    role: Optional[str] = None
    content: Optional[str] = None
    # tool_calls: Optional[List[Dict[str, Any]]] = None # If supporting tool calls in stream

class ChatCompletionStreamChoice(BaseModel): # For streaming
    index: int
    delta: ChatMessage # or ChatCompletionChoiceDelta if more specific
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None # OpenAI spec allows this

class ChatCompletionStreamResponse(BaseModel): # For streaming
    id: str
    object: str = "chat.completion.chunk"
    created: int # Unix timestamp
    model: str
    choices: List[ChatCompletionStreamChoice]
    system_fingerprint: Optional[str] = None
    # usage: Optional[OpenAIUsage] = None # Usage is typically sent at the end of stream or not at all in chunks

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None # OpenAI spec allows this

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: Optional[int] = None # Optional because it might not be present if streaming or error
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str # Unique ID for the request, can be the OpenRouter request ID or our internal one
    object: str = "chat.completion"
    created: int # Unix timestamp
    model: str # The model that generated the response
    choices: List[ChatCompletionChoice]
    usage: Optional[OpenAIUsage] = None
    system_fingerprint: Optional[str] = None
# Legacy Completion Schemas
class CompletionRequest(BaseModel):
    model: Optional[str] = None # Will be overridden or defaulted
    prompt: Union[str, List[str]] # OpenAI allows string or list of strings for batch, we'll focus on single string first
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Other params if needed

class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None # Can be complex
    finish_reason: Optional[str] = None

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int # Unix timestamp
    model: str
    choices: List[CompletionChoice]
    usage: Optional[OpenAIUsage] = None # Re-use OpenAIUsage from chat schemas
    system_fingerprint: Optional[str] = None