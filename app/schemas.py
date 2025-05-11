# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import datetime


# OpenAI-compatible Schemas
class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant", "tool"
    content: Union[
        str, List[Dict[str, Any]]
    ]  # content can be string or list of content parts for multimodal
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None  # Will be overridden or defaulted by our router
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


class ChatCompletionChoiceDelta(BaseModel):  # For streaming
    role: Optional[str] = None
    content: Optional[str] = None
    # tool_calls: Optional[List[Dict[str, Any]]] = None # If supporting tool calls in stream


class ChatCompletionStreamChoice(BaseModel):  # For streaming
    index: int
    delta: ChatMessage  # or ChatCompletionChoiceDelta if more specific
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None  # OpenAI spec allows this


class ChatCompletionStreamResponse(BaseModel):  # For streaming
    id: str
    object: str = "chat.completion.chunk"
    created: int  # Unix timestamp
    model: str
    choices: List[ChatCompletionStreamChoice]
    system_fingerprint: Optional[str] = None
    # usage: Optional[OpenAIUsage] = None # Usage is typically sent at the end of stream or not at all in chunks


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None  # OpenAI spec allows this


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: Optional[int] = (
        None  # Optional because it might not be present if streaming or error
    )
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str  # Unique ID for the request, can be the OpenRouter request ID or our internal one
    object: str = "chat.completion"
    created: int  # Unix timestamp
    model: str  # The model that generated the response
    choices: List[ChatCompletionChoice]
    usage: Optional[OpenAIUsage] = None
    system_fingerprint: Optional[str] = None


# Legacy Completion Schemas
class CompletionRequest(BaseModel):
    model: Optional[str] = None  # Will be overridden or defaulted
    prompt: Union[
        str, List[str]
    ]  # OpenAI allows string or list of strings for batch, we'll focus on single string first
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False  # Added
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
    logprobs: Optional[Any] = None  # Can be complex
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int  # Unix timestamp
    model: str
    choices: List[CompletionChoice]
    usage: Optional[OpenAIUsage] = None  # Re-use OpenAIUsage from chat schemas
    system_fingerprint: Optional[str] = None

# Ollama Schemas

# /api/tags
class OllamaTagModelDetails(BaseModel):
    format: Optional[str] = None
    family: Optional[str] = None
    families: Optional[List[str]] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None

class OllamaTagModel(BaseModel):
    name: str
    model: str # Added based on typical Ollama responses, though not explicitly in /api/tags doc, it's often there.
    modified_at: datetime.datetime = Field(alias="modified_at")
    size: int
    digest: str
    details: OllamaTagModelDetails
    expires_at: Optional[datetime.datetime] = None # Added based on some Ollama versions
    parent_model: Optional[str] = None # Added based on some Ollama versions

class OllamaTagsResponse(BaseModel):
    models: List[OllamaTagModel]

# /api/generate
class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    images: Optional[List[str]] = None # List of base64 encoded images
    format: Optional[str] = None # "json"
    options: Optional[Dict[str, Any]] = None # Runtime parameters
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None # Previous context array
    stream: Optional[bool] = True # Default to true as per Ollama, but allow override
    raw: Optional[bool] = False # Use raw prompt without templating
    keep_alive: Optional[Union[str, float, int]] = Field(default=None, alias="keep_alive") # duration string or number

class OllamaGenerateResponseStreamChunk(BaseModel):
    model: str
    created_at: datetime.datetime = Field(alias="created_at")
    response: str
    done: bool
    # The following fields appear in the *final* chunk when stream=True
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    done_reason: Optional[str] = None # Added based on Ollama docs for final stream object

class OllamaGenerateResponseFinal(BaseModel): # Non-streaming or final summary of stream
    model: str
    created_at: datetime.datetime = Field(alias="created_at")
    response: str
    done: bool
    done_reason: Optional[str] = None # Added based on Ollama docs
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


# /api/chat
class OllamaToolCallDefinition(BaseModel):
    name: str
    arguments: Dict[str, Any]

class OllamaToolDefinition(BaseModel):
    type: str = "function" # Currently, 'function' is the only supported type
    function: OllamaToolCallDefinition

class OllamaChatMessage(BaseModel):
    role: str # "system", "user", "assistant"
    content: str
    images: Optional[List[str]] = None # List of base64 encoded images
    tool_calls: Optional[List[OllamaToolDefinition]] = None # For assistant messages requesting tool use

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaChatMessage]
    format: Optional[str] = None # "json"
    options: Optional[Dict[str, Any]] = None # Runtime parameters
    stream: Optional[bool] = True # Default to true, but allow override
    template: Optional[str] = None # Not explicitly in /api/chat but common
    keep_alive: Optional[Union[str, float, int]] = Field(default=None, alias="keep_alive")
    tools: Optional[List[OllamaToolDefinition]] = None # For function calling

class OllamaChatMessagePart(BaseModel): # For streaming delta
    role: Optional[str] = None
    content: Optional[str] = None
    images: Optional[List[str]] = None # Unlikely in delta, but for completeness
    tool_calls: Optional[List[OllamaToolDefinition]] = None # For streaming tool calls

class OllamaChatResponseStreamChunk(BaseModel):
    model: str
    created_at: datetime.datetime = Field(alias="created_at")
    message: OllamaChatMessagePart # Delta of the message
    done: bool
    # The following fields appear in the *final* chunk when stream=True and done=True
    done_reason: Optional[str] = None # e.g. "stop", "length", "tool_calls"
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

class OllamaChatResponseFinal(BaseModel): # Non-streaming or final summary of stream
    model: str
    created_at: datetime.datetime = Field(alias="created_at")
    message: OllamaChatMessage # Full message object
    done: bool
    done_reason: Optional[str] = None # e.g. "stop", "length", "tool_calls"
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None