# app/main.py (modifications)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import uuid
import aiohttp # Added
from app.database import init_db
from app.logging_config import LOGGER # Import the logger
from app.openrouter_client import OpenRouterClient # Added
from app.routers import chat as chat_router # Import the chat router
from app.routers import completions as completions_router # Import the completions router
from app.routers import ui as ui_router # Import the UI router
from app.routers import ollama as ollama_router # Import the Ollama router

@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("Application startup: Initializing database...")
    await init_db()
    LOGGER.info("Application startup: Database initialized.")
    
    LOGGER.info("Application startup: Creating aiohttp.ClientSession...")
    async with aiohttp.ClientSession() as session:
        app.state.http_session = session # Store session in app.state
        app.state.openrouter_client = OpenRouterClient(session) # Create client instance
        LOGGER.info("Application startup: aiohttp.ClientSession created and OpenRouterClient initialized.")
        yield
    
    LOGGER.info("Application shutdown: aiohttp.ClientSession closed.")
    LOGGER.info("Application shutdown.")

app = FastAPI(title="OpenAI API Router", lifespan=lifespan)

# Global Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    internal_request_id = getattr(request.state, "internal_request_id", "N/A")
    LOGGER.error(
        "Unhandled exception",
        exc_info=True, # Includes stack trace
        extra={
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "internal_request_id": internal_request_id,
            "request_url": str(request.url),
            "request_method": request.method,
        }
    )
    # Return OpenAI-compatible error response
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "An internal server error occurred.",
                "type": "internal_server_error",
                "param": None,
                "code": None,
                "internal_request_id": internal_request_id
            }
        },
    )

# Request/Response Logging Middleware
@app.middleware("http")
async def log_requests_responses(request: Request, call_next):
    start_time = time.time()
    internal_request_id = str(uuid.uuid4())
    request.state.internal_request_id = internal_request_id # Make it available to handlers

    LOGGER.info(
        "Incoming request",
        extra={
            "internal_request_id": internal_request_id,
            "method": request.method,
            "url": str(request.url),
            "client_host": request.client.host,
            "client_port": request.client.port,
            "headers": dict(request.headers) # Be cautious with logging all headers in production
        }
    )

    try:
        response = await call_next(request)
        process_time_ms = (time.time() - start_time) * 1000
        
        LOGGER.info(
            "Outgoing response",
            extra={
                "internal_request_id": internal_request_id,
                "status_code": response.status_code,
                "process_time_ms": round(process_time_ms, 2),
                # "response_headers": dict(response.headers) # Be cautious
            }
        )
        response.headers["X-Internal-Request-ID"] = internal_request_id # Add to response
        return response
    except Exception as e:
        process_time_ms = (time.time() - start_time) * 1000
        LOGGER.error(
            "Exception during request processing in middleware",
            exc_info=True,
            extra={
                "internal_request_id": internal_request_id,
                "process_time_ms": round(process_time_ms, 2),
            }
        )
        raise e


app.include_router(chat_router.router) # Include the chat completions router
app.include_router(completions_router.router) # Include the legacy completions router
app.include_router(ui_router.router) # Include the UI router
app.include_router(ollama_router.router) # Include the Ollama router

@app.get("/")
async def read_root(request: Request): # Added request: Request to access request.state
    internal_request_id = getattr(request.state, "internal_request_id", "N/A_from_root")
    LOGGER.info("Root endpoint called.", extra={"internal_request_id": internal_request_id})
    return {"message": "OpenAI API Router is running"}

# Placeholder for future imports and routes