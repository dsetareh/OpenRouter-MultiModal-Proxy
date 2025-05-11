# OpenAI API Router with OpenRouter Backend

This project is an API router that intercepts multimodal requests (Image/Video), even when the multimodal data is a link in the text body of the request. It intelligently processes requests that include image or video links, logs all activity, tracks costs, and routes requests to appropriate models via OpenRouter.


## Installation Guide

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dsetareh/OpenRouter-MultiModal-Proxy
    cd OpenRouter-MultiModal-Proxy
    ```

2.  **Configure environment variables:**
    *   Copy `.env.example` to `.env` in the root of the project:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your `OPENROUTER_API_KEY` and any other configurations. The `docker-compose.yml` file is set up to pass these variables to the container.

    ```env
    OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY_HERE"
    DEFAULT_TEXT_MODEL="qwen/qwen3-235b-a22b"
    VISION_MODEL="mistralai/mistral-small-3.1-24b-instruct"
    LOG_FILE_PATH="/data/router.log.json"
    OPENROUTER_REFERER="http://localhost:8000"
    OPENROUTER_X_TITLE="OpenRouterProxy"
    DATABASE_URL="sqlite+aiosqlite:////data/openrouter-multimodal-proxy.db"
    WHISPER_MODEL_NAME="distil-medium.en"
    WHISPER_DEVICE="cpu"
    ```

3.  **Build and run the application using Docker Compose:**
    Open your terminal in the project root directory and run:
    ```bash
    docker-compose up -d --build
    ```
*   The application will be available at `http://<docker_host>:8000`.

## Core Features

*   **OpenAI API Compatibility:**
    *   Supports `POST /v1/chat/completions` for chat-based interactions.
    *   Supports `POST /v1/completions` for legacy text completion.
*   **Intelligent Media Processing:**
    *   Detects image and video URLs within request payloads.
    *   Processes images (downloads and converts to base64 if necessary).
    *   Processes videos by downloading, extracting audio, transcribing audio (using a configurable `faster-whisper` model and device), and extracting keyframes.
    *   Transforms requests with media content into a format suitable for multimodal models on OpenRouter.
*   **OpenRouter Integration:**
    *   Routes requests to specified or default models on OpenRouter.
    *   Utilizes `OPENROUTER_API_KEY` for authentication.
    *   Allows configuration of `HTTP-Referer` and `X-Title` for OpenRouter requests.
*   **Comprehensive Logging:**
    *   Logs full request and response bodies in structured JSON format to `router.log.json`.
    *   Includes timestamps, unique request IDs, endpoint details, and processing durations.
    *   Implements log rotation.
*   **Cost Tracking & Analytics:**
    *   Stores detailed request information in a local SQLite database (`.db`).
    *   Tracks model usage, tokens, costs (as reported by OpenRouter), latencies, and potential errors.
*   **Web UI for Tracking:**
    *   Provides a web interface at `/ui/tracking` to view and filter logged API requests.
    *   Allows pagination, sorting, and searching of request data.

## Supported API Endpoints
#### OpenAI
*   `POST /v1/chat/completions`
*   `POST /v1/completions`
#### Ollama (wip)
*   `POST /api/chat`
*   `POST /api/generate`
*   `GET  /api/tags`
#### Internal
*   `GET  /ui/tracking_data`
*   `GET  /ui/tracking`