# OpenAI API Router with OpenRouter Backend

## Objective

This project is an API router designed to emulate OpenAI API endpoints. It intelligently processes requests that include image or video links, logs all activity, tracks costs, and routes requests to appropriate models via OpenRouter.

## Core Features

*   **OpenAI API Compatibility:**
    *   Supports `POST /v1/chat/completions` for chat-based interactions.
    *   Supports `POST /v1/completions` for legacy text completion.
*   **Intelligent Media Processing:**
    *   Detects image and video URLs within request payloads.
    *   Processes images (downloads and converts to base64 if necessary).
    *   Processes videos by downloading, extracting audio, transcribing audio, and extracting keyframes.
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
    *   Stores detailed request information in a local SQLite database (`oai_router.db`).
    *   Tracks model usage, tokens, costs (as reported by OpenRouter), latencies, and potential errors.
*   **Web UI for Tracking:**
    *   Provides a web interface at `/ui/tracking` to view and filter logged API requests.
    *   Allows pagination, sorting, and searching of request data.

## Technology Stack

*   **Language/Framework:** Python 3.x with FastAPI
*   **API Key Management:** `.env` file for `OPENROUTER_API_KEY` and other configurations.
*   **Configuration:** `app/config.py` using `pydantic-settings`.
*   **Asynchronous Operations:** `aiohttp` for non-blocking calls to OpenRouter and media downloads.
*   **Database:** SQLite with SQLAlchemy (async support) for request logging and cost tracking.
*   **Media Processing:**
    *   `yt-dlp`: For downloading videos.
    *   `ffmpeg-python`: For video frame and audio extraction.
    *   `faster-whisper`: For audio transcription.
    *   `Pillow`: For image handling.
*   **Templating:** Jinja2 for the tracking UI.

## Setup and Installation

Follow these steps to set up and run the project locally or using Docker.

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd oai_router
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv env
    # On Windows
    .\env\Scripts\activate
    # On macOS/Linux
    source env/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `ffmpeg` must be installed separately and available in your system's PATH for video processing features to work.*

4.  **Configure environment variables:**
    *   Copy `.env.example` to `.env`:
        ```bash
        copy .env.example .env # Windows
        cp .env.example .env  # macOS/Linux
        ```
    *   Edit `.env` and add your `OPENROUTER_API_KEY` and any other desired configurations:
        ```env
        OPENROUTER_API_KEY="your_openrouter_api_key_here"
        DEFAULT_TEXT_MODEL="openai/gpt-3.5-turbo"
        VISION_MODEL="mistralai/mistral-small-3.1-24b-instruct"
        LOG_FILE_PATH="router.log.json"
        # OPENROUTER_REFERER="https://your-site-url.com" # Optional
        # OPENROUTER_X_TITLE="Your App Name" # Optional
        DATABASE_URL="sqlite+aiosqlite:///./oai_router.db"
        ```

5.  **Run the application:**
    ```bash
    uvicorn app.main:app --reload
    ```
    The API will typically be available at `http://127.0.0.1:8000`.

### Docker Setup

Alternatively, you can run the application using Docker and Docker Compose.

1.  **Ensure Docker is installed:**
    *   Download and install Docker Desktop from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop).

2.  **Clone the repository (if not already done):**
    ```bash
    git clone <repository-url>
    cd oai_router
    ```

3.  **Configure environment variables for Docker:**
    *   Copy `.env.example` to `.env` in the root of the project:
        ```bash
        # On Windows (PowerShell/CMD)
        copy .env.example .env
        # On macOS/Linux
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your `OPENROUTER_API_KEY` and any other configurations. The `docker-compose.yml` file is set up to pass these variables to the container.
        ```env
        OPENROUTER_API_KEY="your_openrouter_api_key_here"
        DEFAULT_TEXT_MODEL="openai/gpt-3.5-turbo"
        VISION_MODEL="mistralai/mistral-small-3.1-24b-instruct"
        LOG_FILE_PATH="/app/router.log.json" # Path inside the container
        DATABASE_URL="sqlite+aiosqlite:///app/oai_router.db" # Path inside the container
        # OPENROUTER_REFERER="https://your-site-url.com" # Optional
        # OPENROUTER_X_TITLE="Your App Name" # Optional
        ```
        *Note: For `LOG_FILE_PATH` and `DATABASE_URL`, use the paths as they will be inside the container (e.g., `/app/router.log.json` and `/app/oai_router.db`). The `docker-compose.yml` handles mounting local files to these container paths.*

4.  **Build and run the application using Docker Compose:**
    Open your terminal in the project root directory and run:
    ```bash
    docker-compose up --build
    ```
    *   The `--build` flag ensures the Docker image is built (or rebuilt if changes are detected).
    *   The application will be available at `http://localhost:8000`.
    *   Logs and the SQLite database will be stored in `router.log.json` and `oai_router.db` in your project directory, as they are mounted as volumes.

5.  **To stop the application:**
    Press `Ctrl+C` in the terminal where `docker-compose up` is running, or run:
    ```bash
    docker-compose down
    ```

## API Endpoints

*   **Chat Completions:** `POST /v1/chat/completions`
*   **Legacy Completions:** `POST /v1/completions`

Refer to the OpenAI API documentation for request and response formats.

## Logging and Tracking UI

*   **Log File:** `router.log.json` (configurable via `.env`)
*   **Tracking UI:** Navigate to `http://127.0.0.1:8000/ui/tracking` in your browser to view API request logs.

