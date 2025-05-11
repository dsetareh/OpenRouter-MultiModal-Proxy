FROM python:3.11-slim

WORKDIR /app

# Install ffmpeg for video processing
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app
COPY .env.example /app/.env.example
COPY router.log.json /app/router.log.json
COPY oai_router.db /app/oai_router.db

# Ensure the log file and database file exist and are writable
# These will be mounted as volumes in docker-compose, but good to have placeholders
RUN touch /app/router.log.json && chmod 666 /app/router.log.json
RUN touch /app/oai_router.db && chmod 666 /app/oai_router.db


EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
