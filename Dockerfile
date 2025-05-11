FROM python:3.11-slim

WORKDIR /app

# Install ffmpeg for video processing
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app
COPY .env.example /app/.env.example


EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
