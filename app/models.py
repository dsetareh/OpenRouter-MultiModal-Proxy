# app/models.py
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, func
from app.database import Base # Import Base from the new database.py

class OpenRouterRequest(Base):
    __tablename__ = "openrouter_requests"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    internal_request_id = Column(String, unique=True, nullable=False, index=True)
    endpoint_called = Column(String, nullable=False)
    client_ip = Column(String, nullable=True)
    model_requested_by_client = Column(String, nullable=True)
    model_routed_to_openrouter = Column(String, nullable=False)
    openrouter_request_id = Column(String, nullable=True, index=True) # From OpenRouter
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    is_multimodal = Column(Boolean, default=False, nullable=False)
    media_type_processed = Column(String, nullable=True) # e.g., 'image', 'video', 'image_link', 'video_link'
    input_char_length = Column(Integer, nullable=True)
    output_char_length = Column(Integer, nullable=True)
    processing_duration_ms = Column(Integer, nullable=True) # Router's own processing time
    openrouter_latency_ms = Column(Integer, nullable=True) # Latency of the call to OpenRouter
    status_code_returned_to_client = Column(Integer, nullable=False)
    error_source = Column(String, nullable=True) # e.g., 'router', 'openrouter', 'media_processing'
    error_message = Column(Text, nullable=True)