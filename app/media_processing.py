# app/media_processing.py
import re
import base64
import aiohttp
import mimetypes  # For guessing image type from URL if not obvious
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urlparse
import asyncio
import subprocess
import tempfile
import os
import shutil
from fastapi.concurrency import run_in_threadpool  # Or asyncio.to_thread for Py 3.9+

from app.logging_config import LOGGER
from app.config import settings
from app.schemas import ChatMessage  # For type hinting

# Regex for common image URLs and base64 data URLs
IMAGE_URL_PATTERN = re.compile(r"https?://\S+\.(?:png|jpe?g|gif|webp)", re.IGNORECASE)
BASE64_IMAGE_PATTERN = re.compile(
    r"data:image/(?:png|jpeg|gif|webp);base64,([A-Za-z0-9+/=]+)"
)
VIDEO_URL_PATTERN = re.compile(
    r"https?://\S+\.(?:mp4|mov|avi|mkv|webm|youtu\.be/|youtube\.com/watch\?v=)",
    re.IGNORECASE,
)

# Placeholder for whisper model (loaded on demand)
_whisper_model = None
_whisper_model_name = (
    settings.WHISPER_MODEL_NAME  # Or tiny.en for faster, less accurate. Or larger models.
)
_whisper_device = settings.WHISPER_DEVICE


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel

            LOGGER.info(
                f"Loading faster-whisper model: {_whisper_model_name} on device: {_whisper_device}"
            )
            # Specify compute_type, e.g., "int8" for CPU, "float16" for GPU if available
            # Device can be "cpu" or "cuda"
            _whisper_model = WhisperModel(
                _whisper_model_name,
                device=_whisper_device,
                compute_type="int8",  # TODO: allow compute_type to be configurable
            )
            LOGGER.info(f"Faster-whisper model {_whisper_model_name} loaded.")
        except ImportError:
            LOGGER.error("faster-whisper library is not installed. Please install it.")
            raise
        except Exception as e:
            LOGGER.error(f"Failed to load faster-whisper model: {e}", exc_info=True)
            raise
    return _whisper_model


async def download_image_to_base64(
    session: aiohttp.ClientSession, url: str
) -> Optional[str]:
    """Downloads an image from a URL and returns it as a base64 data URL."""
    try:
        LOGGER.debug(f"Attempting to download image from URL: {url}")
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=10)
        ) as response:  # 10s timeout for download
            response.raise_for_status()  # Raise an exception for bad status codes
            image_data = await response.read()

            content_type = response.headers.get("Content-Type")
            mime_type = None
            if content_type:
                mime_type = content_type.split(";")[0].strip()

            if not mime_type:
                parsed_url = urlparse(url)
                path = parsed_url.path
                ext = path.split(".")[-1].lower()
                if ext == "jpg":
                    ext = "jpeg"
                if ext in ["png", "jpeg", "gif", "webp"]:
                    mime_type = f"image/{ext}"
                else:
                    LOGGER.warning(
                        f"Could not determine mime type for image URL: {url}, content-type: {content_type}"
                    )
                    return None

            if mime_type not in ["image/png", "image/jpeg", "image/gif", "image/webp"]:
                LOGGER.warning(
                    f"Unsupported image mime type '{mime_type}' from URL: {url}"
                )
                return None

            base64_encoded_data = base64.b64encode(image_data).decode("utf-8")
            data_url = f"data:{mime_type};base64,{base64_encoded_data}"
            LOGGER.info(
                f"Successfully downloaded and base64 encoded image from {url}. Size: {len(image_data)} bytes."
            )
            return data_url
    except aiohttp.ClientError as e:
        LOGGER.error(f"aiohttp.ClientError downloading image {url}: {e}", exc_info=True)
        return None
    except Exception as e:
        LOGGER.error(f"Unexpected error downloading image {url}: {e}", exc_info=True)
        return None


# --- Video Processing Functions (to be run in threadpool) ---


def _download_video_sync(video_url: str, temp_dir: str) -> Optional[str]:
    """Downloads video using yt-dlp. Runs synchronously."""
    try:
        LOGGER.info(f"Starting video download: {video_url}")
        output_template = os.path.join(temp_dir, "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "--no-check-certificate",
            "-f",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o",
            output_template,
            video_url,
        ]
        process = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, check=False
        )

        if process.returncode != 0:
            LOGGER.error(f"yt-dlp failed for {video_url}. Error: {process.stderr}")
            return None

        downloaded_files = [
            f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))
        ]
        if not downloaded_files:
            LOGGER.error(f"yt-dlp ran but no file found in {temp_dir} for {video_url}")
            return None

        video_path = os.path.join(temp_dir, downloaded_files[0])
        LOGGER.info(f"Video downloaded successfully: {video_url} to {video_path}")
        return video_path
    except subprocess.TimeoutExpired:
        LOGGER.error(f"yt-dlp download timed out for {video_url}", exc_info=True)
        return None
    except Exception as e:
        LOGGER.error(
            f"Error downloading video {video_url} with yt-dlp: {e}", exc_info=True
        )
        return None


def _extract_audio_sync(video_path: str, temp_dir: str) -> Optional[str]:
    """Extracts audio from video using ffmpeg. Runs synchronously."""
    try:
        import ffmpeg

        audio_output_path = os.path.join(temp_dir, "extracted_audio.wav")
        LOGGER.info(f"Starting audio extraction from: {video_path}")
        ffmpeg.input(video_path).output(
            audio_output_path, acodec="pcm_s16le", ar="16000", ac=1
        ).run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        LOGGER.info(f"Audio extracted successfully to: {audio_output_path}")
        return audio_output_path
    except Exception as e:
        LOGGER.error(
            f"Error extracting audio from {video_path} with ffmpeg: {e}", exc_info=True
        )
        return None


def _transcribe_audio_sync(audio_path: str) -> Optional[str]:
    """Transcribes audio using faster-whisper. Runs synchronously."""
    try:
        model = get_whisper_model()
        LOGGER.info(f"Starting audio transcription for: {audio_path}")
        segments, info = model.transcribe(audio_path, beam_size=5)
        transcript = "".join([segment.text for segment in segments])
        LOGGER.info(
            f"Audio transcribed successfully. Language: {info.language}, Duration: {info.duration}s"
        )
        return transcript.strip()
    except Exception as e:
        LOGGER.error(
            f"Error transcribing audio {audio_path} with faster-whisper: {e}",
            exc_info=True,
        )
        return None


def _extract_keyframes_sync(
    video_path: str, temp_dir: str, num_frames: int = 3
) -> List[str]:
    """Extracts keyframes from video using ffmpeg and returns them as base64 strings. Runs synchronously."""
    try:
        import ffmpeg
        from PIL import Image
        import io

        frames_base64 = []
        LOGGER.info(
            f"Starting keyframe extraction from: {video_path} ({num_frames} frames)"
        )

        probe = ffmpeg.probe(video_path)
        duration = float(probe["format"]["duration"])

        if duration < 1:
            intervals = [duration / 2]
        else:
            padding = duration * 0.1
            effective_duration = duration - (2 * padding)
            if effective_duration <= 0:
                effective_duration = duration

            if num_frames == 1:
                intervals = [duration / 2]
            else:
                intervals = [
                    padding + (effective_duration / (num_frames - 1)) * i
                    for i in range(num_frames)
                ]
                if not intervals:
                    intervals = [duration / 2]

        for i, time_sec in enumerate(intervals):
            frame_filename = os.path.join(temp_dir, f"keyframe_{i+1}.jpg")
            try:
                (
                    ffmpeg.input(video_path, ss=time_sec)
                    .output(
                        frame_filename,
                        vframes=1,
                        format="image2",
                        vcodec="mjpeg",
                        **{"qscale:v": 2},
                    )
                    .run(
                        capture_stdout=True,
                        capture_stderr=True,
                        overwrite_output=True,  # Ensure capture_stderr is True
                    )
                )
                if os.path.exists(frame_filename):
                    with open(frame_filename, "rb") as image_file:
                        img_bytes = image_file.read()
                    base64_encoded_data = base64.b64encode(img_bytes).decode("utf-8")
                    frames_base64.append(
                        f"data:image/jpeg;base64,{base64_encoded_data}"
                    )
                    LOGGER.info(f"Extracted keyframe {i+1} at {time_sec:.2f}s")
                else:
                    LOGGER.warning(f"Failed to extract keyframe {i+1} for {video_path}")
            except ffmpeg.Error as ffmpeg_frame_exc:  # Catch specific ffmpeg.Error
                stderr_output = (
                    ffmpeg_frame_exc.stderr.decode("utf-8")
                    if ffmpeg_frame_exc.stderr
                    else "No stderr."
                )
                LOGGER.error(
                    f"ffmpeg.Error extracting keyframe {i+1} for {video_path}: {ffmpeg_frame_exc}. FFmpeg stderr: {stderr_output}",
                    exc_info=True,  # Keep exc_info for full traceback
                )
            except Exception as frame_exc:  # General fallback for other errors
                LOGGER.error(
                    f"Non-ffmpeg error extracting keyframe {i+1} for {video_path}: {frame_exc}",
                    exc_info=True,
                )

        LOGGER.info(f"Extracted {len(frames_base64)} keyframes successfully.")
        return frames_base64
    except ffmpeg.Error as ffmpeg_e:  # Catch specific ffmpeg.Error first
        stderr_output = (
            ffmpeg_e.stderr.decode("utf-8") if ffmpeg_e.stderr else "No stderr."
        )
        LOGGER.error(
            f"ffmpeg.Error during keyframe extraction process for {video_path}: {ffmpeg_e}. FFmpeg stderr: {stderr_output}",
            exc_info=True,
        )
        return []  # Return empty list on failure
    except Exception as e:
        LOGGER.error(
            f"General error extracting keyframes from {video_path} with ffmpeg: {e}",
            exc_info=True,
        )
        return []


async def process_video_content(video_url: str) -> Optional[Dict[str, Any]]:
    """Orchestrates video processing steps asynchronously."""
    with tempfile.TemporaryDirectory(prefix="video_proc_") as temp_dir_path:
        try:
            video_path = await run_in_threadpool(
                _download_video_sync, video_url, temp_dir_path
            )
            if not video_path:
                return None

            audio_path = await run_in_threadpool(
                _extract_audio_sync, video_path, temp_dir_path
            )
            if not audio_path:
                return None

            transcript = await run_in_threadpool(_transcribe_audio_sync, audio_path)
            if transcript is None:
                transcript = "[Audio transcription failed]"

            keyframes_base64 = await run_in_threadpool(
                _extract_keyframes_sync, video_path, temp_dir_path, num_frames=3
            )
            if not keyframes_base64:
                LOGGER.warning(
                    f"No keyframes extracted for {video_url}. Proceeding without them."
                )

            return {"transcript": transcript, "keyframes": keyframes_base64}
        except Exception as e:
            LOGGER.error(
                f"Error in video processing pipeline for {video_url}: {e}",
                exc_info=True,
            )
            return None


async def process_messages_for_media(
    messages: List[ChatMessage], http_session: aiohttp.ClientSession
) -> Tuple[List[Dict[str, Any]], bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Processes messages to detect and handle images or videos.
    Prioritizes video if a video URL is found in a message.
    Returns:
        - final_payload_messages: List of message dicts for OpenRouter.
        - media_detected_flag: Boolean.
        - media_type_processed: 'image_url', 'base64_image_data', 'video_url', etc.
        - video_data: Dict with transcript/keyframes if video was processed.
    """
    final_payload_messages = []
    media_detected_flag = False
    media_type_processed = None
    video_data_result = None  # To store transcript and keyframes from video processing

    for msg_idx, message in enumerate(messages):
        current_message_content_parts = []
        message_had_media = False  # Flag for current message

        if message.role == "user":
            if isinstance(message.content, str):
                text_content = message.content

                # 1. Check for Video URLs first
                video_urls_found = VIDEO_URL_PATTERN.findall(text_content)
                if (
                    video_urls_found and not media_detected_flag
                ):  # Process only first video in request for now
                    video_url = video_urls_found[0]
                    LOGGER.info(f"Video URL detected in message {msg_idx}: {video_url}")

                    processed_video_data = await process_video_content(video_url)
                    if processed_video_data:
                        media_detected_flag = True
                        message_had_media = True
                        media_type_processed = "video_url"
                        video_data_result = processed_video_data

                        user_content_parts = [
                            {
                                "type": "text",
                                "text": f"Original prompt: {text_content}\n\nVideo Transcript:\n{video_data_result['transcript']}",
                            }
                        ]
                        for frame_b64 in video_data_result["keyframes"]:
                            user_content_parts.append(
                                {"type": "image_url", "image_url": {"url": frame_b64}}
                            )
                        final_payload_messages.append(
                            {"role": "user", "content": user_content_parts}
                        )
                    else:  # Video processing failed, treat as text
                        LOGGER.warning(
                            f"Video processing failed for {video_url}. Treating message as text."
                        )
                        final_payload_messages.append(
                            {"role": "user", "content": text_content}
                        )
                    continue  # Move to next message as video takes precedence

                # 2. If no video processed for this message, check for images (base64 or URL)
                if (
                    not message_had_media
                ):  # Check if media wasn't already handled (e.g. by video)
                    # Check for base64 images
                    base64_match = BASE64_IMAGE_PATTERN.search(text_content)
                    if base64_match:
                        LOGGER.info(f"Base64 image detected in message {msg_idx}.")
                        message_had_media = True
                        if (
                            not media_detected_flag
                        ):  # Only set global flag if not already set by video
                            media_detected_flag = True
                            media_type_processed = "base64_image_data"

                        current_message_content_parts.append(
                            {"type": "text", "text": text_content}
                        )  # Assuming text is caption
                        for match in BASE64_IMAGE_PATTERN.finditer(text_content):
                            data_url = match.group(0)
                            current_message_content_parts.append(
                                {"type": "image_url", "image_url": {"url": data_url}}
                            )

                    # Check for image URLs if no base64 image was found in this string
                    elif IMAGE_URL_PATTERN.search(
                        text_content
                    ):  # Use search to avoid processing if base64 already did
                        image_urls_found = IMAGE_URL_PATTERN.findall(text_content)
                        if image_urls_found:
                            LOGGER.info(
                                f"Image URLs detected in message {msg_idx}: {image_urls_found}"
                            )
                            message_had_media = True
                            if not media_detected_flag:
                                media_detected_flag = True
                                media_type_processed = "image_url"

                            current_message_content_parts.append(
                                {"type": "text", "text": text_content}
                            )
                            for url in image_urls_found:
                                base64_data_url = await download_image_to_base64(
                                    http_session, url
                                )
                                if base64_data_url:
                                    current_message_content_parts.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": base64_data_url},
                                        }
                                    )
                                else:
                                    LOGGER.warning(
                                        f"Failed to process image URL: {url}"
                                    )

                    if message_had_media:
                        final_payload_messages.append(
                            {"role": "user", "content": current_message_content_parts}
                        )
                    else:  # No media in this string, add as plain text
                        final_payload_messages.append(
                            {"role": "user", "content": text_content}
                        )

            elif isinstance(message.content, list):  # Structured content
                # Process structured content, primarily for images as video URLs are assumed in text for now.
                # This part largely follows the previous image processing logic for lists.
                temp_structured_parts = []
                structured_message_had_image = False
                for part_idx, part in enumerate(message.content):
                    if part.get("type") == "image_url":
                        image_url_obj = part.get("image_url", {})
                        url_val = image_url_obj.get("url", "")
                        if url_val.startswith("http"):
                            LOGGER.info(
                                f"HTTP image URL found in structured content part {part_idx} of message {msg_idx}: {url_val}"
                            )
                            base64_data_url = await download_image_to_base64(
                                http_session, url_val
                            )
                            if base64_data_url:
                                temp_structured_parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": base64_data_url},
                                    }
                                )
                                structured_message_had_image = True
                                if (
                                    not media_detected_flag
                                ):  # Set global flags if not already set by video/other images
                                    media_detected_flag = True
                                    media_type_processed = "image_url_structured"
                            else:
                                LOGGER.warning(
                                    f"Failed to process structured image URL: {url_val}"
                                )
                                # Decide: add original part or skip? For now, skip failed image.
                        elif url_val.startswith("data:image"):
                            LOGGER.info(
                                f"Base64 image data found in structured content part {part_idx} of message {msg_idx}."
                            )
                            temp_structured_parts.append(part)  # Keep as is
                            structured_message_had_image = True
                            if not media_detected_flag:
                                media_detected_flag = True
                                media_type_processed = "base64_structured"
                        else:  # Not http, not data:image, keep as is
                            temp_structured_parts.append(part)
                    else:  # Text part or other, keep as is
                        temp_structured_parts.append(part)

                final_payload_messages.append(
                    {
                        "role": message.role,
                        "content": (
                            temp_structured_parts
                            if temp_structured_parts
                            else message.content
                        ),
                    }
                )
                if structured_message_had_image:
                    message_had_media = (
                        True  # Ensure outer loop knows media was handled
                    )

            else:  # User message with unknown content type
                final_payload_messages.append(
                    {"role": message.role, "content": str(message.content)}
                )

        else:  # system, assistant, tool messages
            # Pass through non-user messages as they are (model_dump for safety)
            final_payload_messages.append(message.model_dump(exclude_none=True))

    # Fallback: if no messages were added to final_payload (e.g. empty input or logic error)
    # or if no media was detected at all, ensure original messages are structured correctly.
    if not final_payload_messages:
        LOGGER.warning(
            "final_payload_messages is empty after processing. Returning original messages."
        )
        final_payload_messages = [m.model_dump(exclude_none=True) for m in messages]

    return (
        final_payload_messages,
        media_detected_flag,
        media_type_processed,
        video_data_result,
    )
