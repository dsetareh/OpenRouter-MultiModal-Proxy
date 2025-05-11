# app/logging_config.py
import logging
import sys
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from app.config import settings

# Ensure python-json-logger is in requirements.txt
# pip install python-json-logger

LOG_FILE = settings.LOG_FILE_PATH

# Custom JsonFormatter to include defaultasctime and other fields
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            log_record['timestamp'] = record.created  # Unix timestamp
        if not log_record.get('asctime'):
            log_record['asctime'] = self.formatTime(record, self.datefmt) # Human-readable time
        if not log_record.get('levelname'):
            log_record['levelname'] = record.levelname
        if not log_record.get('filename'):
            log_record['filename'] = record.filename
        if not log_record.get('lineno'):
            log_record['lineno'] = record.lineno
        if not log_record.get('module'):
            log_record['module'] = record.module
        if not log_record.get('funcName'):
            log_record['funcName'] = record.funcName


def setup_logging():
    logger = logging.getLogger("OpenRouter-MultiModal-Proxy") # Create a specific logger
    logger.setLevel(logging.INFO) # Set default level
    logger.propagate = False # Prevent root logger from handling messages from this logger

    # JSON File Handler
    formatter = CustomJsonFormatter(
        '%(asctime)s %(levelname)s %(filename)s %(lineno)d %(module)s %(funcName)s %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S%z' # ISO 8601 format
    )
    
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Optional: Console Handler for development (non-JSON for readability or also JSON)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)'
    )
    console_handler.setFormatter(console_formatter) # Or use JsonFormatter for console too
    console_handler.setLevel(logging.DEBUG) # Show debug messages on console
    logger.addHandler(console_handler)
    
    return logger

# Initialize and export the logger instance
LOGGER = setup_logging()