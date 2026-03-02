from loguru import logger
import sys
import os
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, f"logs_{TIMESTAMP}.log")

# Remove default logger

# File logger
logger.add(
    LOG_FILE_PATH,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    level="INFO"
)