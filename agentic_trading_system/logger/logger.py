# agentic_trading_system/logger/logger.py
import logging
import os
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, f"logs_{TIMESTAMP}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] \t%(levelname)s \t%(lineno)d \t%(filename)s \t%(funcName)s() \t%(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

# Create the logger instance
logger = logging.getLogger("agentic_trading_system")

# Don't add more handlers if basicConfig already added them
# But make sure the logger is configured
logger.setLevel(logging.INFO)

print(f"Logger created: {logger}") 
print(f"Logger handlers: {logger.handlers}")  