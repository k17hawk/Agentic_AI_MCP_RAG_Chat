import logging
import os
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, f"logs_{TIMESTAMP}.log")

# Create logger
logger = logging.getLogger("project_logger")
logger.setLevel(logging.INFO)

# Prevent duplicate logs
logger.propagate = False

# Create file handler
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    '[%(asctime)s] \t%(levelname)s \t%(lineno)d \t%(filename)s \t%(funcName)s() \t%(message)s'
)

file_handler.setFormatter(formatter)

# Attach handler
logger.addHandler(file_handler)