import logging
import sys
import os
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_NAME = os.path.join(LOG_DIR, f"logs_{TIMESTAMP}.log")

os.makedirs(LOG_DIR,exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE_NAME)

logging.basicConfig(filename=LOG_FILE_PATH,
                    filemode="w",
                    format='[%(asctime)s] \t%(levelname)s \t%(lineno)d \t%(filename)s \t%(funcName)s() \t%(message)s',
                    level=logging.INFO)