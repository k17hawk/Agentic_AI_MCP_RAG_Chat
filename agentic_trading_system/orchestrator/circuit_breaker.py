import asyncio
import time
from enum import Enum
from logger import logging as logger

class CircuitState(Enum):
    CLOSED = "closed"   # Normal: Traffic flows
    OPEN = "open"       # Error: Traffic blocked
    HALF_OPEN = "half_open" # Testing: Limited traffic

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold=5, recovery_timeout=60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time = 0

    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info(f"[{self.name}] Circuit HALF-OPEN - Testing connectivity...")
            else:
                return None # Block the execution

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise e

    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"[{self.name}] Circuit CLOSED - Service restored.")
        self.failures = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self, e):
        self.failures += 1
        self.last_failure_time = time.time()
        logger.error(f"[{self.name}] Failure {self.failures}/{self.failure_threshold}: {e}")
        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.critical(f"[{self.name}] CIRCUIT OPENED! Blocking service to prevent damage.")