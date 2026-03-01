import asyncio
from logger import logging as logger

class TradingScheduler:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.tasks = []

    async def add_job(self, name: str, interval: int, coro_func, *args):
        """Standard job runner for price polling, sentiment checks, etc."""
        logger.info(f"Scheduling job: {name} every {interval}s")
        while self.orchestrator.is_running:
            try:
                await coro_func(*args)
            except Exception as e:
                logger.error(f"Scheduled task {name} failed: {e}")
            await asyncio.sleep(interval)

    def start_background_task(self, name: str, interval: int, coro_func, *args):
        task = asyncio.create_task(self.add_job(name, interval, coro_func, *args))
        self.tasks.append(task)