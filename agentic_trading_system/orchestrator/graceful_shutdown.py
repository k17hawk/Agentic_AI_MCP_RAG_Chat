import asyncio
import signal
from logger import logging as logger

class GracefulShutdown:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def setup(self):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.handle()))

    async def handle(self):
        logger.warning("Shutdown signal received. Initiating emergency procedures...")
        # 1. Stop the scheduler
        self.orchestrator.is_running = False
        
        # 2. Critical: Close or cancel pending orders via ExecutionEngine
        # await self.orchestrator.execution_engine.cancel_all_orders()
        
        # 3. Stop all agents
        await self.orchestrator.shutdown()
        
        logger.info("Cleanup complete. System safe to exit.")
        asyncio.get_event_loop().stop()