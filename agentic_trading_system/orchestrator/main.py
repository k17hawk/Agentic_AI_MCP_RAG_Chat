import asyncio
from logger import logging as logger
from orchestrator.state_manager import StateManager
from orchestrator.graceful_shutdown import GracefulShutdown
from orchestrator.circuit_breaker import CircuitBreaker
from orchestrator.health_check import SystemHealth
from orchestrator.recovery import RecoveryManager
class TradingOrchestrator:
    def __init__(self):
        self.agents = {}
        self.is_running = False
        self.health = SystemHealth(self)
        self.recovery  = RecoveryManager(self)
        self.state_manager = StateManager(self)
        self.shutdown_handler = GracefulShutdown(self)
        self.broker_breaker = CircuitBreaker(name="BrokerAPI")

    def register_agent(self, agent):
        self.agents[agent.name] = agent
        logger.info(f"Agent {agent.name} linked to Orchestrator.")

    async def start(self):
        logger.info("--- Trading System Starting ---")
        self.is_running = True
        self.state_manager.load_from_disk()
        self.shutdown_handler.setup()
        await self.recovery.reconcile_state() # First, check what happened while offline
        asyncio.create_task(self.health.run_forever()) # Start the heartbeat
        
        # Start all agents
        agent_tasks = [asyncio.create_task(a.run()) for a in self.agents.values()]
        
        # Keep the main loop alive
        await asyncio.gather(*agent_tasks)

    async def shutdown(self):
        for agent in self.agents.values():
            await agent.stop()
        logger.info("Orchestration layer stopped.")

if __name__ == "__main__":
    orch = TradingOrchestrator()
    # (Registration of agents happens here)
    try:
        asyncio.run(orch.start())
    except KeyboardInterrupt:
        pass