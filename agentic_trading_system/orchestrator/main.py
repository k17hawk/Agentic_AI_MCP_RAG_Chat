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
        
        # Core components
        self.state_manager = StateManager(storage_path="data/state/system_state.json")
        self.health = SystemHealth(self)
        self.recovery = RecoveryManager(self)
        self.shutdown_handler = GracefulShutdown(self)
        self.broker_breaker = CircuitBreaker(name="BrokerAPI")
        
        # Task tracking for cleanup
        self._background_tasks = set()

    def register_agent(self, agent):
        self.agents[agent.name] = agent
        logger.info(f"Agent {agent.name} linked to Orchestrator.")

    async def start(self):
        logger.info("--- Trading System Starting ---")
        self.is_running = True
        
        # 1. Setup Environment
        self.state_manager.load_from_disk()
        self.shutdown_handler.setup()
        
        # 2. Recovery Logic (Vital for Trading)
        await self.recovery.reconcile_state() 
        
        # 3. Background Monitors
        health_task = asyncio.create_task(self.health.run_forever())
        recovery_task = asyncio.create_task(self.recovery.monitor_and_recover_agents())
        self._background_tasks.add(health_task)
        self._background_tasks.add(recovery_task)
        
        # 4. Start all Agents
        agent_tasks = [asyncio.create_task(a.run()) for a in self.agents.values()]
        
        logger.info(f"Orchestrator online. Managing {len(self.agents)} agents.")
        
        # Wait for all agents to finish (or for shutdown to be called)
        await asyncio.gather(*agent_tasks)

    async def shutdown(self):
        """Standardized cleanup for the trading system"""
        logger.warning("Initiating Orchestrator shutdown sequence...")
        self.is_running = False
        
        # Cancel background monitoring tasks
        for task in self._background_tasks:
            task.cancel()
            
        # Stop each agent gracefully
        for agent in self.agents.values():
            await agent.stop()
            
        logger.info("Orchestration layer fully stopped.")

if __name__ == "__main__":
    orch = TradingOrchestrator()
    # (Registration of agents happens here)
    try:
        asyncio.run(orch.start())
    except KeyboardInterrupt:
        pass