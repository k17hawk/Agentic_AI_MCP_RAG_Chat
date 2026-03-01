import asyncio
from logger import logging as logger

class RecoveryManager:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self._recovery_attempts = {}

    async def reconcile_state(self):
        """
        The 'Cold Start' logic: Compare StateManager data 
        with Broker API actual positions.
        """
        logger.info("Starting System Reconciliation...")
        
        # 1. Load last known state
        last_state = self.orchestrator.state_manager.get_state("active_trades", [])
        
        # 2. In a real scenario, you'd call: 
        # actual_positions = await self.orchestrator.broker.get_positions()
        
        # 3. Logic to handle discrepancies
        if not last_state:
            logger.info("No active trades found in state. Clean start.")
        else:
            logger.warning(f"Recovering {len(last_state)} potential orphaned trades.")

    async def restart_agent(self, agent_name: str):
        """Logic to reboot a specific failed agent with backoff"""
        attempts = self._recovery_attempts.get(agent_name, 0)
        if attempts < 3:
            wait_time = (attempts + 1) * 5
            logger.info(f"Recovering {agent_name} in {wait_time}s...")
            await asyncio.sleep(wait_time)
            
            agent = self.orchestrator.agents.get(agent_name)
            if agent:
                asyncio.create_task(agent.run())
                self._recovery_attempts[agent_name] = attempts + 1
        else:
            logger.critical(f"Agent {agent_name} failed recovery 3 times. Manual intervention required.")