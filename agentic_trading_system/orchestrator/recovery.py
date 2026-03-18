"""
Recovery Manager - Handles system recovery and agent restart
"""
import asyncio
from logger import logging as logger

class RecoveryManager:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self._recovery_attempts = {}
        self._monitoring = True

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
            
            # Placeholder reconciliation logic
            for trade in last_state:
                logger.info(f"  → Checking trade: {trade.get('id', 'unknown')}")

    async def monitor_and_recover_agents(self):
        """
        Background task to monitor agent health and restart failed ones
        This is called from main.py
        """
        logger.info("Starting agent recovery monitor...")
        
        while self.orchestrator.is_running:
            try:
                for agent_name, agent in self.orchestrator.agents.items():
                    # Check if agent is alive (you'd need to implement a health check)
                    # This is a simplified version
                    if hasattr(agent, 'is_running') and not agent.is_running:
                        await self.restart_agent(agent_name)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recovery monitor: {e}")
                await asyncio.sleep(60)

    async def restart_agent(self, agent_name: str):
        """Logic to reboot a specific failed agent with backoff"""
        attempts = self._recovery_attempts.get(agent_name, 0)
        
        if attempts < 3:
            wait_time = (attempts + 1) * 5
            logger.info(f"Recovering {agent_name} in {wait_time}s... (attempt {attempts + 1}/3)")
            await asyncio.sleep(wait_time)
            
            agent = self.orchestrator.agents.get(agent_name)
            if agent:
                # Reset agent state
                agent.is_running = True
                # Restart the agent's run loop
                asyncio.create_task(agent.run())
                self._recovery_attempts[agent_name] = attempts + 1
                logger.info(f"✅ Agent {agent_name} restarted successfully")
        else:
            logger.critical(f"❌ Agent {agent_name} failed recovery 3 times. Manual intervention required.")