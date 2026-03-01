
from typing import Dict, List, Type
import asyncio
from logger import logging  as  logger
from agents.base_agent import BaseAgent, AgentMessage

class AgentOrchestrator:
    """
    Central coordinator for the multi-agent system
    - Starts/stops all agents
    - Routes messages between agents
    - Monitors agent health
    - Handles system-wide events
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_bus = asyncio.Queue()
        self.is_running = False
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    async def start_all(self):
        """Start all registered agents"""
        self.is_running = True
        tasks = []
        
        for agent in self.agents.values():
            task = asyncio.create_task(agent.run())
            tasks.append(task)
            
        # Start message router
        router_task = asyncio.create_task(self.route_messages())
        tasks.append(router_task)
        
        # Start health monitor
        monitor_task = asyncio.create_task(self.health_monitor())
        tasks.append(monitor_task)
        
        logger.info(f"Started {len(self.agents)} agents")
        await asyncio.gather(*tasks)
    
    async def route_messages(self):
        """Route messages between agents"""
        while self.is_running:
            try:
                message = await self.message_bus.get()
                
                if message.receiver == "broadcast":
                    # Send to all agents
                    for agent in self.agents.values():
                        await agent.message_queue.put(message)
                else:
                    # Send to specific agent
                    if message.receiver in self.agents:
                        await self.agents[message.receiver].message_queue.put(message)
                    else:
                        logger.warning(f"Agent {message.receiver} not found")
                        
            except Exception as e:
                logger.error(f"Message routing error: {e}")
    
    async def health_monitor(self):
        """Monitor agent health and restart if needed"""
        while self.is_running:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            for name, agent in self.agents.items():
                health = agent.health_check()
                
                if health["status"] == "degraded":
                    logger.warning(f"Agent {name} degraded - restarting")
                    await agent.stop()
                    await asyncio.sleep(5)
                    asyncio.create_task(agent.run())
                
                elif health["queue_size"] > 100:
                    logger.warning(f"Agent {name} queue backlog: {health['queue_size']}")
    
    async def shutdown(self):
        """Graceful shutdown of all agents"""
        logger.info("Shutting down all agents...")
        self.is_running = False
        
        for agent in self.agents.values():
            await agent.stop()
            
        logger.info("All agents stopped")