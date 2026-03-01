from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
from logger import logging 
from pydantic import BaseModel, Field

class AgentMessage(BaseModel):
    """Standard message format for agent communication"""
    sender: str
    receiver: str
    message_type: str  # 'request', 'response', 'broadcast', 'alert'
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: int = 1  # 1-5, 5 highest
    requires_response: bool = False

class BaseAgent(ABC):
    """
    Abstract base class that all agents inherit from
    Provides core functionality: communication, logging, error handling
    """
    
    def __init__(self, name: str, description: str, config: Dict[str, Any]):
        self.name = name
        self.description = description
        self.config = config
        self.message_queue = asyncio.Queue()
        self.is_running = False
        self.health_status = "healthy"
        self.last_heartbeat = datetime.now()
        
        # Setup logging
        logging.add(f"logs/{name}.log", rotation="1 day")
        
    @abstractmethod
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Main processing logic - each agent must implement this
        """
        pass
    
    async def send_message(self, message: AgentMessage):
        """Send message to another agent"""
        # In production, this would go to a message broker
        await self.message_queue.put(message)
        logging.info(f"{self.name} sent message to {message.receiver}")
    
    async def receive_message(self) -> AgentMessage:
        """Receive message from queue"""
        return await self.message_queue.get()
    
    async def run(self):
        """Main agent loop"""
        self.is_running = True
        logging.info(f"Agent {self.name} started")
        
        while self.is_running:
            try:
                message = await self.receive_message()
                response = await self.process(message)
                
                if response:
                    await self.send_message(response)
                    
                self.last_heartbeat = datetime.now()
                
            except Exception as e:
                logging.error(f"Error in {self.name}: {str(e)}")
                self.health_status = "degraded"
                await asyncio.sleep(5)  # Back off on errors
    
    async def stop(self):
        """Graceful shutdown"""
        self.is_running = False
        logging.info(f"Agent {self.name} stopped")
    
    def health_check(self) -> Dict[str, Any]:
        """Return agent health status"""
        return {
            "name": self.name,
            "status": self.health_status,
            "last_heartbeat": self.last_heartbeat,
            "queue_size": self.message_queue.qsize()
        }