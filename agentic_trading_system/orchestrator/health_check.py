import psutil
import time
from logger import logging as logger
import asyncio
class SystemHealth:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.start_time = time.time()

    def check_system_resources(self):
        """Monitor hardware to ensure execution speed isn't compromised"""
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        
        if cpu > 80:
            logger.warning(f"High CPU usage detected: {cpu}% - Risk of execution delay!")
        if ram > 90:
            logger.error(f"Critical RAM usage: {ram}%")
            
        return {"cpu": cpu, "ram": ram}

    async def run_forever(self, interval=60):
        """Background loop to log health status"""
        while self.orchestrator.is_running:
            metrics = self.check_system_resources()
            uptime = time.time() - self.start_time
            
            # Check Agent Health
            for name, agent in self.orchestrator.agents.items():
                health = agent.health_check()
                if health["status"] != "healthy":
                    logger.warning(f"Health Monitor: {name} is {health['status']}!")
            
            logger.info(f"Heartbeat - Uptime: {int(uptime)}s | CPU: {metrics['cpu']}% | RAM: {metrics['ram']}%")
            await asyncio.sleep(interval)