"""
Base Agent class for AWM system AI agents.
Provides common functionality for all intelligent agents.
"""

import asyncio
import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from shared.mcp_client.base import MCPClient, MCPMessage, MessageType
from shared.database.connection import db_manager, init_database, close_database

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    IDLE = "IDLE"
    PROCESSING = "PROCESSING"
    ERROR = "ERROR"
    MAINTENANCE = "MAINTENANCE"


@dataclass
class AgentTask:
    id: str
    agent_name: str
    task_type: str
    parameters: Dict[str, Any]
    status: str = "PENDING"
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


@dataclass
class AgentMetrics:
    agent_name: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_processing_time: float = 0.0
    last_activity: Optional[datetime] = None
    status: AgentStatus = AgentStatus.IDLE


class BaseAgent(ABC):
    """Base class for all AWM system agents."""
    
    def __init__(self, agent_name: str, config: Optional[Dict[str, Any]] = None):
        self.agent_name = agent_name
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics(agent_name=agent_name)
        self.mcp_client = MCPClient(agent_name)
        self.task_queue = asyncio.Queue()
        self.running = False
        
        # MCP server URLs from environment
        self.server_urls = {
            "market_data": os.getenv("MARKET_DATA_SERVER_URL", "http://market-data-server:8001"),
            "technical_analysis": os.getenv("TECHNICAL_ANALYSIS_SERVER_URL", "http://technical-analysis-server:8002"),
            "portfolio_management": os.getenv("PORTFOLIO_MANAGEMENT_SERVER_URL", "http://portfolio-management-server:8003"),
            "risk_assessment": os.getenv("RISK_ASSESSMENT_SERVER_URL", "http://risk-assessment-server:8004"),
            "trade_execution": os.getenv("TRADE_EXECUTION_SERVER_URL", "http://trade-execution-server:8005"),
            "news": os.getenv("NEWS_SERVER_URL", "http://news-server:8006")
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"agent.{agent_name}")
    
    async def start(self):
        """Start the agent."""
        self.logger.info(f"Starting {self.agent_name} agent...")
        self.running = True
        
        # Initialize database connection
        await init_database()
        
        # Start task processing loop
        asyncio.create_task(self._process_tasks())
        
        # Perform agent-specific initialization
        await self.initialize()
        
        self.status = AgentStatus.IDLE
        self.logger.info(f"{self.agent_name} agent started successfully")
    
    async def stop(self):
        """Stop the agent."""
        self.logger.info(f"Stopping {self.agent_name} agent...")
        self.running = False
        self.status = AgentStatus.MAINTENANCE
        
        # Perform agent-specific cleanup
        await self.cleanup()
        
        # Close database connection
        await close_database()
        
        self.logger.info(f"{self.agent_name} agent stopped")
    
    async def add_task(self, task_type: str, parameters: Dict[str, Any]) -> str:
        """Add a task to the agent's queue."""
        task = AgentTask(
            id=str(uuid.uuid4()),
            agent_name=self.agent_name,
            task_type=task_type,
            parameters=parameters
        )
        
        await self.task_queue.put(task)
        self.logger.info(f"Added task {task.id} of type {task_type}")
        return task.id
    
    async def _process_tasks(self):
        """Process tasks from the queue."""
        while self.running:
            try:
                # Wait for a task with timeout
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                self.status = AgentStatus.PROCESSING
                task.status = "PROCESSING"
                task.started_at = datetime.now(timezone.utc)
                
                self.logger.info(f"Processing task {task.id} of type {task.task_type}")
                
                try:
                    # Process the task
                    result = await self.process_task(task.task_type, task.parameters)
                    
                    # Update task with result
                    task.status = "COMPLETED"
                    task.completed_at = datetime.now(timezone.utc)
                    task.result = result
                    
                    # Update metrics
                    self.metrics.tasks_completed += 1
                    processing_time = (task.completed_at - task.started_at).total_seconds()
                    self._update_average_processing_time(processing_time)
                    
                    self.logger.info(f"Completed task {task.id} in {processing_time:.2f}s")
                    
                except Exception as e:
                    # Handle task error
                    task.status = "FAILED"
                    task.completed_at = datetime.now(timezone.utc)
                    task.error_message = str(e)
                    
                    self.metrics.tasks_failed += 1
                    self.logger.error(f"Task {task.id} failed: {e}")
                
                finally:
                    # Store task result in database
                    await self._store_task_result(task)
                    
                    # Update metrics
                    self.metrics.last_activity = datetime.now(timezone.utc)
                    self.status = AgentStatus.IDLE
                    
            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except Exception as e:
                self.logger.error(f"Error in task processing loop: {e}")
                self.status = AgentStatus.ERROR
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _store_task_result(self, task: AgentTask):
        """Store task result in database."""
        try:
            query = """
                INSERT INTO agent_tasks 
                (id, agent_name, task_type, parameters, status, created_at, started_at, completed_at, result, error_message)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    result = EXCLUDED.result,
                    error_message = EXCLUDED.error_message
            """
            
            await db_manager.execute_query(
                query,
                task.id,
                task.agent_name,
                task.task_type,
                json.dumps(task.parameters),
                task.status,
                task.created_at,
                task.started_at,
                task.completed_at,
                json.dumps(task.result) if task.result else None,
                task.error_message
            )
        except Exception as e:
            self.logger.error(f"Failed to store task result: {e}")
    
    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time metric."""
        if self.metrics.average_processing_time == 0:
            self.metrics.average_processing_time = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_processing_time = (
                alpha * processing_time + 
                (1 - alpha) * self.metrics.average_processing_time
            )
    
    async def call_mcp_server(
        self,
        server_name: str,
        endpoint: str,
        content: Dict[str, Any],
        retries: int = 3
    ) -> Dict[str, Any]:
        """Call an MCP server and return the response content."""
        if server_name not in self.server_urls:
            raise ValueError(f"Unknown server: {server_name}")
        
        server_url = self.server_urls[server_name]
        
        async with self.mcp_client as client:
            response = await client.send_request(
                server_url, endpoint, content, retries=retries
            )
            
            if response.message_type == MessageType.ERROR:
                error_msg = f"MCP server error: {response.error.error_message}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            return response.content
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "metrics": asdict(self.metrics),
            "queue_size": self.task_queue.qsize(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    async def initialize(self):
        """Initialize the agent (called during startup)."""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup the agent (called during shutdown)."""
        pass
    
    @abstractmethod
    async def process_task(self, task_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process a specific task type."""
        pass
