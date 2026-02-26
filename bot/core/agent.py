"""Base Agent class for orchestration."""

from abc import ABC, abstractmethod
from typing import Any

from .context import AgentContext
from ..tools.base import Tool, ToolResult


class Agent(ABC):
    """Base class for agents that orchestrate tools."""
    
    @abstractmethod
    async def process(self, context: AgentContext, input_data: Any) -> Any:
        """
        Process input using available tools.
        
        Args:
            context: AgentContext with current state
            input_data: Input to process
            
        Returns:
            Processed result
        """
        pass
    
    async def execute_tool(self, tool: Tool, context: AgentContext, **kwargs: Any) -> ToolResult:
        """Execute a tool with given context."""
        return await tool.execute(context, **kwargs)
