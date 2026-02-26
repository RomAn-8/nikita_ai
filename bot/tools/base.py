"""Base Tool interface for unified tool access."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result of tool execution."""
    success: bool
    data: Any
    error: str | None = None
    
    @classmethod
    def success_result(cls, data: Any) -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, data=data, error=None)
    
    @classmethod
    def error_result(cls, error: str) -> "ToolResult":
        """Create an error result."""
        return cls(success=False, data=None, error=error)


class Tool(ABC):
    """Base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass
    
    @abstractmethod
    async def execute(self, agent_context: Any, **kwargs: Any) -> ToolResult:
        """
        Execute the tool.
        
        Args:
            agent_context: AgentContext object
            **kwargs: Tool-specific arguments
            
        Returns:
            ToolResult with execution result
        """
        pass
