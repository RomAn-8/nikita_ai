"""Tool registry for managing all available tools."""

from typing import Dict, List
from .base import Tool


class ToolRegistry:
    """Registry for all available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_all(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def list_names(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())


# Global registry instance
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _registry


def register_tool(tool: Tool) -> None:
    """Register a tool in the global registry."""
    _registry.register(tool)
