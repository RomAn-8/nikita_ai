"""MCP tools wrapper."""

from typing import Any
from .base import Tool, ToolResult
from ..core.context import AgentContext
from ..mcp_client import (
    get_git_branch,
    get_weather_via_mcp,
    get_news_via_mcp,
)


class GitBranchTool(Tool):
    """Tool for getting git branch."""
    
    @property
    def name(self) -> str:
        return "git_branch"
    
    @property
    def description(self) -> str:
        return "Get current git branch name"
    
    async def execute(self, agent_context: AgentContext, **kwargs: Any) -> ToolResult:
        """Execute git branch tool."""
        try:
            repo_path = kwargs.get("repo_path")
            branch = await get_git_branch(repo_path)
            if branch:
                return ToolResult.success_result(branch)
            return ToolResult.error_result("Failed to get git branch")
        except Exception as e:
            return ToolResult.error_result(f"Error: {str(e)}")


class WeatherTool(Tool):
    """Tool for getting weather."""
    
    @property
    def name(self) -> str:
        return "get_weather"
    
    @property
    def description(self) -> str:
        return "Get weather for a city"
    
    async def execute(self, agent_context: AgentContext, **kwargs: Any) -> ToolResult:
        """Execute weather tool."""
        try:
            city = kwargs.get("city")
            if not city:
                return ToolResult.error_result("City parameter is required")
            
            weather = await get_weather_via_mcp(city)
            if weather:
                return ToolResult.success_result(weather)
            return ToolResult.error_result("Failed to get weather")
        except Exception as e:
            return ToolResult.error_result(f"Error: {str(e)}")


class NewsTool(Tool):
    """Tool for getting news."""
    
    @property
    def name(self) -> str:
        return "get_news"
    
    @property
    def description(self) -> str:
        return "Get news by topic"
    
    async def execute(self, agent_context: AgentContext, **kwargs: Any) -> ToolResult:
        """Execute news tool."""
        try:
            topic = kwargs.get("topic")
            if not topic:
                return ToolResult.error_result("Topic parameter is required")
            
            news = await get_news_via_mcp(topic)
            if news:
                return ToolResult.success_result(news)
            return ToolResult.error_result("Failed to get news")
        except Exception as e:
            return ToolResult.error_result(f"Error: {str(e)}")
