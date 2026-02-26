"""RAG tool wrapper."""

from typing import Any
from .base import Tool, ToolResult
from ..core.context import AgentContext
from ..embeddings import search_relevant_chunks


class RAGTool(Tool):
    """Tool for RAG search."""
    
    @property
    def name(self) -> str:
        return "rag_search"
    
    @property
    def description(self) -> str:
        return "Search relevant chunks from indexed documentation using RAG"
    
    async def execute(self, agent_context: AgentContext, **kwargs: Any) -> ToolResult:
        """Execute RAG search."""
        try:
            query = kwargs.get("query")
            if not query:
                return ToolResult.error_result("Query parameter is required")
            
            use_filter = kwargs.get("use_filter", True)
            top_k = kwargs.get("top_k", 3)
            
            chunks = search_relevant_chunks(
                query=query,
                use_filter=use_filter,
                top_k=top_k,
            )
            
            if chunks:
                return ToolResult.success_result(chunks)
            return ToolResult.success_result([])  # Empty result is still success
        except Exception as e:
            return ToolResult.error_result(f"Error: {str(e)}")
