from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from mcp import ClientSession

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self):
        """Initialize the base LLM client."""
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None
        self.function_declarations = []

    @abstractmethod
    async def initialize(self, api_key: str) -> None:
        """Initialize the LLM client with the given API key."""
        pass

    @abstractmethod
    async def process_query(self, query: str, system_prompt: Optional[str] = None) -> str:
        """Process a user query using the LLM."""
        pass

    @abstractmethod
    def convert_tools_to_llm_format(self, mcp_tools: List[Any]) -> List[Any]:
        """Convert MCP tools to the format expected by the LLM."""
        pass

    async def connect_to_sse_server(self, server_url: str) -> None:
        """Connect to an MCP server that uses SSE transport."""
        from mcp.client.sse import sse_client
        
        # Open SSE connection
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        # Create MCP session
        self._session_context = ClientSession(*streams)
        self.session = await self._session_context.__aenter__()

        # Initialize session
        await self.session.initialize()

        # Get available tools
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        # Convert tools to LLM format
        self.function_declarations = self.convert_tools_to_llm_format(tools)

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server."""
        try:
            result = await self.session.call_tool(tool_name, tool_args)
            return {"result": result.content}
        except Exception as e:
            return {"error": str(e)} 