from typing import Any, List, Optional
import anthropic
from anthropic.types import Message, MessageParam, Tool, ToolCall

from .base import BaseLLMClient

class AnthropicClient(BaseLLMClient):
    """Anthropic LLM client implementation."""
    
    def __init__(self):
        """Initialize the Anthropic client."""
        super().__init__()
        self.client = None
        self.model_name = 'claude-3-opus-20240229'

    async def initialize(self, api_key: str) -> None:
        """Initialize the Anthropic client with the given API key."""
        self.client = anthropic.Anthropic(api_key=api_key)

    def convert_tools_to_llm_format(self, mcp_tools: List[Any]) -> List[Tool]:
        """Convert MCP tools to Anthropic-compatible format."""
        anthropic_tools = []

        for tool in mcp_tools:
            # Create tool definition
            anthropic_tool = Tool(
                name=tool.name,
                description=tool.description,
                input_schema=tool.inputSchema
            )
            anthropic_tools.append(anthropic_tool)

        return anthropic_tools

    async def process_query(self, query: str, system_prompt: Optional[str] = None) -> str:
        """Process a user query using the Anthropic model."""
        if not self.client:
            raise ValueError("Anthropic client not initialized. Call initialize() first.")

        # Create message
        message = MessageParam(
            role="user",
            content=query
        )

        # Create messages list
        messages = [message]

        # Send query to Anthropic
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            system=system_prompt,
            tools=self.function_declarations
        )

        final_text = []

        # Process response
        for content in response.content:
            if content.type == "text":
                final_text.append(content.text)
            elif content.type == "tool_use":
                # Handle tool call
                tool_call: ToolCall = content.tool_use
                tool_name = tool_call.name
                tool_args = tool_call.input

                # Call the tool
                function_response = await self.call_tool(tool_name, tool_args)

                # Create tool response message
                tool_response = MessageParam(
                    role="assistant",
                    content=f"Tool call result: {function_response}"
                )
                messages.append(tool_response)

                # Get final response from Anthropic
                final_response = self.client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    system=system_prompt,
                    tools=self.function_declarations
                )

                if final_response.content:
                    final_text.append(final_response.content[0].text)

        return "\n".join(final_text) 