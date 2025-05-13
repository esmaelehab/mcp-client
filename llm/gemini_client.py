from typing import Any, List, Optional
from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration

from .base import BaseLLMClient

def clean_schema(schema: dict) -> dict:
    """Remove 'title' fields from a JSON schema."""
    if isinstance(schema, dict):
        schema.pop("title", None)
        if "properties" in schema and isinstance(schema["properties"], dict):
            for key in schema["properties"]:
                schema["properties"][key] = clean_schema(schema["properties"][key])
    return schema

class GeminiClient(BaseLLMClient):
    """Gemini LLM client implementation."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        super().__init__()
        self.genai_client = None
        self.model_name = 'gemini-2.0-flash-001'

    async def initialize(self, api_key: str) -> None:
        """Initialize the Gemini client with the given API key."""
        self.genai_client = genai.Client(api_key=api_key)

    def convert_tools_to_llm_format(self, mcp_tools: List[Any]) -> List[Tool]:
        """Convert MCP tools to Gemini-compatible function declarations."""
        gemini_tools = []

        for tool in mcp_tools:
            # Clean the input schema
            parameters = clean_schema(tool.inputSchema)

            # Create function declaration
            function_declaration = FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=parameters
            )

            # Wrap in Gemini Tool
            gemini_tool = Tool(function_declarations=[function_declaration])
            gemini_tools.append(gemini_tool)

        return gemini_tools

    async def process_query(self, query: str, system_prompt: Optional[str] = None) -> str:
        """Process a user query using the Gemini model."""
        if not self.genai_client:
            raise ValueError("Gemini client not initialized. Call initialize() first.")

        # Create content object with the query
        user_prompt_content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=query)]
        )

        # Add system prompt if provided
        contents = [user_prompt_content]
        if system_prompt:
            system_content = types.Content(
                role='system',
                parts=[types.Part.from_text(text=system_prompt)]
            )
            contents.insert(0, system_content)

        # Send query to Gemini
        response = self.genai_client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=self.function_declarations,
            ),
        )

        final_text = []

        # Process response
        for candidate in response.candidates:
            if candidate.content.parts:
                for part in candidate.content.parts:
                    if part.function_call:
                        # Handle function call
                        tool_name = part.function_call.name
                        tool_args = part.function_call.args

                        # Call the tool
                        function_response = await self.call_tool(tool_name, tool_args)

                        # Create function response part
                        function_response_part = types.Part.from_function_response(
                            name=tool_name,
                            response=function_response
                        )

                        # Create tool response content
                        function_response_content = types.Content(
                            role='tool',
                            parts=[function_response_part]
                        )

                        # Get final response from Gemini
                        final_response = self.genai_client.models.generate_content(
                            model=self.model_name,
                            contents=[
                                user_prompt_content,
                                part,
                                function_response_content,
                            ],
                            config=types.GenerateContentConfig(
                                tools=self.function_declarations,
                            ),
                        )

                        if final_response.candidates and final_response.candidates[0].content.parts:
                            final_text.append(final_response.candidates[0].content.parts[0].text)
                    else:
                        final_text.append(part.text)

        return "\n".join(final_text) 