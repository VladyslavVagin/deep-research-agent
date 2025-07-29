"""
Custom agents framework for the deep research agent.
This provides the basic functionality needed for the research agents.
"""

import asyncio
import uuid
import json
from typing import Any, Callable, Dict, List, Optional, Type, Union
from contextlib import contextmanager
from functools import wraps
import os
import requests
from pydantic import BaseModel
import openai


class ModelSettings(BaseModel):
    """Settings for the model configuration"""
    tool_choice: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class WebSearchTool:
    """Tool for performing web searches"""
    
    def __init__(self, search_context_size: str = "low"):
        self.search_context_size = search_context_size
    
    def __call__(self, query: str) -> str:
        """Perform a web search and return results"""
        # This is a simplified implementation
        # In a real implementation, you'd use a proper search API
        return f"Search results for: {query}\n\nThis is a placeholder for web search results."


def function_tool(func: Callable) -> Callable:
    """Decorator to mark a function as a tool"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper._is_tool = True
    return wrapper


class Agent:
    """Base agent class"""
    
    def __init__(
        self,
        name: str,
        instructions: str,
        tools: Optional[List] = None,
        model: str = "gpt-4o-mini",
        output_type: Optional[Type[BaseModel]] = None,
        model_settings: Optional[ModelSettings] = None
    ):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.model = model
        self.output_type = output_type
        self.model_settings = model_settings or ModelSettings()
    
    async def run(self, input_text: str) -> Any:
        """Run the agent with the given input"""
        # This is a simplified implementation
        # In a real implementation, you'd use OpenAI's API
        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Enhance instructions for structured output if output_type is specified
        enhanced_instructions = self.instructions
        if self.output_type:
            enhanced_instructions += f"\n\nPlease provide your response in JSON format that matches this structure: {self.output_type.model_json_schema()}"
        
        messages = [
            {"role": "system", "content": enhanced_instructions},
            {"role": "user", "content": input_text}
        ]
        
        # Add tool definitions if tools are provided
        tools = []
        if self.tools:
            for tool in self.tools:
                if hasattr(tool, '_is_tool'):
                    # Handle function tools
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.__name__,
                            "description": tool.__doc__ or "",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                    })
                elif isinstance(tool, WebSearchTool):
                    # Handle web search tool
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "description": "Search the web for information",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query"
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    })
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            temperature=self.model_settings.temperature,
            max_tokens=self.model_settings.max_tokens
        )
        
        return AgentResult(response.choices[0].message.content, self.output_type)


class AgentResult:
    """Result from running an agent"""
    
    def __init__(self, content: str, output_type: Optional[Type[BaseModel]] = None):
        self.content = content
        self.output_type = output_type
        self.final_output = content
    
    def final_output_as(self, output_type: Type[BaseModel]) -> BaseModel:
        """Convert the final output to the specified type"""
        try:
            # Try to parse as JSON first
            if isinstance(self.content, str):
                # Look for JSON in the content
                content = self.content.strip()
                if content.startswith('{') and content.endswith('}'):
                    data = json.loads(content)
                elif '```json' in content:
                    # Extract JSON from markdown code block
                    start = content.find('```json') + 7
                    end = content.find('```', start)
                    json_str = content[start:end].strip()
                    data = json.loads(json_str)
                else:
                    # Try to find JSON-like structure
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                    else:
                        # Fallback: create a mock object
                        return self._create_mock_object(output_type)
            else:
                data = self.content
            
            return output_type(**data)
        except (json.JSONDecodeError, ValueError, TypeError):
            # If parsing fails, create a mock object
            return self._create_mock_object(output_type)
    
    def _create_mock_object(self, output_type: Type[BaseModel]) -> BaseModel:
        """Create a mock object of the specified type"""
        # Create default values for the model
        mock_data = {}
        for field_name, field_info in output_type.model_fields.items():
            if hasattr(field_info.annotation, '__origin__') and field_info.annotation.__origin__ is list:
                mock_data[field_name] = []
            elif field_info.annotation == str:
                mock_data[field_name] = f"Mock {field_name}"
            elif field_info.annotation == int:
                mock_data[field_name] = 0
            elif field_info.annotation == bool:
                mock_data[field_name] = False
            else:
                mock_data[field_name] = None
        
        return output_type(**mock_data)


class Runner:
    """Runner for executing agents"""
    
    @staticmethod
    async def run(agent: Agent, input_text: str) -> AgentResult:
        """Run an agent with the given input"""
        return await agent.run(input_text)


def gen_trace_id() -> str:
    """Generate a unique trace ID"""
    return str(uuid.uuid4())


@contextmanager
def trace(name: str, trace_id: str):
    """Context manager for tracing"""
    print(f"Starting trace: {name} (ID: {trace_id})")
    try:
        yield
    finally:
        print(f"Ending trace: {name} (ID: {trace_id})") 