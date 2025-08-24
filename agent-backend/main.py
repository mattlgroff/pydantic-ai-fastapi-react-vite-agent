"""
Minimalist FastAPI + Pydantic-AI Math Agent Server
=================================================

This single-file server demonstrates how to build an AI agent that works with the Vercel AI SDK.
It provides a /agent POST endpoint that streams responses using Server-Sent Events (SSE).

Key Concepts Demonstrated:
- FastAPI with async streaming responses
- Pydantic-AI agent with custom tools
- Vercel AI SDK Data Stream Protocol compatibility
- Message format conversion between AI SDK and Pydantic-AI
- Tool calling and streaming results back to frontend

Run with: fastapi run main.py or fastapi dev main.py
"""

# =============================================================================
# 1. IMPORTS & DEPENDENCIES
# =============================================================================

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

# FastAPI for web server and async streaming
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Pydantic for data validation and models
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# Pydantic-AI for the AI agent functionality
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest, 
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

# Pydantic-AI streaming events
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
)

logger = logging.getLogger(__name__)

# =============================================================================
# 2. SETTINGS & ENVIRONMENT VARIABLES
# =============================================================================

class Settings(BaseSettings):
    """
    Application settings that can be configured via environment variables or .env file.
    
    Required environment variables:
    - GROQ_API_KEY: Your Groq API key for the LLM
    
    This will automatically read from:
    1. Environment variables
    2. A .env file if it exists
    
    If GROQ_API_KEY is not found, the app will fail to start with a clear error.
    """
    groq_api_key: str
    
    # Configuration for reading from .env file
    model_config = SettingsConfigDict(env_file=".env")


# Create a global settings instance
# This will read from environment variables and .env file automatically
settings = Settings()

print(f"🔑 GROQ API key configured: {'✓' if settings.groq_api_key else '❌'}")

# =============================================================================
# 3. PYDANTIC MODELS (Request/Response schemas)
# =============================================================================

class ChatMessageRequest(BaseModel):
    """
    Request model for agent chat messages.
    
    This matches the Vercel AI SDK message format:
    - messages: Array of conversation messages
    
    The 'messages' array contains objects with:
    - id: Unique message ID (optional)
    - role: 'user' or 'assistant'  
    - content: Message text (fallback)
    - parts: Array of message parts (v5 format)
      - type: 'text' for text content, 'tool-call' for tool calls
      - text: The actual text content (for text parts)
    """
    messages: List[Dict[str, Any]]


# =============================================================================
# 3. VERCEL AI SDK COMPATIBILITY LAYER
# =============================================================================

def convert_vercel_messages_to_pydantic(
    messages: List[Dict[str, Any]], 
    system_prompt_content: Optional[str] = None
) -> tuple[str, List[ModelMessage]]:
    """
    Convert Vercel AI SDK message format to Pydantic-AI format.
    
    The Vercel AI SDK uses a different message format than Pydantic-AI:
    
    Vercel AI SDK format:
    {
        "role": "user",
        "content": "What's 2 + 2?",
        "parts": [{"type": "text", "text": "What's 2 + 2?"}]
    }
    
    Pydantic-AI format:
    ModelRequest(parts=[UserPromptPart(content="What's 2 + 2?")])
    
    This function bridges that gap and handles:
    - Extracting the latest user message as the prompt
    - Converting previous messages to conversation history
    - Adding system prompt to the history
    - Handling tool calls and responses
    
    Args:
        messages: List of Vercel AI SDK message objects
        system_prompt_content: System prompt to include in history
        
    Returns:
        Tuple of (latest_user_message, message_history)
    """
    logger.info(f"Converting {len(messages) if messages else 0} messages")
    
    if not messages:
        return "", []
    
    # Extract the latest user message as the prompt
    latest_user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            parts = msg.get("parts", [])
            if parts:
                # v5 format: extract text from parts
                text_parts = [
                    p.get("text", "") for p in parts if p.get("type") == "text"
                ]
                latest_user_message = " ".join(text_parts)
            elif "content" in msg:
                # Fallback to direct content field
                latest_user_message = msg["content"]
            break
    
    # Convert all previous messages (excluding the latest) to ModelMessage format
    message_history = []
    
    # Always include system prompt at the beginning if provided
    if system_prompt_content:
        system_message = ModelRequest(
            parts=[SystemPromptPart(content=system_prompt_content)]
        )
        message_history.append(system_message)
    
    # Process all messages except the last one (which becomes the prompt)
    messages_except_last = messages[:-1] if messages else []
    
    for msg in messages_except_last:
        role = msg.get("role")
        
        if role == "user":
            # Convert user message to ModelRequest with UserPromptPart
            parts = msg.get("parts", [])
            text_content = ""
            if parts:
                text_parts = [
                    p.get("text", "") for p in parts if p.get("type") == "text"
                ]
                text_content = " ".join(text_parts)
            elif "content" in msg:
                text_content = msg["content"]
                
            user_message = ModelRequest(parts=[UserPromptPart(content=text_content)])
            message_history.append(user_message)
            
        elif role == "assistant":
            # Convert assistant message to ModelResponse
            parts = []
            msg_parts = msg.get("parts", [])
            
            # Handle text parts
            if msg_parts:
                text_parts = [
                    p.get("text", "") for p in msg_parts if p.get("type") == "text"
                ]
                if text_parts:
                    parts.append(TextPart(content=" ".join(text_parts)))
            elif "content" in msg:
                parts.append(TextPart(content=msg["content"]))
            
            # Handle tool calls (v5 format)
            tool_calls = []
            if msg_parts:
                for part in msg_parts:
                    if part.get("type") == "tool-call":
                        tool_calls.append(
                            ToolCallPart(
                                tool_name=part.get("toolName", ""),
                                args=part.get("input", {}),  # v5 uses 'input'
                                tool_call_id=part.get("toolCallId", ""),
                            )
                        )
            
            parts.extend(tool_calls)
            
            if parts:
                assistant_message = ModelResponse(parts=parts)
                message_history.append(assistant_message)
                
                # Add tool return messages for completed tool calls
                for tool_call in tool_calls:
                    # Look for tool results in the message parts
                    result = None
                    if msg_parts:
                        for part in msg_parts:
                            if (
                                part.get("type") == "tool-result"
                                and part.get("toolCallId") == tool_call.tool_call_id
                            ):
                                result = part.get("output")  # v5 uses 'output'
                                break
                    
                    if result is not None:
                        tool_return = ModelRequest(
                            parts=[
                                ToolReturnPart(
                                    tool_name=tool_call.tool_name,
                                    content=result,
                                    tool_call_id=tool_call.tool_call_id,
                                )
                            ]
                        )
                        message_history.append(tool_return)
    
    return latest_user_message, message_history


async def to_data_stream_protocol(node, run):
    """Convert Pydantic AI agent stream node to Vercel AI SDK Data Stream Protocol.
    
    This implementation handles text streaming and tool calls for the math agent,
    emitting proper tool-input-available and tool-output-available events.
    
    Args:
        node: Agent stream node from agent.iter()
        run: Agent run context

    Yields:
        str: Data stream protocol formatted chunks
    """
    from pydantic_ai import Agent
    from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent
    
    if not hasattr(run, "_tool_calls_pending"):
        run._tool_calls_pending = {}
    if not hasattr(run, "_tool_name_map"):
        run._tool_name_map = {}

    if Agent.is_user_prompt_node(node):
        # User prompts are handled by the frontend, skip
        pass
    elif Agent.is_model_request_node(node):
        async with node.stream(run.ctx) as request_stream:
            async for event in request_stream:
                logger.info(f"📊 Event type: {type(event).__name__}")
                
                if isinstance(event, PartStartEvent):
                    if event.part.part_kind == "text":
                        if not hasattr(run, "_text_id"):
                            run._text_id = "text-" + str(id(event))
                        chunk = "data: {}\n\n".format(
                            json.dumps({"type": "text-start", "id": run._text_id})
                        )
                        yield chunk

                        # Check if PartStartEvent contains initial content
                        if hasattr(event.part, "content") and event.part.content:
                            initial_content = event.part.content
                            initial_chunk = "data: {}\n\n".format(
                                json.dumps({
                                    "type": "text-delta",
                                    "id": run._text_id,
                                    "delta": initial_content,
                                })
                            )
                            yield initial_chunk
                            
                    elif event.part.part_kind == "tool-call":
                        run._tool_calls_pending[event.part.tool_call_id] = {
                            "toolName": event.part.tool_name,
                            "args_parts": [],
                        }
                        
                elif isinstance(event, PartDeltaEvent):
                    if event.delta.part_delta_kind == "text":
                        if not hasattr(run, "_text_id"):
                            run._text_id = "text-main"
                        chunk = "data: {}\n\n".format(
                            json.dumps({
                                "type": "text-delta",
                                "id": run._text_id,
                                "delta": event.delta.content_delta,
                            })
                        )
                        yield chunk
                        
    elif Agent.is_call_tools_node(node):
        # Handle tool calls with proper event emission
        async with node.stream(run.ctx) as tool_stream:
            async for event in tool_stream:
                if isinstance(event, FunctionToolCallEvent):
                    print(f"🔧 TOOL CALL STARTED: {event.part.tool_name}")
                    print(f"🔧 TOOL INPUT: {json.dumps(event.part.args, indent=2)}")
                    
                    # Store tool name mapping
                    run._tool_name_map[event.part.tool_call_id] = event.part.tool_name
                    
                    # Emit tool-input-available event
                    yield "data: {}\n\n".format(
                        json.dumps({
                            "type": "tool-input-available",
                            "toolCallId": event.part.tool_call_id,
                            "toolName": event.part.tool_name,
                            "input": event.part.args,
                        })
                    )
                    
                elif isinstance(event, FunctionToolResultEvent):
                    print(f"🔧 TOOL RESULT RECEIVED for call_id: {event.result.tool_call_id}")
                    print(f"🔧 TOOL OUTPUT: {event.result.content}")
                    
                    # Emit tool-output-available event
                    result_content = (
                        event.result.content.to_dict()
                        if hasattr(event.result.content, "to_dict")
                        else (
                            event.result.content.model_dump()
                            if hasattr(event.result.content, "model_dump")
                            else event.result.content
                        )
                    )
                    
                    yield "data: {}\n\n".format(
                        json.dumps({
                            "type": "tool-output-available", 
                            "toolCallId": event.result.tool_call_id,
                            "output": result_content,
                        })
                    )
                        
    elif Agent.is_end_node(node):
        # Send text-end if we were streaming text
        if hasattr(run, "_text_id"):
            chunk = "data: {}\n\n".format(
                json.dumps({"type": "text-end", "id": run._text_id})
            )
            yield chunk
            delattr(run, "_text_id")


async def old_to_data_stream_protocol(agent_stream, run_context):
    """
    Convert Pydantic-AI agent stream to Vercel AI SDK Data Stream Protocol.
    
    The Vercel AI SDK expects Server-Sent Events (SSE) in a specific format:
    
    For text streaming:
    - data: {"type": "text-start", "id": "text-123"}
    - data: {"type": "text-delta", "id": "text-123", "delta": "Hello"}  
    - data: {"type": "text-end", "id": "text-123"}
    
    For tool calls:
    - data: {"type": "tool-input-available", "toolCallId": "call-123", "toolName": "sum_numbers", "input": {...}}
    - data: {"type": "tool-output-available", "toolCallId": "call-123", "output": 42}
    
    This function converts Pydantic-AI's streaming events to this format.
    
    Args:
        agent_stream: The agent stream iterator from agent.run()
        run_context: Agent run context for tracking state
        
    Yields:
        str: SSE-formatted data chunks
    """
    # Track tool calls and text streaming
    if not hasattr(run_context, "_tool_calls_pending"):
        run_context._tool_calls_pending = {}
    
    text_id = None
    
    try:
        async for event in agent_stream:
            logger.info(f"Processing event: {type(event).__name__}")
            
            if isinstance(event, PartStartEvent):
                if event.part.part_kind == "text":
                    # Start streaming text
                    text_id = f"text-{id(event)}"
                    yield f"data: {json.dumps({'type': 'text-start', 'id': text_id})}\n\n"
                    
                    # If there's initial content, send it
                    if hasattr(event.part, "content") and event.part.content:
                        yield f"data: {json.dumps({'type': 'text-delta', 'id': text_id, 'delta': event.part.content})}\n\n"
                        
                elif event.part.part_kind == "tool-call":
                    # Track tool call start
                    tool_call_id = getattr(event.part, "tool_call_id", f"call-{id(event)}")
                    run_context._tool_calls_pending[tool_call_id] = {
                        "toolName": getattr(event.part, "tool_name", "unknown"),
                        "input_parts": []
                    }
            
            elif isinstance(event, PartDeltaEvent):
                if event.delta.part_delta_kind == "text" and text_id:
                    # Stream text delta
                    delta_content = getattr(event.delta, "content_delta", "")
                    yield f"data: {json.dumps({'type': 'text-delta', 'id': text_id, 'delta': delta_content})}\n\n"
                    
                elif event.delta.part_delta_kind == "tool-call":
                    # Accumulate tool call arguments
                    tool_call_id = getattr(event.part, "tool_call_id", None)
                    if tool_call_id and tool_call_id in run_context._tool_calls_pending:
                        args_delta = getattr(event.delta, "args_delta", "")
                        run_context._tool_calls_pending[tool_call_id]["input_parts"].append(args_delta)
            
            # Handle completed tool calls and text endings
            # This is simplified - in the full version, you'd handle tool completion events
            
    except Exception as e:
        logger.error(f"Stream processing error: {e}")
        error_id = "error-text"
        yield f"data: {json.dumps({'type': 'text-start', 'id': error_id})}\n\n"
        yield f"data: {json.dumps({'type': 'text-delta', 'id': error_id, 'delta': f'Error: {str(e)}'})}\n\n"
        yield f"data: {json.dumps({'type': 'text-end', 'id': error_id})}\n\n"
    
    finally:
        # End text streaming if active
        if text_id:
            yield f"data: {json.dumps({'type': 'text-end', 'id': text_id})}\n\n"


# =============================================================================
# 5. PYDANTIC-AI MATH AGENT SETUP  
# =============================================================================

def create_math_agent() -> Agent:
    """
    Create and configure the math agent.
    
    This function demonstrates how to:
    1. Set up a Pydantic-AI agent with a custom system prompt
    2. Configure the LLM model (using Groq in this case)
    3. Add custom tools that the agent can call
    4. Return a configured agent ready for use
    
    The agent will automatically decide when to use tools based on the
    system prompt instructions and the user's input.
    """
    
    # System prompt that tells the agent how to behave
    system_prompt = """You are a helpful math assistant with access to calculation tools.

IMPORTANT INSTRUCTIONS:
- When given math problems, always use your included tools for accurate calculations
- Use the sum_numbers tool for addition operations
- Show your work and explain the calculation process
- Be friendly and educational in your responses
- If asked to do math that your tools can't handle, explain what you can help with

Available tools:
- sum_numbers: Add multiple numbers together

Examples of when to use sum_numbers:
- "What's 15 + 27 + 8?" -> Use sum_numbers(15, 27, 8)
- "Add 100.5 and 200.25" -> Use sum_numbers(100.5, 200.25)
- "Sum these: 1, 2, 3, 4, 5" -> Use sum_numbers(1, 2, 3, 4, 5)
"""
    
    # Configure the LLM model
    # Using Groq for fast inference with the API key from settings
    # The API key is loaded from environment variables or .env file
    provider = GroqProvider(api_key=settings.groq_api_key)
    model = GroqModel("qwen/qwen3-32b", provider=provider)
    
    # Create the agent with the model and system prompt
    agent = Agent(model, system_prompt=system_prompt)
    
    # Add the sum_numbers tool to the agent
    @agent.tool_plain
    def sum_numbers(*numbers: float) -> float:
        """
        Add multiple numbers together using Python's built-in math.
        
        This tool demonstrates how to create custom functions that your AI agent
        can call. The agent will automatically use this when it detects the user
        wants to add numbers together.
        
        Args:
            *numbers: Variable number of float arguments to sum
            
        Returns:
            The sum of all provided numbers
            
        Examples:
            sum_numbers(1, 2, 3) -> 6.0
            sum_numbers(10.5, 20.25) -> 30.75
        """
        print(f"🔢 TOOL CALLED: sum_numbers({', '.join(map(str, numbers))})")
        result = sum(numbers)
        print(f"🔢 RESULT: {result}")
        return result
    
    return agent


# =============================================================================
# 5. FASTAPI APPLICATION SETUP
# =============================================================================

# Create the FastAPI application
app = FastAPI(
    title="Math Agent - Pydantic-AI Demo",
    description="A minimalist math agent built with FastAPI and Pydantic-AI, compatible with Vercel AI SDK",
    version="1.0.0",
)

print("🚀 FastAPI Math Agent Server Starting!")

# Add CORS middleware to allow frontend connections
# This enables your Vite frontend to make requests to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

print("🌐 CORS middleware configured for frontend connections")


@app.post("/agent")
async def chat_with_agent(request: ChatMessageRequest) -> StreamingResponse:
    """
    The main (and only) endpoint for this minimalist demonstration.
    
    This shows how to host a Pydantic-AI agent in FastAPI for chat with tool calling
    using the Vercel AI SDK on the frontend. This minimalist approach focuses on the 
    core concepts without authentication, conversation persistence, or other features.
    
    In a real application, you could add:
    - Health check endpoint (/health)
    - Conversation history endpoints (/conversations)
    - User authentication and authorization
    - Rate limiting and request validation
    - Logging and monitoring
    - Multiple agent types
    
    This endpoint demonstrates:
    1. Receiving Vercel AI SDK message format
    2. Converting to Pydantic-AI format  
    3. Running the agent with tool calling
    4. Streaming responses back via Server-Sent Events
    5. Handling conversation history
    
    Example request from Vercel AI SDK:
    {
        "messages": [
            {"role": "user", "content": "What's 15 + 27?", "parts": [{"type": "text", "text": "What's 15 + 27?"}]}
        ]
    }
    """
    print(f"🚀 Received chat request with {len(request.messages)} messages")
    
    try:
        # Create the math agent
        agent = create_math_agent()
        
        # Get the system prompt content as string (from the agent creation)
        system_prompt_content = """You are a helpful math assistant with access to calculation tools.

IMPORTANT INSTRUCTIONS:
- When given math problems, always use your included tools for accurate calculations
- Use the sum_numbers tool for addition operations
- Show your work and explain the calculation process
- Be friendly and educational in your responses
- If asked to do math that your tools can't handle, explain what you can help with

Available tools:
- sum_numbers: Add multiple numbers together

Examples of when to use sum_numbers:
- "What's 15 + 27 + 8?" -> Use sum_numbers(15, 27, 8)
- "Add 100.5 and 200.25" -> Use sum_numbers(100.5, 200.25)
- "Sum these: 1, 2, 3, 4, 5" -> Use sum_numbers(1, 2, 3, 4, 5)
"""
        
        # Convert Vercel AI SDK messages to Pydantic-AI format
        user_message, message_history = convert_vercel_messages_to_pydantic(
            request.messages,
            system_prompt_content=system_prompt_content
        )
        
        print(f"📝 User message: {user_message[:100]}...")
        print(f"📚 Message history: {len(message_history)} previous messages")
        
        # Create the streaming response function
        async def stream_agent_response():
            """
            This async generator function handles the streaming response.
            
            It runs the agent and converts the stream to the format expected
            by the Vercel AI SDK on the frontend.
            """
            try:
                # Run the agent with conversation context - matching the working pattern
                async with agent.iter(user_message, message_history=message_history) as agent_run:
                    async for node in agent_run:
                        # Convert to data stream protocol format - use the same pattern as the working version
                        async for chunk in to_data_stream_protocol(node, agent_run):
                            yield chunk
                
            except Exception as e:
                print(f"❌ Error in agent stream: {str(e)}")
                # Send error as a text stream
                error_id = "error-text"
                error_msg = f"I apologize, but I encountered an error: {str(e)}"
                
                yield f"data: {json.dumps({'type': 'text-start', 'id': error_id})}\n\n"
                yield f"data: {json.dumps({'type': 'text-delta', 'id': error_id, 'delta': error_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'text-end', 'id': error_id})}\n\n"
        
        # Return the streaming response with proper headers
        response = StreamingResponse(
            stream_agent_response(),
            media_type="text/event-stream"
        )
        
        # Set headers required for Server-Sent Events
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["Content-Type"] = "text/event-stream"
        
        return response
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 7. ENTRY POINT
# =============================================================================

# No need for explicit uvicorn or __main__ block!
# FastAPI CLI automatically detects the app and runs it with uvicorn.
#
# Run with:
# - fastapi dev main.py    (development with auto-reload)
# - fastapi run main.py    (production)
