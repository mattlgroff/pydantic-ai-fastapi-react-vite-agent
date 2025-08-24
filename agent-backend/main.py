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
from datetime import datetime
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

class UserContext(BaseModel):
    """
    User context information from authentication.
    
    This is compatible with Vercel AI SDK's user context format.
    In a real app, this would come from your authentication middleware.
    """
    first_name: str
    last_name: str
    email: str
    timezone: Optional[str] = None  # IANA timezone name (e.g., "America/Chicago")


class ChatMessageRequest(BaseModel):
    """
    Request model for agent chat messages.
    
    This matches the Vercel AI SDK message format exactly:
    - messages: Array of conversation messages
    - conversation_id: Unique identifier for the conversation
    - user_context: Optional user information
    
    The 'messages' array contains objects with:
    - id: Unique message ID
    - role: 'user' or 'assistant'  
    - content: Message text (fallback)
    - parts: Array of message parts (v5 format)
      - type: 'text' for text content, 'tool-call' for tool calls
      - text: The actual text content (for text parts)
    """
    messages: List[Dict[str, Any]]
    conversation_id: str
    user_context: Optional[UserContext] = None


# =============================================================================
# 3. VERCEL AI SDK COMPATIBILITY LAYER
# =============================================================================

def convert_vercel_messages_to_pydantic(
    messages: List[Dict[str, Any]], 
    user_context: Optional[UserContext] = None,
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
        user_context: Optional user context to prepend to user prompt
        system_prompt_content: System prompt to include in history
        
    Returns:
        Tuple of (latest_user_prompt, message_history)
    """
    logger.info(f"Converting {len(messages) if messages else 0} messages")
    
    if not messages:
        return "", []
    
    # Extract the latest user message as the prompt
    latest_user_prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            parts = msg.get("parts", [])
            if parts:
                # v5 format: extract text from parts
                text_parts = [
                    p.get("text", "") for p in parts if p.get("type") == "text"
                ]
                latest_user_prompt = " ".join(text_parts)
            elif "content" in msg:
                # Fallback to direct content field
                latest_user_prompt = msg["content"]
            break
    
    # Prepend user context to the latest prompt if available
    if user_context and latest_user_prompt:
        user_name = f"{user_context.first_name} {user_context.last_name}"
        context_prefix = f"""### User Context
Signed-In User: {user_name} - {user_context.email}

---

### User Message

"""
        latest_user_prompt = context_prefix + latest_user_prompt
    
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
    
    return latest_user_prompt, message_history


async def to_data_stream_protocol(agent_stream, run_context):
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
    model = GroqModel("llama-3.1-70b-versatile", api_key=settings.groq_api_key)
    
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
# This enables your frontend (React, Next.js, etc.) to make requests to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Next.js default
        "http://localhost:5173",   # Vite default  
        "http://localhost:5174",   # Vite alternative
        "http://localhost:8000",   # FastAPI default
        "http://localhost:8001",   # FastAPI alternative
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

print("🌐 CORS middleware configured for frontend connections")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    This is useful for:
    - Checking if the server is running
    - Load balancer health checks  
    - Monitoring and uptime checks
    """
    return {
        "status": "healthy",
        "service": "Math Agent",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "capabilities": ["addition", "tool_calling", "streaming"]
    }


# =============================================================================
# 6. MAIN /agent ENDPOINT - This is where the magic happens!
# =============================================================================

@app.post("/agent")
async def chat_with_agent(request: ChatMessageRequest) -> StreamingResponse:
    """
    Main chat endpoint that's compatible with Vercel AI SDK.
    
    This endpoint:
    1. Receives messages in Vercel AI SDK format
    2. Converts them to Pydantic-AI format
    3. Runs the math agent
    4. Streams the response back in AI SDK format
    5. Handles tool calls and streaming text
    
    The Vercel AI SDK will POST to this endpoint with a request like:
    {
        "messages": [
            {
                "role": "user", 
                "content": "What's 15 + 27?",
                "parts": [{"type": "text", "text": "What's 15 + 27?"}]
            }
        ],
        "conversation_id": "conv-123",
        "user_context": {"first_name": "John", "last_name": "Doe", "email": "john@example.com"}
    }
    
    The response will be a Server-Sent Events stream that the AI SDK can consume.
    """
    print(f"🚀 Received chat request with {len(request.messages)} messages")
    
    try:
        # Create the math agent
        agent = create_math_agent()
        
        # Convert Vercel AI SDK messages to Pydantic-AI format
        user_prompt, message_history = convert_vercel_messages_to_pydantic(
            request.messages,
            request.user_context,
            agent.system_prompt  # Pass the agent's system prompt
        )
        
        print(f"📝 User prompt: {user_prompt[:100]}...")
        print(f"📚 Message history: {len(message_history)} previous messages")
        
        # Create the streaming response function
        async def stream_agent_response():
            """
            This async generator function handles the streaming response.
            
            It runs the agent and converts the stream to the format expected
            by the Vercel AI SDK on the frontend.
            """
            try:
                # Run the agent with conversation context
                agent_run = agent.run_stream(user_prompt, message_history=message_history)
                
                # Track the response content for logging
                content_parts = []
                text_id = None
                
                async for event in agent_run:
                    print(f"🔄 Processing event: {type(event).__name__}")
                    
                    # Handle text streaming events
                    if hasattr(event, 'part') and hasattr(event.part, 'part_kind'):
                        if event.part.part_kind == "text":
                            if isinstance(event, PartStartEvent):
                                text_id = f"text-{id(event)}"
                                yield f"data: {json.dumps({'type': 'text-start', 'id': text_id})}\n\n"
                                
                            elif isinstance(event, PartDeltaEvent):
                                if hasattr(event.delta, 'content_delta'):
                                    delta = event.delta.content_delta
                                    content_parts.append(delta)
                                    yield f"data: {json.dumps({'type': 'text-delta', 'id': text_id, 'delta': delta})}\n\n"
                    
                    # Handle tool calls - simplified for this demo
                    # In the full implementation, you'd handle ToolCallStartEvent, etc.
                    
                # End the text stream
                if text_id:
                    yield f"data: {json.dumps({'type': 'text-end', 'id': text_id})}\n\n"
                
                # Log the complete response
                complete_response = ''.join(content_parts)
                print(f"✅ Agent response complete: {complete_response[:200]}...")
                
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
# 7. MAIN ENTRY POINT
# =============================================================================

# No need for explicit uvicorn or __main__ block!
# FastAPI CLI automatically detects the app and runs it with uvicorn.
#
# Run with:
# - fastapi dev main.py    (development with auto-reload)
# - fastapi run main.py    (production)
