# This script demonstrates how to create a simple MCP server using FastMCP.
# It exposes a single tool that generates a greeting.
# 1. Make sure you have FastMCP installed:
# pip install fastmcp
from fastmcp import FastMCP

# Initialize the FastMCP server.
mcp_server = FastMCP("GreeterServer")


# Define a simple tool function.
# The `@mcp_server.tool` decorator registers this Python function as an MCPtool.
# The docstring becomes the tool's description for the LLM.
@mcp_server.tool
def greet(name: str) -> str:
    """
    Generates a personalized greeting.
    Args:
    11
    name: The name of the person to greet.
    Returns:
    A greeting string.
    """
    return f"Hello, {name}! Nice to meet you."


# Or if you want to run it from the script:
if __name__ == "__main__":
    mcp_server.run(transport="http", host="127.0.0.1", port=8001)
