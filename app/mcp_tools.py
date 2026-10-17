"""Load tools from an MCP server and make them callable from our SYNC graph.

Discovery: the client asks the server "tools/list" and gets name + description + JSON schema.
The adapter turns each into a LangChain tool, but those tools are async-only (coroutine, no
func). Our graph, checkpointer and scripts are sync, so each tool gets a thin sync wrapper that
runs one short-lived event loop per call. The cost is milliseconds, small next to an LLM call.

Limit: asyncio.run() cannot be used while an event loop is already running in the same thread
(for example inside Jupyter). Our CLI, scripts and Streamlit have no running loop.
"""

import asyncio

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient


def _as_text(result) -> str:
    """MCP results can be plain text or a list of content blocks; the model needs one string."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = [b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "") for b in result]
        return "".join(parts)
    return str(result)


def _to_sync(tool: BaseTool) -> StructuredTool:
    def call(**kwargs) -> str:
        return _as_text(asyncio.run(tool.ainvoke(kwargs)))

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,  # the server's JSON schema, passed through unchanged
        func=call,
    )


def load_mcp_tools(url: str, server_name: str = "billing") -> list[BaseTool]:
    """Connect to a streamable-HTTP MCP server and return its tools as sync LangChain tools."""
    client = MultiServerMCPClient({server_name: {"url": url, "transport": "streamable_http"}})
    try:
        tools = asyncio.run(client.get_tools())
    except Exception as exc:  # connection refused, wrong URL, server not started
        raise RuntimeError(
            f"Cannot reach the '{server_name}' MCP server at {url}. "
            "Start it from the project root with:  python -m servers.billing_server"
        ) from exc
    return [_to_sync(tool) for tool in tools]
