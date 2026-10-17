"""Tools the specialist agents can call.

The model only *requests* a tool call (name + arguments). Our code runs it (via ToolNode).
The docstring is effectively a prompt: the model reads it to decide when to use the tool.

SECURITY NOTE (revisit in M14): the arguments come from the model, so they are untrusted.
Real tools must authorize the caller using trusted session data, not model-supplied IDs.

Billing tools are NOT defined in this codebase. They live on the billing MCP server
(servers/billing_server.py) and are discovered at startup from BILLING_MCP_URL.
If the server is unreachable, billing starts with no tools (degraded mode, logged as a warning).

Memory is NOT a tool: facts are extracted after each reply by the remember node.
"""

import logging
import os
from dotenv import load_dotenv
from langchain_core.tools import tool

logger = logging.getLogger(__name__)
load_dotenv()

# Fixture data: stand-in for a real identity service (accounts stay local for now).
_ACCOUNTS = {
    "ACC001": "Account status: active. Login enabled: yes.",
    "ACC002": "Account status: locked. Login enabled: no.",
    "ACC003": "Account status: suspended. Login enabled: no.",
}


def _normalize(raw_id: str) -> str:
    return raw_id.strip().upper()


@tool
def check_account_status(account_id: str) -> str:
    """Look up an account by account ID (format ACC followed by digits, e.g. ACC002).

    Use when the customer has a login or access problem and has given an account ID.
    Returns the account status and whether login is enabled.
    """
    key = _normalize(account_id)
    return _ACCOUNTS.get(key, f"No record found for account ID '{account_id}'.")


# Tools that cause real side effects: these need human approval before they run (M11).
# Names only: the tools themselves come from the MCP server, but the policy stays on our side.
SENSITIVE_TOOLS = {"issue_refund"}


def _billing_tools() -> list:
    """Discover billing tools from the MCP server. Returns [] (degraded mode) if unavailable."""
    url = os.getenv("BILLING_MCP_URL")
    if not url:
        logger.warning("BILLING_MCP_URL is not set: billing agent starts WITHOUT tools (degraded).")
        return []
    from app.mcp_tools import load_mcp_tools

    try:
        return load_mcp_tools(url, "billing")
    except RuntimeError as exc:
        logger.warning("%s Billing agent starts WITHOUT tools (degraded).", exc)
        return []


# Single source of truth: the real tools each specialist is bound to AND its ToolNode can run.
# General and technical have no tools, so they never see tool instructions (avoids tool_use_failed).
SPECIALIST_TOOLS = {
    "general": [],
    "technical": [],
    "billing": _billing_tools(),
    "account": [check_account_status],
}
