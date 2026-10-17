"""Billing MCP server. Run:  python -m servers.billing_server   (listens on http://127.0.0.1:8765/mcp)

Any MCP client can connect, call tools/list, and use these tools. The agent app needs no
billing code of its own: it discovers the tools (name, description, JSON schema) at runtime.

The function signature becomes the JSON schema and the docstring becomes the description the
model reads, the same mechanism as LangChain's @tool.

KNOWN GAP: no authentication and no server-side approval check (revisit in M14).
"""

import logging

from mcp.server.fastmcp import FastMCP

from servers import billing_logic

mcp = FastMCP("billing", host="127.0.0.1", port=8765)
logger = logging.getLogger("billing")  # FastMCP already set up root logging at INFO


@mcp.tool()
def check_payment_status(transaction_id: str) -> str:
    """Look up a payment by transaction ID (format TXN followed by digits, e.g. TXN002).

    Use when the customer reports a payment problem and has given a transaction ID.
    Returns the payment status and whether the course was activated.
    """
    logger.info("TOOL CALL check_payment_status(transaction_id=%r)", transaction_id)
    return billing_logic.check_payment_status(transaction_id)


@mcp.tool()
def issue_refund(transaction_id: str, amount: float | None = None) -> str:
    """Queue a refund for a completed payment. This moves money, so a human reviews it first.

    Use ONLY when the customer asks for a refund AND has given a transaction ID
    (format TXN followed by digits, e.g. TXN002). Leave amount empty for a full refund;
    set it only if the customer states a specific amount. Do not use it to answer
    questions about refund policy.
    """
    logger.info("TOOL CALL issue_refund(transaction_id=%r, amount=%r)", transaction_id, amount)
    result = billing_logic.issue_refund(transaction_id, amount)
    logger.info("TOOL RESULT issue_refund -> %s", result)
    return result


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
