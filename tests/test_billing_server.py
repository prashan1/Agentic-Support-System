"""The billing MCP server, tested in-process (no network, no LLM).

list_tools / call_tool on the FastMCP object are the same handlers a remote client reaches
over the wire, so this checks what a client would discover and what comes back.
"""

import asyncio

import pytest

from servers import billing_logic
from servers.billing_server import mcp


@pytest.fixture(autouse=True)
def clean_refunds():
    billing_logic._REFUNDS.clear()
    yield
    billing_logic._REFUNDS.clear()


def tools_by_name() -> dict:
    return {t.name: t for t in asyncio.run(mcp.list_tools())}


def call(name: str, **args) -> str:
    result = asyncio.run(mcp.call_tool(name, args))
    blocks = result[0] if isinstance(result, tuple) else result  # newer SDKs add structured output
    return "".join(b.text for b in blocks if hasattr(b, "text"))


def test_server_advertises_exactly_the_billing_tools():
    assert set(tools_by_name()) == {"check_payment_status", "issue_refund"}


def test_tools_describe_themselves():
    for tool in tools_by_name().values():
        assert tool.description and len(tool.description) > 40  # the model reads this


def test_schemas_come_from_the_function_signatures():
    tools = tools_by_name()
    check_schema = tools["check_payment_status"].inputSchema
    assert check_schema["required"] == ["transaction_id"]
    refund_schema = tools["issue_refund"].inputSchema
    assert "transaction_id" in refund_schema["required"]
    assert "amount" not in refund_schema.get("required", [])  # optional: None means full refund


def test_check_payment_status_over_the_protocol():
    assert "completed" in call("check_payment_status", transaction_id="TXN002")


def test_unknown_transaction():
    assert "No record found" in call("check_payment_status", transaction_id="TXN999")


def test_refund_goes_through_the_service_rules():
    assert "queued" in call("issue_refund", transaction_id="TXN002").lower()
    assert "already" in call("issue_refund", transaction_id="TXN002").lower()  # idempotent
    assert "Cannot refund" in call("issue_refund", transaction_id="TXN001", amount=100000)
