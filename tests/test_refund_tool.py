"""Tests for the billing service logic (server side) and the tool policy (agent side).

Pure python: no LLM, no network, no MCP.   python -m pytest tests/test_refund_tool.py -q
"""

import pytest

from app.tools import SENSITIVE_TOOLS, SPECIALIST_TOOLS
from servers import billing_logic


@pytest.fixture(autouse=True)
def clean_refunds():
    billing_logic._REFUNDS.clear()
    yield
    billing_logic._REFUNDS.clear()


def refund(transaction_id, amount=None) -> str:
    return billing_logic.issue_refund(transaction_id, amount)


# ---- refund rules ----
def test_full_refund_for_completed_payment():
    result = refund("TXN002").lower()
    assert "queued" in result and "txn002" in result
    assert "TXN002" in billing_logic._REFUNDS


def test_partial_refund_amount_is_echoed():
    assert "100" in refund("TXN001", 100)


def test_id_is_normalized():
    assert "queued" in refund("  txn001 ").lower()
    assert "TXN001" in billing_logic._REFUNDS


def test_second_refund_is_idempotent():
    refund("TXN002")
    assert "already" in refund("TXN002").lower()


def test_unknown_transaction():
    assert "No record found" in refund("TXN999")
    assert "TXN999" not in billing_logic._REFUNDS


@pytest.mark.parametrize("txn", ["TXN003", "TXN004"])  # pending / failed payments
def test_only_completed_payments_are_refundable(txn):
    assert "queued" not in refund(txn).lower()
    assert txn not in billing_logic._REFUNDS


@pytest.mark.parametrize("amount", [0, -5, 100000])  # zero, negative, more than was paid
def test_invalid_amounts_are_refused(amount):
    assert "queued" not in refund("TXN002", amount).lower()
    assert "TXN002" not in billing_logic._REFUNDS


# ---- agent-side policy ----
def test_sensitive_tools_policy():
    assert SENSITIVE_TOOLS == {"issue_refund"}  # names (strings), read-only tools are NOT gated


def test_billing_is_bound_to_both_tools_but_others_are_not():
    billing_names = {t.name for t in SPECIALIST_TOOLS["billing"]}
    assert billing_names == {"check_payment_status", "issue_refund"}
    for name in ("general", "technical", "account"):
        assert "issue_refund" not in {t.name for t in SPECIALIST_TOOLS[name]}
