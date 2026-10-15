"""Spec for M11 Step 1: the refund tool (the side-effect action that needs human approval).

Implement in app/tools.py so these pass:   python -m pytest tests/test_refund_tool.py -q
Pure python: no LLM, no network.
"""

import pytest

from app import tools
from app.tools import SENSITIVE_TOOLS, SPECIALIST_TOOLS, issue_refund


@pytest.fixture(autouse=True)
def clean_refunds():
    """Your fixture state must be a module-level set named _REFUNDS (ids already refunded)."""
    tools._REFUNDS.clear()
    yield
    tools._REFUNDS.clear()


def call(**args) -> str:
    return issue_refund.invoke(args)


def test_full_refund_for_completed_payment():
    result = call(transaction_id="TXN002").lower()
    assert "queued" in result and "txn002" in result
    assert "TXN002" in tools._REFUNDS


def test_partial_refund_amount_is_echoed():
    assert "100" in call(transaction_id="TXN001", amount=100)


def test_id_is_normalized():
    assert "queued" in call(transaction_id="  txn001 ").lower()
    assert "TXN001" in tools._REFUNDS


def test_second_refund_is_idempotent():
    call(transaction_id="TXN002")
    assert "already" in call(transaction_id="TXN002").lower()


def test_unknown_transaction():
    assert "No record found" in call(transaction_id="TXN999")
    assert "TXN999" not in tools._REFUNDS


@pytest.mark.parametrize("txn", ["TXN003", "TXN004"])  # pending / failed payments
def test_only_completed_payments_are_refundable(txn):
    result = call(transaction_id=txn).lower()
    assert "queued" not in result
    assert txn not in tools._REFUNDS


@pytest.mark.parametrize("amount", [0, -5, 100000])  # zero, negative, more than was paid
def test_invalid_amounts_are_refused(amount):
    result = call(transaction_id="TXN002", amount=amount).lower()
    assert "queued" not in result
    assert "TXN002" not in tools._REFUNDS


def test_sensitive_tools_policy():
    assert SENSITIVE_TOOLS == {"issue_refund"}  # names (strings), read-only tools are NOT gated


def test_billing_is_bound_to_both_tools_but_others_are_not():
    billing_names = {t.name for t in SPECIALIST_TOOLS["billing"]}
    assert billing_names == {"check_payment_status", "issue_refund"}
    for name in ("general", "technical", "account"):
        assert "issue_refund" not in {t.name for t in SPECIALIST_TOOLS[name]}
