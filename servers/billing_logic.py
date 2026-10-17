"""Billing domain logic. This is the billing SERVICE's code, not the agent's.

The MCP server (servers/billing_server.py) exposes these functions as tools. The agent app only
runs them in "local" mode (no BILLING_MCP_URL), which is handy for tests and offline work.
"""

# Fixture data: stand-ins for a real payments database.
_PAYMENTS = {
    "TXN001": "Payment status: completed. Course activated: yes.",
    "TXN002": "Payment status: completed. Course activated: NO (activation failed).",
    "TXN003": "Payment status: pending. Course activated: no.",
    "TXN004": "Payment status: failed. Course activated: no.",
}
_PAID_AMOUNTS = {"TXN001": 499.0, "TXN002": 499.0, "TXN003": 499.0, "TXN004": 499.0}
_REFUNDS: set[str] = set()  # transaction IDs already refunded (idempotency guard)


def _normalize(raw_id: str) -> str:
    return raw_id.strip().upper()


def check_payment_status(transaction_id: str) -> str:
    key = _normalize(transaction_id)
    print(f"Checking payment status for key: {key}")
    return _PAYMENTS.get(key, f"No record found for transaction ID '{transaction_id}'.")


def issue_refund(transaction_id: str, amount: float | None = None) -> str:
    # KNOWN GAP: nothing here checks that a human approved. The approval gate lives in the agent
    # (app/approval.py), so any other caller of this service can refund. Revisit in M14.
    key = _normalize(transaction_id)
    if key not in _PAYMENTS:
        return f"No record found for transaction ID '{transaction_id}'."
    if "Payment status: completed" not in _PAYMENTS[key]:
        return f"Cannot refund {key}: the payment is not completed."
    paid = _PAID_AMOUNTS[key]
    if amount is not None and not 0 < amount <= paid:
        return f"Cannot refund {key}: amount must be between 0 and {paid}."
    if key in _REFUNDS:  # a resumed or retried run must never pay twice
        return f"Refund already issued for {key}."
    _REFUNDS.add(key)
    return f"Refund of {paid if amount is None else amount} queued for {key}."
