"""Tools the specialist agents can call.

The model only *requests* a tool call (name + arguments). Our code runs it (via ToolNode).
The docstring is effectively a prompt: the model reads it to decide when to use the tool.

SECURITY NOTE (revisit in M9): the arguments come from the model, so they are untrusted.
Real tools must authorize the caller using trusted session data, not model-supplied IDs.

Memory is NOT a tool any more: facts are extracted after each reply by the remember node
(see app/nodes.py), so saving never depends on the model choosing to call a tool.
"""

from langchain_core.tools import tool

# Fixture data: stand-ins for real billing / identity services.
_PAYMENTS = {
    "TXN001": "Payment status: completed. Course activated: yes.",
    "TXN002": "Payment status: completed. Course activated: NO (activation failed).",
    "TXN003": "Payment status: pending. Course activated: no.",
    "TXN004": "Payment status: failed. Course activated: no.",
}

_ACCOUNTS = {
    "ACC001": "Account status: active. Login enabled: yes.",
    "ACC002": "Account status: locked. Login enabled: no.",
    "ACC003": "Account status: suspended. Login enabled: no.",
}


def _normalize(raw_id: str) -> str:
    return raw_id.strip().upper()


@tool
def check_payment_status(transaction_id: str) -> str:
    """Look up a payment by transaction ID (format TXN followed by digits, e.g. TXN002).

    Use when the customer reports a payment problem and has given a transaction ID.
    Returns the payment status and whether the course was activated.
    """
    key = _normalize(transaction_id)
    return _PAYMENTS.get(key, f"No record found for transaction ID '{transaction_id}'.")


@tool
def check_account_status(account_id: str) -> str:
    """Look up an account by account ID (format ACC followed by digits, e.g. ACC002).

    Use when the customer has a login or access problem and has given an account ID.
    Returns the account status and whether login is enabled.
    """
    key = _normalize(account_id)
    return _ACCOUNTS.get(key, f"No record found for account ID '{account_id}'.")


_PAID_AMOUNTS = {"TXN001": 499.0, "TXN002": 499.0, "TXN003": 499.0, "TXN004": 499.0}
_REFUNDS: set[str] = set()  # transaction IDs already refunded (idempotency guard)


@tool
def issue_refund(transaction_id: str, amount: float | None = None) -> str:
    """Queue a refund for a completed payment. This moves money, so a human reviews it first.

    Use ONLY when the customer asks for a refund AND has given a transaction ID
    (format TXN followed by digits, e.g. TXN002). Leave amount empty for a full refund;
    set it only if the customer states a specific amount. Do not use it to answer
    questions about refund policy.
    """
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


# Tools that cause real side effects: these need human approval before they run (M11).
SENSITIVE_TOOLS = {"issue_refund"}

# Single source of truth: the real tools each specialist is bound to AND its ToolNode can run.
# General and technical have no tools, so they never see tool instructions (avoids tool_use_failed).
SPECIALIST_TOOLS = {
    "general": [],
    "technical": [],
    "billing": [check_payment_status, issue_refund],
    "account": [check_account_status],
}
