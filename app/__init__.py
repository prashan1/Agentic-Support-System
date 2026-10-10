"""
app package.

The warning filter below must run before langgraph is imported anywhere.
It silences one known, harmless notice that langgraph itself triggers on import
(a future default change for `allowed_objects` in its checkpoint serializer).

We import LangChain's warning class first because langchain_core registers its own
filter for it on import; ours must be added afterwards so it takes precedence.
"""

import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
    message=r".*allowed_objects.*",
)
