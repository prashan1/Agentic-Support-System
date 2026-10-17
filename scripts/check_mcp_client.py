"""Live check: discover the billing server's tools and call them over HTTP.

Terminal 1 (project root):  python -m servers.billing_server
Terminal 2 (project root):  python -m scripts.check_mcp_client
"""

import os
import sys

from app.mcp_tools import load_mcp_tools

sys.stdout.reconfigure(encoding="utf-8")

URL = os.getenv("BILLING_MCP_URL", "http://127.0.0.1:8765/mcp")

tools = load_mcp_tools(URL)
print(f"discovered {len(tools)} tools from {URL}")
for tool in tools:
    print(f"  - {tool.name}: {tool.description.strip().splitlines()[0]}")

by_name = {tool.name: tool for tool in tools}
print("CHECK TXN002 ->", by_name["check_payment_status"].invoke({"transaction_id": "TXN002"}))
print("CHECK TXN999 ->", by_name["check_payment_status"].invoke({"transaction_id": "TXN999"}))
