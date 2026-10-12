---
category: technical
source: Coding sandbox and API console
---

## KB-SBX-01: The sandbox console does not run my Python code
If pressing Run does nothing in the coding sandbox or API console, first check the status indicator in the top right: it must say Connected. If it says Disconnected, reload the page to start a new session. Make sure you are using a supported browser with extensions and ad blockers turned off for the platform, because they can block the sandbox connection. Idle sessions are stopped after 20 minutes, so reload if you were away.

## KB-SBX-02: Supported Python versions and packages in the sandbox
The sandbox runs Python 3.11 with common data science libraries preinstalled (numpy, pandas, requests). You cannot install extra packages with pip inside the sandbox, and there is no access to the internet except the course API endpoints. If your code needs another library, check the course notes for the list of allowed packages.

## KB-SBX-03: Sandbox says execution timed out
Each run is limited to 30 seconds and 512 MB of memory. Infinite loops, very large loops, or heavy computations hit this limit. Add a stopping condition, test with a smaller input first, and print progress to see where time is spent. A timeout does not count against any course attempt.
