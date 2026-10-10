# M0: Project Setup Guide

## Folder Structure

```
support-agent-lab/
├── app/                       # Core agent logic (grows module by module)
│   ├── __init__.py
│   ├── llm.py                # LLM factory (M0) — created
│   ├── state.py              # State schema (M1)
│   ├── prompts.py            # System prompts (M2)
│   ├── nodes.py              # Agent node functions (M2+)
│   ├── tools.py              # @tool functions (M5)
│   └── graph.py              # LangGraph definition (M1+)
│
├── evals/                      # Evaluation harness (M8)
│   ├── __init__.py
│   ├── dataset.py            # Labeled examples
│   ├── evaluators.py         # Metrics
│   ├── run_experiment.py     # Full evaluation
│   ├── baseline_router.py    # Naive router
│   └── compare_routers.py    # vs. Supervisor
│
├── tests/                      # Pytest scripts (M8+)
│   ├── __init__.py
│   ├── test_nodes.py
│   ├── test_graph.py
│   ├── test_memory.py
│   └── ...
│
├── hello_llm.py              # M0 checkpoint script (created)
├── requirements.txt          # Dependencies (created)
├── .env                       # Secrets (you create, git-ignored)
├── .env.example              # Placeholder template (created)
├── .gitignore                # Git ignore rules (created)
└── README.md                 # Project notes
```

## Setup Steps

### 1. Create a `.env` file

Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

Then edit `.env` and fill in your real credentials:
```
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
CHAT_MODEL=llama-3.3-70b-versatile

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_your_actual_key_here
LANGSMITH_PROJECT=support-agent-lab
```

**Important:** `.env` is in `.gitignore`, so it will never be committed. Keep real keys here only.

### 2. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Test the setup

```powershell
python hello_llm.py
```

Expected output:
```
============================================================
M0 Checkpoint: Hello LLM (with LangSmith tracing)
============================================================

1. Creating LLM...
   ✓ Created llama-3.3-70b-versatile at https://api.groq.com/openai/v1

2. Sending message to LLM...
   Query: Hello! What is 2 + 2?

3. Awaiting response...

4. Response:
   2 + 2 = 4

============================================================
✓ Success!
============================================================

Next: Check LangSmith for a trace at:
https://smith.langchain.com/

Look for the project 'support-agent-lab' and find this run.
You should see the message sent and the model's reply.
```

## LangSmith Tracing Setup

### What is LangSmith?

LangSmith is a platform that automatically captures:
- **Traces:** A tree of every LLM call, node execution, and tool call.
- **Latency:** How long each part took.
- **Tokens and cost:** How many tokens were used and how much it cost.
- **Inputs and outputs:** The exact messages sent and received.

### How to view traces

1. Go to https://smith.langchain.com/
2. Sign in with your LangSmith account.
3. Navigate to the **Projects** tab.
4. Find your project (e.g., `support-agent-lab`).
5. Click on a run to see the full trace tree.

### Example trace for `hello_llm.py`

When you run `python hello_llm.py`, a single trace appears:
```
support-agent-lab
└── ChatOpenAI (call)
    ├── Input: [HumanMessage(content="Hello! What is 2 + 2?")]
    ├── Output: AIMessage(content="2 + 2 = 4")
    └── Timing: ~0.5s, ~50 tokens, ~$0.0001
```

Later modules will show more complex traces with multiple nodes, tool calls, and branches.

### Tracing is automatic

The environment variables do the work:
- `LANGSMITH_TRACING=true` — enables tracing.
- `LANGSMITH_API_KEY` — authenticates with LangSmith.
- `LANGSMITH_PROJECT` — organizes runs by project.

Every LangChain and LangGraph operation is automatically traced. No code changes needed.

## Why this structure?

- **`app/` grows incrementally:** Each module adds one new file (`state.py`, then `nodes.py`, then `graph.py`, etc.). Old files don't change (except during fixes).
- **`evals/` is separate:** Evaluation logic is isolated and can be run independently.
- **`llm.py` is central:** All model creation goes through one place, so swapping providers is a one-file change.
- **`.env` is hidden:** Secrets never leak into git.
- **`hello_llm.py` tests the foundation:** If this doesn't run, nothing else will.

## Next steps

After you confirm the checkpoint runs, we move to **M1: LangGraph fundamentals** and design the state schema.
