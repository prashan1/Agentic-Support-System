"""
Hello LLM checkpoint for M0.

A minimal script to:
1. Load config from .env
2. Create a chat model
3. Send one message
4. Print the reply
5. See a trace in LangSmith
"""

from langchain_core.messages import HumanMessage
from app.llm import get_llm


def main():
    print("=" * 60)
    print("M0 Checkpoint: Hello LLM (with LangSmith tracing)")
    print("=" * 60)

    # Create the LLM
    print("\n1. Creating LLM...")
    llm = get_llm(temperature=0)
    print(f"   ✓ Created {llm.model_name} at {llm.base_url}")

    # Send a simple message
    print("\n2. Sending message to LLM...")
    message = HumanMessage(content="Hello! What is 2 + 2?")
    print(f"   Query: {message.content}")

    # Invoke the model
    print("\n3. Awaiting response...")
    response = llm.invoke([message])
    print(f"\n4. Response:")
    print(f"   {response.content}")

    print("\n" + "=" * 60)
    print("✓ Success!")
    print("=" * 60)
    print("\nNext: Check LangSmith for a trace at:")
    print("https://smith.langchain.com/")
    print("\nLook for the project 'support-agent-lab' and find this run.")
    print("You should see the message sent and the model's reply.")


if __name__ == "__main__":
    main()
