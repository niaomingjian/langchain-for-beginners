"""
Token Usage Tracking Example
Run: python 02-chat-models/code/06_token_tracking.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How can I track token usage across multiple API calls in a conversation?"
- "How would I calculate the cost based on token usage?"
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def track_token_usage():
    model = ChatOpenAI(
            model=os.getenv("AI_MODEL"),
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )

    print("📊 Token Usage Tracking Example\n")

    # Make a request
    response = model.invoke("Explain what Python is in 2 sentences. 使用中文回答。")

    # Extract token usage from metadata (usage_metadata in Python)
    usage = response.usage_metadata

    if usage:
        print("Token Breakdown:")
        print(f"  Prompt tokens:     {usage.get('input_tokens', 'N/A')}")
        print(f"  Completion tokens: {usage.get('output_tokens', 'N/A')}")
        print(f"  Total tokens:      {usage.get('total_tokens', 'N/A')}")
    else:
        print("⚠️  Token usage information not available in response metadata.")

    print("\n📝 Response:")
    print(response.content)


if __name__ == "__main__":
    track_token_usage()
