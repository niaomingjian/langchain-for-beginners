"""
Setup Test - Verify AI Provider Access
"""
import json
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def fetch_github_models(endpoint: str, api_key: str) -> list[str]:
    """Fetch supported GitHub Models from the inference endpoint."""
    url = endpoint.rstrip("/") + "/models"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = response.read().decode("utf-8")
        data = json.loads(payload)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return []

    if isinstance(data, dict) and isinstance(data.get("data"), list):
        items = data["data"]
    elif isinstance(data, list):
        items = data
    else:
        return []

    models = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        model_name = item.get("name") or item.get("display_name") or model_id
        if model_name:
            models.append(model_name)
    return models


def test_setup():
    """Test AI provider connection and configuration."""
    print("🚀 Testing AI provider connection...\n")
    
    # Load environment variables
    load_dotenv()
    
    # Check if required variables are set
    if not os.getenv("AI_API_KEY"):
        print("❌ ERROR: AI_API_KEY not found in .env file")
        sys.exit(1)
    
    if not os.getenv("AI_ENDPOINT"):
        print("❌ ERROR: AI_ENDPOINT not found in .env file")
        sys.exit(1)
    
    try:
        models = fetch_github_models(
            os.getenv("AI_ENDPOINT"),
            os.getenv("AI_API_KEY"),
        )
        if models:
            print("📚 GitHub Models available:")
            for model in models:
                print(f"   - {model}")
            print("")
        else:
            print("⚠️  Could not fetch GitHub Models list (continuing)...\n")

        model = ChatOpenAI(
            model=os.getenv("AI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )
        
        response = model.invoke("Say 'Setup successful!'")
        
        print("✅ SUCCESS! Your AI provider is working!")
        print(f"   Provider: {os.getenv('AI_ENDPOINT')}")
        print(f"   Model: {os.getenv('AI_MODEL', 'gpt-4o-mini')}")
        print(f"\nModel response: {response.content}")
        print("\n🎉 You're ready to start the course!")
    except Exception as error:
        print(f"❌ ERROR: {str(error)}")
        print("\nTroubleshooting:")
        print("1. Check your AI_API_KEY in .env file")
        print("2. Verify the AI_ENDPOINT is correct")
        print("3. Ensure the AI_MODEL is valid for your provider")
        print("4. Verify the token/key has no extra spaces")
        sys.exit(1)


if __name__ == "__main__":
    test_setup()
