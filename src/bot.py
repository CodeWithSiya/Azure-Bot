import os
from openai import AzureOpenAI
from dotenv import load_dotenv

"""
Testing Azure OpenAI functionality with a simple Python script.

Author: Siyabonga Madondo
Version: 07/07/2025
"""

def load_credentials() -> dict:
    """Load Azure OpenAI API credentials from the .env file."""
    load_dotenv()
    return {
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_API_BASE": os.getenv("AZURE_OPENAI_API_BASE"),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
        "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    }

def test_azure_openai():
    """Test Azure OpenAI API functionality using the updated SDK."""
    credentials = load_credentials()

    client = AzureOpenAI(
        api_key=credentials["AZURE_OPENAI_API_KEY"],
        api_version=credentials["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=credentials["AZURE_OPENAI_API_BASE"]
    )

    response = client.chat.completions.create(
        model=credentials["AZURE_OPENAI_DEPLOYMENT_NAME"],
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"}
        ],
        temperature=0.0
    )

    print("Response from Azure OpenAI:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    print("Testing Azure OpenAI API...")
    test_azure_openai()