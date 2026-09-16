import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

print("API key loaded:", bool(api_key), flush=True)

client = OpenAI(
    api_key=api_key,
    base_url="https://integrate.api.nvidia.com/v1",
    timeout=30.0
)

print("Sending request...", flush=True)

try:
    response = client.chat.completions.create(
        model="moonshotai/kimi-k3",

        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: NVIDIA WORKS"
            }
        ],

        max_tokens=20,
        stream=False
    )

    print("\nSUCCESS!", flush=True)
    print(response.choices[0].message.content, flush=True)

except Exception as e:
    print("\nFAILED!", flush=True)
    print(type(e).__name__, flush=True)
    print(e, flush=True)