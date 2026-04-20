import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["FEATHERLESS_API_KEY"],
    base_url=os.environ.get("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1"),
)

model = "moonshotai/Kimi-K2.5"

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What's the weather in Toronto? Use the tool."}],
    tools=tools,
    tool_choice="auto",
)

print("=== RAW RESPONSE ===")
print(resp.model_dump_json(indent=2))
print()
msg = resp.choices[0].message
print("=== CONTENT ===", repr(msg.content))
print("=== TOOL_CALLS ===", msg.tool_calls)
