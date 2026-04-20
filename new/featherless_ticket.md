# Tool calls not parsed for `Qwen/Qwen3.5-397B-A17B`

## Summary
When calling `Qwen/Qwen3.5-397B-A17B` via the OpenAI-compatible Chat Completions endpoint with `tools`, the model's tool invocation is **not parsed** into the `tool_calls` field. Instead, the raw Hermes/XML-style tool syntax is returned verbatim in `message.content`, and `finish_reason` is `"stop"` instead of `"tool_calls"`.

This breaks any OpenAI-SDK-based agent framework, since `tool_calls` is `None`.

For comparison, `Qwen/Qwen3-8B` on the same endpoint *does* parse tool calls into `tool_calls` (though with a separate minor bug — the `arguments` string is `'{}{"city":"Toronto"}'`, two concatenated JSON objects).

## Environment
- Endpoint: `https://api.featherless.ai/v1`
- SDK: `openai` Python SDK
- Broken model: `Qwen/Qwen3.5-397B-A17B` — `system_fingerprint: fp1-nst-nes`
- Working-ish model: `Qwen/Qwen3-8B` — `system_fingerprint: fp1-rss-g3-herm`

## Reproduction

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["FEATHERLESS_API_KEY"],
    base_url="https://api.featherless.ai/v1",
)

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
    model="Qwen/Qwen3.5-397B-A17B",
    messages=[{"role": "user", "content": "What's the weather in Toronto? Use the tool."}],
    tools=tools,
    tool_choice="auto",
)
print(resp.model_dump_json(indent=2))
```

## Actual response (abridged)

```json
{
  "choices": [{
    "finish_reason": "stop",
    "message": {
      "role": "assistant",
      "content": "<tool_call>\n<function=get_weather>\n<parameter=city>\nToronto\n</parameter>\n</function>\n</tool_call>",
      "tool_calls": null
    }
  }],
  "model": "Qwen/Qwen3.5-397B-A17B",
  "system_fingerprint": "fp1-nst-nes"
}
```

## Expected
`finish_reason: "tool_calls"` and a populated `tool_calls` array:

```json
"tool_calls": [{
  "id": "...",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"city\":\"Toronto\"}"
  }
}]
```

## Likely cause
The `fp1-nst-nes` serving profile for `Qwen3.5-397B-A17B` appears to have no tool-call parser attached, so the model's native tool syntax passes through as plain text. The `fp1-rss-g3-herm` profile used for `Qwen3-8B` does parse (though imperfectly — see the `'{}{"city":"Toronto"}'` issue worth tracking separately).

## Request
Please enable a tool-call parser (Hermes/XML, matching Qwen3.5's output format) on the serving profile for `Qwen/Qwen3.5-397B-A17B`.
