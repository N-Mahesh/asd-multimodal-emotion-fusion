from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="OPENAI_ROUTER_API_KEY",
)

completion = client.chat.completions.create(
  model="openai/gpt-4.1",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)

print(completion.choices[0].message.content)
