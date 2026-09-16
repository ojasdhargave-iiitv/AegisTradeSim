import ollama

response = ollama.chat(
    model='qwen2.5:1.5b',
    messages=[
        {
            'role': 'system',
            'content': 'You are a technical stock market analyst.'
        },
        {
            'role': 'user',
            'content': '''
            NVIDIA:
            Price: 180
            SMA50: 172
            SMA200: 150
            RSI: 58
            MACD: positive

            Analyze the technical situation.
            Separate Evidence, Interpretation, and Risks.
            '''
        }
    ]
)

print(response['message']['content'])