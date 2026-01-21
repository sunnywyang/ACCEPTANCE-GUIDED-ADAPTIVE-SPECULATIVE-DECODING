file_path = "/qa/ess-llama-2-chat-70b-fp16-temperature-1.0.jsonl"

import json
new_tokens_sum = 0
idxs_sum = 0

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 解析JSON数据
        data = json.loads(line.strip())
        if "choices" in data:
            for choice in data["choices"]:
                new_tokens_sum += sum(choice.get("new_tokens", []))
                idxs_sum += sum(choice.get("idxs", []))

result = new_tokens_sum / idxs_sum if idxs_sum != 0 else None

print(f"New Tokens Sum: {new_tokens_sum}")
print(f"Idxs Sum: {idxs_sum}")
print(f"Result (New Tokens / Idxs): {result}")