#Adds reasoning traces to training data

import json

Input = "Training/Training_Data_reasoning_trace.json"
Output = "Training/Training_Data_final_reasoning_trace.json"

with open(Input, "r", encoding="utf-8") as file:
    reasoning_traces = json.load(file)

questions = []

for element in (reasoning_traces,):
    for data in element:
        questions.append({
            "id": data["id"],
            "problem": data["problem"],
            "attempt": data["attempt"],
            "solution": data["solution"],
            "question_type": data["question_type"],
            "category": data["category"],
            "reasoning_trace": data["reasoning_trace"],
            "text":f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{data["problem"]} <|im_end|>\n<|im_start|>assistant\n<|im_start|>think\n{data["reasoning_trace"]}\n**Final Answer**\n<|im_start|>answer\n{data["solution"]}<|im_end|>",
        })

with open(Output, "w", encoding="utf-8") as file:
    json.dump(questions, file, ensure_ascii=False, indent=2)
