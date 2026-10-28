#Combines VSQuestions and SustainableQA to the final training dataset

import json

Inputs = {
    "VSQuestions": "Training/VSQuestions/VSQuestions_Selection_2.json",
    "SustainableQA": "Training/SustainableQA/SustainableQA_Selection_2.json",
}
Output = "Training/Training_Data_final.json"

with open(Inputs["VSQuestions"], "r", encoding="utf-8") as file:
    VSQuestions = json.load(file)

with open(Inputs["SustainableQA"], "r", encoding="utf-8") as file:
    SustainableQA = json.load(file)["selected_questions"]

questions = []

for element in (SustainableQA, VSQuestions):
    for data in element:
        questions.append({
            "id": data["id"],
            "problem": data["problem"],
            "attempt": data["attempt"],
            "solution": data["solution"],
            "question_type": data["question_type"],
            "category": data["category"],
            "text":f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{data["problem"]} <|im_end|>\n<|im_start|>assistant\n{data["solution"]}<|im_end|>",
        })

with open(Output, "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)
