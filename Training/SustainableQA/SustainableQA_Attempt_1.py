#Generation of answers to the SustainableQA questions

import getpass
import json
import os

from huggingface_hub import InferenceClient


Model = "Qwen/Qwen2.5-14B-Instruct:featherless-ai"
Input = "Training/SustainableQA/SustainableQA_Selection_1.json"
Output = "Training/SustainableQA/SustainableQA_Attempt_1.json"

#Load the preselected questions
def load_tasks():
    with open(Input, "r", encoding="utf-8") as file:
        preselection = json.load(file)
    questions = preselection["selected_questions"]
    return preselection, questions

#Safe attempt as json
def save_json(preselection, results):
    output = dict(preselection)
    output["selected_questions"] = results

    with open(Output, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        token = getpass.getpass("Enter Hugging Face token: ")

    client = InferenceClient(api_key=token)
    preselection, tasks = load_tasks()
    results = []

    for task in tasks:
        task_id = task["question_id"]
        problem = task["question"]

        # Only the question is sent to the model.
        messages = [
            {
                "role": "user",
                "content": problem,
            }
        ]

        print(f"Processing task {task_id} ...")

        try:
            completion = client.chat.completions.create(
                model=Model,
                messages=messages,
                max_tokens=1024,
                temperature=0,
            )
            usage = completion.usage

            print(f"Input tokens: {usage.prompt_tokens}")
            print(f"Output tokens: {usage.completion_tokens}")
            print(f"Total tokens: {usage.total_tokens}")

            attempt = completion.choices[0].message.content or ""
        except Exception:
            attempt = None

        result = dict(task)
        result.update(
            {
                "id": task_id,
                "problem": problem,
                "solution": task["answer"],
                "attempt": attempt,
            }
        )
        results.append(result)
        save_json(preselection, results)

    print(f"\nJSON result: {Output}")


if __name__ == "__main__":
    main()
