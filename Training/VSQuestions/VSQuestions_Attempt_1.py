#Generation of answers to the VSQuestions
import getpass
import json
import os

from huggingface_hub import InferenceClient


Model = "Qwen/Qwen2.5-14B-Instruct:featherless-ai"
Input = "Training/VSQuestions/VSQuestions_Selection_1.json"
Output = "Training/VSQuestions/VSQuestions_Attempt_1.json"
Output_fields = ["id", "problem", "solution", "attempt", "question_type", "category"]

#Load the preselected questions
def load_tasks():
    with open(Input, "r", encoding="utf-8") as file:
        preselection = json.load(file)
    return preselection["selected_questions"]

#Build new json structure for output
def prepare_results_for_output(results):
    return [
        {field: result[field] for field in Output_fields}
        for result in results
    ]

#Save attempt as json
def save_json(results):
    with open(Output, "w", encoding="utf-8") as file:
        json.dump(
            prepare_results_for_output(results),
            file,
            ensure_ascii=False,
            indent=2,
        )


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        token = getpass.getpass("Enter Hugging Face token: ")

    client = InferenceClient(api_key=token)
    tasks = load_tasks()
    results = []

    for task in tasks:
        task_id = task["id"]
        problem = task["problem"]

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

        results.append(
            {
                "id": task_id,
                "problem": problem,
                "solution": task["solution"],
                "attempt": attempt,
                "question_type": task["question_type"],
                "category": task["category"],
            }
        )
        save_json(results)


if __name__ == "__main__":
    main()
