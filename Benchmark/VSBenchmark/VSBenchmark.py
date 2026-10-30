#Benchmark of the Base Model on VSBenchmark
import getpass
import json
import os

from huggingface_hub import InferenceClient


Model = "Qwen/Qwen2.5-14B-Instruct:featherless-ai"
Input = "Benchmark/VSBenchmark/VSBenchmark.json"
Output = "Benchmark/VSBenchmark/VSBenchmark_1.json"
Output_fields = ["id", "use case", "problem", "solution", "attempt"]

#Build new json structure for output
def prepare_results_for_output(results):
    return [
        {field: result[field] for field in Output_fields}
        for result in results
    ]


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        token = getpass.getpass("Enter Hugging Face token: ")

    client = InferenceClient(api_key=token)
    with open(Input, "r", encoding="utf-8") as file:
        tasks = json.load(file)
    results = []

    for position, task in enumerate(tasks, start=1):
        task_id = task.get("id", str(position))
        problem = task.get("problem", "")

        messages = [
            {
                "role": "user",
                "content": problem,
            }
        ]

        print(f"Processing task {task_id}")

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

        attempt = completion.choices[0].message.content

        result = dict(task)
        result["attempt"] = attempt
        results.append(result)
        with open(Output, "w", encoding="utf-8") as file:
            json.dump(
                prepare_results_for_output(results),
                file,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
