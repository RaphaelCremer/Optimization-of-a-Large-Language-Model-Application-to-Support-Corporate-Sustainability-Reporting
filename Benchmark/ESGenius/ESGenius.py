#Benchmark of the Base Model on ESGenius
#ESGenius Prompt adapted from He, C, Zhou, X, Wu, Y, Yu, X, Zhang, Y, Zhang, L, Di Wang, Lyu, S, Xu, H, Wang, X, Liu, W & Miao, C 2025, ESGenius: Benchmarking LLMs on Environmental, Social, and Governance (ESG) and Sustainability Knowledge, Available from: https://github.com/ANGEL-NTU/ESGenius [27 October 2026] under Apache License Version 2.0, January 2004 found under /Third_Party_Licenses

import getpass
import json
import os

from huggingface_hub import InferenceClient

Input = "Benchmark/ESGenius/ESGenius_1136q.json"
Output = "Benchmark/ESGenius/ESGenius_1.json"
Output_fields = ["id", "question", "options", "correct_answer", "ref_page", "ref_doc", "source_text", "new_id", "attempt",]

Model = "Qwen/Qwen2.5-14B-Instruct:featherless-ai"
System_prompt = (
    "You are answering a multiple-choice ESG benchmark question. "
    "Select the single best option from A, B, C, D, or Z. "
    "Return only the option letter and no explanation."
)
User_prompt_template = """[Question text]
{question}

[Option A text]
{option_a}

[Option B text]
{option_b}

[Option C text]
{option_c}

[Option D text]
{option_d}

[Option Z text]
{option_z}
"""

#Build new json structure for output
def prepare_json_results(results):
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

#Build user prompt with question and answer possibilities
    for task in tasks:
        user_prompt = User_prompt_template.format(
            question=task["question"],
            option_a=task["options"]["A"],
            option_b=task["options"]["B"],
            option_c=task["options"]["C"],
            option_d=task["options"]["D"],
            option_z=task["options"]["Z"],
        )
        messages = [
            {
                "role": "system",
                "content": System_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ]

        print(f"Processing task {task['new_id']}")

        completion = client.chat.completions.create(
            model=Model,
            messages=messages,
            max_tokens=16,
            temperature=0,
            top_p=1,
            extra_body={
                "do_sample": False,
                "top_k": 0,
            },
        )
        usage = completion.usage

        if usage is not None:
            print(f"Input tokens: {usage.prompt_tokens}")
            print(f"Output tokens: {usage.completion_tokens}")
            print(f"Total tokens: {usage.total_tokens}")

        attempt = completion.choices[0].message.content

        result = dict(task)
        result["attempt"] = attempt
        results.append(result)
        with open(Output, "w", encoding="utf-8") as file:
            json.dump(
                prepare_json_results(results),
                file,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
