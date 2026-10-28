#Generates synthetic reasoning traces supplementing the training data with Gemma4 

import getpass
import json
import os

from huggingface_hub import InferenceClient


Model = "google/gemma-4-26B-A4B-it:scaleway"
Input = "Training/Training_Data_final.json"
Output = "Training/Training_Data_reasoning_trace.json"

#Generate synthetic reasoning traces
def generate_trace(client, record):
    completion = client.chat.completions.create(
        model=Model,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are an AI tutor demonstrating how difficult ESG problems "
                    "can be solved through an English inner monologue. Use only the "
                    "problem. The reference solution is reserved for a separate "
                    "evaluator and must not be visible during generation. First "
                    "clarify the question, ambiguous terms, and what a correct "
                    "solution must explain. Generate several genuinely different "
                    "solution paths instead of starting from a predetermined list. "
                    "Develop and test them using definitions, causal consequences, "
                    "constraints, and counterexamples. Include plausible wrong "
                    "hypotheses, genuine doubt, interruptions, and changes of "
                    "direction. Reject each wrong path for a concrete reason. Every "
                    "reconsideration must introduce new information or improve the "
                    "hypothesis; avoid repetitive \"Wait\" or \"Maybe\" loops. Do not "
                    "select ideas because they sound professional, and do not invent "
                    "company-specific facts. Distinguish objectives, actors, "
                    "mechanisms, procedures, and outputs. End after identifying and "
                    "stress-testing the strongest solution direction, but do not "
                    "state the final answer. Output only continuous plain text "
                    "without an introduction, headings, lists, speaker labels, or "
                    "quotation marks. Never exceed 12,000 characters.\n\n"
                    f"Problem:\n{record['problem']}\n\n"
                    f"Reference solution:\n{record['solution']}"
                ),
            }
        ],
        max_tokens=8000,
        temperature=1.0,
        extra_body={"enable_thinking": False},
    )
    reasoning_trace = completion.choices[0].message.content
    usage = completion.usage
    return reasoning_trace.strip(), usage.prompt_tokens, usage.completion_tokens


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        token = getpass.getpass("Enter Hugging Face token: ")

    with open(Input, "r", encoding="utf-8") as file:
        records = json.load(file)

    for record in records:
        text = record.pop("text")
        record["reasoning_trace"] = ""
        record["text"] = text

    client = InferenceClient(api_key=token)

    for record in records:
        reasoning_trace, input_tokens, output_tokens = generate_trace(
            client,
            record,
        )
        record["reasoning_trace"] = reasoning_trace
        print(
            f"{record['id']}: "
            f"input tokens={input_tokens}, output tokens={output_tokens}",
            flush=True,
        )

        with open(Output, "w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)
        print(
            f"{sum(bool(record['reasoning_trace']) for record in records)}/{len(records)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
