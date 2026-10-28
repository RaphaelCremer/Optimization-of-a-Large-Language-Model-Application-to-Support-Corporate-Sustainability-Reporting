#Benchmarking the SFT model on ESGenius and VSBenchmark
#ESGenius Prompt adapted from He, C, Zhou, X, Wu, Y, Yu, X, Zhang, Y, Zhang, L, Di Wang, Lyu, S, Xu, H, Wang, X, Liu, W & Miao, C 2025, ESGenius: Benchmarking LLMs on Environmental, Social, and Governance (ESG) and Sustainability Knowledge, Available from: https://github.com/ANGEL-NTU/ESGenius [27 October 2026] under Apache License Version 2.0, January 2004 found under /Third_Party_Licenses
import re
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


Model = "/hpcwork/x/qwen_Training_Data_final.json_bs16_lr1e-5_epoch5_wd1e-4_20260829-163512.out"

ESGenius = "Benchmark/ESGenius/ESGenius_1136q.json"
ESGenius_output = "Benchmark/ESGenius/ESGenius_FullSFT.json"

VSBenchmark = "Benchmark/VSBenchmark/VSBenchmark.json"
VSBenchmark_output = "Benchmark/VSBenchmark/VSBenchmark_FullSFT.json"

System_prmpt_ESGenius = (
    "You are answering a multiple-choice ESG benchmark question. "
    "Select the single best option from A, B, C, D, or Z. "
    "Return only the option letter and no explanation."
)

User_prompt_template_ESGenius = """[Question text]
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

#Generation of answers for benchmark questions 
def generate_answers(model, tokenizer, prompts, batch_size, max_new_tokens):
    answers = []

    for start in range(0, len(prompts), batch_size):
        inputs = tokenizer(
            prompts[start:start + batch_size],
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = output[:, inputs["input_ids"].shape[1]:]
        answers.extend(
            tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        )

    return answers 

#Saving benchmarking results
def save_results(results, json_path):
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def main():
    tokenizer = AutoTokenizer.from_pretrained(Model)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        Model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    model.eval()

    with open(ESGenius, "r", encoding="utf-8") as file:
        esgenius_tasks = json.load(file)

    esgenius_prompts = []
    for task in esgenius_tasks:
        user_prompt = User_prompt_template_ESGenius.format(
            question=task["question"],
            option_a=task["options"]["A"],
            option_b=task["options"]["B"],
            option_c=task["options"]["C"],
            option_d=task["options"]["D"],
            option_z=task["options"]["Z"],
        )
        messages = [
            {"role": "system", "content": System_prmpt_ESGenius},
            {"role": "user", "content": user_prompt},
        ]
        esgenius_prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    esgenius_answers = generate_answers(model, tokenizer, esgenius_prompts, batch_size=32, max_new_tokens=16,)

    esgenius_results = []
    for task, answer in zip(esgenius_tasks, esgenius_answers):
        result = dict(task)
        answer = answer.strip()
        match = re.fullmatch(r"(?:OPTION\s+)?([ABCDZ])[\s.)]*", answer.upper())
        result["attempt"] = match.group(1) if match else answer
        esgenius_results.append(result)

    save_results(
        esgenius_results,
        ESGenius_output,
    )

    with open(VSBenchmark, "r", encoding="utf-8") as file:
        vsbenchmark_tasks = json.load(file)

    vsbenchmark_prompts = []
    for task in vsbenchmark_tasks:
        messages = [
            {
                "role": "user",
                "content": task["problem"],
            }
        ]
        vsbenchmark_prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    vsbenchmark_answers = generate_answers(model, tokenizer, vsbenchmark_prompts, batch_size=8, max_new_tokens=1024,)

    vsbenchmark_results = []
    for task, answer in zip(vsbenchmark_tasks, vsbenchmark_answers):
        result = dict(task)
        answer = answer.strip()
        result["attempt"] = answer
        vsbenchmark_results.append(result)

    save_results(vsbenchmark_results, VSBenchmark_output,)


if __name__ == "__main__":
    main()
