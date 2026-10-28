#Benchmarking the reasoning SFT model on ESGenius and VSBenchmark
#Adapted and modified from Muennighoff, N, Yang, Z, Shi, W, Li, XL, Fei-Fei, L, Hajishirzi, H, Zettlemoyer, L, Liang, P, Candès, E & Hashimoto, T 2025, s1: Simple test-time scaling. Available from: https://github.com/simplescaling/s1 [19. September 2026] under Apache License Version 2.0, January 2004 found under /Third_Party_Licenses
#ESGenius Prompt from He, C, Zhou, X, Wu, Y, Yu, X, Zhang, Y, Zhang, L, Di Wang, Lyu, S, Xu, H, Wang, X, Liu, W & Miao, C 2025, ESGenius: Benchmarking LLMs on Environmental, Social, and Governance (ESG) and Sustainability Knowledge, Available from: https://github.com/ANGEL-NTU/ESGenius [27 October 2026] under Apache License Version 2.0, January 2004 found under /Third_Party_Licenses

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import re

def save_results(results, json_path):
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

Model = "/hpcwork/x/qwen_Training_Data_final_reasoning_trace.json_bs16_lr1e-5_epoch5_wd1e-4_20260905-140040.out"

ESGenius = "Benchmark/ESGenius/ESGenius_1136q.json"
ESGenius_output = "Benchmark/ESGenius/ESGenius_BFSFT.json"

VSBenchmark = "Benchmark/VSBenchmark/VSBenchmark.json"
VSBenchmark_output = "Benchmark/VSBenchmark/VSBenchmark_BFSFT.json"

#Assign model path, gpu and tokenizer
model = LLM(
    Model,
    tensor_parallel_size=1, 
    enable_prefix_caching=True,
    gpu_memory_utilization=0.90,
)
tok = AutoTokenizer.from_pretrained(
    Model
)

#Get Benchmark Data
with open(ESGenius, "r", encoding="utf-8") as file:
    esgenius_tasks = json.load(file)

with open(VSBenchmark, "r", encoding="utf-8") as file:
    vsbenchmark_tasks = json.load(file)

#ESGenius Prompt
System_prompt_ESGenius = (
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
esgenius_prompts = []
esgenius_results = []

for task in esgenius_tasks:
    user_prompt = User_prompt_template_ESGenius.format(
        question=task["question"],
        option_a=task["options"]["A"],
        option_b=task["options"]["B"],
        option_c=task["options"]["C"],
        option_d=task["options"]["D"],
        option_z=task["options"]["Z"],
    )
    esgenius_prompts.append(user_prompt)
    
#Generation of reasoning trace
esgenius_model_prompts = []
for task, p in zip(esgenius_tasks, esgenius_prompts):
    prompt = "<|im_start|>system\n" + System_prompt_ESGenius + "<|im_end|>\n<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"
    prompt += "<|im_start|>think"
    esgenius_model_prompts.append(prompt)

stop_token_ids = tok("<|im_start|><|im_end|>")["input_ids"]
sampling_params = SamplingParams(
    max_tokens=2048,
    min_tokens=0,
    stop_token_ids=stop_token_ids,
    skip_special_tokens=False,
    temperature=0.0,
)
o = model.generate(
    esgenius_model_prompts,
    sampling_params=sampling_params,
)

esgenius_reasoning = [
    output.outputs[0].text
    for output in o
]

### Final answer ###
esgenius_model_prompts = [
    prompt
    + output.outputs[0].text
    + (
        "\n**Final Answer**\n<|im_start|>answer\n"
        if output.outputs[0].finish_reason == "length"
        else ""
    )
    for prompt, output in zip(esgenius_model_prompts, o)
]

stop_token_ids = tok("<|im_end|>")["input_ids"]
sampling_params = SamplingParams(
    max_tokens=16,
    min_tokens=0,
    stop_token_ids=stop_token_ids,
    skip_special_tokens=False,
    temperature=0.0,
)
o = model.generate(
    esgenius_model_prompts,
    sampling_params=sampling_params,
)

#Extract an answer option for different answering formats
for task, output, reasoning in zip(esgenius_tasks, o, esgenius_reasoning):
    final_output = output.outputs[0].text.strip()

    cleaned_output = re.sub(r"^\s*<\|im_start\|>answer\s*","",final_output,flags=re.IGNORECASE,)
    cleaned_output = re.sub(r"\s*<\|im_end\|>\s*$","",cleaned_output,flags=re.IGNORECASE,).strip()
    match = re.fullmatch(r"(?:OPTION\s+)?([ABCDZ])[\s.)]*",cleaned_output,flags=re.IGNORECASE,)

    result = dict(task)
    result["attempt"] = (
        match.group(1).upper()
        if match
        else final_output
    )
    result["reasoning"] = reasoning
    esgenius_results.append(result)

save_results(
    esgenius_results,
    ESGenius_output,
)

#VSBenchmark

vsbenchmark_prompts = []
vsbenchmark_results = []

for task in vsbenchmark_tasks:
    vsbenchmark_prompts.append(task["problem"])

vsbenchmark_model_prompts = []
for task, p in zip(vsbenchmark_tasks, vsbenchmark_prompts):
    prompt = (
        "<|im_start|>user\n"
        + p
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )
    prompt += "<|im_start|>think"
    vsbenchmark_model_prompts.append(prompt)

# Start reasoning
stop_token_ids = tok("<|im_start|><|im_end|>")["input_ids"]
sampling_params = SamplingParams(
    max_tokens=2048,
    min_tokens=0,
    stop_token_ids=stop_token_ids,
    skip_special_tokens=False,
    temperature=0.0,
)
o = model.generate(
    vsbenchmark_model_prompts,
    sampling_params=sampling_params,
)

vsbenchmark_reasoning = [
    output.outputs[0].text
    for output in o
]

### Final answer ###
vsbenchmark_model_prompts = [
    prompt
    + output.outputs[0].text
    + (
        "\n**Final Answer**\n<|im_start|>answer\n"
        if output.outputs[0].finish_reason == "length"
        else ""
    )
    for prompt, output in zip(vsbenchmark_model_prompts, o)
]
stop_token_ids = tok("<|im_end|>")["input_ids"]
sampling_params = SamplingParams(
    max_tokens=1024,
    min_tokens=0,
    stop_token_ids=stop_token_ids,
    skip_special_tokens=False,
    temperature=0.0,
)
o = model.generate(
    vsbenchmark_model_prompts,
    sampling_params=sampling_params,
)

for task, output, reasoning in zip(vsbenchmark_tasks, o, vsbenchmark_reasoning):
    final_output = output.outputs[0].text.strip()

    # Remove optional Qwen answer delimiters
    cleaned_output = re.sub(r"^\s*<\|im_start\|>answer\s*","",final_output,flags=re.IGNORECASE,)

    cleaned_output = re.sub(r"\s*<\|im_end\|>\s*$","",cleaned_output,flags=re.IGNORECASE,).strip()

    result = dict(task)
    result["attempt"] = cleaned_output
    result["reasoning"] = reasoning
    vsbenchmark_results.append(result)


save_results(
    vsbenchmark_results,
    VSBenchmark_output,
)
